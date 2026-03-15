"""WebSocket server — bridges Electron frontend to Browser/Desktop agent.

Supports:
- Browser mode (Playwright via Gemini Computer Use)
- Desktop mode (proxied through clawd-cursor REST API)
- Voice input via Google Cloud STT
"""
import asyncio
import json
import base64
import os
import threading
import time
import queue
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import websockets

from agent import BrowserAgent
from computers import PlaywrightComputer, EnvState
from voice_input import VoiceTranscriber
from clawd_bridge import ClawdBridge
from google.genai.types import (
    FinishReason, FunctionResponse, Content, Part,
)
from google.genai import types

PLAYWRIGHT_SCREEN_SIZE = (1440, 900)
LIVE_FEED_FPS = 5


class AgentSession:
    """Manages agent lifecycle for both browser and desktop modes."""

    def __init__(self, ws, loop):
        self._ws = ws
        self._loop = loop
        self._voice = VoiceTranscriber()
        self._clawd = ClawdBridge()
        self._closed = False
        self._agent_running = False
        self._cmd_queue = queue.Queue()
        self._playwright_env = None

        # Worker thread for agent execution
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        print(f"[SESSION] Created")

    def _send(self, data):
        """Thread-safe send over async websocket."""
        try:
            asyncio.run_coroutine_threadsafe(
                self._ws.send(json.dumps(data)), self._loop
            )
        except Exception as e:
            print(f"[WS ERROR] {e}")

    def _worker_loop(self):
        """Processes commands from the queue."""
        print("[WORKER] Started")
        while not self._closed:
            try:
                cmd = self._cmd_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            action = cmd.get("action")
            print(f"[WORKER] Command: {action}")

            if action == "run_agent":
                self._run_agent(cmd["query"], cmd["model"], cmd.get("mode", "browser"))
            elif action == "shutdown":
                break

        print("[WORKER] Ended")

    def _run_agent(self, query: str, model: str, mode: str = "browser"):
        """Run the agent loop in the appropriate mode."""
        if self._agent_running:
            self._send({"type": "error", "message": "Agent already running"})
            return

        self._agent_running = True

        if mode == "desktop":
            self._run_desktop_agent(query)
        else:
            self._run_browser_agent(query, model)

        self._agent_running = False

    def _run_browser_agent(self, query: str, model: str):
        """Run Playwright-based browser agent with Gemini Computer Use."""
        self._send({"type": "thinking", "text": f"Starting browser agent with `{model}`..."})

        try:
            env = PlaywrightComputer(
                screen_size=PLAYWRIGHT_SCREEN_SIZE,
                initial_url="https://www.google.com",
            )
            with env as browser_computer:
                self._playwright_env = browser_computer
                agent = FrontendBrowserAgent(
                    browser_computer=browser_computer,
                    query=query,
                    model_name=model,
                    ws_send=self._send,
                )
                agent.agent_loop()
                result = agent.final_reasoning or "Task completed."
                self._send({"type": "complete", "result": result})
        except Exception as e:
            print(f"[AGENT] Browser error: {e}")
            import traceback
            traceback.print_exc()
            self._send({"type": "error", "message": str(e)})
        finally:
            self._playwright_env = None

    def _run_desktop_agent(self, query: str):
        """Run desktop agent via clawd-cursor REST API."""
        self._send({"type": "thinking", "text": "Sending task to clawd-cursor desktop agent..."})

        if not self._clawd.is_available():
            self._send({"type": "error", "message": "clawd-cursor is not running. Start it with: clawdcursor start"})
            return

        try:
            # Submit task to clawd-cursor
            result = self._clawd.submit_task(query)
            if not result.get("accepted"):
                self._send({"type": "error", "message": result.get("error", "Task rejected by clawd-cursor")})
                return

            self._send({"type": "thinking", "text": "Task accepted by clawd-cursor. Monitoring progress..."})

            # Poll status and stream screenshots
            iteration = 0
            while not self._closed:
                time.sleep(1)
                status = self._clawd.get_status()
                agent_status = status.get("status", "idle")

                # Stream screenshot
                screenshot_b64 = self._clawd.get_screenshot()
                if screenshot_b64:
                    self._send({
                        "type": "live_frame",
                        "image": screenshot_b64,
                        "url": status.get("currentStep", "Desktop"),
                    })

                # Update iteration
                steps_done = status.get("stepsCompleted", 0)
                if steps_done > iteration:
                    iteration = steps_done
                    self._send({"type": "iteration", "count": iteration})
                    step_desc = status.get("currentStep", "")
                    if step_desc:
                        self._send({"type": "action", "name": "desktop_action", "args": {"step": step_desc}})

                # Check if waiting for confirmation
                if agent_status == "waiting_confirm":
                    self._send({"type": "thinking", "text": "⚠️ clawd-cursor is waiting for confirmation. Auto-approving..."})
                    self._clawd.confirm(True)

                # Check if done
                if agent_status == "idle" and iteration > 0:
                    self._send({"type": "complete", "result": "Desktop task completed via clawd-cursor."})
                    break

                if agent_status not in ("idle", "thinking", "acting", "waiting_confirm", "paused"):
                    self._send({"type": "error", "message": f"Unexpected agent status: {agent_status}"})
                    break

        except Exception as e:
            print(f"[AGENT] Desktop error: {e}")
            import traceback
            traceback.print_exc()
            self._send({"type": "error", "message": str(e)})

    def start_agent(self, query: str, model: str, mode: str = "browser"):
        """Queue agent run."""
        self._cmd_queue.put({"action": "run_agent", "query": query, "model": model, "mode": mode})

    def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio data to text."""
        if not self._voice.available:
            return ""
        return self._voice.transcribe_audio(audio_data)

    def close(self):
        """Shutdown the session."""
        self._closed = True
        self._cmd_queue.put({"action": "shutdown"})
        print("[SESSION] Closed")


class FrontendBrowserAgent(BrowserAgent):
    """Extends BrowserAgent to stream reasoning/actions to the Electron frontend."""

    def __init__(self, browser_computer, query, model_name, ws_send):
        super().__init__(
            browser_computer=browser_computer,
            query=query,
            model_name=model_name,
            verbose=False,
        )
        self._ws_send = ws_send
        self._iteration = 0

    def run_one_iteration(self):
        self._iteration += 1
        self._ws_send({"type": "iteration", "count": self._iteration})
        print(f"[AGENT] Iteration {self._iteration}")

        try:
            response = self.get_model_response()
        except Exception as e:
            print(f"[AGENT] Model error: {e}")
            self._ws_send({"type": "error", "message": str(e)})
            return "COMPLETE"

        if not response.candidates:
            self._ws_send({"type": "error", "message": "Empty response from model"})
            return "COMPLETE"

        candidate = response.candidates[0]
        if candidate.content:
            self._contents.append(candidate.content)

        reasoning = self.get_text(candidate)
        function_calls = self.extract_function_calls(candidate)

        # Send reasoning to frontend
        if reasoning:
            self._ws_send({"type": "thinking", "text": reasoning})

        if (
            not function_calls
            and not reasoning
            and candidate.finish_reason == FinishReason.MALFORMED_FUNCTION_CALL
        ):
            return "CONTINUE"

        if not function_calls:
            self.final_reasoning = reasoning
            return "COMPLETE"

        # Execute actions and send to frontend
        function_responses = []
        for fc in function_calls:
            args_dict = dict(fc.args) if fc.args else {}
            self._ws_send({"type": "action", "name": fc.name, "args": args_dict})

            try:
                fc_result = self.handle_action(fc)
            except Exception as e:
                print(f"[AGENT] Action error ({fc.name}): {e}")
                # On error, still get current state so Gemini has a screenshot
                try:
                    fc_result = self._browser_computer.current_state()
                except Exception:
                    fc_result = {"error": str(e)}

            if isinstance(fc_result, EnvState):
                # Stream the screenshot to frontend
                screenshot_b64 = base64.b64encode(fc_result.screenshot).decode("utf-8")
                self._ws_send({
                    "type": "live_frame",
                    "image": screenshot_b64,
                    "url": fc_result.url,
                })

                function_responses.append(
                    Part(function_response=FunctionResponse(
                        name=fc.name,
                        response={"url": fc_result.url},
                    ))
                )
                function_responses.append(
                    Part(inline_data=types.Blob(
                        mime_type="image/png",
                        data=fc_result.screenshot,
                    ))
                )
            else:
                function_responses.append(
                    Part(function_response=FunctionResponse(
                        name=fc.name,
                        response=fc_result if isinstance(fc_result, dict) else {"result": str(fc_result)},
                    ))
                )

        self._contents.append(Content(role="user", parts=function_responses))

        # Prune old screenshots to keep context manageable
        self._prune_screenshots()
        return "CONTINUE"

    def _prune_screenshots(self):
        """Remove old screenshots, keeping only the most recent ones."""
        MAX_RECENT = 3
        screenshot_indices = []
        for i in range(len(self._contents) - 1, -1, -1):
            content = self._contents[i]
            if not content.parts:
                continue
            for j in range(len(content.parts) - 1, -1, -1):
                p = content.parts[j]
                if hasattr(p, 'inline_data') and p.inline_data:
                    screenshot_indices.append((i, j))

        to_remove = screenshot_indices[MAX_RECENT:]
        for content_idx, part_idx in sorted(to_remove, reverse=True):
            content = self._contents[content_idx]
            parts_list = list(content.parts)
            parts_list.pop(part_idx)
            self._contents[content_idx] = Content(
                role=content.role,
                parts=parts_list if parts_list else [Part(text="[screenshot removed]")]
            )

        if to_remove:
            print(f"[AGENT] Pruned {len(to_remove)} old screenshots")


# ── WebSocket handler ──

async def handler(websocket):
    print(f"[WS] Client connected")
    loop = asyncio.get_event_loop()
    session = AgentSession(websocket, loop)

    try:
        async for message in websocket:
            # Binary = audio data
            if isinstance(message, bytes):
                print(f"[WS] Received audio: {len(message)} bytes")
                transcript = session.transcribe_audio(message)
                if transcript:
                    await websocket.send(json.dumps({
                        "type": "transcription",
                        "text": transcript,
                    }))
                else:
                    await websocket.send(json.dumps({
                        "type": "transcription",
                        "text": "",
                        "error": "Could not transcribe audio",
                    }))
                continue

            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "start":
                query = data.get("query", "")
                model = data.get("model", "gemini-3-flash-preview")
                mode = data.get("mode", "browser")
                print(f"[WS] Start: mode={mode}, model={model}, query={query[:60]}")
                session.start_agent(query, model, mode)

            elif msg_type == "voice_data":
                audio_b64 = data.get("audio", "")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    transcript = session.transcribe_audio(audio_bytes)
                    await websocket.send(json.dumps({
                        "type": "transcription",
                        "text": transcript,
                    }))

    except websockets.exceptions.ConnectionClosed:
        print("[WS] Client disconnected")
    finally:
        session.close()


async def main():
    print("=" * 60)
    print("  Gemini AI Agent Server")
    print("  WebSocket: ws://localhost:8765")
    print("  Browser mode: Playwright + Gemini Computer Use")
    print("  Desktop mode: clawd-cursor API bridge")
    print("=" * 60)
    async with websockets.serve(handler, "localhost", 8765, max_size=50 * 1024 * 1024):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
