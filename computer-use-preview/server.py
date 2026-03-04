"""WebSocket server — bridges Electron frontend to Desktop/Browser agent.

Supports:
- Full desktop control mode (pyautogui + Win32)
- Browser-only mode (Playwright, legacy)
- Voice input via Google Cloud STT
- Session memory via Firestore
"""
import asyncio
import json
import base64
import os
import threading
import time
import queue
import io
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

import websockets
from PIL import Image

from agent import BrowserAgent
from desktop_agent import DesktopAgent
from computers import PlaywrightComputer, DesktopComputer, EnvState
from screen_capture import WindowCapture
from memory import SessionMemory
from voice_input import VoiceTranscriber
from google.genai.types import (
    FinishReason, FunctionResponse, Content, Part,
)
from google.genai import types

SCREEN_SIZE = (1920, 1200)
LIVE_FEED_FPS = 10


class DesktopSession:
    """Manages desktop agent lifecycle — captures full screen, runs agent."""

    def __init__(self, ws, loop):
        self._ws = ws
        self._loop = loop
        self._desktop: DesktopComputer | None = None
        self._memory = SessionMemory()
        self._voice = VoiceTranscriber()
        self._closed = False
        self._agent_running = False
        self._cmd_queue = queue.Queue()

        # Start threads
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self._feed_thread = threading.Thread(target=self._feed_loop, daemon=True)
        self._feed_thread.start()
        print(f"[SESSION] Created, memory session: {self._memory.session_id}")

    def _send(self, data):
        """Thread-safe send over async websocket."""
        try:
            asyncio.run_coroutine_threadsafe(
                self._ws.send(json.dumps(data)), self._loop
            )
        except Exception as e:
            print(f"[WS ERROR] {e}")

    def _feed_loop(self):
        """Captures desktop screen and streams frames to frontend."""
        interval = 1.0 / LIVE_FEED_FPS
        frame_count = 0
        target_w, target_h = SCREEN_SIZE

        while not self._closed:
            if self._desktop is not None:
                try:
                    raw = self._desktop._take_screenshot()
                    if raw:
                        b64 = base64.b64encode(raw).decode("utf-8")
                        # Get current window title
                        title = "Desktop"
                        if self._desktop.focused_hwnd:
                            try:
                                import ctypes
                                hwnd = self._desktop.focused_hwnd
                                length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                                buf = ctypes.create_unicode_buffer(length + 1)
                                ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                                title = buf.value or "Desktop"
                            except Exception:
                                pass
                        self._send({
                            "type": "live_frame",
                            "image": b64,
                            "url": title,
                        })
                        frame_count += 1
                        if frame_count == 1:
                            print(f"[FEED] First frame sent ({len(raw)} bytes)")
                        if frame_count % 200 == 0:
                            print(f"[FEED] {frame_count} frames sent")
                except Exception as e:
                    if frame_count == 0:
                        print(f"[FEED] Capture error: {e}")
            time.sleep(interval)
        print("[FEED] Loop ended")

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

            if action == "init_desktop":
                self._init_desktop()
            elif action == "run_agent":
                self._run_agent(cmd["query"], cmd["model"], cmd.get("mode", "desktop"))
            elif action == "shutdown":
                break

        print("[WORKER] Ended")

    def _init_desktop(self):
        """Initialize the desktop computer interface."""
        if self._desktop is not None:
            return
        self._send({"type": "thinking", "text": "Initializing desktop control (background mode)..."})
        try:
            self._desktop = DesktopComputer(
                screen_size=SCREEN_SIZE,
                capture_mode="desktop",
                background_mode=True,  # Agent interacts without stealing focus
            )
            self._send({"type": "thinking", "text": "Desktop control ready — background mode active. You can keep working while the agent operates."})
            print("[DESKTOP] Initialized (background mode)")
        except Exception as e:
            print(f"[DESKTOP] Init failed: {e}")
            self._send({"type": "error", "message": f"Desktop init failed: {e}"})

    def _run_agent(self, query: str, model: str, mode: str = "desktop"):
        """Run the agent loop."""
        if self._agent_running:
            self._send({"type": "error", "message": "Agent already running"})
            return

        self._agent_running = True

        if mode == "desktop":
            if self._desktop is None:
                self._init_desktop()
            if self._desktop is None:
                self._send({"type": "error", "message": "Desktop not available"})
                self._agent_running = False
                return

            self._send({"type": "thinking", "text": f"Starting desktop agent with `{model}`..."})
            try:
                agent = FrontendDesktopAgent(
                    desktop=self._desktop,
                    query=query,
                    model_name=model,
                    memory=self._memory,
                    ws_send=self._send,
                )
                agent.agent_loop()
                result = agent.final_reasoning or "Task completed."
                self._send({"type": "complete", "result": result})
            except Exception as e:
                print(f"[AGENT] Error: {e}")
                import traceback
                traceback.print_exc()
                self._send({"type": "error", "message": str(e)})
        else:
            # Legacy browser mode
            self._send({"type": "error", "message": "Browser mode deprecated. Use desktop mode."})

        self._agent_running = False

    def start_agent(self, query: str, model: str, mode: str = "desktop"):
        """Queue agent run."""
        self._cmd_queue.put({"action": "init_desktop"})
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


class FrontendDesktopAgent(DesktopAgent):
    """Extends DesktopAgent to stream reasoning/actions to the frontend."""

    def __init__(self, desktop, query, model_name, memory, ws_send):
        super().__init__(
            desktop=desktop,
            query=query,
            model_name=model_name,
            memory=memory,
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

        # Send reasoning
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

        # Execute actions
        function_responses = []
        for fc in function_calls:
            args_dict = dict(fc.args) if fc.args else {}
            self._ws_send({"type": "action", "name": fc.name, "args": args_dict})

            # Auto-acknowledge safety
            if fc.name == "acknowledge_safety_decision":
                function_responses.append(
                    Part(function_response=FunctionResponse(
                        name=fc.name,
                        response={"decision": args_dict.get("safety_decision", "ACCEPT")},
                    ))
                )
                continue

            try:
                result = self.handle_action(fc)
            except Exception as e:
                print(f"[AGENT] Action error ({fc.name}): {e}")
                result = {"error": str(e)}

            if isinstance(result, EnvState):
                function_responses.append(
                    Part(function_response=FunctionResponse(
                        name=fc.name,
                        response={"url": result.url},
                    ))
                )
                function_responses.append(
                    Part(inline_data=types.Blob(
                        mime_type="image/jpeg",
                        data=result.screenshot,
                    ))
                )
            else:
                function_responses.append(
                    Part(function_response=FunctionResponse(
                        name=fc.name,
                        response=result if isinstance(result, dict) else {"result": str(result)},
                    ))
                )

        self._contents.append(Content(role="user", parts=function_responses))
        self._prune_screenshots()
        return "CONTINUE"


# ── WebSocket handler ──

async def handler(websocket):
    print(f"[WS] Client connected")
    loop = asyncio.get_event_loop()
    session = DesktopSession(websocket, loop)

    try:
        async for message in websocket:
            # Check if binary (audio data)
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
                mode = data.get("mode", "desktop")
                print(f"[WS] Start: mode={mode}, model={model}, query={query[:60]}")
                session.start_agent(query, model, mode)

            elif msg_type == "voice_data":
                # Base64 encoded audio
                audio_b64 = data.get("audio", "")
                if audio_b64:
                    audio_bytes = base64.b64decode(audio_b64)
                    transcript = session.transcribe_audio(audio_bytes)
                    await websocket.send(json.dumps({
                        "type": "transcription",
                        "text": transcript,
                    }))

            elif msg_type == "list_apps":
                if session._desktop:
                    apps = session._desktop.list_open_apps()
                    await websocket.send(json.dumps({
                        "type": "app_list",
                        "apps": [{"title": a["title"], "process": a["process"]} for a in apps],
                    }))

            elif msg_type == "close_browser":
                # Legacy compat
                pass

    except websockets.exceptions.ConnectionClosed:
        print("[WS] Client disconnected")
    finally:
        session.close()


async def main():
    print("=" * 60)
    print("  Desktop AI Agent Server")
    print("  WebSocket: ws://localhost:8765")
    print("=" * 60)
    async with websockets.serve(handler, "localhost", 8765, max_size=50 * 1024 * 1024):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())


