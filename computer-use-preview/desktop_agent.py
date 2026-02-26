"""Desktop Agent — extends BrowserAgent for full OS control.

Uses Gemini's computer_use tool in ENVIRONMENT_DESKTOP mode with
additional custom functions for app management.
"""
import os
import time
from typing import Literal, Optional, Union

from google import genai
from google.genai import types
from google.genai.types import (
    Part, GenerateContentConfig, Content, Candidate,
    FunctionResponse, FinishReason,
)

from computers import EnvState
from computers.desktop.desktop import DesktopComputer
from memory import SessionMemory

MAX_RECENT_TURN_WITH_SCREENSHOTS = 3

FunctionResponseT = Union[EnvState, dict]


# ── Custom function declarations ──

def open_app(app_name: str) -> dict:
    """Opens an application on the computer by name. If the app is already running, it focuses the existing window instead of opening a new one.
    Common app names: brave, chrome, firefox, calculator, notepad, explorer, terminal, vscode, spotify, discord.
    You can also use any executable name or app name installed on the system."""
    return {"app_name": app_name}


def close_app(app_name: str) -> dict:
    """Closes an application by name."""
    return {"app_name": app_name}


def switch_to_app(app_name: str) -> dict:
    """Switches focus to an already running application. Use this instead of open_app when you know the app is already open."""
    return {"app_name": app_name}


def list_open_apps() -> dict:
    """Lists all currently open/visible applications and windows on the desktop. Use this to check what's running before opening new apps."""
    return {}


def run_shell_command(command: str) -> dict:
    """Runs a shell command on the system and returns the output. Use for file operations, system info, etc."""
    return {"command": command}


def calculator_compute(expression: str) -> dict:
    """Presses calculator buttons to compute an expression. Use this when the Calculator app is open.
    The expression should use digits and operators: +, -, *, /, %, =, C (clear).
    Example: '287487*63%=' computes 63% of 287487.
    Example: '123+456=' computes 123+456.
    Always end with = to get the result."""
    return {"expression": expression}


class DesktopAgent:
    """AI agent that can control the entire desktop."""

    def __init__(
        self,
        desktop: DesktopComputer,
        query: str,
        model_name: str,
        memory: SessionMemory,
        verbose: bool = False,
    ):
        self._desktop = desktop
        self._query = query
        self._model_name = model_name
        self._memory = memory
        self._verbose = verbose
        self.final_reasoning: Optional[str] = None

        self._client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
            vertexai=os.environ.get("USE_VERTEXAI", "0").lower() in ["true", "1"],
            project=os.environ.get("VERTEXAI_PROJECT"),
            location=os.environ.get("VERTEXAI_LOCATION"),
        )

        # Build context with memory
        context = self._memory.get_context_summary()
        system_prompt = (
            "You are a desktop AI agent that can control the entire computer. "
            "You can open apps, switch between them, click, type, scroll, and run commands. "
            "You operate in BACKGROUND MODE — the user can keep working on other apps "
            "while you interact with your target app. Your clicks and keystrokes are sent "
            "directly to the target window without stealing focus.\n\n"
            "IMPORTANT RULES:\n"
            "1. Before opening an app, check if it's already open using list_open_apps or your memory.\n"
            "2. If an app is already open, use switch_to_app instead of open_app.\n"
            "3. When the user says 'open brave' they mean the Brave Browser application, not a browser search.\n"
            "4. The screenshot shows the target app's window content, even if it's behind other windows.\n"
            "5. Coordinates are relative to the target window's screenshot.\n"
            "6. Be efficient — don't repeat actions unnecessarily.\n"
            "7. After switch_to_app, you'll see that app's window in screenshots and can interact with it.\n"
            "8. The user does NOT need to have the app in foreground for you to work with it.\n"
            "9. For Calculator: use the calculator_compute function with math expressions instead of clicking individual buttons. Example: '287487*63%=' for 63% of 287487.\n"
        )
        if context:
            system_prompt += f"\nCurrent session state:\n{context}\n"

        self._contents: list[Content] = [
            Content(
                role="user",
                parts=[Part(text=f"{system_prompt}\n\nUser request: {self._query}")],
            )
        ]

        # Custom functions for desktop control
        custom_functions = [
            types.FunctionDeclaration.from_callable(client=self._client, callable=open_app),
            types.FunctionDeclaration.from_callable(client=self._client, callable=close_app),
            types.FunctionDeclaration.from_callable(client=self._client, callable=switch_to_app),
            types.FunctionDeclaration.from_callable(client=self._client, callable=list_open_apps),
            types.FunctionDeclaration.from_callable(client=self._client, callable=run_shell_command),
            types.FunctionDeclaration.from_callable(client=self._client, callable=calculator_compute),
        ]

        self._generate_content_config = GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            tools=[
                types.Tool(
                    computer_use=types.ComputerUse(
                        environment=types.Environment.ENVIRONMENT_BROWSER,
                        excluded_predefined_functions=[],
                    ),
                ),
                types.Tool(function_declarations=custom_functions),
            ],
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )

    def denormalize_x(self, x: int) -> int:
        return int(x / 1000 * self._desktop.screen_size()[0])

    def denormalize_y(self, y: int) -> int:
        return int(y / 1000 * self._desktop.screen_size()[1])

    def handle_action(self, action: types.FunctionCall) -> FunctionResponseT:
        """Handle a function call from the model."""
        name = action.name
        args = action.args or {}

        # Desktop-specific custom functions
        if name == "open_app":
            result = self._desktop.open_app(args["app_name"])
            app_name = args["app_name"].lower()
            if self._desktop.focused_hwnd:
                self._memory.register_app(
                    app_name, self._desktop.focused_hwnd,
                    "", self._desktop.focused_hwnd  # simplified
                )
            self._memory.log_action("open_app", args)
            return result

        elif name == "close_app":
            result = self._desktop.close_app(args["app_name"])
            self._memory.unregister_app(args["app_name"].lower())
            self._memory.log_action("close_app", args)
            return result

        elif name == "switch_to_app":
            result = self._desktop.switch_to_app(args["app_name"])
            self._memory.set_focused_app(args["app_name"].lower())
            self._memory.log_action("switch_to_app", args)
            return result

        elif name == "list_open_apps":
            apps = self._desktop.list_open_apps()
            self._memory.log_action("list_open_apps")
            # Flatten to simple string to avoid nested list issues with API
            app_strs = [f"{a['process']}: {a['title']}" for a in apps]
            return {"apps": "\n".join(app_strs), "count": str(len(apps))}

        elif name == "run_shell_command":
            result = self._desktop.run_command(args["command"])
            self._memory.log_action("run_shell_command", args, result.get("stdout", ""))
            return result

        elif name == "calculator_compute":
            from computers.desktop.uia_helper import calc_press_sequence, get_display_text
            expression = args["expression"]
            hwnd = self._desktop.focused_hwnd
            if hwnd:
                success = calc_press_sequence(hwnd, expression)
                time.sleep(0.3)
                display = get_display_text(hwnd)
                self._memory.log_action("calculator_compute", args, display)
                if success:
                    # Return the result as text + screenshot
                    state = self._desktop.current_state()
                    return EnvState(screenshot=state.screenshot, url=f"Calculator: {display}")
                else:
                    return {"error": "Could not press calculator buttons via UIA", "display": display}
            return {"error": "No calculator window targeted"}

        # Built-in computer_use actions
        elif name == "click_at":
            x = self.denormalize_x(args["x"])
            y = self.denormalize_y(args["y"])
            self._memory.log_action("click_at", {"x": x, "y": y})
            return self._desktop.click_at(x, y)

        elif name == "hover_at":
            x = self.denormalize_x(args["x"])
            y = self.denormalize_y(args["y"])
            return self._desktop.hover_at(x, y)

        elif name == "type_text_at":
            x = self.denormalize_x(args["x"])
            y = self.denormalize_y(args["y"])
            return self._desktop.type_text_at(
                x, y, args["text"],
                press_enter=args.get("press_enter", False),
                clear_before_typing=args.get("clear_before_typing", True),
            )

        elif name == "scroll_document":
            return self._desktop.scroll_document(args["direction"])

        elif name == "scroll_at":
            x = self.denormalize_x(args["x"])
            y = self.denormalize_y(args["y"])
            mag = args.get("magnitude", 800)
            return self._desktop.scroll_at(x, y, args["direction"], mag)

        elif name == "wait_5_seconds":
            return self._desktop.wait_5_seconds()

        elif name == "go_back":
            return self._desktop.go_back()

        elif name == "go_forward":
            return self._desktop.go_forward()

        elif name == "search":
            return self._desktop.search()

        elif name == "navigate":
            return self._desktop.navigate(args["url"])

        elif name == "key_combination":
            keys = args["keys"]
            if isinstance(keys, str):
                keys = keys.split("+")
            return self._desktop.key_combination(keys)

        elif name == "drag_and_drop":
            x = self.denormalize_x(args["x"])
            y = self.denormalize_y(args["y"])
            dx = self.denormalize_x(args["destination_x"])
            dy = self.denormalize_y(args["destination_y"])
            return self._desktop.drag_and_drop(x, y, dx, dy)

        elif name == "open_web_browser":
            return self._desktop.open_web_browser()

        else:
            raise ValueError(f"Unknown action: {name}")


    def get_model_response(self, max_retries=5, base_delay_s=1):
        for attempt in range(max_retries):
            try:
                # Debug: log contents structure
                total_img_bytes = 0
                for c in self._contents:
                    if c.parts:
                        for p in c.parts:
                            if hasattr(p, 'inline_data') and p.inline_data and p.inline_data.data:
                                total_img_bytes += len(p.inline_data.data)
                print(f"[AGENT] API call: model={self._model_name}, turns={len(self._contents)}, img_bytes={total_img_bytes}")
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=self._contents,
                    config=self._generate_content_config,
                )
                return response
            except Exception as e:
                err_str = str(e)
                print(f"[AGENT] API error (attempt {attempt+1}): {err_str}")
                # If invalid argument, dump contents structure for debugging
                if attempt == 0 and "INVALID_ARGUMENT" in err_str:
                    for i, c in enumerate(self._contents):
                        parts_desc = []
                        if c.parts:
                            for p in c.parts:
                                if p.text:
                                    parts_desc.append(f"text({len(p.text)})")
                                elif hasattr(p, 'function_call') and p.function_call:
                                    parts_desc.append(f"fc({p.function_call.name})")
                                elif hasattr(p, 'function_response') and p.function_response:
                                    parts_desc.append(f"fr({p.function_response.name})")
                                elif hasattr(p, 'inline_data') and p.inline_data:
                                    parts_desc.append(f"img({len(p.inline_data.data) if p.inline_data.data else 0})")
                                else:
                                    parts_desc.append("unknown")
                        print(f"  [{i}] role={c.role} parts=[{', '.join(parts_desc)}]")
                if attempt < max_retries - 1:
                    time.sleep(base_delay_s * (2 ** attempt))
                else:
                    raise

    def get_text(self, candidate: Candidate) -> Optional[str]:
        if not candidate.content or not candidate.content.parts:
            return None
        text = [p.text for p in candidate.content.parts if p.text]
        return " ".join(text) or None

    def extract_function_calls(self, candidate: Candidate) -> list[types.FunctionCall]:
        if not candidate.content or not candidate.content.parts:
            return []
        return [p.function_call for p in candidate.content.parts if p.function_call]

    def _prune_screenshots(self):
        """Remove old screenshots, keeping only the most recent ones."""
        # Count all inline_data parts from newest to oldest
        screenshot_indices = []  # (content_idx, part_idx)
        for i in range(len(self._contents) - 1, -1, -1):
            content = self._contents[i]
            if not content.parts:
                continue
            for j in range(len(content.parts) - 1, -1, -1):
                p = content.parts[j]
                if hasattr(p, 'inline_data') and p.inline_data:
                    screenshot_indices.append((i, j))

        # Remove all but the most recent N screenshots
        to_remove = screenshot_indices[MAX_RECENT_TURN_WITH_SCREENSHOTS:]
        for content_idx, part_idx in sorted(to_remove, reverse=True):
            content = self._contents[content_idx]
            parts_list = list(content.parts)
            parts_list.pop(part_idx)
            # Rebuild the content with remaining parts
            self._contents[content_idx] = Content(
                role=content.role,
                parts=parts_list if parts_list else [Part(text="[screenshot removed]")]
            )

        if to_remove:
            print(f"[AGENT] Pruned {len(to_remove)} old screenshots, kept {MAX_RECENT_TURN_WITH_SCREENSHOTS}")

    def run_one_iteration(self) -> Literal["COMPLETE", "CONTINUE"]:
        try:
            response = self.get_model_response()
        except Exception as e:
            print(f"[AGENT] Model error: {e}")
            return "COMPLETE"

        if not response.candidates:
            return "COMPLETE"

        candidate = response.candidates[0]
        if candidate.content:
            self._contents.append(candidate.content)

        reasoning = self.get_text(candidate)
        function_calls = self.extract_function_calls(candidate)

        if (
            not function_calls
            and not reasoning
            and candidate.finish_reason == FinishReason.MALFORMED_FUNCTION_CALL
        ):
            return "CONTINUE"

        if not function_calls:
            self.final_reasoning = reasoning
            return "COMPLETE"

        function_responses = []
        for fc in function_calls:
            try:
                result = self.handle_action(fc)
            except Exception as e:
                print(f"[AGENT] Action error ({fc.name}): {e}")
                result = {"error": str(e)}

            if isinstance(result, EnvState):
                function_responses.append(
                    Part(
                        function_response=FunctionResponse(
                            name=fc.name,
                            response={"url": result.url},
                        )
                    )
                )
                function_responses.append(
                    Part(
                        inline_data=types.Blob(
                            mime_type="image/jpeg",
                            data=result.screenshot,
                        )
                    )
                )
            else:
                function_responses.append(
                    Part(
                        function_response=FunctionResponse(
                            name=fc.name,
                            response=result if isinstance(result, dict) else {"result": str(result)},
                        )
                    )
                )

        self._contents.append(Content(role="user", parts=function_responses))
        self._prune_screenshots()
        return "CONTINUE"

    def agent_loop(self, max_iterations: int = 50):
        """Run the agent loop until complete or max iterations."""
        for i in range(max_iterations):
            status = self.run_one_iteration()
            if status == "COMPLETE":
                break
        return self.final_reasoning
