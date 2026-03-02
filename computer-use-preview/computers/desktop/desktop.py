"""Full desktop computer control with BACKGROUND interaction support.

Uses Win32 PostMessage/SendMessage for mouse/keyboard input to target windows
WITHOUT stealing focus. The user can keep working while the agent operates
on a different window in the background.

Screenshot capture uses PrintWindow API which also works without focus.
"""
import ctypes
import ctypes.wintypes
import io
import os
import subprocess
import time
from typing import Literal

import pyautogui
from PIL import Image

from ..computer import Computer, EnvState
from .uia_helper import is_uwp_app, UIA_AVAILABLE

# Disable pyautogui failsafe
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.05

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
gdi32 = ctypes.windll.gdi32

# Win32 constants
SW_RESTORE = 9
SW_SHOW = 5
SW_MINIMIZE = 6
HWND_TOP = 0
SWP_SHOWWINDOW = 0x0040
GW_OWNER = 4
GA_ROOTOWNER = 3

# Window messages for background input
WM_LBUTTONDOWN = 0x0201
WM_LBUTTONUP = 0x0202
WM_RBUTTONDOWN = 0x0204
WM_RBUTTONUP = 0x0205
WM_MOUSEMOVE = 0x0200
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_CHAR = 0x0102
WM_CLOSE = 0x0010
WM_SETTEXT = 0x000C
WM_MOUSEWHEEL = 0x020A
WM_MOUSEHWHEEL = 0x020E
MK_LBUTTON = 0x0001

# Virtual key codes
VK_RETURN = 0x0D
VK_BACK = 0x08
VK_DELETE = 0x2E
VK_TAB = 0x09
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_LEFT = 0x25
VK_UP = 0x26
VK_RIGHT = 0x27
VK_DOWN = 0x28
VK_HOME = 0x24
VK_END = 0x23
VK_PRIOR = 0x21  # Page Up
VK_NEXT = 0x22   # Page Down
VK_CONTROL = 0x11
VK_SHIFT = 0x10
VK_MENU = 0x12   # Alt
VK_LWIN = 0x5B

# DPI awareness
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass


MAKELONG = lambda lo, hi: (lo & 0xFFFF) | ((hi & 0xFFFF) << 16)
MAKELPARAM = MAKELONG


def _get_screen_size():
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def _capture_full_screen() -> bytes | None:
    try:
        img = pyautogui.screenshot()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def _capture_window_by_hwnd(hwnd: int) -> bytes | None:
    from screen_capture import capture_window
    return capture_window(hwnd)


def _find_windows_by_title(*keywords: str) -> list[dict]:
    results = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
    )
    def enum_cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        if any(kw.lower() in title.lower() for kw in keywords):
            pid = ctypes.wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            results.append({"hwnd": hwnd, "title": title, "pid": pid.value})
        return True
    user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
    return results


def _get_all_visible_windows() -> list[dict]:
    results = []
    WNDENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
    )
    def enum_cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        if not title or title in ("Program Manager", "Windows Input Experience"):
            return True
        pid = ctypes.wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        results.append({"hwnd": hwnd, "title": title, "pid": pid.value})
        return True
    user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
    return results


def _focus_window(hwnd: int):
    """Bring a window to the foreground (only used for open_app initial launch)."""
    if user32.IsIconic(hwnd):
        user32.ShowWindow(hwnd, SW_RESTORE)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.3)


def _get_process_name(pid: int) -> str:
    try:
        import psutil
        return psutil.Process(pid).name()
    except Exception:
        return "unknown"


# ── Background Input Helpers ──

def _bg_click(hwnd: int, x: int, y: int):
    """Send a click to a window WITHOUT focusing it."""
    lparam = MAKELPARAM(x, y)
    user32.PostMessageW(hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam)
    time.sleep(0.05)
    user32.PostMessageW(hwnd, WM_LBUTTONUP, 0, lparam)


def _bg_right_click(hwnd: int, x: int, y: int):
    """Send a right-click to a window WITHOUT focusing it."""
    lparam = MAKELPARAM(x, y)
    user32.PostMessageW(hwnd, WM_RBUTTONDOWN, MK_LBUTTON, lparam)
    time.sleep(0.05)
    user32.PostMessageW(hwnd, WM_RBUTTONUP, 0, lparam)


def _bg_mouse_move(hwnd: int, x: int, y: int):
    """Send mouse move to a window WITHOUT focusing it."""
    lparam = MAKELPARAM(x, y)
    user32.PostMessageW(hwnd, WM_MOUSEMOVE, 0, lparam)


def _bg_type_char(hwnd: int, char: str):
    """Send a single character to a window via WM_CHAR."""
    user32.PostMessageW(hwnd, WM_CHAR, ord(char), 0)


def _bg_type_text(hwnd: int, text: str):
    """Type text into a window WITHOUT focusing it."""
    for ch in text:
        if ch == '\n':
            _bg_key_press(hwnd, VK_RETURN)
        else:
            _bg_type_char(hwnd, ch)
        time.sleep(0.01)


def _bg_key_press(hwnd: int, vk_code: int, scan_code: int = 0):
    """Send a key press (down + up) to a window WITHOUT focusing it."""
    lparam_down = (scan_code << 16) | 1
    lparam_up = (scan_code << 16) | 1 | (1 << 30) | (1 << 31)
    user32.PostMessageW(hwnd, WM_KEYDOWN, vk_code, lparam_down)
    time.sleep(0.02)
    user32.PostMessageW(hwnd, WM_KEYUP, vk_code, lparam_up)


def _bg_hotkey(hwnd: int, *vk_codes: int):
    """Send a key combination (e.g. Ctrl+A) to a window WITHOUT focusing it."""
    # Press all modifier keys down
    for vk in vk_codes[:-1]:
        user32.PostMessageW(hwnd, WM_KEYDOWN, vk, 1)
        time.sleep(0.01)
    # Press and release the main key
    main_vk = vk_codes[-1]
    user32.PostMessageW(hwnd, WM_KEYDOWN, main_vk, 1)
    time.sleep(0.02)
    user32.PostMessageW(hwnd, WM_KEYUP, main_vk, 1 | (1 << 30) | (1 << 31))
    # Release modifiers in reverse
    for vk in reversed(vk_codes[:-1]):
        user32.PostMessageW(hwnd, WM_KEYUP, vk, 1 | (1 << 30) | (1 << 31))
        time.sleep(0.01)


def _bg_scroll(hwnd: int, x: int, y: int, delta: int, horizontal: bool = False):
    """Send scroll to a window WITHOUT focusing it."""
    lparam = MAKELPARAM(x, y)
    wparam = MAKELONG(0, delta)
    msg = WM_MOUSEHWHEEL if horizontal else WM_MOUSEWHEEL
    user32.PostMessageW(hwnd, msg, wparam, lparam)


def _name_to_vk(key_name: str) -> int:
    """Convert a key name string to a Win32 virtual key code."""
    KEY_MAP = {
        "ctrl": VK_CONTROL, "control": VK_CONTROL,
        "shift": VK_SHIFT,
        "alt": VK_MENU, "menu": VK_MENU,
        "win": VK_LWIN, "meta": VK_LWIN, "command": VK_LWIN,
        "enter": VK_RETURN, "return": VK_RETURN,
        "backspace": VK_BACK, "back": VK_BACK,
        "delete": VK_DELETE, "del": VK_DELETE,
        "tab": VK_TAB,
        "escape": VK_ESCAPE, "esc": VK_ESCAPE,
        "space": VK_SPACE,
        "left": VK_LEFT, "arrowleft": VK_LEFT,
        "right": VK_RIGHT, "arrowright": VK_RIGHT,
        "up": VK_UP, "arrowup": VK_UP,
        "down": VK_DOWN, "arrowdown": VK_DOWN,
        "home": VK_HOME, "end": VK_END,
        "pageup": VK_PRIOR, "pagedown": VK_NEXT,
    }
    k = key_name.lower().strip()
    if k in KEY_MAP:
        return KEY_MAP[k]
    # Single character → VkKeyScan
    if len(k) == 1:
        vk = user32.VkKeyScanW(ord(k)) & 0xFF
        return vk
    # F-keys
    if k.startswith("f") and k[1:].isdigit():
        n = int(k[1:])
        if 1 <= n <= 24:
            return 0x70 + (n - 1)  # VK_F1 = 0x70
    return 0


# Common app registry
APP_REGISTRY = {
    "brave": {
        "exe": "brave.exe",
        "paths": [
            os.path.expandvars(r"%LOCALAPPDATA%\BraveSoftware\Brave-Browser\Application\brave.exe"),
            os.path.expandvars(r"%PROGRAMFILES%\BraveSoftware\Brave-Browser\Application\brave.exe"),
            os.path.expandvars(r"%PROGRAMFILES(X86)%\BraveSoftware\Brave-Browser\Application\brave.exe"),
        ],
        "window_keywords": ["Brave"],
    },
    "chrome": {
        "exe": "chrome.exe",
        "paths": [
            os.path.expandvars(r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%PROGRAMFILES(X86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        ],
        "window_keywords": ["Chrome"],
    },
    "firefox": {
        "exe": "firefox.exe",
        "paths": [
            os.path.expandvars(r"%PROGRAMFILES%\Mozilla Firefox\firefox.exe"),
            os.path.expandvars(r"%PROGRAMFILES(X86)%\Mozilla Firefox\firefox.exe"),
        ],
        "window_keywords": ["Firefox"],
    },
    "calculator": {
        "exe": "calc.exe",
        "paths": ["calc.exe"],
        "window_keywords": ["Calculator"],
    },
    "notepad": {
        "exe": "notepad.exe",
        "paths": ["notepad.exe"],
        "window_keywords": ["Notepad"],
    },
    "explorer": {
        "exe": "explorer.exe",
        "paths": ["explorer.exe"],
        "window_keywords": ["File Explorer"],
    },
    "terminal": {
        "exe": "wt.exe",
        "paths": ["wt.exe"],
        "window_keywords": ["Terminal", "PowerShell", "Command Prompt"],
    },
    "vscode": {
        "exe": "code.exe",
        "paths": [
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe"),
        ],
        "window_keywords": ["Visual Studio Code"],
    },
    "spotify": {
        "exe": "spotify.exe",
        "paths": [os.path.expandvars(r"%APPDATA%\Spotify\Spotify.exe")],
        "window_keywords": ["Spotify"],
    },
    "discord": {
        "exe": "discord.exe",
        "paths": [os.path.expandvars(r"%LOCALAPPDATA%\Discord\Update.exe")],
        "window_keywords": ["Discord"],
    },
}


class DesktopComputer(Computer):
    """Full desktop control with background interaction support.
    
    When background_mode=True (default), all mouse/keyboard actions are sent
    via Win32 PostMessage to the target window handle — no focus stealing.
    The user can keep working on other apps while the agent operates.
    
    Screenshots use PrintWindow API which also works without focus.
    """

    def __init__(
        self,
        screen_size: tuple[int, int] = (1920, 1080),
        capture_mode: str = "desktop",
        background_mode: bool = True,
    ):
        self._screen_size = screen_size
        self._capture_mode = capture_mode
        self._background_mode = background_mode
        self._actual_screen = _get_screen_size()
        self._focused_hwnd: int | None = None
        self._focused_app: str | None = None
        print(f"[DESKTOP] Initialized. Screen: {self._actual_screen}, output: {self._screen_size}, background: {background_mode}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def screen_size(self) -> tuple[int, int]:
        return self._screen_size

    @property
    def focused_app(self) -> str | None:
        return self._focused_app

    @property
    def focused_hwnd(self) -> int | None:
        return self._focused_hwnd

    def _take_screenshot(self) -> bytes:
        """Capture target window via PrintWindow (works in background)."""
        raw = None
        if self._focused_hwnd and user32.IsWindow(self._focused_hwnd):
            raw = _capture_window_by_hwnd(self._focused_hwnd)
        if raw is None:
            raw = _capture_full_screen()
        if raw is None:
            img = Image.new("RGB", self._screen_size, (0, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        try:
            img = Image.open(io.BytesIO(raw))
            img = img.resize(self._screen_size, Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            return buf.getvalue()
        except Exception:
            return raw

    def _get_window_rect(self) -> tuple[int, int, int, int]:
        """Get (left, top, width, height) of focused window."""
        if self._focused_hwnd and user32.IsWindow(self._focused_hwnd):
            rect = ctypes.wintypes.RECT()
            user32.GetWindowRect(self._focused_hwnd, ctypes.byref(rect))
            return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top
        return 0, 0, self._actual_screen[0], self._actual_screen[1]

    def _to_window_coords(self, x: int, y: int) -> tuple[int, int]:
        """Convert from target resolution coords to window-relative coords for PostMessage."""
        if self._focused_hwnd and user32.IsWindow(self._focused_hwnd):
            _, _, win_w, win_h = self._get_window_rect()
            # Map from screenshot space to window client area
            wx = int(x * win_w / self._screen_size[0])
            wy = int(y * win_h / self._screen_size[1])
            return wx, wy
        # Fallback: map to screen coords (for pyautogui)
        return (
            int(x * self._actual_screen[0] / self._screen_size[0]),
            int(y * self._actual_screen[1] / self._screen_size[1]),
        )

    def _to_screen_coords(self, x: int, y: int) -> tuple[int, int]:
        """Convert from target resolution to absolute screen coords (for pyautogui fallback)."""
        if self._focused_hwnd and user32.IsWindow(self._focused_hwnd):
            left, top, win_w, win_h = self._get_window_rect()
            return (
                left + int(x * win_w / self._screen_size[0]),
                top + int(y * win_h / self._screen_size[1]),
            )
        return (
            int(x * self._actual_screen[0] / self._screen_size[0]),
            int(y * self._actual_screen[1] / self._screen_size[1]),
        )

    def _can_bg(self) -> bool:
        """Check if we can use background PostMessage for the current target.
        Returns False for UWP apps (they need UIA or pyautogui instead).
        """
        return (
            self._background_mode
            and self._focused_hwnd is not None
            and user32.IsWindow(self._focused_hwnd)
            and not self._is_uwp_target()
        )

    def _is_uwp_target(self) -> bool:
        """Check if the current target app is a UWP app."""
        if self._focused_app:
            return is_uwp_app(self._focused_app)
        return False

    def _use_uia(self) -> bool:
        """Check if we should use UI Automation (for UWP apps in background)."""
        return (
            self._background_mode
            and self._focused_hwnd is not None
            and user32.IsWindow(self._focused_hwnd)
            and self._is_uwp_target()
            and UIA_AVAILABLE
        )

    def current_state(self) -> EnvState:
        screenshot = self._take_screenshot()
        title = ""
        if self._focused_hwnd and user32.IsWindow(self._focused_hwnd):
            length = user32.GetWindowTextLengthW(self._focused_hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(self._focused_hwnd, buf, length + 1)
            title = buf.value
        return EnvState(screenshot=screenshot, url=title or "Desktop")

    # ── Core interaction (background-aware) ──

    def open_web_browser(self) -> EnvState:
        os.startfile("https://www.google.com")
        time.sleep(2)
        return self.current_state()

    def click_at(self, x: int, y: int) -> EnvState:
        if self._use_uia():
            # UWP app: use UIA ElementFromPoint to find and invoke the control
            sx, sy = self._to_screen_coords(x, y)
            self._uia_click_at_screen(sx, sy)
            print(f"[UIA] click at screen({sx},{sy})")
        elif self._can_bg():
            wx, wy = self._to_window_coords(x, y)
            _bg_click(self._focused_hwnd, wx, wy)
            print(f"[BG] click at window({wx},{wy})")
        else:
            sx, sy = self._to_screen_coords(x, y)
            pyautogui.click(sx, sy)
        time.sleep(0.3)
        return self.current_state()

    def _uia_click_at_screen(self, screen_x: int, screen_y: int):
        """Use UI Automation to find and invoke the element at screen coordinates.
        Works for UWP apps without stealing focus.
        """
        try:
            import uiautomation as uia
            element = uia.ControlFromPoint(screen_x, screen_y)
            if element:
                # Try InvokePattern first (buttons)
                try:
                    invoke = element.GetInvokePattern()
                    if invoke:
                        invoke.Invoke()
                        return
                except Exception:
                    pass
                # Try TogglePattern (checkboxes, toggle buttons)
                try:
                    toggle = element.GetTogglePattern()
                    if toggle:
                        toggle.Toggle()
                        return
                except Exception:
                    pass
                # Try SelectionItemPattern (radio buttons, list items)
                try:
                    sel = element.GetSelectionItemPattern()
                    if sel:
                        sel.Select()
                        return
                except Exception:
                    pass
                # Fallback: use SetFocus + click for this specific element
                try:
                    element.SetFocus()
                    time.sleep(0.05)
                    element.Click()
                    return
                except Exception:
                    pass
            # Last resort: pyautogui click
            pyautogui.click(screen_x, screen_y)
        except Exception as e:
            print(f"[UIA] click failed at ({screen_x},{screen_y}): {e}")
            pyautogui.click(screen_x, screen_y)

    def hover_at(self, x: int, y: int) -> EnvState:
        if self._can_bg():
            wx, wy = self._to_window_coords(x, y)
            _bg_mouse_move(self._focused_hwnd, wx, wy)
        else:
            sx, sy = self._to_screen_coords(x, y)
            pyautogui.moveTo(sx, sy)
        time.sleep(0.2)
        return self.current_state()

    def type_text_at(
        self, x: int, y: int, text: str,
        press_enter: bool = False, clear_before_typing: bool = True,
    ) -> EnvState:
        if self._use_uia():
            # UWP: click the target field via UIA, then type via UIA ValuePattern
            sx, sy = self._to_screen_coords(x, y)
            self._uia_type_at_screen(sx, sy, text, press_enter, clear_before_typing)
            print(f"[UIA] typed '{text[:30]}...' at screen({sx},{sy})")
        elif self._can_bg():
            wx, wy = self._to_window_coords(x, y)
            _bg_click(self._focused_hwnd, wx, wy)
            time.sleep(0.15)
            if clear_before_typing:
                _bg_hotkey(self._focused_hwnd, VK_CONTROL, _name_to_vk("a"))
                time.sleep(0.05)
                _bg_key_press(self._focused_hwnd, VK_DELETE)
                time.sleep(0.05)
            _bg_type_text(self._focused_hwnd, text)
            if press_enter:
                _bg_key_press(self._focused_hwnd, VK_RETURN)
            print(f"[BG] typed '{text[:30]}...' at window({wx},{wy})")
        else:
            sx, sy = self._to_screen_coords(x, y)
            pyautogui.click(sx, sy)
            time.sleep(0.15)
            if clear_before_typing:
                pyautogui.hotkey("ctrl", "a")
                time.sleep(0.05)
                pyautogui.press("delete")
                time.sleep(0.05)
            if text.isascii():
                pyautogui.typewrite(text, interval=0.02)
            else:
                pyautogui.write(text)
            if press_enter:
                pyautogui.press("enter")
        time.sleep(0.3)
        return self.current_state()

    def _uia_type_at_screen(self, screen_x: int, screen_y: int, text: str,
                             press_enter: bool, clear_before: bool):
        """Type text into a UWP control using UI Automation."""
        try:
            import uiautomation as uia
            element = uia.ControlFromPoint(screen_x, screen_y)
            if element:
                # Try ValuePattern (text fields)
                try:
                    vp = element.GetValuePattern()
                    if vp:
                        if clear_before:
                            vp.SetValue("")
                        vp.SetValue(text)
                        if press_enter:
                            _bg_key_press(self._focused_hwnd, VK_RETURN)
                        return
                except Exception:
                    pass
            # Fallback: click and type with pyautogui
            pyautogui.click(screen_x, screen_y)
            time.sleep(0.1)
            if clear_before:
                pyautogui.hotkey("ctrl", "a")
                pyautogui.press("delete")
            if text.isascii():
                pyautogui.typewrite(text, interval=0.02)
            else:
                pyautogui.write(text)
            if press_enter:
                pyautogui.press("enter")
        except Exception as e:
            print(f"[UIA] type failed: {e}")
            pyautogui.click(screen_x, screen_y)
            time.sleep(0.1)
            pyautogui.typewrite(text, interval=0.02)

    def scroll_document(self, direction: Literal["up", "down", "left", "right"]) -> EnvState:
        if self._can_bg():
            delta = 120 * 3 if direction in ("up", "left") else -120 * 3
            horiz = direction in ("left", "right")
            _, _, win_w, win_h = self._get_window_rect()
            _bg_scroll(self._focused_hwnd, win_w // 2, win_h // 2, delta, horiz)
        else:
            clicks = 5 if direction in ("down", "right") else -5
            if direction in ("up", "down"):
                pyautogui.scroll(clicks)
            else:
                pyautogui.hscroll(clicks)
        time.sleep(0.3)
        return self.current_state()

    def scroll_at(
        self, x: int, y: int,
        direction: Literal["up", "down", "left", "right"],
        magnitude: int = 800,
    ) -> EnvState:
        if self._can_bg():
            wx, wy = self._to_window_coords(x, y)
            delta = int(magnitude * 120 / 800)
            if direction in ("down", "right"):
                delta = -delta
            horiz = direction in ("left", "right")
            _bg_scroll(self._focused_hwnd, wx, wy, delta, horiz)
        else:
            sx, sy = self._to_screen_coords(x, y)
            pyautogui.moveTo(sx, sy)
            clicks = int(magnitude / 100)
            if direction in ("up", "left"):
                clicks = -clicks
            if direction in ("up", "down"):
                pyautogui.scroll(clicks)
            else:
                pyautogui.hscroll(clicks)
        time.sleep(0.3)
        return self.current_state()

    def wait_5_seconds(self) -> EnvState:
        time.sleep(5)
        return self.current_state()

    def go_back(self) -> EnvState:
        if self._can_bg():
            _bg_hotkey(self._focused_hwnd, VK_MENU, VK_LEFT)
        else:
            pyautogui.hotkey("alt", "left")
        time.sleep(0.5)
        return self.current_state()

    def go_forward(self) -> EnvState:
        if self._can_bg():
            _bg_hotkey(self._focused_hwnd, VK_MENU, VK_RIGHT)
        else:
            pyautogui.hotkey("alt", "right")
        time.sleep(0.5)
        return self.current_state()

    def search(self) -> EnvState:
        # Windows search — this one needs foreground (OS-level)
        pyautogui.hotkey("win", "s")
        time.sleep(0.5)
        return self.current_state()

    def navigate(self, url: str) -> EnvState:
        if url.startswith("http") or "." in url:
            os.startfile(url)
            time.sleep(2)
        return self.current_state()

    def key_combination(self, keys: list[str]) -> EnvState:
        if self._use_uia():
            # UWP: UIA doesn't have a great key combo mechanism,
            # but we can try SendKeys on the focused element
            try:
                import uiautomation as uia
                ctrl = uia.ControlFromHandle(self._focused_hwnd)
                if ctrl:
                    # Map keys to SendKeys format
                    sk_map = {
                        "ctrl": "^", "control": "^", "shift": "+",
                        "alt": "%", "enter": "{Enter}", "return": "{Enter}",
                        "tab": "{Tab}", "escape": "{Esc}", "esc": "{Esc}",
                        "delete": "{Delete}", "backspace": "{Backspace}",
                        "space": " ", "up": "{Up}", "down": "{Down}",
                        "left": "{Left}", "right": "{Right}",
                    }
                    sk = ""
                    for k in keys:
                        mapped = sk_map.get(k.lower().strip(), k)
                        sk += mapped
                    ctrl.SendKeys(sk)
                    print(f"[UIA] SendKeys: {sk}")
            except Exception as e:
                print(f"[UIA] key_combination failed: {e}")
                # Fallback to pyautogui
                pyautogui.hotkey(*[k.lower().strip() for k in keys])
        elif self._can_bg():
            vk_codes = [_name_to_vk(k) for k in keys]
            vk_codes = [vk for vk in vk_codes if vk != 0]
            if vk_codes:
                _bg_hotkey(self._focused_hwnd, *vk_codes)
                print(f"[BG] hotkey: {keys}")
        else:
            mapped = []
            for k in keys:
                k_lower = k.lower().strip()
                PYAG_MAP = {
                    "control": "ctrl", "command": "ctrl", "meta": "win",
                    "return": "enter", "space": " ", "arrowleft": "left",
                    "arrowright": "right", "arrowup": "up", "arrowdown": "down",
                    "backspace": "backspace", "delete": "delete", "escape": "esc",
                    "tab": "tab", "shift": "shift", "alt": "alt",
                }
                mapped.append(PYAG_MAP.get(k_lower, k_lower))
            pyautogui.hotkey(*mapped)
        time.sleep(0.3)
        return self.current_state()

    def drag_and_drop(
        self, x: int, y: int, destination_x: int, destination_y: int,
    ) -> EnvState:
        if self._can_bg():
            wx, wy = self._to_window_coords(x, y)
            dx, dy = self._to_window_coords(destination_x, destination_y)
            lparam_start = MAKELPARAM(wx, wy)
            lparam_end = MAKELPARAM(dx, dy)
            user32.PostMessageW(self._focused_hwnd, WM_LBUTTONDOWN, MK_LBUTTON, lparam_start)
            time.sleep(0.05)
            # Interpolate mouse moves
            steps = 10
            for i in range(1, steps + 1):
                mx = wx + (dx - wx) * i // steps
                my = wy + (dy - wy) * i // steps
                lp = MAKELPARAM(mx, my)
                user32.PostMessageW(self._focused_hwnd, WM_MOUSEMOVE, MK_LBUTTON, lp)
                time.sleep(0.02)
            user32.PostMessageW(self._focused_hwnd, WM_LBUTTONUP, 0, lparam_end)
        else:
            sx, sy = self._to_screen_coords(x, y)
            dx, dy = self._to_screen_coords(destination_x, destination_y)
            pyautogui.moveTo(sx, sy)
            time.sleep(0.1)
            pyautogui.mouseDown()
            time.sleep(0.1)
            pyautogui.moveTo(dx, dy, duration=0.5)
            pyautogui.mouseUp()
        time.sleep(0.3)
        return self.current_state()

    # ── Desktop-specific methods ──

    def open_app(self, app_name: str) -> EnvState:
        """Open an application. Focuses it if already running.
        NOTE: open_app is the ONE action that brings a window to foreground
        briefly so the OS can initialize it. After that, all interaction is background.
        """
        app_key = app_name.lower().strip()
        registry_entry = APP_REGISTRY.get(app_key)

        # Check if already running
        if registry_entry:
            keywords = registry_entry["window_keywords"]
            windows = _find_windows_by_title(*keywords)
            if windows:
                win = windows[0]
                self._focused_hwnd = win["hwnd"]
                self._focused_app = app_key
                print(f"[DESKTOP] App '{app_key}' already running, targeted: {win['title']}")
                time.sleep(0.3)
                return self.current_state()

        # Launch from registry
        if registry_entry:
            for p in registry_entry["paths"]:
                if os.path.exists(p) or os.sep not in p:
                    try:
                        if app_key == "discord":
                            subprocess.Popen([p, "--processStart", "Discord.exe"])
                        else:
                            subprocess.Popen([p])
                        print(f"[DESKTOP] Launched: {p}")
                        time.sleep(2)
                        windows = _find_windows_by_title(*registry_entry["window_keywords"])
                        if windows:
                            self._focused_hwnd = windows[0]["hwnd"]
                            self._focused_app = app_key
                        return self.current_state()
                    except Exception as e:
                        print(f"[DESKTOP] Failed to launch {p}: {e}")
                        continue

        # Fallback: os.startfile
        try:
            os.startfile(app_name)
            time.sleep(2)
            return self.current_state()
        except Exception:
            pass

        # Last resort: Windows search (needs foreground)
        pyautogui.hotkey("win", "s")
        time.sleep(0.5)
        pyautogui.typewrite(app_name, interval=0.03)
        time.sleep(1)
        pyautogui.press("enter")
        time.sleep(2)
        return self.current_state()

    def close_app(self, app_name: str) -> EnvState:
        """Close an application by name (background — sends WM_CLOSE)."""
        app_key = app_name.lower().strip()
        registry_entry = APP_REGISTRY.get(app_key)
        keywords = registry_entry["window_keywords"] if registry_entry else [app_name]
        windows = _find_windows_by_title(*keywords)
        for win in windows:
            try:
                user32.PostMessageW(win["hwnd"], WM_CLOSE, 0, 0)
            except Exception:
                pass
        if self._focused_app == app_key:
            self._focused_hwnd = None
            self._focused_app = None
        time.sleep(0.5)
        return self.current_state()

    def switch_to_app(self, app_name: str) -> EnvState:
        """Switch the agent's target to a different app (NO focus steal).
        The agent will now send all inputs to this app's window handle
        and capture screenshots from it, but it stays in the background.
        """
        app_key = app_name.lower().strip()
        registry_entry = APP_REGISTRY.get(app_key)
        keywords = registry_entry["window_keywords"] if registry_entry else [app_name]
        windows = _find_windows_by_title(*keywords)
        if windows:
            win = windows[0]
            self._focused_hwnd = win["hwnd"]
            self._focused_app = app_key
            print(f"[DESKTOP] Agent now targeting: {win['title']} (background)")
            time.sleep(0.3)
            return self.current_state()
        print(f"[DESKTOP] App '{app_name}' not found running")
        return self.current_state()

    def list_open_apps(self) -> list[dict]:
        windows = _get_all_visible_windows()
        apps = []
        for w in windows:
            proc_name = _get_process_name(w["pid"])
            apps.append({
                "title": w["title"],
                "process": proc_name,
                "pid": w["pid"],
                "hwnd": w["hwnd"],
                "focused": w["hwnd"] == self._focused_hwnd,
            })
        return apps

    def focus_window(self, title_keyword: str) -> EnvState:
        """Set agent target by title keyword (background)."""
        windows = _find_windows_by_title(title_keyword)
        if windows:
            win = windows[0]
            self._focused_hwnd = win["hwnd"]
            print(f"[DESKTOP] Agent targeting: {win['title']}")
            time.sleep(0.3)
        return self.current_state()

    def run_command(self, command: str) -> dict:
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            return {
                "stdout": result.stdout[:2000],
                "stderr": result.stderr[:500],
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"stdout": "", "stderr": "Command timed out", "returncode": -1}
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "returncode": -1}



