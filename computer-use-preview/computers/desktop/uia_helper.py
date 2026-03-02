"""UI Automation helper for interacting with UWP apps (like Calculator) in the background.

UWP apps don't respond to PostMessage for input. Instead, we use the
Windows UI Automation API to find controls and invoke them directly —
no mouse, no keyboard, no focus needed.
"""
import time
from typing import Optional

try:
    import uiautomation as uia
    UIA_AVAILABLE = True
except ImportError:
    UIA_AVAILABLE = False
    print("[UIA] uiautomation not installed, UWP background interaction disabled")


# Apps known to be UWP / not respond to PostMessage
UWP_APPS = {
    "calculator", "calc", "photos", "settings", "store",
    "mail", "calendar", "maps", "weather", "clock",
    "alarms", "camera", "xbox", "groove", "movies",
}


def is_uwp_app(app_name: str) -> bool:
    """Check if an app is known to be UWP (needs UIA instead of PostMessage)."""
    return app_name.lower().strip() in UWP_APPS


def get_window_control(hwnd: int) -> Optional[object]:
    """Get a UIA control from a window handle."""
    if not UIA_AVAILABLE:
        return None
    try:
        ctrl = uia.ControlFromHandle(hwnd)
        return ctrl
    except Exception as e:
        print(f"[UIA] Failed to get control from hwnd {hwnd}: {e}")
        return None


def find_and_click_button(hwnd: int, button_name: str) -> bool:
    """Find a button in a UWP window by name/AutomationId and invoke it.
    
    Works without focus — uses UIA InvokePattern.
    Returns True if button was found and clicked.
    """
    if not UIA_AVAILABLE:
        return False
    try:
        ctrl = uia.ControlFromHandle(hwnd)
        if not ctrl:
            return False
        # Search by Name first
        btn = ctrl.ButtonControl(searchDepth=8, Name=button_name)
        if btn and btn.Exists(maxSearchSeconds=1):
            invoke = btn.GetInvokePattern()
            if invoke:
                invoke.Invoke()
                return True
        # Try AutomationId
        btn = ctrl.ButtonControl(searchDepth=8, AutomationId=button_name)
        if btn and btn.Exists(maxSearchSeconds=1):
            invoke = btn.GetInvokePattern()
            if invoke:
                invoke.Invoke()
                return True
        return False
    except Exception as e:
        print(f"[UIA] Button click failed for '{button_name}': {e}")
        return False


def find_and_set_text(hwnd: int, text: str, control_type: str = "Edit") -> bool:
    """Set text in a UWP text field using ValuePattern.
    
    Works without focus.
    """
    if not UIA_AVAILABLE:
        return False
    try:
        ctrl = uia.ControlFromHandle(hwnd)
        if not ctrl:
            return False
        if control_type == "Edit":
            edit = ctrl.EditControl(searchDepth=8)
        else:
            edit = ctrl.Control(searchDepth=8, ControlType=control_type)
        if edit and edit.Exists(maxSearchSeconds=1):
            vp = edit.GetValuePattern()
            if vp:
                vp.SetValue(text)
                return True
        return False
    except Exception as e:
        print(f"[UIA] Set text failed: {e}")
        return False


def get_display_text(hwnd: int) -> str:
    """Get the display/result text from a UWP app (e.g., Calculator result)."""
    if not UIA_AVAILABLE:
        return ""
    try:
        ctrl = uia.ControlFromHandle(hwnd)
        if not ctrl:
            return ""
        # Calculator-specific: look for the result display
        # The Calculator result has AutomationId "CalculatorResults"
        result = ctrl.TextControl(searchDepth=8, AutomationId="CalculatorResults")
        if result and result.Exists(maxSearchSeconds=1):
            return result.Name or ""
        # Generic: try to find any text control with a value
        text_ctrl = ctrl.TextControl(searchDepth=5)
        if text_ctrl and text_ctrl.Exists(maxSearchSeconds=1):
            return text_ctrl.Name or ""
        return ""
    except Exception as e:
        print(f"[UIA] Get display text failed: {e}")
        return ""


# Calculator button name mapping
CALC_BUTTON_MAP = {
    "0": "Zero", "1": "One", "2": "Two", "3": "Three",
    "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
    "8": "Eight", "9": "Nine",
    "+": "Plus", "-": "Minus", "*": "Multiply", "/": "Divide",
    "×": "Multiply", "÷": "Divide",
    "=": "Equals", ".": "Decimal point",
    "%": "Percent", "C": "Clear", "CE": "Clear entry",
    "⌫": "Backspace",
}


def calc_press_sequence(hwnd: int, expression: str) -> bool:
    """Press a sequence of calculator buttons for an expression like '63%*287487='.
    
    Uses UIA to invoke buttons directly — works in background.
    """
    if not UIA_AVAILABLE:
        return False
    
    # First clear
    find_and_click_button(hwnd, "Clear")
    time.sleep(0.1)
    
    for char in expression:
        btn_name = CALC_BUTTON_MAP.get(char)
        if btn_name:
            success = find_and_click_button(hwnd, btn_name)
            if not success:
                print(f"[UIA] Calculator: couldn't press '{char}' ('{btn_name}')")
                return False
            time.sleep(0.05)
        elif char == ' ':
            continue
        else:
            print(f"[UIA] Calculator: unknown char '{char}'")
    return True

