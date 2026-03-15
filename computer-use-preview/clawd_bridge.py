"""Bridge to clawd-cursor REST API for desktop control.

clawd-cursor runs as a separate process (clawdcursor start) and exposes
a REST API on port 3847. This module wraps those endpoints so our
WebSocket server can proxy desktop tasks through it.
"""
import base64
import json
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError


CLAWD_BASE_URL = "http://127.0.0.1:3847"
TIMEOUT = 5  # seconds


def _request(method: str, path: str, body: dict | None = None) -> dict:
    """Make an HTTP request to clawd-cursor."""
    url = f"{CLAWD_BASE_URL}{path}"
    data = json.dumps(body).encode("utf-8") if body else None
    req = Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        return {"error": f"clawd-cursor unreachable: {e}"}
    except Exception as e:
        return {"error": str(e)}


class ClawdBridge:
    """Proxy for clawd-cursor desktop agent API."""

    def is_available(self) -> bool:
        """Check if clawd-cursor is running."""
        result = _request("GET", "/health")
        return result.get("status") == "ok"

    def get_version(self) -> str:
        """Get clawd-cursor version."""
        result = _request("GET", "/health")
        return result.get("version", "unknown")

    def submit_task(self, task: str) -> dict:
        """Submit a desktop task. Returns {"accepted": true} on success."""
        return _request("POST", "/task", {"task": task})

    def get_status(self) -> dict:
        """Get current agent state."""
        return _request("GET", "/status")

    def get_screenshot(self) -> Optional[str]:
        """Get current screen as base64 PNG. Returns None if unavailable."""
        try:
            url = f"{CLAWD_BASE_URL}/screenshot"
            req = Request(url, method="GET")
            with urlopen(req, timeout=TIMEOUT) as resp:
                content_type = resp.headers.get("Content-Type", "")
                raw = resp.read()
                if "image" in content_type:
                    return base64.b64encode(raw).decode("utf-8")
                # If JSON response with base64
                try:
                    data = json.loads(raw)
                    return data.get("image") or data.get("screenshot")
                except Exception:
                    return base64.b64encode(raw).decode("utf-8")
        except Exception:
            return None

    def confirm(self, approved: bool) -> dict:
        """Approve or reject a pending safety confirmation."""
        return _request("POST", "/confirm", {"approved": approved})

    def abort(self) -> dict:
        """Abort the current task."""
        return _request("POST", "/abort")

    def get_logs(self) -> list:
        """Get recent log entries."""
        result = _request("GET", "/logs")
        return result if isinstance(result, list) else []
