"""Session memory backed by Google Cloud Firestore.

Tracks open apps, user preferences, and conversation history across sessions.
Falls back to in-memory storage if Firestore is unavailable.
"""
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any


class SessionMemory:
    """Persistent session memory with Firestore backend."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self._firestore_available = False
        self._db = None
        self._local: dict[str, Any] = {
            "open_apps": {},       # app_name -> {hwnd, title, pid, opened_at}
            "focused_app": None,
            "history": [],         # recent actions
            "preferences": {},     # user preferences learned over time
        }
        self._init_firestore()

    def _init_firestore(self):
        """Try to connect to Firestore."""
        try:
            from google.cloud import firestore
            project = os.environ.get("VERTEXAI_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
            if project:
                self._db = firestore.Client(project=project)
                # Test connection
                self._db.collection("sessions").document("_ping").set(
                    {"ts": firestore.SERVER_TIMESTAMP}, merge=True
                )
                self._firestore_available = True
                print(f"[MEMORY] Firestore connected, session: {self.session_id}")
            else:
                print("[MEMORY] No project ID, using local memory only")
        except Exception as e:
            print(f"[MEMORY] Firestore unavailable ({e}), using local memory")

    def _session_ref(self):
        if self._db:
            return self._db.collection("sessions").document(self.session_id)
        return None

    # ── App tracking ──

    def register_app(self, app_name: str, hwnd: int, title: str, pid: int):
        """Register an app as open."""
        entry = {
            "hwnd": hwnd,
            "title": title,
            "pid": pid,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
        self._local["open_apps"][app_name] = entry
        self._local["focused_app"] = app_name
        self._sync_to_firestore()

    def unregister_app(self, app_name: str):
        """Mark an app as closed."""
        self._local["open_apps"].pop(app_name, None)
        if self._local["focused_app"] == app_name:
            self._local["focused_app"] = None
        self._sync_to_firestore()

    def set_focused_app(self, app_name: str):
        self._local["focused_app"] = app_name
        self._sync_to_firestore()

    def get_focused_app(self) -> str | None:
        return self._local["focused_app"]

    def get_open_apps(self) -> dict:
        return self._local["open_apps"]

    def is_app_open(self, app_name: str) -> bool:
        return app_name.lower() in self._local["open_apps"]

    def get_app_info(self, app_name: str) -> dict | None:
        return self._local["open_apps"].get(app_name.lower())

    # ── Action history ──

    def log_action(self, action_name: str, args: dict | None = None, result: str | None = None):
        """Log an agent action."""
        entry = {
            "action": action_name,
            "args": args or {},
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._local["history"].append(entry)
        # Keep last 100 actions in memory
        if len(self._local["history"]) > 100:
            self._local["history"] = self._local["history"][-100:]
        self._sync_action_to_firestore(entry)

    def get_recent_actions(self, n: int = 20) -> list[dict]:
        return self._local["history"][-n:]

    # ── Preferences ──

    def set_preference(self, key: str, value: Any):
        self._local["preferences"][key] = value
        self._sync_to_firestore()

    def get_preference(self, key: str, default: Any = None) -> Any:
        return self._local["preferences"].get(key, default)

    # ── Context for agent ──

    def get_context_summary(self) -> str:
        """Generate a context string for the agent about current state."""
        parts = []
        open_apps = self._local["open_apps"]
        if open_apps:
            app_list = ", ".join(
                f"{name} ('{info['title']}')" for name, info in open_apps.items()
            )
            parts.append(f"Currently open apps: {app_list}")
        focused = self._local["focused_app"]
        if focused:
            parts.append(f"Currently focused on: {focused}")
        recent = self._local["history"][-5:]
        if recent:
            action_strs = [f"  - {a['action']}({a.get('args', {})})" for a in recent]
            parts.append("Recent actions:\n" + "\n".join(action_strs))
        return "\n".join(parts) if parts else "No apps open. Fresh session."

    # ── Firestore sync ──

    def _sync_to_firestore(self):
        if not self._firestore_available:
            return
        try:
            ref = self._session_ref()
            if ref:
                # Firestore can't store int keys or complex nested dicts easily
                # so we serialize the open_apps
                data = {
                    "open_apps": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in self._local["open_apps"].items()},
                    "focused_app": self._local["focused_app"],
                    "preferences": self._local["preferences"],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                ref.set(data, merge=True)
        except Exception as e:
            print(f"[MEMORY] Firestore sync error: {e}")

    def _sync_action_to_firestore(self, entry: dict):
        if not self._firestore_available:
            return
        try:
            ref = self._session_ref()
            if ref:
                # Store actions in a subcollection
                ref.collection("actions").add(entry)
        except Exception as e:
            print(f"[MEMORY] Firestore action log error: {e}")

    def load_from_firestore(self):
        """Load session state from Firestore (for session resumption)."""
        if not self._firestore_available:
            return
        try:
            ref = self._session_ref()
            if ref:
                doc = ref.get()
                if doc.exists:
                    data = doc.to_dict()
                    self._local["open_apps"] = data.get("open_apps", {})
                    self._local["focused_app"] = data.get("focused_app")
                    self._local["preferences"] = data.get("preferences", {})
                    print(f"[MEMORY] Loaded session from Firestore: {self.session_id}")
        except Exception as e:
            print(f"[MEMORY] Firestore load error: {e}")

