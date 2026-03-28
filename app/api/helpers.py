"""
app/api/helpers.py
Shared utilities: session extraction, JSON responses, error wrapping.
"""
from __future__ import annotations
import uuid
import functools
from flask import session, current_app, jsonify, request


def get_session_id() -> str:
    """Return (and persist) a stable session ID for this browser session."""
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


def get_training_session():
    """Fetch the TrainingSession for the current browser session."""
    from app.core.session_manager import SessionManager
    mgr: SessionManager = current_app.extensions.get("session_manager")
    if mgr is None:
        # lazy-init
        mgr = SessionManager()
        current_app.extensions["session_manager"] = mgr
    sid = get_session_id()
    return mgr.get_or_create(sid)


def get_registry():
    return current_app.extensions["module_registry"]


def ok(data: dict | list | None = None, **kwargs) -> tuple:
    payload = {"ok": True}
    if data is not None:
        payload["data"] = data
    payload.update(kwargs)
    return jsonify(payload), 200


def err(message: str, code: int = 400) -> tuple:
    return jsonify({"ok": False, "error": message}), code


def api_route(f):
    """Decorator: catch exceptions and return JSON error automatically."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except KeyError as e:
            return err(f"Not found: {e}", 404)
        except ValueError as e:
            return err(str(e), 400)
        except RuntimeError as e:
            return err(str(e), 500)
        except Exception as e:
            current_app.logger.exception("Unhandled error in API route")
            return err(f"Internal error: {e}", 500)
    return wrapper
