"""
app/api/helpers.py
Shared utilities: session extraction, JSON responses, error wrapping.
"""
from __future__ import annotations
import uuid
import functools
from flask import session, current_app, jsonify, request


def get_session_id() -> str:
    """Return a stable key for this visitor's in-memory training session.

    A signed-in user is keyed by their account, so the same network is still
    there after a refresh, in a second tab, or in a browser that refuses to
    store our session cookie (cross-site iframe previews) — the token auth path
    has no cookie to hang a random id on.  Anonymous visitors fall back to a
    random id kept in the session cookie.
    """
    from flask_login import current_user

    if current_user.is_authenticated:
        return f"user:{current_user.id}"
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


def get_session_manager():
    """Get the global SessionManager instance."""
    from app.core.session_manager import SessionManager
    mgr: SessionManager = current_app.extensions.get("session_manager")
    if mgr is None:
        # lazy-init
        mgr = SessionManager()
        current_app.extensions["session_manager"] = mgr
    return mgr


def get_registry():
    from ..modules.registry import get_registry as _get_reg
    return _get_reg()


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
