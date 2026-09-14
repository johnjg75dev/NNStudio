"""
app/auth_tokens.py
Signed bearer tokens — the cookie-free half of authentication.

Why this exists
───────────────
The studio is routinely opened inside an iframe on someone else's origin (a live
preview, a portal, a docs page).  Browsers increasingly refuse to *store* a
session cookie in that third-party context even with `SameSite=None; Secure`, so
`POST /login` succeeds and the very next request is anonymous again — the SPA
then bounces back to /login forever.

`localStorage` still works inside those iframes (partitioned per top-level site),
so the login response also carries a signed token.  The SPA stores it and sends
it back as `X-Session-Token`; `login_manager.request_loader` turns it into
`current_user`.  Nothing here replaces the cookie — both paths are honoured, and
the cookie wins when it is present.

The token is just the user id signed with the app's secret key, so there is no
server-side state to store or expire.  Revocation = changing SECRET_KEY.
"""
from __future__ import annotations

from flask import current_app
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

SALT = "nnstudio.session-token"
MAX_AGE_SECONDS = 30 * 24 * 60 * 60  # 30 days


def _serializer(app=None) -> URLSafeTimedSerializer:
    app = app or current_app
    return URLSafeTimedSerializer(app.secret_key, salt=SALT)


def make_token(user_id: int, app=None) -> str:
    """Sign a user id into a token the SPA can keep in localStorage."""
    return _serializer(app).dumps(int(user_id))


def read_token(token: str | None, app=None) -> int | None:
    """Return the user id a token was issued for, or None if it is not valid."""
    if not token or not isinstance(token, str):
        return None
    try:
        return int(_serializer(app).loads(token, max_age=MAX_AGE_SECONDS))
    except (BadSignature, SignatureExpired, ValueError, TypeError):
        return None


def token_from_request(req) -> str | None:
    """Pull the token out of a request: header first, query string as a fallback."""
    return req.headers.get("X-Session-Token") or req.args.get("session_token")
