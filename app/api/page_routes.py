"""
app/api/page_routes.py
Serves the React single-page application.

The front-end lives in `frontend/` and is built with Vite into `frontend/dist`.
Flask owns exactly one origin: it serves the compiled bundle and every
non-API route falls through to `index.html` so React Router can handle it.

Development workflow
────────────────────
    cd frontend && npm install && npm run dev      # Vite HMR on :5173, proxies /api → :5000
    python run.py server                           # Flask API on :5000

Production / preview workflow
─────────────────────────────
    cd frontend && npm run build                   # emits frontend/dist
    python run.py server                           # Flask serves the bundle on :5000
"""
from __future__ import annotations

import os

from flask import Blueprint, abort, current_app, send_from_directory

page_bp = Blueprint("pages", __name__)

# Routes owned by other blueprints — never shadow them with the SPA fallback.
RESERVED_PREFIXES = ("api/", "static/", "admin/", "instance/")


def dist_dir() -> str:
    """Absolute path to the built React bundle."""
    root = os.path.abspath(os.path.join(current_app.root_path, os.pardir))
    return os.path.join(root, "frontend", "dist")


def spa_ready() -> bool:
    return os.path.isfile(os.path.join(dist_dir(), "index.html"))


def _missing_bundle_notice() -> str:
    """Friendly placeholder rendered when the front-end has not been built."""
    return """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>NNStudio — front-end not built</title>
<style>
  body{margin:0;height:100vh;display:grid;place-items:center;background:#07090f;color:#eef2fb;
       font:15px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}
  .box{max-width:520px;padding:32px 34px;border:1px solid rgba(255,255,255,.1);border-radius:16px;
       background:#10141f;box-shadow:0 24px 60px -20px rgba(0,0,0,.85)}
  h1{font-size:20px;margin:0 0 10px}
  code{display:block;margin-top:14px;padding:12px 14px;border-radius:8px;background:#070a11;
       color:#6ea0ff;font:13px/1.7 ui-monospace,Menlo,monospace;white-space:pre}
  a{color:#6ea0ff}
</style></head><body><div class="box">
<h1>&#11042; NNStudio front-end not built</h1>
<p>The React bundle is missing. Build it once and refresh:</p>
<code>cd frontend
npm install
npm run build</code>
<p style="color:#a3adc4;font-size:13px;margin-top:14px">
For hot-reload development run <code style="display:inline;padding:1px 5px">npm run dev</code>
inside <b>frontend/</b> and open the Vite URL (it proxies the API to Flask).</p>
</div></body></html>"""


@page_bp.get("/")
def index():
    if not spa_ready():
        return _missing_bundle_notice(), 200
    return send_from_directory(dist_dir(), "index.html")


@page_bp.get("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(os.path.join(dist_dir(), "assets"), filename)


# The SPA embeds its icon as a data URI, so the bundle ships no favicon file.
# Serve one anyway (browsers and crawlers ask for it) — inline, always available.
_FAVICON_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<text y='26' font-size='26'>&#x2B21;</text></svg>"
)


@page_bp.get("/favicon.svg")
def favicon():
    candidate = os.path.join(dist_dir(), "favicon.svg")
    if os.path.isfile(candidate):
        return send_from_directory(dist_dir(), "favicon.svg")
    return current_app.response_class(_FAVICON_SVG, mimetype="image/svg+xml")


@page_bp.get("/<path:path>")
def spa_fallback(path: str):
    """Serve a built static file if it exists, otherwise the SPA shell."""
    if path.startswith(RESERVED_PREFIXES):
        abort(404)
    if not spa_ready():
        return _missing_bundle_notice(), 200

    candidate = os.path.join(dist_dir(), path)
    if os.path.isfile(candidate):
        return send_from_directory(dist_dir(), path)
    return send_from_directory(dist_dir(), "index.html")
