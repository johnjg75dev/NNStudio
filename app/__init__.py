"""
app/__init__.py
Flask application factory.  Registers all blueprints and wires the
module-registry so every folder-based module is discovered automatically.
"""
from flask import Flask

from .modules.registry import ModuleRegistry
from .api.session_routes import session_bp
from .api.train_routes   import train_bp
from .api.module_routes  import module_bp
from .api.page_routes    import page_bp


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = "nn-trainer-dev-key"

    # ── custom config overrides (e.g. from tests) ──
    if config:
        app.config.update(config)

    # ── build and attach the module registry ──
    registry = ModuleRegistry()
    registry.discover()                # scans all module sub-folders
    app.extensions["module_registry"] = registry

    # ── blueprints ──
    app.register_blueprint(page_bp)
    app.register_blueprint(session_bp, url_prefix="/api/session")
    app.register_blueprint(train_bp,   url_prefix="/api/train")
    app.register_blueprint(module_bp,  url_prefix="/api/modules")

    return app
