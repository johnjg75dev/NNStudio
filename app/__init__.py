"""
app/__init__.py
Flask application factory.  Registers all blueprints and wires the
module-registry so every folder-based module is discovered automatically.
"""
import os
from flask import Flask, has_request_context, jsonify, redirect, request, url_for
from flask.sessions import SecureCookieSessionInterface
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

from .modules.registry import ModuleRegistry

db = SQLAlchemy()
login_manager = LoginManager()


class AdaptiveSessionInterface(SecureCookieSessionInterface):
    """Session cookies that survive being embedded in another site.

    The studio is usually opened through an HTTPS proxy/iframe (a live preview,
    a portal, a reverse proxy on another domain).  A default ``SameSite=Lax``
    cookie is *not* sent on cross-site iframe requests, so every call would look
    anonymous and the SPA would bounce between /train and /login forever.

    ``SameSite=None`` fixes that, but browsers only honour it together with
    ``Secure`` — so both flags are switched on when the request really arrives
    over HTTPS, and left at the same-site defaults for a plain
    ``http://localhost:5000`` run.
    """

    @staticmethod
    def _over_https() -> bool:
        if not has_request_context():
            return False
        forwarded = request.headers.get("X-Forwarded-Proto", "")
        return bool(request.is_secure or "https" in forwarded.lower())

    def get_cookie_secure(self, app):
        # Flask's own default is False, so only an explicit True pins it on.
        if app.config.get("SESSION_COOKIE_SECURE"):
            return True
        return self._over_https()

    def get_cookie_samesite(self, app):
        configured = app.config.get("SESSION_COOKIE_SAMESITE")
        if configured:
            return configured
        return "None" if self._over_https() else "Lax"

def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = "nn-trainer-dev-key"
    
    # Database Configuration
    basedir = os.path.abspath(os.path.dirname(__file__))
    instance_dir = os.path.join(basedir, "..", "instance")
    os.makedirs(instance_dir, exist_ok=True)   # a fresh clone has no instance/ folder
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(instance_dir, "nnstudio.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # ── custom config overrides (e.g. from tests) ──
    if config:
        app.config.update(config)

    # ── session cookies that work behind an HTTPS proxy / inside an iframe ──
    app.session_interface = AdaptiveSessionInterface()
    if os.environ.get("SESSION_COOKIE_SAMESITE"):
        app.config["SESSION_COOKIE_SAMESITE"] = os.environ["SESSION_COOKIE_SAMESITE"]
    if os.environ.get("SESSION_COOKIE_SECURE"):
        app.config["SESSION_COOKIE_SECURE"] = (
            os.environ["SESSION_COOKIE_SECURE"].lower() in ("1", "true", "yes", "on")
        )

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    # ── auth failures: JSON for the SPA, redirect for a plain browser ──
    # Flask-Login's default is a 302 to the login page. `fetch()` follows that
    # transparently, so an API call would end up parsing the login HTML and
    # fail with a confusing error. The SPA already turns a 401 into a
    # client-side bounce to /login?next=<path>.
    @login_manager.unauthorized_handler
    def _unauthorised():
        wants_json = (
            request.path.startswith("/api/")
            or request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or request.accept_mimetypes.best == "application/json"
        )
        if wants_json:
            return jsonify(ok=False, error="Authentication required."), 401
        return redirect(url_for("auth.login", next=request.full_path.rstrip("?")))

    # ── build and attach the module registry ──
    registry = ModuleRegistry()
    registry.discover()                # scans all module sub-folders
    app.extensions["module_registry"] = registry
    
    # Load built-in architectures from database (after db is initialized)
    with app.app_context():
        registry.load_architectures_from_database()

    # ── blueprints ──
    from .api.page_routes import page_bp
    from .api.session_routes import session_bp
    from .api.train_routes import train_bp
    from .api.module_routes import module_bp
    from .api.auth_routes import auth_bp
    from .api.preset_routes import preset_bp
    from .api.model_routes import model_bp
    from .api.custom_function_routes import custom_function_bp
    from .api.dataset_routes import dataset_bp
    from .api.admin_routes import admin_bp

    app.register_blueprint(page_bp)
    app.register_blueprint(session_bp, url_prefix="/api/session")
    app.register_blueprint(train_bp,   url_prefix="/api/train")
    app.register_blueprint(module_bp,  url_prefix="/api/modules")
    app.register_blueprint(auth_bp)
    app.register_blueprint(preset_bp,  url_prefix="/api/presets")
    app.register_blueprint(model_bp,   url_prefix="/api/models")
    app.register_blueprint(custom_function_bp, url_prefix="/api/functions/custom")
    app.register_blueprint(dataset_bp, url_prefix="/api/datasets")
    app.register_blueprint(admin_bp)

    # User loader
    from .models import User, Preset
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Create database tables
    with app.app_context():
        db.create_all()

    return app
