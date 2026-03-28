"""
app/__init__.py
Flask application factory.  Registers all blueprints and wires the
module-registry so every folder-based module is discovered automatically.
"""
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

from .modules.registry import ModuleRegistry

db = SQLAlchemy()
login_manager = LoginManager()

def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = "nn-trainer-dev-key"
    
    # Database Configuration
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(basedir, "..", "instance", "nnstudio.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # ── custom config overrides (e.g. from tests) ──
    if config:
        app.config.update(config)

    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    # ── build and attach the module registry ──
    registry = ModuleRegistry()
    registry.discover()                # scans all module sub-folders
    app.extensions["module_registry"] = registry

    # ── blueprints ──
    from .api.page_routes import page_bp
    from .api.session_routes import session_bp
    from .api.train_routes import train_bp
    from .api.module_routes import module_bp
    from .api.auth_routes import auth_bp

    app.register_blueprint(page_bp)
    app.register_blueprint(session_bp, url_prefix="/api/session")
    app.register_blueprint(train_bp,   url_prefix="/api/train")
    app.register_blueprint(module_bp,  url_prefix="/api/modules")
    app.register_blueprint(auth_bp)

    # User loader
    from .models import User
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # Create database tables
    with app.app_context():
        db.create_all()

    return app
