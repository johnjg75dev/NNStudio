"""
app/api/auth_routes.py
Authentication: sign up, sign in, sign out and "who am I".

The React front-end talks to these endpoints with `Accept: application/json`
and gets JSON back. Plain browser form posts still work (they redirect), so the
routes are usable without JavaScript too.
"""
import json

from flask import Blueprint, jsonify, redirect, request, url_for
from flask_login import current_user, login_user, logout_user

from .. import db
from ..auth_tokens import make_token
from ..models import (
    ArchitectureDefinition,
    Dataset,
    LayerDefinition,
    Preset,
    User,
)
from .helpers import api_route, get_registry, ok

auth_bp = Blueprint("auth", __name__)


def _auth_payload(user: User) -> dict:
    """What the SPA gets back after a successful sign-in.

    `token` is the cookie-free credential: browsers that refuse to store a
    third-party session cookie (embedded previews) keep this in localStorage and
    replay it as `X-Session-Token` instead.  Cookie-based browsers ignore it.
    """
    return {
        "ok": True,
        "id": user.id,
        "username": user.username,
        "is_admin": bool(getattr(user, "is_admin", False)),
        "token": make_token(user.id),
    }


def _wants_json() -> bool:
    if request.is_json:
        return True
    accept = request.headers.get("Accept", "")
    xhr = request.headers.get("X-Requested-With", "")
    return "application/json" in accept or xhr == "fetch"


def _unauthorised(message: str, code: int = 400):
    if _wants_json():
        return jsonify({"ok": False, "error": message}), code
    from flask import flash

    flash(message)
    return redirect(request.referrer or url_for("auth.login"))


@auth_bp.get("/api/me")
def me():
    """Current user, or `authenticated: false` so the SPA can route to /login."""
    if not current_user.is_authenticated:
        return ok({"authenticated": False})
    return ok({"authenticated": True, **_auth_payload(current_user)})


@auth_bp.route("/check-username")
@api_route
def check_username():
    username = request.args.get("username", "").strip()
    if not username:
        return ok({"available": False, "message": "Username cannot be empty"})

    if len(username) < 3:
        return ok({"available": False, "message": "Username too short (min 3 chars)"})

    user = User.query.filter_by(username=username).first()
    if user:
        return ok({"available": False, "message": "Username is already taken"})

    return ok({"available": True, "message": "Username is available"})


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        # The SPA renders the login screen; React Router owns the route.
        from .page_routes import index as spa_index

        return spa_index()

    if current_user.is_authenticated:
        return _redirect_or_json("/", _auth_payload(current_user))

    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""

    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return _unauthorised("Invalid username or password", 401)

    login_user(user)
    next_url = request.args.get("next") or request.form.get("next") or "/"
    return _redirect_or_json(next_url, _auth_payload(user))


@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        from .page_routes import index as spa_index

        return spa_index()

    if current_user.is_authenticated:
        return _redirect_or_json("/", _auth_payload(current_user))

    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""

    if len(username) < 3:
        return _unauthorised("Username must be at least 3 characters", 400)
    if len(password) < 6:
        return _unauthorised("Password must be at least 6 characters", 400)
    if User.query.filter_by(username=username).first():
        return _unauthorised("Username already exists", 409)

    new_user = User(username=username)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()

    _seed_account(new_user)
    login_user(new_user)
    return _redirect_or_json("/", _auth_payload(new_user))


@auth_bp.route("/logout")
def logout():
    """End the cookie session.

    A token held in localStorage cannot be revoked from here (it is stateless),
    so the SPA drops it itself before calling this — see `signOut` in AppShell.
    """
    if current_user.is_authenticated:
        logout_user()
    if _wants_json():
        return jsonify({"ok": True})
    return redirect(url_for("auth.login"))


def _redirect_or_json(location: str, payload: dict):
    if _wants_json():
        return jsonify(payload)
    return redirect(location)


def _seed_account(new_user: User) -> None:
    """Give a brand-new account the default layers, architectures, presets and datasets."""
    registry = get_registry()

    db.session.add(
        LayerDefinition(
            user_id=new_user.id,
            name="dense",
            label="Dense (Fully Connected)",
            description=(
                "Standard fully connected layer where every input neuron connects "
                "to every output neuron."
            ),
            type="dense",
            default_activation="tanh",
            default_neurons=4,
        )
    )

    for a in registry.all_of_category("architectures"):
        db.session.add(
            ArchitectureDefinition(
                user_id=new_user.id,
                name=a.key,
                label=a.label,
                description=a.description,
                accent_color=getattr(a, "accent_color", "#58a6ff"),
                diagram_type=getattr(a, "diagram_type", "generic"),
                trainable=getattr(a, "trainable", False),
                is_autoencoder=getattr(a, "is_autoencoder", False),
            )
        )

    for p in registry.all_of_category("presets"):
        db.session.add(
            Preset(
                user_id=new_user.id,
                label=p.label,
                description=p.description,
                arch_key=p.arch_key,
                func_key=p.func_key,
                layers=json.dumps(p.layers),
                activation=getattr(p, "activation", "tanh"),
                optimizer=getattr(p, "optimizer", "adam"),
                loss=getattr(p, "loss", "bce"),
                lr=getattr(p, "lr", 0.01),
                dropout=getattr(p, "dropout", 0.0),
                weight_decay=getattr(p, "weight_decay", 0.0),
            )
        )

    db.session.add(
        Dataset(
            user_id=new_user.id,
            name="MNIST Digits",
            description="The classic dataset of 28x28 handwritten digits.",
            ds_type="mnist",
            num_inputs=784,
            num_outputs=10,
            width=28,
            height=28,
            is_predefined=True,
            downloaded=False,
        )
    )

    db.session.commit()
