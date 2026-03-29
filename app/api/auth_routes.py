"""
app/api/auth_routes.py
Authentication routes for registration and login.
"""
import json
from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from ..models import User, Preset, LayerDefinition, ArchitectureDefinition, Dataset
from .. import db
from .helpers import api_route, ok, get_registry

auth_bp = Blueprint("auth", __name__)

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
    if current_user.is_authenticated:
        return redirect(url_for("pages.index"))
        
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for("pages.index"))
        else:
            flash("Invalid username or password")
            
    return render_template("pages/login.html")

@auth_bp.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("pages.index"))
        
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        
        if User.query.filter_by(username=username).first():
            flash("Username already exists")
        else:
            new_user = User(username=username)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()

            # Initialize registry for seeding defaults
            registry = get_registry()

            # Seed default Layers
            # We hardcode the initial defaults for Dense here, or can pull from a Registry if we had one for Layers
            db_l = LayerDefinition(
                user_id=new_user.id,
                name="dense",
                label="Dense (Fully Connected)",
                description="Standard fully connected layer where every input neuron connects to every output neuron.",
                type="dense",
                default_activation="tanh",
                default_neurons=4
            )
            db.session.add(db_l)

            # Seed default Architectures
            arch_defaults = registry.all_of_category("architectures")
            for a in arch_defaults:
                db_a = ArchitectureDefinition(
                    user_id=new_user.id,
                    name=a.key,
                    label=a.label,
                    description=a.description,
                    accent_color=getattr(a, "accent_color", "#58a6ff"),
                    diagram_type=getattr(a, "diagram_type", "generic"),
                    trainable=getattr(a, "trainable", False),
                    is_autoencoder=getattr(a, "is_autoencoder", False)
                )
                db.session.add(db_a)

            # Seed default presets
            defaults = registry.all_of_category("presets")
            for p in defaults:
                db_p = Preset(
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
                    weight_decay=getattr(p, "weight_decay", 0.0)
                )
                db.session.add(db_p)
            
            # Seed default Datasets
            # Predefined MNIST
            mnist = Dataset(
                user_id=new_user.id,
                name="MNIST Digits",
                description="The classic dataset of 28x28 handwritten digits.",
                ds_type="mnist",
                num_inputs=784,
                num_outputs=10,
                width=28,
                height=28,
                is_predefined=True,
                downloaded=False
            )
            db.session.add(mnist)

            db.session.commit()

            login_user(new_user)
            return redirect(url_for("pages.index"))
            
    return render_template("pages/signup.html")

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))
