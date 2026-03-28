"""
app/api/auth_routes.py
Authentication routes for registration and login.
"""
from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from ..models import User
from .. import db
from .helpers import api_route, ok

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
            login_user(new_user)
            return redirect(url_for("pages.index"))
            
    return render_template("pages/signup.html")

@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))
