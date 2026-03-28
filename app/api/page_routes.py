"""
app/api/page_routes.py
Serves the single-page application shell.
"""
from flask import Blueprint, render_template
from flask_login import login_required
from .helpers import get_registry

page_bp = Blueprint("pages", __name__)


@page_bp.get("/")
@login_required
def index():
    registry = get_registry()
    return render_template(
        "pages/index.html",
        registry=registry.to_dict(),
    )
