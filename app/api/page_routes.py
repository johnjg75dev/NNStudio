"""
app/api/page_routes.py
Serves the single-page application shell.
"""
from flask import Blueprint, render_template
from .helpers import get_registry

page_bp = Blueprint("pages", __name__)


@page_bp.get("/")
def index():
    registry = get_registry()
    data = registry.to_dict()
    print(data)
    return render_template(
        "pages/index.html",
        registry=data,
    )
