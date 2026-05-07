"""
app/api/page_routes.py

Page routes for the redesigned NNStudio UI.

Each route renders a top-level surface in the new IA:
  /                → home (use-case gallery + recent projects)
  /projects        → project management list
  /projects/new    → new-project wizard
  /playground      → live model playground (N8N-style)
  /data            → data pipeline studio
  /train           → training + HPO surface
  /evaluate        → evaluation studio
  /export          → export + quantization + QAT studio
  /layers          → layer library reference
  /settings        → app settings

The legacy /classic route remains available for the original SPA shell.
"""
from flask import Blueprint, render_template
from flask_login import login_required
from .helpers import get_registry

page_bp = Blueprint("pages", __name__)


@page_bp.get("/")
@login_required
def home():
    return render_template("pages/home.html")


@page_bp.get("/projects")
@login_required
def projects():
    return render_template("pages/projects.html")


@page_bp.get("/projects/new")
@login_required
def new_project():
    return render_template("pages/new_project.html")


@page_bp.get("/playground")
@login_required
def playground():
    registry = get_registry()
    return render_template(
        "pages/playground.html",
        registry=registry.to_dict(),
    )


@page_bp.get("/data")
@login_required
def data_studio():
    return render_template("pages/data_studio.html")


@page_bp.get("/train")
@login_required
def train_studio():
    return render_template("pages/train_studio.html")


@page_bp.get("/evaluate")
@login_required
def eval_studio():
    return render_template("pages/eval_studio.html")


@page_bp.get("/export")
@login_required
def quant_studio():
    return render_template("pages/quant_studio.html")


@page_bp.get("/layers")
@login_required
def layer_library():
    return render_template("pages/layer_library.html")


@page_bp.get("/settings")
@login_required
def settings():
    return render_template("pages/settings.html")


@page_bp.get("/admin/usecases")
@login_required
def admin_usecases():
    return render_template("pages/admin_usecases.html")


# Legacy SPA shell — kept available for direct access while the redesign
# is rolled out.  Redirects do NOT happen from "/" because the new home
# IS the new "/", per the IA in design.
@page_bp.get("/classic")
@login_required
def classic_index():
    registry = get_registry()
    return render_template(
        "pages/index.html",
        registry=registry.to_dict(),
    )
