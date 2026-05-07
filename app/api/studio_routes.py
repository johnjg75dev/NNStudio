"""
app/api/studio_routes.py
/api/studio/*  — API endpoints for the new studio surfaces
(data pipeline, training, evaluation, export/quantization).

These are placeholder endpoints that will be expanded as each
studio surface is developed.
"""
from flask import Blueprint, request
from flask_login import login_required
from .helpers import ok, err, api_route

studio_bp = Blueprint("studio", __name__)


@studio_bp.get("/status")
@login_required
@api_route
def studio_status():
    """Return the status of the studio surfaces."""
    return ok({
        "surfaces": {
            "data":     {"ready": True},
            "train":    {"ready": True},
            "evaluate": {"ready": True},
            "export":   {"ready": True},
        }
    })
