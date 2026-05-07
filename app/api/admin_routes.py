"""
app/api/admin_routes.py
Admin utilities for managing built-in modules (architectures, functions, optimizers, etc.)
"""

from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from .. import db
from ..models import BuiltinArchitecture, User

admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")


def is_admin(user: User) -> bool:
    """Check if user is admin."""
    # Check if user has admin privileges
    return user is not None and user.is_admin


@admin_bp.route("/architectures", methods=["GET"])
@login_required
def list_architectures():
    """List all built-in architectures."""
    if not is_admin(request.user if hasattr(request, "user") else None):
        return jsonify({"error": "Unauthorized"}), 403

    architectures = BuiltinArchitecture.query.all()
    return jsonify({"architectures": [arch.to_dict() for arch in architectures]})


@admin_bp.route("/architectures/<key>", methods=["PUT"])
@login_required
def update_architecture(key: str):
    """Update a built-in architecture."""
    user = current_user if current_user.is_authenticated else None
    if not is_admin(user):
        return jsonify({"error": "Unauthorized"}), 403

    arch = BuiltinArchitecture.query.filter_by(key=key).first()
    if not arch:
        return jsonify({"error": "Architecture not found"}), 404

    data = request.get_json()

    # Only allow editing certain fields
    if "label" in data:
        arch.label = data["label"]
    if "description" in data:
        arch.description = data["description"]
    if "accent_color" in data:
        arch.accent_color = data["accent_color"]
    if "diagram_type" in data:
        arch.diagram_type = data["diagram_type"]

    db.session.commit()

    return jsonify(
        {"message": f"Architecture '{key}' updated", "architecture": arch.to_dict()}
    )


@admin_bp.route("/architectures", methods=["POST"])
@login_required
def create_architecture():
    """Create a new built-in architecture."""
    user = current_user if current_user.is_authenticated else None
    if not is_admin(user):
        return jsonify({"error": "Unauthorized"}), 403

    data = request.get_json()
    required_fields = ["key", "label", "description"]

    if not all(field in data for field in required_fields):
        return jsonify(
            {"error": f"Missing required fields: {', '.join(required_fields)}"}
        ), 400

    # Check if key already exists
    existing = BuiltinArchitecture.query.filter_by(key=data["key"]).first()
    if existing:
        return jsonify({"error": f"Architecture '{data['key']}' already exists"}), 409

    arch = BuiltinArchitecture(
        key=data["key"],
        label=data["label"],
        description=data["description"],
        accent_color=data.get("accent_color", "#58a6ff"),
        diagram_type=data.get("diagram_type", "generic"),
        trainable=data.get("trainable", False),
        is_autoencoder=data.get("is_autoencoder", False),
    )

    db.session.add(arch)
    db.session.commit()

    return jsonify(
        {
            "message": f"Architecture '{data['key']}' created",
            "architecture": arch.to_dict(),
        }
    ), 201
