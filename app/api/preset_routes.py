"""
app/api/preset_routes.py
CRUD routes for user-specific presets.
"""
from flask import Blueprint, request
from flask_login import login_required, current_user
from .helpers import ok, err, api_route
from ..models import Preset
from .. import db

preset_bp = Blueprint("presets", __name__)

@preset_bp.post("/save")
@login_required
@api_route
def save_preset():
    """Create a new preset from the provided config."""
    body = request.get_json(force=True)
    
    label = body.get("label", "My Preset").strip()
    if not label:
        return err("Preset label cannot be empty")
        
    preset = Preset(
        user_id=current_user.id,
        label=label,
        description=body.get("description", ""),
        arch_key=body.get("arch_key", "mlp"),
        func_key=body.get("func_key", "xor"),
        hidden_layers=int(body.get("hidden_layers", 1)),
        neurons=int(body.get("neurons", 4)),
        activation=body.get("activation", "tanh"),
        optimizer=body.get("optimizer", "adam"),
        loss=body.get("loss", "bce"),
        lr=float(body.get("lr", 0.01)),
        dropout=float(body.get("dropout", 0.0)),
        weight_decay=float(body.get("weight_decay", 0.0))
    )
    
    db.session.add(preset)
    db.session.commit()
    
    return ok(preset.to_dict())

@preset_bp.delete("/<int:preset_id>")
@login_required
@api_route
def delete_preset(preset_id: int):
    """Delete a preset if it belongs to the current user."""
    preset = Preset.query.filter_by(id=preset_id, user_id=current_user.id).first()
    if not preset:
        return err("Preset not found or access denied", 404)
        
    db.session.delete(preset)
    db.session.commit()
    
    return ok({"message": "Preset deleted"})
