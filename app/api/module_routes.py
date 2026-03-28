"""
app/api/module_routes.py
/api/modules/*  — expose the ModuleRegistry to the frontend.
"""
from flask import Blueprint, request
from flask_login import current_user
from .helpers import ok, err, api_route, get_registry
from ..models import Preset

module_bp = Blueprint("modules", __name__)


@module_bp.get("/all")
@api_route
def all_modules():
    """Return the full registry grouped by category."""
    registry = get_registry()
    all_data = registry.to_dict()
    
    # Override "presets" with user-specific ones from DB
    if current_user.is_authenticated:
        user_presets = Preset.query.filter_by(user_id=current_user.id).all()
        all_data["presets"] = [p.to_dict() for p in user_presets]
        
        user_archs = ArchitectureDefinition.query.filter_by(user_id=current_user.id).all()
        all_data["architectures"] = [a.to_dict() for a in user_archs]
        
        user_layers = LayerDefinition.query.filter_by(user_id=current_user.id).all()
        all_data["layers"] = [l.to_dict() for l in user_layers]
    
    return ok(all_data)


@module_bp.get("/category/<category>")
@api_route
def by_category(category: str):
    registry = get_registry()
    modules  = registry.all_of_category(category)
    if not modules:
        return err(f"No modules in category '{category}'", 404)
    return ok([m.to_dict() for m in modules])


@module_bp.get("/<key>")
@api_route
def single_module(key: str):
    registry = get_registry()
    mod      = registry.get(key)
    if mod is None:
        return err(f"Module '{key}' not found", 404)
    return ok(mod.to_dict())


@module_bp.get("/functions/<key>/dataset")
@api_route
def function_dataset(key: str):
    """Return the raw dataset for a training function."""
    registry = get_registry()
    mod      = registry.get(key)
    if mod is None:
        return err(f"Function '{key}' not found", 404)
    if not hasattr(mod, "generate_dataset"):
        return err(f"'{key}' is not a TrainingFunction", 400)
    return ok(mod.generate_dataset())
