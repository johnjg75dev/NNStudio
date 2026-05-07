"""
app/api/usecase_routes.py — Admin CRUD API for managing usecases and categories.
All endpoints require admin authentication (to be secured later).
"""
from flask import Blueprint, request, jsonify
from app import db
from app.models.usecase import Usecase, UsecaseCategory
from app.api.helpers import ok, err, api_route

usecase_bp = Blueprint("usecases", __name__, url_prefix="/api/usecases")


# ─────────────────────────────────────────────────────────────────────────
# PUBLIC: Gallery and category listing
# ─────────────────────────────────────────────────────────────────────────

@usecase_bp.route("/categories", methods=["GET"])
@api_route
def list_categories():
    """List all usecase categories with count of usecases in each."""
    cats = UsecaseCategory.query.order_by(UsecaseCategory.display_order).all()
    return ok(
        {
            "categories": [
                {
                    **c.to_dict(),
                    "usecase_count": Usecase.query.filter_by(category_id=c.id, is_active=True).count(),
                }
                for c in cats
            ]
        }
    )


@usecase_bp.route("/gallery", methods=["GET"])
@api_route
def list_usecases_gallery():
    """List all active usecases grouped by category. Used by landing gallery and wizard."""
    tag_filter = request.args.get("tag")  # Optional filter by tag
    
    usecases = Usecase.query.filter_by(is_active=True).order_by(Usecase.display_order).all()
    
    # Apply tag filter if provided
    if tag_filter:
        usecases = [u for u in usecases if tag_filter in (u.tags or "").split(",")]
    
    return ok(
        {
            "categories": [c.to_dict() for c in UsecaseCategory.query.order_by(UsecaseCategory.display_order).all()],
            "usecases": [u.to_dict() for u in usecases],
        }
    )


@usecase_bp.route("/<key>", methods=["GET"])
@api_route
def get_usecase(key):
    """Get a single usecase by key."""
    uc = Usecase.query.filter_by(key=key, is_active=True).first()
    if not uc:
        return err(f"Usecase '{key}' not found", 404)
    return ok({"usecase": uc.to_dict()})


# ─────────────────────────────────────────────────────────────────────────
# ADMIN: Category management
# ─────────────────────────────────────────────────────────────────────────

@usecase_bp.route("/admin/categories", methods=["POST"])
@api_route
def create_category():
    """Create a new usecase category. [ADMIN]"""
    data = request.get_json() or {}
    
    # Validate required fields
    for field in ["key", "label", "blurb", "icon", "color"]:
        if not data.get(field):
            return err(f"Missing required field: {field}")
    
    # Check for duplicate key
    if UsecaseCategory.query.filter_by(key=data["key"]).first():
        return err(f"Category key '{data['key']}' already exists")
    
    cat = UsecaseCategory(
        key=data["key"],
        label=data["label"],
        blurb=data["blurb"],
        icon=data["icon"],
        color=data["color"],
        display_order=data.get("display_order", 0),
    )
    db.session.add(cat)
    db.session.commit()
    
    return ok({"category": cat.to_dict()}, 201)


@usecase_bp.route("/admin/categories/<int:cat_id>", methods=["PUT"])
@api_route
def update_category(cat_id):
    """Update a usecase category. [ADMIN]"""
    cat = UsecaseCategory.query.get(cat_id)
    if not cat:
        return err("Category not found", 404)
    
    data = request.get_json() or {}
    cat.label = data.get("label", cat.label)
    cat.blurb = data.get("blurb", cat.blurb)
    cat.icon = data.get("icon", cat.icon)
    cat.color = data.get("color", cat.color)
    cat.display_order = data.get("display_order", cat.display_order)
    
    db.session.commit()
    return ok({"category": cat.to_dict()})


@usecase_bp.route("/admin/categories/<int:cat_id>", methods=["DELETE"])
@api_route
def delete_category(cat_id):
    """Delete a usecase category (soft delete via archiving). [ADMIN]"""
    cat = UsecaseCategory.query.get(cat_id)
    if not cat:
        return err("Category not found", 404)
    
    # Soft delete: mark all usecases in this category as inactive
    Usecase.query.filter_by(category_id=cat_id).update({"is_active": False})
    db.session.delete(cat)
    db.session.commit()
    
    return ok({"message": "Category deleted"})


# ─────────────────────────────────────────────────────────────────────────
# ADMIN: Usecase management
# ─────────────────────────────────────────────────────────────────────────

@usecase_bp.route("/admin", methods=["GET"])
@api_route
def list_usecases_admin():
    """List all usecases (including inactive) for admin panel. [ADMIN]"""
    usecases = Usecase.query.order_by(Usecase.display_order).all()
    return ok(
        {
            "usecases": [u.to_admin_dict() for u in usecases],
            "total": len(usecases),
        }
    )


@usecase_bp.route("/admin", methods=["POST"])
@api_route
def create_usecase():
    """Create a new usecase. [ADMIN]"""
    data = request.get_json() or {}
    
    # Validate required fields
    for field in ["key", "category_id", "title", "description", "icon", "tags"]:
        if field not in data or not data[field]:
            return err(f"Missing required field: {field}")
    
    # Check for duplicate key
    if Usecase.query.filter_by(key=data["key"]).first():
        return err(f"Usecase key '{data['key']}' already exists")
    
    # Verify category exists
    cat = UsecaseCategory.query.get(data["category_id"])
    if not cat:
        return err("Category not found", 404)
    
    uc = Usecase(
        key=data["key"],
        category_id=data["category_id"],
        title=data["title"],
        description=data["description"],
        icon=data["icon"],
        tags=data.get("tags", "Local"),
        starter_config=data.get("starter_config", {}),
        is_active=data.get("is_active", True),
        display_order=data.get("display_order", 0),
    )
    db.session.add(uc)
    db.session.commit()
    
    return ok({"usecase": uc.to_admin_dict()}, 201)


@usecase_bp.route("/admin/<int:uc_id>", methods=["PUT"])
@api_route
def update_usecase(uc_id):
    """Update a usecase. [ADMIN]"""
    uc = Usecase.query.get(uc_id)
    if not uc:
        return err("Usecase not found", 404)
    
    data = request.get_json() or {}
    
    # Update fields (allow partial updates)
    if "category_id" in data:
        cat = UsecaseCategory.query.get(data["category_id"])
        if not cat:
            return err("Category not found", 404)
        uc.category_id = data["category_id"]
    
    uc.title = data.get("title", uc.title)
    uc.description = data.get("description", uc.description)
    uc.icon = data.get("icon", uc.icon)
    uc.tags = data.get("tags", uc.tags)
    uc.starter_config = data.get("starter_config", uc.starter_config)
    uc.is_active = data.get("is_active", uc.is_active)
    uc.display_order = data.get("display_order", uc.display_order)
    
    db.session.commit()
    return ok({"usecase": uc.to_admin_dict()})


@usecase_bp.route("/admin/<int:uc_id>", methods=["DELETE"])
@api_route
def delete_usecase(uc_id):
    """Delete a usecase (soft delete via is_active flag). [ADMIN]"""
    uc = Usecase.query.get(uc_id)
    if not uc:
        return err("Usecase not found", 404)
    
    uc.is_active = False
    db.session.commit()
    
    return ok({"message": "Usecase archived"})


@usecase_bp.route("/admin/<int:uc_id>/activate", methods=["POST"])
@api_route
def activate_usecase(uc_id):
    """Re-activate an archived usecase. [ADMIN]"""
    uc = Usecase.query.get(uc_id)
    if not uc:
        return err("Usecase not found", 404)
    
    uc.is_active = True
    db.session.commit()
    
    return ok({"usecase": uc.to_admin_dict()})


@usecase_bp.route("/admin/reorder", methods=["POST"])
@api_route
def reorder_usecases():
    """Reorder usecases. Expects { "ids": [id1, id2, ...] } where order = index. [ADMIN]"""
    data = request.get_json() or {}
    ids = data.get("ids", [])
    
    if not ids:
        return err("Missing 'ids' array")
    
    for idx, uc_id in enumerate(ids):
        uc = Usecase.query.get(uc_id)
        if uc:
            uc.display_order = idx
    
    db.session.commit()
    return ok({"message": "Usecases reordered"})
