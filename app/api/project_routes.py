"""
app/api/project_routes.py

REST endpoints for the project-management surface.

GET    /api/projects                 → list
POST   /api/projects                 → create
GET    /api/projects/<id>            → fetch (with model_graph)
PATCH  /api/projects/<id>            → update (any subset of fields)
DELETE /api/projects/<id>            → delete
POST   /api/projects/<id>/duplicate  → clone with " (copy)" suffix
POST   /api/projects/<id>/touch      → bump last_opened_at
"""
import json
import re
from datetime import datetime
from flask import Blueprint, request
from flask_login import login_required, current_user

from .. import db
from ..models.project import Project
from .helpers import ok, err

project_bp = Blueprint("projects", __name__)


def _slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s or f"project-{int(datetime.utcnow().timestamp())}"


def _query_for_user():
    if current_user.is_authenticated:
        return Project.query.filter(
            (Project.user_id == current_user.id) | (Project.user_id.is_(None))
        )
    return Project.query.filter(Project.user_id.is_(None))


@project_bp.get("")
@login_required
def list_projects():
    limit = request.args.get("limit", type=int)
    q = _query_for_user().order_by(Project.last_opened_at.desc())
    if limit:
        q = q.limit(limit)
    items = [p.to_dict() for p in q.all()]
    return ok({"projects": items, "count": len(items)})


@project_bp.post("")
@login_required
def create_project():
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip() or "Untitled project"
    usecase  = body.get("usecase")  or "custom"
    category = body.get("category") or "tools"

    p = Project(
        user_id=current_user.id if current_user.is_authenticated else None,
        name=name,
        slug=_slugify(name),
        description=body.get("description", ""),
        usecase=usecase,
        category=category,
        model_graph=json.dumps(body.get("model_graph") or {}),
        dataset_cfg=json.dumps(body.get("dataset_cfg") or {}),
        train_cfg=json.dumps(body.get("train_cfg") or {}),
        eval_cfg=json.dumps(body.get("eval_cfg") or {}),
        export_cfg=json.dumps(body.get("export_cfg") or {}),
    )
    p.tags = body.get("tags") or []
    db.session.add(p)
    db.session.commit()
    return ok({"project": p.to_dict(include_graph=True)})


@project_bp.get("/<int:pid>")
@login_required
def get_project(pid):
    p = _query_for_user().filter_by(id=pid).first()
    if not p:
        return err("Not found", 404)
    return ok({"project": p.to_dict(include_graph=True)})


@project_bp.patch("/<int:pid>")
@login_required
def patch_project(pid):
    p = _query_for_user().filter_by(id=pid).first()
    if not p:
        return err("Not found", 404)
    body = request.get_json(silent=True) or {}
    for key in ("name", "description", "usecase", "category"):
        if key in body and body[key] is not None:
            setattr(p, key, body[key])
    if "tags" in body:
        p.tags = body["tags"] or []
    if "starred" in body:
        p.starred = bool(body["starred"])
    if "archived" in body:
        p.archived = bool(body["archived"])
    for key in ("model_graph", "dataset_cfg", "train_cfg", "eval_cfg", "export_cfg"):
        if key in body and body[key] is not None:
            setattr(p, key, json.dumps(body[key]))
    if "name" in body and body["name"]:
        p.slug = _slugify(body["name"])
    db.session.commit()
    return ok({"project": p.to_dict(include_graph=True)})


@project_bp.delete("/<int:pid>")
@login_required
def delete_project(pid):
    p = _query_for_user().filter_by(id=pid).first()
    if not p:
        return err("Not found", 404)
    db.session.delete(p)
    db.session.commit()
    return ok({"deleted": pid})


@project_bp.post("/<int:pid>/duplicate")
@login_required
def duplicate_project(pid):
    p = _query_for_user().filter_by(id=pid).first()
    if not p:
        return err("Not found", 404)
    clone = Project(
        user_id=p.user_id,
        name=f"{p.name} (copy)",
        slug=_slugify(f"{p.name}-copy"),
        description=p.description,
        usecase=p.usecase,
        category=p.category,
        model_graph=p.model_graph,
        dataset_cfg=p.dataset_cfg,
        train_cfg=p.train_cfg,
        eval_cfg=p.eval_cfg,
        export_cfg=p.export_cfg,
    )
    clone.tags = p.tags
    db.session.add(clone)
    db.session.commit()
    return ok({"project": clone.to_dict()})


@project_bp.post("/<int:pid>/touch")
@login_required
def touch_project(pid):
    p = _query_for_user().filter_by(id=pid).first()
    if not p:
        return err("Not found", 404)
    p.last_opened_at = datetime.utcnow()
    db.session.commit()
    return ok({"project": p.to_dict()})
