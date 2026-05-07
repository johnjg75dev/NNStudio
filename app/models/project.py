"""
app/models/project.py

Project ORM model. Stores everything needed to re-open a workspace:
- identity (name, slug, owner)
- scenario / use-case starter that scaffolded it
- model graph definition (JSON: layers, connections, hyperparams)
- dataset reference + augmentation graph (JSON)
- training, eval, export, and quant settings (JSON)
- soft state: starred, archived, tags
- audit: created_at, updated_at, last_opened_at

The schema is portable: the `JSON` columns are TEXT under SQLite and MySQL,
so a future move from SQLite → MySQL is a config-only change.
"""
from datetime import datetime
import json
from .. import db


class Project(db.Model):
    __tablename__ = "projects"

    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True, index=True)

    name          = db.Column(db.String(160), nullable=False)
    slug          = db.Column(db.String(180), nullable=False, index=True)
    description   = db.Column(db.Text, default="")

    usecase       = db.Column(db.String(80), default="custom", index=True)   # e.g. "tab_classify"
    category      = db.Column(db.String(40), default="tools",   index=True)  # e.g. "tabular"

    # JSON-encoded blobs, stored as TEXT for portability
    model_graph   = db.Column(db.Text, default="{}")        # nodes/edges/hyperparams for the playground
    dataset_cfg   = db.Column(db.Text, default="{}")        # source, splits, augmentation
    train_cfg     = db.Column(db.Text, default="{}")        # optimizer, schedule, hpo space
    eval_cfg      = db.Column(db.Text, default="{}")
    export_cfg    = db.Column(db.Text, default="{}")        # quant matrix + qat budget

    tags_csv      = db.Column(db.String(400), default="")   # comma-separated tags
    starred       = db.Column(db.Boolean, default=False)
    archived      = db.Column(db.Boolean, default=False)

    created_at    = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at    = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_opened_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # ── helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _loads(s):
        try:
            return json.loads(s) if s else {}
        except Exception:
            return {}

    @property
    def tags(self):
        return [t.strip() for t in (self.tags_csv or "").split(",") if t.strip()]

    @tags.setter
    def tags(self, value):
        self.tags_csv = ",".join((value or []))

    def to_dict(self, include_graph: bool = False):
        d = {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "slug": self.slug,
            "description": self.description or "",
            "usecase": self.usecase,
            "category": self.category,
            "tags": self.tags,
            "starred": bool(self.starred),
            "archived": bool(self.archived),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_opened_at": self.last_opened_at.isoformat() if self.last_opened_at else None,
        }
        if include_graph:
            d.update({
                "model_graph": self._loads(self.model_graph),
                "dataset_cfg": self._loads(self.dataset_cfg),
                "train_cfg":   self._loads(self.train_cfg),
                "eval_cfg":    self._loads(self.eval_cfg),
                "export_cfg":  self._loads(self.export_cfg),
            })
        return d
