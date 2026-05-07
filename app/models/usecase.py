"""
Usecase model — represents a pre-built scenario/use case that users can
select from the landing gallery. Admins can CRUD these via admin interface.
"""
from app import db
from datetime import datetime
import json


class UsecaseCategory(db.Model):
    """Represents a category of usecases (Tabular, Vision, NLP, etc.)"""
    __tablename__ = "usecase_categories"

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False)  # e.g., "tabular", "vision"
    label = db.Column(db.String(255), nullable=False)  # e.g., "Tabular & Structured Data"
    blurb = db.Column(db.Text, nullable=False)  # category description
    icon = db.Column(db.String(50), nullable=False)  # icon name from ICONS dict
    color = db.Column(db.String(50), nullable=False)  # CSS var or hex color
    display_order = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    usecases = db.relationship("Usecase", backref="category", lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "key": self.key,
            "label": self.label,
            "blurb": self.blurb,
            "icon": self.icon,
            "color": self.color,
            "display_order": self.display_order,
        }


class Usecase(db.Model):
    """Represents a single usecase (e.g., "Image Classification", "XOR Gate")"""
    __tablename__ = "usecases"

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(50), unique=True, nullable=False)  # e.g., "img_classify", "fnd_xor"
    category_id = db.Column(db.Integer, db.ForeignKey("usecase_categories.id"), nullable=False)
    title = db.Column(db.String(255), nullable=False)  # e.g., "Image Classification"
    description = db.Column(db.Text, nullable=False)  # rich description
    icon = db.Column(db.String(50), nullable=False)  # icon name
    tags = db.Column(db.String(255), default="Local")  # comma-separated: "Local,Cloud,Preview"
    
    # Starter config (JSON): { "arch": "cnn", "loss": "ce", "layers": [...], "task": "xor", "blank": true }
    starter_config = db.Column(db.JSON, nullable=True)
    
    # Metadata
    is_active = db.Column(db.Boolean, default=True)
    display_order = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Serialize to frontend format (matches usecases.js structure)"""
        return {
            "id": self.id,
            "key": self.key,
            "cat": self.category.key,
            "title": self.title,
            "desc": self.description,
            "icon": self.icon,
            "tags": [t.strip() for t in self.tags.split(",")] if self.tags else ["Local"],
            "starter": self.starter_config or {},
        }

    def to_admin_dict(self):
        """Serialize for admin interface (includes internal metadata)"""
        d = self.to_dict()
        d.update({
            "id": self.id,
            "category_id": self.category_id,
            "is_active": self.is_active,
            "display_order": self.display_order,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        })
        return d
