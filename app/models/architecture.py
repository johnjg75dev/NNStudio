from .. import db


class BuiltinArchitecture(db.Model):
    """
    Stores built-in architecture metadata in the database.
    Global, admin-managed architectures (not user-specific).
    Allows admin to manage architectures without modifying Python files.
    """
    id = db.Column(db.Integer, primary_key=True)
    
    # Core identifier (must be unique)
    key = db.Column(db.String(50), unique=True, nullable=False, index=True)
    label = db.Column(db.String(100), nullable=False)
    
    # Visual & metadata
    description = db.Column(db.Text, nullable=False, default="")
    accent_color = db.Column(db.String(7), default="#58a6ff")  # hex color
    diagram_type = db.Column(db.String(50), default="generic")
    
    # Behavior flags
    trainable = db.Column(db.Boolean, default=False)
    is_autoencoder = db.Column(db.Boolean, default=False)
    
    # Track source (for data migration from Python files)
    is_migrated = db.Column(db.Boolean, default=False)
    
    def to_dict(self):
        """Convert to the format expected by the frontend & registry."""
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "category": "architectures",
            "trainable": self.trainable,
            "accent_color": self.accent_color,
            "diagram_type": self.diagram_type,
            "is_autoencoder": self.is_autoencoder,
        }
