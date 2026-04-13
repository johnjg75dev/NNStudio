"""
app/models/saved_model.py
Database model for persisted trained neural networks.
"""
from datetime import datetime
from .. import db


class SavedModel(db.Model):
    """
    Stores trained neural network models in the database.
    Includes metadata, serialized architecture, and weights.
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Metadata
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    
    # Model configuration and weights (JSON serialization)
    model_data = db.Column(db.JSON, nullable=False)  # Stores to_dict() output
    
    # Tracking
    architecture_name = db.Column(db.String(50), nullable=True)
    function_name = db.Column(db.String(50), nullable=True)
    epochs_trained = db.Column(db.Integer, default=0)
    final_loss = db.Column(db.Float, nullable=True)
    final_accuracy = db.Column(db.Float, nullable=True)
    
    # History and snapshots
    history = db.Column(db.JSON, default=[])  # Periodic evaluation logs
    snapshots = db.Column(db.JSON, default=[]) # Model state snapshots (modifications)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Serialize model metadata (not the full weights)."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "architecture_name": self.architecture_name,
            "function_name": self.function_name,
            "epochs_trained": self.epochs_trained,
            "final_loss": self.final_loss,
            "final_accuracy": self.final_accuracy,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_dict_full(self):
        """Serialize model including weights (for export/sharing)."""
        result = self.to_dict()
        result["model_data"] = self.model_data
        return result
