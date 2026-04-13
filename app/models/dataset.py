"""
app/models/dataset.py
Database model for user-defined and predefined datasets.
"""
from datetime import datetime
from .. import db


class Dataset(db.Model):
    """
    Stores datasets which can be tabular, image-based, or predefined.
    Datasets can be "Input-Only" (labels provided by Custom Function)
    or complete (Input + Ground Truth).
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Metadata
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    ds_type = db.Column(db.String(20), nullable=False)  # 'tabular', 'image', 'predefined'
    
    # Content type
    is_input_only = db.Column(db.Boolean, default=False)
    
    # Structure
    num_inputs = db.Column(db.Integer, nullable=False)
    num_outputs = db.Column(db.Integer, nullable=True)  # Null if input_only
    input_labels = db.Column(db.JSON, default=[])
    output_labels = db.Column(db.JSON, default=[])
    
    # For Image types
    width = db.Column(db.Integer, nullable=True)
    height = db.Column(db.Integer, nullable=True)
    channels = db.Column(db.Integer, default=1)
    
    # Storage
    # For 'tabular' and 'image' (small): stored directly as JSON
    # For 'predefined': stores reference/key
    data = db.Column(db.JSON, nullable=True)
    file_path = db.Column(db.String(255), nullable=True)
    
    # Predefined info
    is_predefined = db.Column(db.Boolean, default=False)
    downloaded = db.Column(db.Boolean, default=False)
    
    # Tracking
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        data_length = len(self.data) if self.data else 0
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "ds_type": self.ds_type,
            "is_input_only": self.is_input_only,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "input_labels": self.input_labels or [],
            "output_labels": self.output_labels or [],
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "is_predefined": self.is_predefined,
            "downloaded": self.downloaded,
            "data_length": data_length,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_dict_full(self):
        result = self.to_dict()
        result["data"] = self.data
        result["file_path"] = self.file_path
        return result
