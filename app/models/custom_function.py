"""
app/models/custom_function.py
Database model for user-defined training functions.
"""
from datetime import datetime
from .. import db


class CustomTrainingFunction(db.Model):
    """
    Stores user-defined training functions in Python or JavaScript.
    Functions take a fixed-length input array and return an output array.
    """
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Metadata
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    
    # Function definition
    language = db.Column(db.String(20), nullable=False)  # 'python' or 'javascript'
    code = db.Column(db.Text, nullable=False)  # User's function code
    
    # Input/Output specification
    num_inputs = db.Column(db.Integer, nullable=False)  # Fixed input array length
    num_outputs = db.Column(db.Integer, nullable=False)  # Fixed output array length
    input_labels = db.Column(db.JSON, default=[])  # e.g., ["A", "B"]
    output_labels = db.Column(db.JSON, default=[])  # e.g., ["Sum"]
    
    # Function behavior
    is_classification = db.Column(db.Boolean, default=False)
    # For classification: sample inputs to generate dataset (or user provides dataset)
    sample_strategy = db.Column(db.String(50), default='linspace')  # 'linspace', 'random', 'custom'
    
    # Dataset (optional - for custom sample generation)
    custom_dataset = db.Column(db.JSON, nullable=True)  # [{"x": [...], "y": [...]}, ...]
    
    # Tracking
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Validation
    last_test_result = db.Column(db.JSON, nullable=True)  # Last test execution result
    is_valid = db.Column(db.Boolean, default=False)  # Passes basic syntax/test check
    
    def to_dict(self):
        """Serialize metadata (not code)."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "language": self.language,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "input_labels": self.input_labels or [],
            "output_labels": self.output_labels or [],
            "is_classification": self.is_classification,
            "is_valid": self.is_valid,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }
    
    def to_dict_full(self):
        """Serialize including code."""
        result = self.to_dict()
        result["code"] = self.code
        result["sample_strategy"] = self.sample_strategy
        result["custom_dataset"] = self.custom_dataset
        result["last_test_result"] = self.last_test_result
        return result
