"""
app/modules/architectures/database_architecture.py
Dynamic architecture module loaded from the BuiltinArchitecture database.
"""
from ..base import BaseModule


class DatabaseArchitecture(BaseModule):
    """
    Wraps a BuiltinArchitecture database record to look like a module.
    """
    category = "architectures"
    
    def __init__(self, arch_record):
        """
        Args:
            arch_record: BuiltinArchitecture database model instance
        """
        self.arch_record = arch_record
        self.key = arch_record.key
        self.label = arch_record.label
        self.description = arch_record.description
        self.accent_color = arch_record.accent_color
        self.diagram_type = arch_record.diagram_type
        self.trainable = arch_record.trainable
        self.is_autoencoder = arch_record.is_autoencoder
    
    def to_dict(self):
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "category": self.category,
            "trainable": self.trainable,
            "accent_color": self.accent_color,
            "diagram_type": self.diagram_type,
            "is_autoencoder": self.is_autoencoder,
        }
