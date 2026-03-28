"""
app/modules/architectures/base_architecture.py
Abstract base for architecture descriptor modules.
These are educational/visual — they carry diagram data sent to the
frontend canvas renderer but do NOT run training themselves (except MLP).
"""
from __future__ import annotations
from ..base import BaseModule


class ArchitectureModule(BaseModule):
    """
    Describes a neural network architecture.

    Attributes
    ----------
    trainable       : True only for MLP & Autoencoder (run live in-browser)
    accent_color    : hex colour used in the UI badge
    diagram_type    : slug used by the JS canvas renderer to pick draw function
    is_autoencoder  : hint to force symmetric topology
    """
    category       = "architectures"
    trainable:      bool  = False
    accent_color:   str   = "#58a6ff"
    diagram_type:   str   = "generic"
    is_autoencoder: bool  = False

    def to_dict(self) -> dict:
        return {
            "key":           self.key,
            "label":         self.label,
            "description":   self.description,
            "category":      self.category,
            "trainable":     self.trainable,
            "accent_color":  self.accent_color,
            "diagram_type":  self.diagram_type,
            "is_autoencoder": self.is_autoencoder,
        }
