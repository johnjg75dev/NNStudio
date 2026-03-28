"""
app/modules/functions/base_function.py
Abstract base for all training function modules.
Each function defines its dataset, I/O labels, and metadata.
"""
from __future__ import annotations
import numpy as np
from abc import abstractmethod
from ..base import BaseModule


class TrainingFunction(BaseModule):
    """
    A training function module describes one learnable task
    (e.g. XOR, Sine, 7-Segment) and generates its dataset.
    """
    category    = "functions"
    inputs:  int = 2
    outputs: int = 1
    input_labels:  list[str] = []
    output_labels: list[str] = []
    is_classification: bool  = True
    # Default recommended architecture config
    recommended = {
        "layers":        [{"neurons": 4, "activation": "tanh", "type": "dense"}],
        "activation":    "tanh",
        "optimizer":     "adam",
        "loss":          "bce",
        "dropout":       0.0,
        "lr":            0.01,
    }

    @abstractmethod
    def generate_dataset(self) -> list[dict]:
        """Return list of {"x": [...], "y": [...]} dicts."""

    def to_dict(self) -> dict:
        return {
            "key":              self.key,
            "label":            self.label,
            "description":      self.description,
            "category":         self.category,
            "inputs":           self.inputs,
            "outputs":          self.outputs,
            "input_labels":     self.input_labels,
            "output_labels":    self.output_labels,
            "is_classification": self.is_classification,
            "recommended":      self.recommended,
        }
