"""
app/core/layers/dropout.py
Dropout layer for regularization.
"""
from __future__ import annotations
import numpy as np
from .base import Layer
from ..optimizers import BaseOptimizer


class DropoutLayer(Layer):
    """
    Dropout layer for regularization.
    Randomly sets a fraction of input units to 0 at each update during training.
    
    During training: output = input * mask / (1 - p)
    During inference: output = input
    """

    def __init__(self, rate: float = 0.5):
        """
        Args:
            rate: Fraction of input units to drop (0.0 to 1.0)
        """
        self.rate = float(rate)
        self.keep_prob = 1.0 - self.rate
        
        # Cached values
        self._mask: np.ndarray | None = None
        self._x: np.ndarray | None = None
        self._dW: np.ndarray | None = None  # Always None (no weights)
        self._db: np.ndarray | None = None  # Always None (no bias)

    @property
    def n_in(self) -> int:
        return 0  # Dynamic, depends on input

    @property
    def n_out(self) -> int:
        return 0  # Dynamic, depends on input

    @property
    def is_output(self) -> bool:
        return False

    @property
    def param_count(self) -> int:
        return 0  # No trainable parameters

    # ── forward ──
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self._x = x
        
        if training and self.rate > 0.0:
            # Create inverted dropout mask
            self._mask = (np.random.rand(*x.shape) > self.rate).astype(np.float64)
            # Scale by keep_prob to maintain expected value
            return x * self._mask / self.keep_prob
        else:
            self._mask = None
            return x

    # ── backward ──
    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Propagate gradient through dropout mask.
        """
        if self._mask is not None:
            # Scale gradient by mask and keep_prob
            return delta * self._mask / self.keep_prob
        return delta

    # ── update ──
    def update(self, optimizer: BaseOptimizer, layer_idx: int):
        """No-op: Dropout has no trainable parameters."""
        pass

    # ── serialisation ──
    def to_dict(self) -> dict:
        return {
            "type": "dropout",
            "rate": self.rate,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DropoutLayer":
        return cls(rate=d.get("rate", 0.5))

    # ── inspection helpers ──
    def weight_snapshot(self) -> dict:
        return {
            "W": None,
            "b": None,
            "dW": None,
            "db": None,
            "activation": self._x.tolist() if self._x is not None else None,
            "mask": self._mask.tolist() if self._mask is not None else None,
        }
