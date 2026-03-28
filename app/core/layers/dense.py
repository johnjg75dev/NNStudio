from __future__ import annotations
import numpy as np
from .base import Layer
from ..activations import ACTIVATIONS, Activation
from ..optimizers import BaseOptimizer

class DenseLayer(Layer):
    """
    Fully-connected layer:
        z = W·x + b
        a = activation(z)   [hidden layers]
        a = sigmoid(z)      [output layer — always]
    """

    def __init__(self,
                 n_in:      int,
                 n_out:     int,
                 activation: Activation,
                 is_output: bool  = False):
        self.n_in      = n_in
        self.n_out     = n_out
        self.activation = activation
        self.is_output = is_output

        # He initialisation
        scale  = np.sqrt(2.0 / n_in)
        self.W = np.random.randn(n_out, n_in).astype(np.float64) * scale
        self.b = np.zeros(n_out, dtype=np.float64)

        # Cached values (set during forward/backward)
        self._x:    np.ndarray | None = None   # input
        self._z:    np.ndarray | None = None   # pre-activation
        self._a:    np.ndarray | None = None   # post-activation
        self._dW:   np.ndarray | None = None
        self._db:   np.ndarray | None = None

    # ── forward ──
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self._x = x
        self._z = self.W @ x + self.b

        if self.is_output:
            # sigmoid output (works for BCE and MSE)
            a = 1.0 / (1.0 + np.exp(-np.clip(self._z, -500, 500)))
        else:
            a = self.activation.forward(self._z)

        self._a = a
        return a

    # ── backward ──
    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        delta is the gradient w.r.t. pre-activation z of *this* layer
        (already computed by the caller for output layer, or propagated here).
        Returns delta for the previous layer.
        """
        if not self.is_output:
            act_d = self.activation.derivative(self._z)
            delta = delta * act_d

        self._dW = np.outer(delta, self._x)
        self._db = delta.copy()

        return self.W.T @ delta  # downstream delta

    # ── update ──
    def update(self, optimizer: BaseOptimizer, layer_idx: int):
        optimizer.tick()
        prefix = f"L{layer_idx}"
        self.W = optimizer.step(self.W, self._dW, key=f"{prefix}_W")
        self.b = optimizer.step(self.b, self._db, key=f"{prefix}_b")

    # ── serialisation ──
    def to_dict(self) -> dict:
        return {
            "type":       "dense",
            "n_in":       self.n_in,
            "n_out":      self.n_out,
            "activation": self.activation.name,
            "is_output":  self.is_output,
            "W":          self.W.tolist(),
            "b":          self.b.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DenseLayer":
        act = ACTIVATIONS[d["activation"]]
        layer = cls(d["n_in"], d["n_out"], act,
                    is_output=d.get("is_output", False))
        layer.W = np.array(d["W"], dtype=np.float64)
        layer.b = np.array(d["b"], dtype=np.float64)
        return layer

    @property
    def param_count(self) -> int:
        return self.W.size + self.b.size

    # ── inspection helpers ──
    def weight_snapshot(self) -> dict:
        return {
            "W":    self.W.tolist(),
            "b":    self.b.tolist(),
            "dW":   self._dW.tolist() if self._dW is not None else None,
            "db":   self._db.tolist() if self._db is not None else None,
            "activation": self._a.tolist() if self._a is not None else None,
        }
