"""
app/core/layers/batch_norm.py
Batch Normalization layer.
"""
from __future__ import annotations
import numpy as np
from .base import Layer
from ..optimizers import BaseOptimizer


class BatchNormLayer(Layer):
    """
    Batch Normalization layer.
    Normalizes the input to have zero mean and unit variance, then applies
    learnable scale (gamma) and shift (beta) parameters.
    
    During training: uses batch statistics
    During inference: uses running averages
    """

    def __init__(self, n_features: int, momentum: float = 0.9, eps: float = 1e-8):
        """
        Args:
            n_features: Number of features to normalize
            momentum: Momentum for running averages
            eps: Small constant for numerical stability
        """
        self.n_features = n_features
        self.momentum = momentum
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(n_features, dtype=np.float64)  # Scale
        self.beta = np.zeros(n_features, dtype=np.float64)   # Shift
        
        # Running statistics (for inference)
        self.running_mean = np.zeros(n_features, dtype=np.float64)
        self.running_var = np.ones(n_features, dtype=np.float64)
        
        # Cached values for backprop
        self._x: np.ndarray | None = None
        self._x_norm: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._dgamma: np.ndarray | None = None
        self._dbeta: np.ndarray | None = None
        self._dW: np.ndarray | None = None  # Alias for _dgamma
        self._db: np.ndarray | None = None  # Alias for _dbeta

    @property
    def n_in(self) -> int:
        return self.n_features

    @property
    def n_out(self) -> int:
        return self.n_features

    @property
    def is_output(self) -> bool:
        return False

    @property
    def param_count(self) -> int:
        return self.n_features * 2  # gamma + beta

    # ── forward ──
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self._x = x
        
        if training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Normalize
            self._std = np.sqrt(batch_var + self.eps)
            self._x_norm = (x - batch_mean) / self._std
            
            # Scale and shift
            out = self.gamma * self._x_norm + self.beta
            
            # Update running averages
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
            
            # Cache for backward
            self._batch_mean = batch_mean
        else:
            # Use running statistics for inference
            std = np.sqrt(self.running_var + self.eps)
            x_norm = (x - self.running_mean) / std
            out = self.gamma * x_norm + self.beta
            self._x_norm = x_norm
            self._std = std
        
        return out

    # ── backward ──
    def backward(self, delta: np.ndarray) -> np.ndarray:
        """
        Compute gradients for batch normalization.
        """
        N = delta.shape[0] if delta.ndim > 1 else 1
        
        # Gradients for gamma and beta
        self._dgamma = np.sum(delta * self._x_norm, axis=0) if self._x_norm is not None else np.zeros_like(self.gamma)
        self._dbeta = np.sum(delta, axis=0)
        
        # Gradient for input
        if self._x_norm is not None:
            d_xnorm = delta * self.gamma
            
            # Gradient through normalization
            d_var = np.sum(d_xnorm * (self._x - self._batch_mean) * (-0.5) * (self._std ** -3), axis=0)
            d_mean = np.sum(d_xnorm * (-1 / self._std), axis=0) + d_var * np.mean(-2 * (self._x - self._batch_mean), axis=0)
            
            dx = d_xnorm / self._std + d_var * 2 * (self._x - self._batch_mean) / N + d_mean / N
        else:
            dx = delta * self.gamma
        
        return dx

    # ── update ──
    def update(self, optimizer: BaseOptimizer, layer_idx: int):
        optimizer.tick()
        prefix = f"L{layer_idx}"
        self.gamma = optimizer.step(self.gamma, self._dgamma, key=f"{prefix}_gamma")
        self.beta = optimizer.step(self.beta, self._dbeta, key=f"{prefix}_beta")

    # ── serialisation ──
    def to_dict(self) -> dict:
        return {
            "type": "batchnorm",
            "n_features": self.n_features,
            "momentum": self.momentum,
            "eps": self.eps,
            "gamma": self.gamma.tolist(),
            "beta": self.beta.tolist(),
            "running_mean": self.running_mean.tolist(),
            "running_var": self.running_var.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BatchNormLayer":
        layer = cls(
            n_features=d["n_features"],
            momentum=d.get("momentum", 0.9),
            eps=d.get("eps", 1e-8)
        )
        layer.gamma = np.array(d["gamma"], dtype=np.float64)
        layer.beta = np.array(d["beta"], dtype=np.float64)
        layer.running_mean = np.array(d.get("running_mean", np.zeros(d["n_features"])), dtype=np.float64)
        layer.running_var = np.array(d.get("running_var", np.ones(d["n_features"])), dtype=np.float64)
        return layer

    # ── inspection helpers ──
    def weight_snapshot(self) -> dict:
        return {
            "W": self.gamma.tolist(),
            "b": self.beta.tolist(),
            "dW": self._dgamma.tolist() if self._dgamma is not None else None,
            "db": self._dbeta.tolist() if self._dbeta is not None else None,
            "activation": self._x_norm.tolist() if self._x_norm is not None else None,
        }
