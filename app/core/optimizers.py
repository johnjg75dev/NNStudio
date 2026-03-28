"""
app/core/optimizers.py
Optimizer hierarchy.  Every optimizer is a class that holds its own state
(moments, etc.) and exposes .step(param, grad) → updated_param.

OptimizerFactory.build(name, lr, **kwargs) is the public entry point.
"""
from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Any


# ═══════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════
class BaseOptimizer(ABC):
    label:       str = ""
    description: str = ""

    def __init__(self, lr: float = 0.01, **kwargs):
        self.lr = lr
        self.t  = 0

    def tick(self):
        """Increment global timestep (call once per weight-update pass)."""
        self.t += 1

    @abstractmethod
    def step(self, param: np.ndarray, grad: np.ndarray,
             key: str) -> np.ndarray:
        """Return updated parameter array."""

    def state_dict(self) -> dict:
        return {"t": self.t, "lr": self.lr}

    def load_state(self, d: dict):
        self.t  = d.get("t", 0)
        self.lr = d.get("lr", self.lr)


# ═══════════════════════════════════════════════════════════════════════
# Concrete optimizers
# ═══════════════════════════════════════════════════════════════════════
class SGD(BaseOptimizer):
    label       = "SGD"
    description = "Vanilla stochastic gradient descent. Simple, predictable, sensitive to LR."

    def step(self, param, grad, key=""):
        return param - self.lr * grad


class SGDMomentum(BaseOptimizer):
    label       = "SGD + Momentum"
    description = "SGD with exponential moving average of gradients. Smooths oscillations."

    def __init__(self, lr=0.01, momentum=0.9, **kw):
        super().__init__(lr, **kw)
        self.momentum = momentum
        self._velocity: dict[str, np.ndarray] = {}

    def step(self, param, grad, key=""):
        v = self._velocity.get(key, np.zeros_like(param))
        v = self.momentum * v + self.lr * grad
        self._velocity[key] = v
        return param - v

    def state_dict(self):
        d = super().state_dict()
        d["velocity"] = {k: v.tolist() for k, v in self._velocity.items()}
        return d

    def load_state(self, d):
        super().load_state(d)
        self._velocity = {k: np.array(v) for k, v in d.get("velocity", {}).items()}


class RMSProp(BaseOptimizer):
    label       = "RMSProp"
    description = "Adaptive per-parameter LR. Divides by RMS of recent gradients."

    def __init__(self, lr=0.01, rho=0.9, eps=1e-8, **kw):
        super().__init__(lr, **kw)
        self.rho = rho
        self.eps = eps
        self._cache: dict[str, np.ndarray] = {}

    def step(self, param, grad, key=""):
        c = self._cache.get(key, np.zeros_like(param))
        c = self.rho * c + (1 - self.rho) * grad ** 2
        self._cache[key] = c
        return param - self.lr * grad / (np.sqrt(c) + self.eps)

    def state_dict(self):
        d = super().state_dict()
        d["cache"] = {k: v.tolist() for k, v in self._cache.items()}
        return d

    def load_state(self, d):
        super().load_state(d)
        self._cache = {k: np.array(v) for k, v in d.get("cache", {}).items()}


class Adam(BaseOptimizer):
    label       = "Adam"
    description = "Adaptive Moment Estimation. Best all-round default. Less sensitive to LR than SGD."

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, **kw):
        super().__init__(lr, **kw)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}

    def step(self, param, grad, key=""):
        m = self._m.get(key, np.zeros_like(param))
        v = self._v.get(key, np.zeros_like(param))
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * grad ** 2
        self._m[key] = m
        self._v[key] = v
        mh = m / (1 - self.beta1 ** max(self.t, 1))
        vh = v / (1 - self.beta2 ** max(self.t, 1))
        return param - self.lr * mh / (np.sqrt(vh) + self.eps)

    def state_dict(self):
        d = super().state_dict()
        d["m"] = {k: v.tolist() for k, v in self._m.items()}
        d["v"] = {k: v.tolist() for k, v in self._v.items()}
        return d

    def load_state(self, d):
        super().load_state(d)
        self._m = {k: np.array(v) for k, v in d.get("m", {}).items()}
        self._v = {k: np.array(v) for k, v in d.get("v", {}).items()}


class AdamW(Adam):
    label       = "AdamW"
    description = "Adam with decoupled weight decay. Best for generalisation."

    def __init__(self, lr=0.001, weight_decay=0.01, **kw):
        super().__init__(lr, **kw)
        self.weight_decay = weight_decay

    def step(self, param, grad, key=""):
        updated = super().step(param, grad, key)
        return updated - self.lr * self.weight_decay * param


# ═══════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════
_REGISTRY: dict[str, type[BaseOptimizer]] = {
    "sgd":      SGD,
    "momentum": SGDMomentum,
    "rmsprop":  RMSProp,
    "adam":     Adam,
    "adamw":    AdamW,
}


class OptimizerFactory:
    @staticmethod
    def build(name: str, lr: float, **kwargs) -> BaseOptimizer:
        cls = _REGISTRY.get(name)
        if cls is None:
            raise ValueError(f"Unknown optimizer '{name}'. "
                             f"Available: {list(_REGISTRY)}")
        return cls(lr=lr, **kwargs)

    @staticmethod
    def available() -> list[dict]:
        return [{"key": k, "label": v.label, "description": v.description}
                for k, v in _REGISTRY.items()]
