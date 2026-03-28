from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from ..optimizers import BaseOptimizer

class Layer(ABC):
    """Base class for all layer types."""

    @abstractmethod
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray: ...

    @abstractmethod
    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Receive upstream delta, store gradients, return downstream delta."""

    @abstractmethod
    def update(self, optimizer: BaseOptimizer, layer_idx: int): ...

    @abstractmethod
    def to_dict(self) -> dict: ...

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> "Layer": ...

    @property
    @abstractmethod
    def param_count(self) -> int: ...
