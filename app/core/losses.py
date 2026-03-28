"""
app/core/losses.py
Loss functions as a registry dict.  Each LossFunction carries .compute()
(scalar loss) and .output_delta() (gradient w.r.t. pre-sigmoid output).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class LossFunction:
    name:         str
    label:        str
    description:  str
    # scalar loss given predictions p and targets y (both 1-D arrays)
    compute:      Callable[[np.ndarray, np.ndarray], float]
    # gradient δ at the final layer (already folded with output activation)
    output_delta: Callable[[np.ndarray, np.ndarray], np.ndarray]


def _sigmoid(z):
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


LOSSES: dict[str, LossFunction] = {
    "mse": LossFunction(
        name="mse", label="MSE",
        description="Mean Squared Error. Penalises large errors heavily. Good for regression.",
        compute=lambda p, y: float(np.mean((p - y) ** 2)),
        output_delta=lambda p, y: (p - y) * p * (1.0 - p),   # sigmoid out
    ),
    "bce": LossFunction(
        name="bce", label="Binary Cross-Entropy",
        description="Best for binary classification (0/1 targets). Use with sigmoid output.",
        compute=lambda p, y: float(-np.mean(
            y * np.log(np.clip(p, 1e-9, 1)) +
            (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))),
        output_delta=lambda p, y: p - y,   # combined BCE + sigmoid gradient
    ),
    "mae": LossFunction(
        name="mae", label="MAE",
        description="Mean Absolute Error. Robust to outliers. Slower to converge.",
        compute=lambda p, y: float(np.mean(np.abs(p - y))),
        output_delta=lambda p, y: np.sign(p - y) * p * (1.0 - p),
    ),
}
