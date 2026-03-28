"""
app/core/activations.py
Pure-NumPy activation functions exposed as a registry-style dict.
Each entry is an Activation dataclass with .forward() and .derivative().
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Activation:
    name:        str
    label:       str
    forward:     Callable[[np.ndarray], np.ndarray]
    derivative:  Callable[[np.ndarray], np.ndarray]
    description: str = ""


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def _gelu(z: np.ndarray) -> np.ndarray:
    return 0.5 * z * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z ** 3)))


def _gelu_d(z: np.ndarray) -> np.ndarray:
    t = np.tanh(0.7978845608 * (z + 0.044715 * z ** 3))
    return 0.5 * (1 + t) + 0.5 * z * (1 - t ** 2) * 0.7978845608 * (1 + 3 * 0.044715 * z ** 2)


def _swish(z: np.ndarray) -> np.ndarray:
    return z * _sigmoid(z)


def _swish_d(z: np.ndarray) -> np.ndarray:
    s = _sigmoid(z)
    return s + z * s * (1.0 - s)


ACTIVATIONS: dict[str, Activation] = {
    "relu": Activation(
        name="relu", label="ReLU",
        forward=lambda z: np.maximum(0.0, z),
        derivative=lambda z: (z > 0).astype(float),
        description="Fast, sparse. Risk of dead neurons at high LR."),
    "leakyrelu": Activation(
        name="leakyrelu", label="Leaky ReLU",
        forward=lambda z: np.where(z > 0, z, 0.01 * z),
        derivative=lambda z: np.where(z > 0, 1.0, 0.01),
        description="Fixes dead neuron problem with small negative slope."),
    "tanh": Activation(
        name="tanh", label="Tanh",
        forward=np.tanh,
        derivative=lambda z: 1.0 - np.tanh(z) ** 2,
        description="Zero-centered. Good default for shallow nets."),
    "sigmoid": Activation(
        name="sigmoid", label="Sigmoid",
        forward=_sigmoid,
        derivative=lambda z: _sigmoid(z) * (1.0 - _sigmoid(z)),
        description="Saturates at extremes. Use BCE loss with it."),
    "gelu": Activation(
        name="gelu", label="GELU",
        forward=_gelu,
        derivative=_gelu_d,
        description="Smooth. Used in GPT, BERT, modern Transformers."),
    "swish": Activation(
        name="swish", label="Swish",
        forward=_swish,
        derivative=_swish_d,
        description="Self-gated. Outperforms ReLU in deep nets."),
}
