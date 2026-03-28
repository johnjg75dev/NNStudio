"""
app/core/network.py
Object-oriented neural network engine.

Class hierarchy:
    Layer           — abstract base
    DenseLayer      — fully-connected layer
    NeuralNetwork   — holds a list of Layer, drives fwd/bwd/update
    NetworkBuilder  — constructs NeuralNetwork from a topology dict
    NetworkSnapshot — serialisable representation for save/load
"""
from __future__ import annotations

import json
import numpy as np
from abc import ABC, abstractmethod
from typing import Any

from .activations import ACTIVATIONS, Activation
from .losses      import LOSSES, LossFunction
from .optimizers  import BaseOptimizer, OptimizerFactory


# ═══════════════════════════════════════════════════════════════════════
# Abstract Layer
# ═══════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════
# Dense (fully-connected) Layer
# ═══════════════════════════════════════════════════════════════════════
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
                 dropout:   float = 0.0,
                 is_output: bool  = False):
        self.n_in      = n_in
        self.n_out     = n_out
        self.activation = activation
        self.dropout   = dropout
        self.is_output = is_output

        # He initialisation
        scale  = np.sqrt(2.0 / n_in)
        self.W = np.random.randn(n_out, n_in).astype(np.float64) * scale
        self.b = np.zeros(n_out, dtype=np.float64)

        # Cached values (set during forward/backward)
        self._x:    np.ndarray | None = None   # input
        self._z:    np.ndarray | None = None   # pre-activation
        self._a:    np.ndarray | None = None   # post-activation
        self._mask: np.ndarray | None = None   # dropout mask
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
            if training and self.dropout > 0.0:
                keep = 1.0 - self.dropout
                self._mask = (np.random.rand(*a.shape) < keep).astype(float) / keep
                a = a * self._mask
            else:
                self._mask = None

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
            if self._mask is not None:
                delta = delta * self._mask
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
            "dropout":    self.dropout,
            "is_output":  self.is_output,
            "W":          self.W.tolist(),
            "b":          self.b.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DenseLayer":
        act = ACTIVATIONS[d["activation"]]
        layer = cls(d["n_in"], d["n_out"], act,
                    dropout=d.get("dropout", 0.0),
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


# ═══════════════════════════════════════════════════════════════════════
# NeuralNetwork
# ═══════════════════════════════════════════════════════════════════════
class NeuralNetwork:
    """
    Container for an ordered list of Layer objects.
    Drives training: forward → backward → update.
    """

    def __init__(self,
                 layers:    list[Layer],
                 optimizer: BaseOptimizer,
                 loss_fn:   LossFunction):
        self.layers    = layers
        self.optimizer = optimizer
        self.loss_fn   = loss_fn
        self.epoch:    int   = 0
        self.loss_history: list[float] = []

    # ── forward pass ──
    def predict(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        a = x.astype(np.float64)
        for layer in self.layers:
            a = layer.forward(a, training=training)
        return a

    # ── single sample train step ──
    def train_step(self, x: np.ndarray, y: np.ndarray):
        p     = self.predict(x, training=True)
        delta = self.loss_fn.output_delta(p, y.astype(np.float64))

        # backprop through layers in reverse
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                # output layer: pass combined loss+sigmoid delta directly
                layer._dW = np.outer(delta, layer._x)
                layer._db = delta.copy()
                delta     = layer.W.T @ delta
            else:
                delta = layer.backward(delta)
            layer.update(self.optimizer, layer_idx=i)

    # ── epoch over dataset ──
    def train_epoch(self, dataset: list[dict], lr: float | None = None):
        if lr is not None:
            self.optimizer.lr = lr
        indices = np.random.permutation(len(dataset))
        for i in indices:
            sample = dataset[i]
            self.train_step(np.array(sample["x"]), np.array(sample["y"]))
        self.epoch += 1

    # ── metrics ──
    def compute_loss(self, dataset: list[dict]) -> float:
        total = 0.0
        for s in dataset:
            p = self.predict(np.array(s["x"]))
            total += self.loss_fn.compute(p, np.array(s["y"]))
        return total / len(dataset)

    def compute_accuracy(self, dataset: list[dict], threshold: float = 0.5) -> float:
        correct = 0
        for s in dataset:
            p = self.predict(np.array(s["x"]))
            y = np.array(s["y"])
            if np.all(np.abs((p > threshold).astype(float) - np.round(y)) < 0.5):
                correct += 1
        return correct / len(dataset)

    # ── topology helpers ──
    @property
    def topology(self) -> list[int]:
        if not self.layers:
            return []
        topo = [self.layers[0].n_in]
        topo += [layer.n_out for layer in self.layers]
        return topo

    @property
    def param_count(self) -> int:
        return sum(l.param_count for l in self.layers)

    # ── activation snapshot (for visualisation) ──
    def activation_snapshot(self, x: np.ndarray) -> list[list[float]]:
        """Returns activations at every layer including input."""
        self.predict(x)
        snaps = [x.tolist()]
        snaps += [l._a.tolist() for l in self.layers if l._a is not None]
        return snaps

    # ── serialisation ──
    def to_dict(self) -> dict:
        return {
            "layers":       [l.to_dict() for l in self.layers],
            "optimizer":    self.optimizer.__class__.__name__.lower(),
            "optimizer_lr": self.optimizer.lr,
            "loss":         self.loss_fn.name,
            "epoch":        self.epoch,
            "loss_history": self.loss_history[-200:],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NeuralNetwork":
        layers = [DenseLayer.from_dict(ld) for ld in d["layers"]]
        opt    = OptimizerFactory.build(d["optimizer"], d["optimizer_lr"])
        loss   = LOSSES[d["loss"]]
        net    = cls(layers, opt, loss)
        net.epoch        = d.get("epoch", 0)
        net.loss_history = d.get("loss_history", [])
        return net


# ═══════════════════════════════════════════════════════════════════════
# NetworkBuilder
# ═══════════════════════════════════════════════════════════════════════
class NetworkBuilder:
    """
    Builds a NeuralNetwork from a plain config dict.

    config = {
        "inputs":       2,
        "outputs":      1,
        "hidden_layers": 2,
        "neurons":      8,
        "activation":   "tanh",
        "optimizer":    "adam",
        "lr":           0.01,
        "loss":         "bce",
        "dropout":      0.0,
        "weight_decay": 0.0,
    }
    """

    @staticmethod
    def build(config: dict) -> NeuralNetwork:
        act  = ACTIVATIONS.get(config["activation"], ACTIVATIONS["tanh"])
        loss = LOSSES.get(config["loss"], LOSSES["mse"])
        opt  = OptimizerFactory.build(
            config["optimizer"],
            config["lr"],
            weight_decay=config.get("weight_decay", 0.0),
        )

        n_in  = int(config["inputs"])
        n_out = int(config["outputs"])
        hl    = int(config["hidden_layers"])
        hw    = int(config["neurons"])
        drop  = float(config.get("dropout", 0.0))

        layers: list[Layer] = []
        sizes = [n_in] + [hw] * hl + [n_out]

        for i in range(1, len(sizes)):
            is_out = (i == len(sizes) - 1)
            layers.append(DenseLayer(
                n_in=sizes[i - 1],
                n_out=sizes[i],
                activation=act,
                dropout=0.0 if is_out else drop,
                is_output=is_out,
            ))

        return NeuralNetwork(layers, opt, loss)
