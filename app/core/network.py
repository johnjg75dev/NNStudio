"""
app/core/network.py
Object-oriented neural network engine.
"""
from __future__ import annotations

import json
import numpy as np
from typing import Any

from .activations import ACTIVATIONS
from .losses      import LOSSES, LossFunction
from .optimizers  import BaseOptimizer, OptimizerFactory
from .layers      import Layer, DenseLayer, LAYER_TYPES


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
    Builds a NeuralNetwork from a flexible config.
    """

    @staticmethod
    def build(config: dict) -> NeuralNetwork:
        loss = LOSSES.get(config["loss"], LOSSES["mse"])
        opt  = OptimizerFactory.build(
            config["optimizer"],
            config["lr"],
            weight_decay=config.get("weight_decay", 0.0),
        )

        n_in  = int(config["inputs"])
        n_out = int(config["outputs"])

        # Use explicit layers list
        layers_config = config.get("layers", [])

        layers: list[Layer] = []
        curr_in = n_in

        for i, lc in enumerate(layers_config):
            l_type = lc.get("type", "dense")
            l_cls  = LAYER_TYPES.get(l_type, DenseLayer)

            n_neurons = int(lc.get("neurons", 4))
            act       = ACTIVATIONS.get(lc.get("activation", "tanh"), ACTIVATIONS["tanh"])
            drop      = float(lc.get("dropout", 0.0))

            layers.append(l_cls(
                n_in=curr_in,
                n_out=n_neurons,
                activation=act,
                dropout=drop,
                is_output=False
            ))
            curr_in = n_neurons

        # Always add final output layer
        layers.append(DenseLayer(
            n_in=curr_in,
            n_out=n_out,
            activation=ACTIVATIONS["sigmoid"], # Sigmoid for final
            dropout=0.0,
            is_output=True
        ))

        return NeuralNetwork(layers, opt, loss)
