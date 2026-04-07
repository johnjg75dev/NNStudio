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
    def predict(self, x: np.ndarray, training: bool = False, start_layer: int = 0, end_layer: int | None = None) -> np.ndarray:
        a = x.astype(np.float64)
        end_idx = end_layer if end_layer is not None else len(self.layers)
        for i in range(start_layer, end_idx):
            a = self.layers[i].forward(a, training=training)
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
        """
        Returns the network topology as a list of feature counts per layer.
        Only includes layers with meaningful dimensions for visualization.
        Skips: Dropout, BatchNorm, Flatten, LayerNorm, PositionalEncoding, MaxPool
        """
        if not self.layers:
            return []
        
        # Start with input size - get from first layer's appropriate attribute
        first_layer = self.layers[0]
        if hasattr(first_layer, 'n_in') and first_layer.n_in > 0 and first_layer.__class__.__name__ != 'Conv2DLayer':
            input_size = first_layer.n_in
        elif hasattr(first_layer, 'embed_dim'):
            input_size = first_layer.embed_dim
        elif hasattr(first_layer, 'in_channels') and hasattr(first_layer, 'W'):
            # Conv2D: input_size = in_channels * spatial^2 (assume 8x8)
            input_size = first_layer.in_channels * 64
        else:
            input_size = 64  # Default
        
        topo = [input_size]
        
        # Add output sizes for layers that have meaningful dimensions
        for layer in self.layers:
            layer_type = layer.__class__.__name__
            
            # Skip non-visualizable layers
            if layer_type in ['DropoutLayer', 'BatchNormLayer', 'FlattenLayer',
                              'LayerNorm', 'PositionalEncoding', 'MaxPool2DLayer']:
                continue
            
            # Conv2D: use total feature count (channels * spatial^2)
            if layer_type == 'Conv2DLayer':
                # Estimate from weight shape: (out_ch, in_ch, k, k)
                if hasattr(layer, 'W') and layer.W is not None:
                    out_ch = layer.W.shape[0]
                    # Assume 4x4 spatial for visualization
                    topo.append(out_ch * 16)
                else:
                    topo.append(layer.out_channels * 16)
            # Embedding: use embed_dim
            elif layer_type == 'EmbeddingLayer':
                topo.append(layer.embed_dim)
            # LSTM/RNN: use hidden_size
            elif layer_type in ['LSTMLayer', 'SimpleRNNLayer']:
                topo.append(layer.hidden_size)
            # Attention: use embed_dim
            elif layer_type == 'MultiHeadAttention':
                topo.append(layer.embed_dim)
            # Default: use n_out
            elif hasattr(layer, 'n_out'):
                topo.append(layer.n_out)
        
        return topo

    @property
    def param_count(self) -> int:
        return sum(l.param_count for l in self.layers)

    # ── activation snapshot (for visualisation) ──
    def activation_snapshot(self, x: np.ndarray) -> list[list[float]]:
        """Returns activations at every layer including input.
        Only includes layers with meaningful activations (skips Dropout, BatchNorm, etc.)
        """
        self.predict(x)
        snaps = [x.tolist()]
        for l in self.layers:
            # Only include activations from layers that have them
            layer_type = l.__class__.__name__
            if layer_type in ['DropoutLayer', 'BatchNormLayer', 'LayerNorm', 
                              'FlattenLayer', 'PositionalEncoding']:
                continue
            if hasattr(l, '_a') and l._a is not None:
                snaps.append(l._a.flatten().tolist())
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
        
        # Track spatial dimensions for Conv2D
        spatial_size = int(np.sqrt(n_in))  # Assume square input
        in_channels = 1  # Assume single channel for first conv

        for i, lc in enumerate(layers_config):
            l_type = lc.get("type", "dense")
            l_cls  = LAYER_TYPES.get(l_type, DenseLayer)

            if l_type == "dropout":
                # Dropout layer: no input/output size needed
                rate = float(lc.get("rate", 0.5))
                layers.append(l_cls(rate=rate))
            elif l_type == "batchnorm":
                # BatchNorm layer: needs feature count
                layers.append(l_cls(n_features=curr_in))
            elif l_type == "conv2d":
                # Conv2D layer
                out_channels = int(lc.get("out_channels", 32))
                kernel_size = int(lc.get("kernel_size", 3))
                stride = int(lc.get("stride", 1))
                padding = int(lc.get("padding", 1))
                act_name = lc.get("activation", "relu")
                act = ACTIVATIONS.get(act_name, ACTIVATIONS["relu"])
                
                layers.append(l_cls(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    activation=act
                ))
                
                # Update spatial size: out = (in + 2*pad - kernel) / stride + 1
                spatial_size = (spatial_size + 2 * padding - kernel_size) // stride + 1
                in_channels = out_channels
                curr_in = out_channels * spatial_size * spatial_size  # Total features
            elif l_type == "maxpool2d":
                # MaxPool2D layer
                pool_size = int(lc.get("pool_size", 2))
                stride = int(lc.get("stride", pool_size))
                layers.append(l_cls(pool_size=pool_size, stride=stride))
                
                # Update spatial size
                spatial_size = (spatial_size - pool_size) // stride + 1
                curr_in = in_channels * spatial_size * spatial_size
            elif l_type == "flatten":
                # Flatten layer
                layers.append(l_cls())
                # curr_in stays the same (just reshaping)
            elif l_type == "simple_rnn":
                # Simple RNN layer
                hidden_size = int(lc.get("hidden_size", 64))
                act_name = lc.get("activation", "tanh")
                act = ACTIVATIONS.get(act_name, ACTIVATIONS["tanh"])
                layers.append(l_cls(
                    input_size=curr_in,
                    hidden_size=hidden_size,
                    activation=act,
                    return_sequences=lc.get("return_sequences", True)
                ))
                curr_in = hidden_size
            elif l_type == "lstm":
                # LSTM layer
                hidden_size = int(lc.get("hidden_size", 64))
                layers.append(l_cls(
                    input_size=curr_in,
                    hidden_size=hidden_size,
                    return_sequences=lc.get("return_sequences", True)
                ))
                curr_in = hidden_size
            elif l_type == "embedding":
                # Embedding layer
                vocab_size = int(lc.get("vocab_size", 1000))
                embed_dim = int(lc.get("embed_dim", 128))
                layers.append(l_cls(vocab_size=vocab_size, embed_dim=embed_dim))
                curr_in = embed_dim
            elif l_type == "layernorm":
                # LayerNorm layer: doesn't change dimensions
                layers.append(l_cls(normalized_shape=curr_in))
                # curr_in stays the same
            elif l_type == "multihead_attention":
                # Multi-Head Attention layer
                embed_dim = int(lc.get("embed_dim", curr_in))
                num_heads = int(lc.get("num_heads", 4))
                layers.append(l_cls(embed_dim=embed_dim, num_heads=num_heads))
                curr_in = embed_dim
            elif l_type == "positional_encoding":
                # Positional Encoding layer: doesn't change dimensions
                max_seq_len = int(lc.get("max_seq_len", 512))
                embed_dim = int(lc.get("embed_dim", curr_in))
                layers.append(l_cls(max_seq_len=max_seq_len, embed_dim=embed_dim))
                # curr_in stays the same
            else:
                # Dense layer (default)
                n_neurons = int(lc.get("neurons", 4))
                act       = ACTIVATIONS.get(lc.get("activation", "tanh"), ACTIVATIONS["tanh"])

                layers.append(l_cls(
                    n_in=curr_in,
                    n_out=n_neurons,
                    activation=act,
                    is_output=False
                ))
                curr_in = n_neurons

        # Always add final output layer
        layers.append(DenseLayer(
            n_in=curr_in,
            n_out=n_out,
            activation=ACTIVATIONS["sigmoid"],  # Sigmoid for final
            is_output=True
        ))

        return NeuralNetwork(layers, opt, loss)
