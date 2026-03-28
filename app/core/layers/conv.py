"""
app/core/layers/conv.py
Convolutional and Pooling layers for CNN architectures.
"""
from __future__ import annotations
import numpy as np
from .base import Layer
from ..activations import ACTIVATIONS, Activation
from ..optimizers import BaseOptimizer


class Conv2DLayer(Layer):
    """
    2D Convolutional layer.
    Applies learnable filters to input feature maps.
    
    For simplicity, this implementation assumes:
    - Input shape: (channels, height, width) flattened to 1D
    - Output: flattened feature maps
    """

    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 activation: Activation = None,
                 use_bias: bool = True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation or ACTIVATIONS["relu"]
        self.use_bias = use_bias
        
        # He initialization for weights
        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)
        self.W = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ).astype(np.float64) * scale
        
        if use_bias:
            self.b = np.zeros(out_channels, dtype=np.float64)
        
        # Cached values
        self._x: np.ndarray | None = None
        self._z: np.ndarray | None = None
        self._a: np.ndarray | None = None
        self._dW: np.ndarray | None = None
        self._db: np.ndarray | None = None
        self._padded_x: np.ndarray | None = None

    @property
    def n_in(self) -> int:
        return self.in_channels

    @property
    def n_out(self) -> int:
        return self.out_channels

    @property
    def is_output(self) -> bool:
        return False

    @property
    def param_count(self) -> int:
        params = self.in_channels * self.out_channels * self.kernel_size * self.kernel_size
        if self.use_bias:
            params += self.out_channels
        return params

    def _pad_input(self, x: np.ndarray) -> np.ndarray:
        """Pad input for convolution."""
        if self.padding == 0:
            return x
        # Reshape to (C, H, W) if needed
        if x.ndim == 1:
            # Assume square input
            side = int(np.sqrt(len(x) / self.in_channels))
            x = x.reshape(self.in_channels, side, side)
        
        return np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), 
                     mode='constant', constant_values=0)

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass through convolution.
        Input x can be 1D (flattened) or 3D (C, H, W).
        """
        # Store input
        self._x = x
        
        # Reshape to (C, H, W) if 1D
        if x.ndim == 1:
            side = int(np.sqrt(len(x) / self.in_channels))
            x = x.reshape(self.in_channels, side, side)
        
        # Pad input
        self._padded_x = self._pad_input(x)
        C, H, W = self._padded_x.shape
        
        # Output dimensions
        out_h = (H - self.kernel_size) // self.stride + 1
        out_w = (W - self.kernel_size) // self.stride + 1
        
        # Perform convolution
        self._z = np.zeros((self.out_channels, out_h, out_w))
        
        for oc in range(self.out_channels):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size
                    
                    patch = self._padded_x[:, h_start:h_end, w_start:w_end]
                    self._z[oc, i, j] = np.sum(patch * self.W[oc]) + (self.b[oc] if self.use_bias else 0)
        
        # Apply activation
        self._a = self.activation.forward(self._z)
        
        # Flatten output
        return self._a.flatten()

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Backward pass through convolution."""
        # Reshape delta to (out_channels, out_h, out_w)
        if delta.ndim == 1:
            out_h = out_w = int(np.sqrt(len(delta) / self.out_channels))
            delta = delta.reshape(self.out_channels, out_h, out_w)
        
        # Gradient through activation
        if self._z is not None:
            delta = delta * self.activation.derivative(self._z)
        
        # Compute weight gradients
        C, H, W = self._padded_x.shape
        self._dW = np.zeros_like(self.W)
        
        for oc in range(self.out_channels):
            for i in range(delta.shape[1]):
                for j in range(delta.shape[2]):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size
                    
                    self._dW[oc] += delta[oc, i, j] * self._padded_x[:, h_start:h_end, w_start:w_end]
        
        if self.use_bias:
            self._db = np.sum(delta, axis=(1, 2))
        
        # Compute input gradient (simplified - full implementation would unpad)
        dx = np.zeros_like(self._padded_x)
        for oc in range(self.out_channels):
            for i in range(delta.shape[1]):
                for j in range(delta.shape[2]):
                    h_start = i * self.stride
                    h_end = h_start + self.kernel_size
                    w_start = j * self.stride
                    w_end = w_start + self.kernel_size
                    dx[:, h_start:h_end, w_start:w_end] += delta[oc, i, j] * self.W[oc]
        
        # Remove padding
        if self.padding > 0:
            dx = dx[:, self.padding:-self.padding, self.padding:-self.padding]
        
        return dx.flatten()

    def update(self, optimizer: BaseOptimizer, layer_idx: int):
        optimizer.tick()
        prefix = f"L{layer_idx}"
        self.W = optimizer.step(self.W, self._dW, key=f"{prefix}_W")
        if self.use_bias:
            self.b = optimizer.step(self.b, self._db, key=f"{prefix}_b")

    def to_dict(self) -> dict:
        return {
            "type": "conv2d",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "activation": self.activation.name if self.activation else None,
            "use_bias": self.use_bias,
            "W": self.W.tolist(),
            "b": self.b.tolist() if self.use_bias else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Conv2DLayer":
        act = ACTIVATIONS.get(d.get("activation"), ACTIVATIONS["relu"]) if d.get("activation") else None
        layer = cls(
            in_channels=d["in_channels"],
            out_channels=d["out_channels"],
            kernel_size=d.get("kernel_size", 3),
            stride=d.get("stride", 1),
            padding=d.get("padding", 1),
            activation=act,
            use_bias=d.get("use_bias", True)
        )
        layer.W = np.array(d["W"], dtype=np.float64)
        if d.get("b") is not None:
            layer.b = np.array(d["b"], dtype=np.float64)
        return layer

    def weight_snapshot(self) -> dict:
        return {
            "W": self.W.tolist(),
            "b": self.b.tolist() if self.use_bias else None,
            "dW": self._dW.tolist() if self._dW is not None else None,
            "db": self._db.tolist() if self._db is not None else None,
            "activation": self._a.flatten().tolist() if self._a is not None else None,
        }


class MaxPool2DLayer(Layer):
    """
    2D Max Pooling layer.
    Downsamples spatial dimensions by taking maximum in each window.
    """

    def __init__(self, 
                 pool_size: int = 2,
                 stride: int = 2,
                 padding: int = 0):
        self.pool_size = pool_size
        self.stride = stride
        self.padding = padding
        
        # Cached values
        self._x: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._dW: np.ndarray | None = None  # Always None (no weights)
        self._db: np.ndarray | None = None  # Always None (no bias)

    @property
    def n_in(self) -> int:
        return 0  # Dynamic

    @property
    def n_out(self) -> int:
        return 0  # Dynamic

    @property
    def is_output(self) -> bool:
        return False

    @property
    def param_count(self) -> int:
        return 0

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass through max pooling."""
        self._x = x
        
        # Reshape to (C, H, W) if 1D
        if x.ndim == 1:
            # Assume some number of channels
            n_channels = 4  # Default assumption
            side = int(np.sqrt(len(x) / n_channels))
            x = x.reshape(n_channels, side, side)
        
        C, H, W = x.shape
        
        # Output dimensions
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1
        
        # Perform max pooling
        output = np.zeros((C, out_h, out_w))
        self._mask = np.zeros_like(output, dtype=int)
        
        for c in range(C):
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    h_end = h_start + self.pool_size
                    w_start = j * self.stride
                    w_end = w_start + self.pool_size
                    
                    patch = x[c, h_start:h_end, w_start:w_end]
                    max_idx = np.argmax(patch)
                    max_i, max_j = divmod(max_idx, self.pool_size)
                    
                    output[c, i, j] = patch[max_i, max_j]
                    self._mask[c, i, j] = h_start * W + w_start + max_i * W + max_j
        
        self._output_shape = (C, out_h, out_w)
        return output.flatten()

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Backward pass through max pooling."""
        # Reshape delta
        if delta.ndim == 1:
            C, out_h, out_w = self._output_shape
            delta = delta.reshape(C, out_h, out_w)
        
        C, H, W = self._x.shape
        dx = np.zeros_like(self._x)
        
        for c in range(C):
            for i in range(delta.shape[1]):
                for j in range(delta.shape[2]):
                    flat_idx = self._mask[c, i, j]
                    orig_i = flat_idx // W
                    orig_j = flat_idx % W
                    dx[c, orig_i, orig_j] += delta[c, i, j]
        
        return dx.flatten()

    def update(self, optimizer: BaseOptimizer, layer_idx: int):
        pass  # No parameters to update

    def to_dict(self) -> dict:
        return {
            "type": "maxpool2d",
            "pool_size": self.pool_size,
            "stride": self.stride,
            "padding": self.padding,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MaxPool2DLayer":
        return cls(
            pool_size=d.get("pool_size", 2),
            stride=d.get("stride", 2),
            padding=d.get("padding", 0)
        )

    def weight_snapshot(self) -> dict:
        return {
            "W": None,
            "b": None,
            "dW": None,
            "db": None,
            "activation": self._x.flatten().tolist() if self._x is not None else None,
        }


class FlattenLayer(Layer):
    """
    Flatten layer.
    Reshapes multi-dimensional input to 1D.
    """

    def __init__(self):
        self._input_shape: tuple | None = None
        self._dW: np.ndarray | None = None
        self._db: np.ndarray | None = None

    @property
    def n_in(self) -> int:
        return 0  # Dynamic

    @property
    def n_out(self) -> int:
        return 0  # Dynamic

    @property
    def is_output(self) -> bool:
        return False

    @property
    def param_count(self) -> int:
        return 0

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Flatten input to 1D."""
        self._input_shape = x.shape
        self._x = x
        return x.flatten()

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Reshape gradient to original shape."""
        if self._input_shape:
            return delta.reshape(self._input_shape)
        return delta

    def update(self, optimizer: BaseOptimizer, layer_idx: int):
        pass  # No parameters

    def to_dict(self) -> dict:
        return {
            "type": "flatten",
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FlattenLayer":
        return cls()

    def weight_snapshot(self) -> dict:
        return {
            "W": None,
            "b": None,
            "dW": None,
            "db": None,
            "activation": self._x.flatten().tolist() if self._x is not None else None,
        }
