"""
app/core/layers/transformer.py
Transformer architecture layers: Embedding, MultiHeadAttention, LayerNorm, etc.
"""
from __future__ import annotations
import numpy as np
from .base import Layer
from ..activations import ACTIVATIONS
from ..optimizers import BaseOptimizer


class EmbeddingLayer(Layer):
    """
    Embedding layer.
    Maps discrete indices to dense vectors.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Initialize embeddings
        scale = np.sqrt(2.0 / embed_dim)
        self.W = np.random.randn(vocab_size, embed_dim).astype(np.float64) * scale
        
        # Cached values
        self._x: np.ndarray | None = None
        self._dW: np.ndarray | None = None

    @property
    def n_in(self) -> int:
        return 1  # Single index input

    @property
    def n_out(self) -> int:
        return self.embed_dim

    @property
    def is_output(self) -> bool:
        return False

    @property
    def param_count(self) -> int:
        return self.W.size

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Lookup embeddings for input indices."""
        self._x = x
        
        # Handle single index or sequence
        if np.isscalar(x) or (hasattr(x, '__len__') and len(np.shape(x)) == 0):
            return self.W[int(x)]
        else:
            x = np.array(x, dtype=int)
            return self.W[x].reshape(-1)

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Compute gradient for embeddings."""
        if self._x is None:
            return np.zeros(1)
        
        # Initialize gradient
        self._dW = np.zeros_like(self.W)
        
        # Handle single index or sequence
        if np.isscalar(self._x) or (hasattr(self._x, '__len__') and len(np.shape(self._x)) == 0):
            idx = int(self._x)
            self._dW[idx] = delta
            return np.zeros(1)  # No gradient to input (it's an index)
        else:
            x = np.array(self._x, dtype=int).flatten()
            delta = delta.reshape(len(x), -1)
            for i, idx in enumerate(x):
                self._dW[idx] += delta[i]
            return np.zeros(len(x))  # No gradient to input indices

    def update(self, optimizer: BaseOptimizer, layer_idx: int):
        optimizer.tick()
        prefix = f"L{layer_idx}"
        self.W = optimizer.step(self.W, self._dW, key=f"{prefix}_W")

    def to_dict(self) -> dict:
        return {
            "type": "embedding",
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
            "W": self.W.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmbeddingLayer":
        layer = cls(d["vocab_size"], d["embed_dim"])
        layer.W = np.array(d["W"], dtype=np.float64)
        return layer

    def weight_snapshot(self) -> dict:
        return {
            "W": self.W.tolist(),
            "b": None,
            "dW": self._dW.tolist() if self._dW is not None else None,
            "db": None,
            "activation": self.forward(self._x).tolist() if self._x is not None else None,
        }


class LayerNorm(Layer):
    """
    Layer Normalization.
    Normalizes across feature dimension.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(normalized_shape, dtype=np.float64)
        self.beta = np.zeros(normalized_shape, dtype=np.float64)
        
        # Cached values
        self._x: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._x_norm: np.ndarray | None = None
        self._dW: np.ndarray | None = None
        self._db: np.ndarray | None = None

    @property
    def n_in(self) -> int:
        return self.normalized_shape

    @property
    def n_out(self) -> int:
        return self.normalized_shape

    @property
    def is_output(self) -> bool:
        return False

    @property
    def param_count(self) -> int:
        return self.gamma.size + self.beta.size

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Layer normalization forward pass."""
        self._x = x
        
        # Compute mean and std
        self._mean = np.mean(x)
        self._std = np.std(x) + self.eps
        
        # Normalize
        self._x_norm = (x - self._mean) / self._std
        
        # Scale and shift
        return self.gamma * self._x_norm + self.beta

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Layer normalization backward pass."""
        N = self._x.size
        
        # Gradients for gamma and beta
        self._dW = delta * self._x_norm  # gamma gradient
        self._db = delta  # beta gradient
        
        # Gradient for input
        d_xnorm = delta * self.gamma
        d_var = np.sum(d_xnorm * (self._x - self._mean) * (-0.5) * (self._std ** -3))
        d_mean = np.sum(d_xnorm * (-1 / self._std)) + d_var * np.mean(-2 * (self._x - self._mean))
        
        dx = d_xnorm / self._std + d_var * 2 * (self._x - self._mean) / N + d_mean / N
        
        return dx

    def update(self, optimizer: BaseOptimizer, layer_idx: int):
        optimizer.tick()
        prefix = f"L{layer_idx}"
        self.gamma = optimizer.step(self.gamma, self._dW, key=f"{prefix}_gamma")
        self.beta = optimizer.step(self.beta, self._db, key=f"{prefix}_beta")

    def to_dict(self) -> dict:
        return {
            "type": "layernorm",
            "normalized_shape": self.normalized_shape,
            "eps": self.eps,
            "gamma": self.gamma.tolist(),
            "beta": self.beta.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LayerNorm":
        layer = cls(d["normalized_shape"], d.get("eps", 1e-6))
        layer.gamma = np.array(d["gamma"], dtype=np.float64)
        layer.beta = np.array(d["beta"], dtype=np.float64)
        return layer

    def weight_snapshot(self) -> dict:
        return {
            "W": self.gamma.tolist(),
            "b": self.beta.tolist(),
            "dW": self._dW.tolist() if self._dW is not None else None,
            "db": self._db.tolist() if self._db is not None else None,
            "activation": self._x_norm.tolist() if self._x_norm is not None else None,
        }


class MultiHeadAttention(Layer):
    """
    Multi-Head Self-Attention layer.
    Simplified implementation for visualization purposes.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # Projection matrices
        scale = np.sqrt(2.0 / embed_dim)
        self.W_q = np.random.randn(embed_dim, embed_dim).astype(np.float64) * scale
        self.W_k = np.random.randn(embed_dim, embed_dim).astype(np.float64) * scale
        self.W_v = np.random.randn(embed_dim, embed_dim).astype(np.float64) * scale
        self.W_o = np.random.randn(embed_dim, embed_dim).astype(np.float64) * scale
        
        # Cached values
        self._x: np.ndarray | None = None
        self._attention: np.ndarray | None = None
        self._dW: np.ndarray | None = None
        self._db: np.ndarray | None = None

    @property
    def n_in(self) -> int:
        return self.embed_dim

    @property
    def n_out(self) -> int:
        return self.embed_dim

    @property
    def is_output(self) -> bool:
        return False

    @property
    def param_count(self) -> int:
        return self.W_q.size + self.W_k.size + self.W_v.size + self.W_o.size

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass through multi-head attention.
        Simplified: treats input as single sequence.
        """
        self._x = x
        
        # Project to Q, K, V
        q = self.W_q @ x
        k = self.W_k @ x
        v = self.W_v @ x
        
        # Split into heads (simplified)
        q = q.reshape(self.num_heads, self.head_dim)
        k = k.reshape(self.num_heads, self.head_dim)
        v = v.reshape(self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = (q @ k.T) / np.sqrt(self.head_dim)
        self._attention = self._softmax(scores)
        
        # Apply attention to values
        attended = self._attention @ v
        attended = attended.reshape(self.embed_dim)
        
        # Output projection
        output = self.W_o @ attended
        
        return output

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Backward pass through attention."""
        # Simplified gradient computation
        # Full implementation would require storing more intermediates
        
        # Gradient through output projection
        attended = self.W_o.T @ delta
        
        # Gradient through attention (simplified)
        dx = self.W_q.T @ delta + self.W_k.T @ delta + self.W_v.T @ delta
        
        # Store weight gradients (simplified)
        self._dW = np.outer(delta, attended)
        
        return dx

    def update(self, optimizer: BaseOptimizer, layer_idx: int):
        optimizer.tick()
        prefix = f"L{layer_idx}"
        self.W_q = optimizer.step(self.W_q, self._dW, key=f"{prefix}_W_q")
        self.W_k = optimizer.step(self.W_k, self._dW, key=f"{prefix}_W_k")
        self.W_v = optimizer.step(self.W_v, self._dW, key=f"{prefix}_W_v")
        self.W_o = optimizer.step(self.W_o, self._dW, key=f"{prefix}_W_o")

    def to_dict(self) -> dict:
        return {
            "type": "multihead_attention",
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "W_q": self.W_q.tolist(),
            "W_k": self.W_k.tolist(),
            "W_v": self.W_v.tolist(),
            "W_o": self.W_o.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MultiHeadAttention":
        layer = cls(d["embed_dim"], d.get("num_heads", 4))
        layer.W_q = np.array(d["W_q"], dtype=np.float64)
        layer.W_k = np.array(d["W_k"], dtype=np.float64)
        layer.W_v = np.array(d["W_v"], dtype=np.float64)
        layer.W_o = np.array(d["W_o"], dtype=np.float64)
        return layer

    def weight_snapshot(self) -> dict:
        return {
            "W": np.vstack([self.W_q, self.W_k, self.W_v, self.W_o]).tolist(),
            "b": None,
            "dW": self._dW.tolist() if self._dW is not None else None,
            "db": None,
            "activation": self.forward(self._x).tolist() if self._x is not None else None,
        }


class PositionalEncoding(Layer):
    """
    Positional Encoding layer.
    Adds position information to embeddings.
    """

    def __init__(self, max_seq_len: int = 512, embed_dim: int = 128):
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # Create positional encoding matrix
        pe = np.zeros((max_seq_len, embed_dim))
        position = np.arange(0, max_seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = pe
        
        # No trainable parameters
        self._dW: np.ndarray | None = None
        self._db: np.ndarray | None = None

    @property
    def n_in(self) -> int:
        return self.embed_dim

    @property
    def n_out(self) -> int:
        return self.embed_dim

    @property
    def is_output(self) -> bool:
        return False

    @property
    def param_count(self) -> int:
        return 0  # Not trainable

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Add positional encoding to input."""
        # For single vector, use first position
        if x.ndim == 1:
            return x + self.pe[0]
        else:
            seq_len = min(x.shape[0] if len(x.shape) > 1 else 1, self.max_seq_len)
            return x + self.pe[:seq_len].flatten()

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Gradient passes through unchanged."""
        return delta

    def update(self, optimizer: BaseOptimizer, layer_idx: int):
        pass  # No parameters to update

    def to_dict(self) -> dict:
        return {
            "type": "positional_encoding",
            "max_seq_len": self.max_seq_len,
            "embed_dim": self.embed_dim,
            "pe": self.pe.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PositionalEncoding":
        layer = cls(d.get("max_seq_len", 512), d.get("embed_dim", 128))
        layer.pe = np.array(d.get("pe", layer.pe), dtype=np.float64)
        return layer

    def weight_snapshot(self) -> dict:
        return {
            "W": None,
            "b": None,
            "dW": None,
            "db": None,
            "activation": self.pe[0].tolist(),
        }
