"""
app/core/layers/rnn.py
Recurrent Neural Network layers: SimpleRNN, LSTM, GRU.
"""
from __future__ import annotations
import numpy as np
from .base import Layer
from ..activations import ACTIVATIONS, Activation
from ..optimizers import BaseOptimizer


class SimpleRNNLayer(Layer):
    """
    Simple Recurrent Neural Network layer.
    h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b)
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 activation: Activation = None,
                 return_sequences: bool = True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation or ACTIVATIONS["tanh"]
        self.return_sequences = return_sequences
        
        # Xavier initialization
        scale_xh = np.sqrt(2.0 / (input_size + hidden_size))
        scale_hh = np.sqrt(2.0 / (hidden_size + hidden_size))
        
        self.W_xh = np.random.randn(hidden_size, input_size).astype(np.float64) * scale_xh
        self.W_hh = np.random.randn(hidden_size, hidden_size).astype(np.float64) * scale_hh
        self.b = np.zeros(hidden_size, dtype=np.float64)
        
        # Cached values
        self._x: np.ndarray | None = None  # Input sequence (T, input_size)
        self._h: list[np.ndarray] = []     # Hidden states
        self._dW_xh: np.ndarray | None = None
        self._dW_hh: np.ndarray | None = None
        self._db: np.ndarray | None = None

    @property
    def n_in(self) -> int:
        return self.input_size

    @property
    def n_out(self) -> int:
        return self.hidden_size

    @property
    def is_output(self) -> bool:
        return False

    @property
    def param_count(self) -> int:
        return self.W_xh.size + self.W_hh.size + self.b.size

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Forward pass through RNN.
        x: (T, input_size) sequence or (input_size,) single step
        """
        # Handle single step input
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        self._x = x
        T = x.shape[0]
        
        # Initialize hidden state
        h = np.zeros(self.hidden_size)
        self._h = [h.copy()]
        
        # Process sequence
        for t in range(T):
            x_t = x[t]
            h = self.activation.forward(self.W_xh @ x_t + self.W_hh @ h + self.b)
            self._h.append(h.copy())
        
        # Return last hidden state or all
        if self.return_sequences:
            return np.array(self._h[1:]).flatten()
        else:
            return self._h[-1]

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Backpropagation Through Time (BPTT)."""
        if self._x is None or len(self._h) < 2:
            return np.zeros(self.input_size)
        
        T = self._x.shape[0]
        
        # Initialize gradients
        self._dW_xh = np.zeros_like(self.W_xh)
        self._dW_hh = np.zeros_like(self.W_hh)
        self._db = np.zeros_like(self.b)
        
        # Reshape delta if needed
        if self.return_sequences:
            delta = delta.reshape(T, self.hidden_size)
        else:
            delta = delta.reshape(1, self.hidden_size)
        
        # BPTT
        dh_next = np.zeros(self.hidden_size)
        dx = np.zeros_like(self._x)
        
        for t in reversed(range(T)):
            dh = delta[t] + dh_next
            h_t = self._h[t]
            x_t = self._x[t]
            
            # Gradient through activation
            if self._h[t + 1] is not None:
                dz = dh * self.activation.derivative(self.W_xh @ x_t + self.W_hh @ h_t + self.b)
            else:
                dz = dh
            
            # Accumulate gradients
            self._dW_xh += np.outer(dz, x_t)
            self._dW_hh += np.outer(dz, h_t)
            self._db += dz
            
            # Gradient for input and previous hidden state
            dx[t] = self.W_xh.T @ dz
            dh_next = self.W_hh.T @ dz
        
        return dx.flatten() if self.return_sequences else dx[0]

    def update(self, optimizer: BaseOptimizer, layer_idx: int):
        optimizer.tick()
        prefix = f"L{layer_idx}"
        self.W_xh = optimizer.step(self.W_xh, self._dW_xh, key=f"{prefix}_W_xh")
        self.W_hh = optimizer.step(self.W_hh, self._dW_hh, key=f"{prefix}_W_hh")
        self.b = optimizer.step(self.b, self._db, key=f"{prefix}_b")

    def to_dict(self) -> dict:
        return {
            "type": "simple_rnn",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "activation": self.activation.name,
            "return_sequences": self.return_sequences,
            "W_xh": self.W_xh.tolist(),
            "W_hh": self.W_hh.tolist(),
            "b": self.b.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SimpleRNNLayer":
        act = ACTIVATIONS.get(d.get("activation"), ACTIVATIONS["tanh"])
        layer = cls(
            input_size=d["input_size"],
            hidden_size=d["hidden_size"],
            activation=act,
            return_sequences=d.get("return_sequences", True)
        )
        layer.W_xh = np.array(d["W_xh"], dtype=np.float64)
        layer.W_hh = np.array(d["W_hh"], dtype=np.float64)
        layer.b = np.array(d["b"], dtype=np.float64)
        return layer

    def weight_snapshot(self) -> dict:
        return {
            "W": np.vstack([self.W_xh, self.W_hh]).tolist(),
            "b": self.b.tolist(),
            "dW": np.vstack([self._dW_xh, self._dW_hh]).tolist() if self._dW_xh is not None else None,
            "db": self._db.tolist() if self._db is not None else None,
            "activation": np.array(self._h[1:]).flatten().tolist() if self._h else None,
        }


class LSTMLayer(Layer):
    """
    Long Short-Term Memory layer.
    Implements forget, input, and output gates.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 return_sequences: bool = True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.return_sequences = return_sequences
        
        # Xavier initialization for all gates
        combined_size = input_size + hidden_size
        scale = np.sqrt(2.0 / combined_size)
        
        # Forget gate
        self.W_f = np.random.randn(hidden_size, combined_size).astype(np.float64) * scale
        self.b_f = np.ones(hidden_size, dtype=np.float64)  # Initialize to 1 for forget gate
        
        # Input gate
        self.W_i = np.random.randn(hidden_size, combined_size).astype(np.float64) * scale
        self.b_i = np.zeros(hidden_size, dtype=np.float64)
        
        # Cell candidate
        self.W_c = np.random.randn(hidden_size, combined_size).astype(np.float64) * scale
        self.b_c = np.zeros(hidden_size, dtype=np.float64)
        
        # Output gate
        self.W_o = np.random.randn(hidden_size, combined_size).astype(np.float64) * scale
        self.b_o = np.zeros(hidden_size, dtype=np.float64)
        
        # Cached values
        self._x: np.ndarray | None = None
        self._h: list[np.ndarray] = []
        self._c: list[np.ndarray] = []
        self._gates: list[dict] = []

    @property
    def n_in(self) -> int:
        return self.input_size

    @property
    def n_out(self) -> int:
        return self.hidden_size

    @property
    def is_output(self) -> bool:
        return False

    @property
    def param_count(self) -> int:
        return (self.W_f.size + self.b_f.size + 
                self.W_i.size + self.b_i.size + 
                self.W_c.size + self.b_c.size + 
                self.W_o.size + self.b_o.size)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _sigmoid_deriv(self, x: np.ndarray) -> np.ndarray:
        s = self._sigmoid(x)
        return s * (1 - s)

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass through LSTM."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        self._x = x
        T = x.shape[0]
        
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        self._h = [h.copy()]
        self._c = [c.copy()]
        self._gates = []
        
        for t in range(T):
            x_t = x[t]
            combined = np.concatenate([x_t, h])
            
            # Gates
            f_t = self._sigmoid(self.W_f @ combined + self.b_f)  # Forget
            i_t = self._sigmoid(self.W_i @ combined + self.b_i)  # Input
            c_tilde = np.tanh(self.W_c @ combined + self.b_c)    # Cell candidate
            o_t = self._sigmoid(self.W_o @ combined + self.b_o)  # Output
            
            # Update cell and hidden state
            c = f_t * c + i_t * c_tilde
            h = o_t * np.tanh(c)
            
            self._h.append(h.copy())
            self._c.append(c.copy())
            self._gates.append({
                'f': f_t, 'i': i_t, 'c_tilde': c_tilde, 'o': o_t,
                'combined': combined
            })
        
        if self.return_sequences:
            return np.array(self._h[1:]).flatten()
        else:
            return self._h[-1]

    def backward(self, delta: np.ndarray) -> np.ndarray:
        """Backpropagation Through Time for LSTM."""
        if self._x is None or len(self._h) < 2:
            return np.zeros(self.input_size)
        
        T = self._x.shape[0]
        
        # Initialize gradients
        dW_f = np.zeros_like(self.W_f)
        dW_i = np.zeros_like(self.W_i)
        dW_c = np.zeros_like(self.W_c)
        dW_o = np.zeros_like(self.W_o)
        db_f = np.zeros_like(self.b_f)
        db_i = np.zeros_like(self.b_i)
        db_c = np.zeros_like(self.b_c)
        db_o = np.zeros_like(self.b_o)
        
        # Reshape delta
        if self.return_sequences:
            delta = delta.reshape(T, self.hidden_size)
        else:
            delta = delta.reshape(1, self.hidden_size)
        
        dh_next = np.zeros(self.hidden_size)
        dc_next = np.zeros(self.hidden_size)
        dx = np.zeros_like(self._x)
        
        for t in reversed(range(T)):
            dh = delta[t] + dh_next
            c_t = self._c[t + 1]
            c_prev = self._c[t]
            gates = self._gates[t]
            
            # Output gate gradient
            do = dh * np.tanh(c_t)
            do_raw = do * self._sigmoid_deriv(self.W_o @ gates['combined'] + self.b_o)
            
            # Cell gradient
            dc = dh * gates['o'] * (1 - np.tanh(c_t) ** 2) + dc_next
            
            # Forget gate gradient
            df = dc * c_prev
            df_raw = df * self._sigmoid_deriv(self.W_f @ gates['combined'] + self.b_f)
            
            # Input gate gradient
            di = dc * gates['c_tilde']
            di_raw = di * self._sigmoid_deriv(self.W_i @ gates['combined'] + self.b_i)
            
            # Cell candidate gradient
            dc_tilde = dc * gates['i']
            dc_tilde_raw = dc_tilde * (1 - gates['c_tilde'] ** 2)
            
            # Accumulate weight gradients
            dW_f += np.outer(do_raw, gates['combined'])
            dW_i += np.outer(di_raw, gates['combined'])
            dW_c += np.outer(dc_tilde_raw, gates['combined'])
            dW_o += np.outer(do_raw, gates['combined'])
            
            db_f += df_raw
            db_i += di_raw
            db_c += dc_tilde_raw
            db_o += do_raw
            
            # Gradient for input and previous hidden state
            d_combined = (self.W_f.T @ df_raw + self.W_i.T @ di_raw + 
                         self.W_c.T @ dc_tilde_raw + self.W_o.T @ do_raw)
            dx[t] = d_combined[:self.input_size]
            dh_next = d_combined[self.input_size:]
            dc_next = dc * gates['f']
        
        # Store gradients
        self._dW = dW_f + dW_i + dW_c + dW_o  # Combined for update
        self._db = db_f + db_i + db_c + db_o
        
        return dx.flatten() if self.return_sequences else dx[0]

    def update(self, optimizer: BaseOptimizer, layer_idx: int):
        optimizer.tick()
        prefix = f"L{layer_idx}"
        self.W_f = optimizer.step(self.W_f, self._dW, key=f"{prefix}_W_f")
        self.W_i = optimizer.step(self.W_i, self._dW, key=f"{prefix}_W_i")
        self.W_c = optimizer.step(self.W_c, self._dW, key=f"{prefix}_W_c")
        self.W_o = optimizer.step(self.W_o, self._dW, key=f"{prefix}_W_o")
        self.b_f = optimizer.step(self.b_f, self._db, key=f"{prefix}_b_f")
        self.b_i = optimizer.step(self.b_i, self._db, key=f"{prefix}_b_i")
        self.b_c = optimizer.step(self.b_c, self._db, key=f"{prefix}_b_c")
        self.b_o = optimizer.step(self.b_o, self._db, key=f"{prefix}_b_o")

    def to_dict(self) -> dict:
        return {
            "type": "lstm",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "return_sequences": self.return_sequences,
            "W_f": self.W_f.tolist(),
            "W_i": self.W_i.tolist(),
            "W_c": self.W_c.tolist(),
            "W_o": self.W_o.tolist(),
            "b_f": self.b_f.tolist(),
            "b_i": self.b_i.tolist(),
            "b_c": self.b_c.tolist(),
            "b_o": self.b_o.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LSTMLayer":
        layer = cls(
            input_size=d["input_size"],
            hidden_size=d["hidden_size"],
            return_sequences=d.get("return_sequences", True)
        )
        layer.W_f = np.array(d["W_f"], dtype=np.float64)
        layer.W_i = np.array(d["W_i"], dtype=np.float64)
        layer.W_c = np.array(d["W_c"], dtype=np.float64)
        layer.W_o = np.array(d["W_o"], dtype=np.float64)
        layer.b_f = np.array(d["b_f"], dtype=np.float64)
        layer.b_i = np.array(d["b_i"], dtype=np.float64)
        layer.b_c = np.array(d["b_c"], dtype=np.float64)
        layer.b_o = np.array(d["b_o"], dtype=np.float64)
        return layer

    def weight_snapshot(self) -> dict:
        return {
            "W": np.vstack([self.W_f, self.W_i, self.W_c, self.W_o]).tolist(),
            "b": np.concatenate([self.b_f, self.b_i, self.b_c, self.b_o]).tolist(),
            "dW": None,  # Simplified
            "db": None,
            "activation": np.array(self._h[1:]).flatten().tolist() if self._h else None,
        }
