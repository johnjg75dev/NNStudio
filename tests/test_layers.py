"""
tests/test_layers.py
Unit tests for neural network layers.
"""
import pytest
import numpy as np
from app.core.activations import ACTIVATIONS
from app.core.optimizers import Adam
from app.core.layers.dense import DenseLayer


class TestDenseLayerInit:
    """Test DenseLayer initialization."""

    def test_layer_creation(self):
        """Should create a dense layer with correct dimensions."""
        layer = DenseLayer(n_in=10, n_out=5, activation=ACTIVATIONS["relu"])
        assert layer.n_in == 10
        assert layer.n_out == 5
        assert layer.W.shape == (5, 10)
        assert layer.b.shape == (5,)

    def test_weight_initialization(self):
        """Weights should be initialized with He initialization."""
        np.random.seed(42)
        layer = DenseLayer(n_in=100, n_out=10, activation=ACTIVATIONS["relu"])
        
        # He initialization: std = sqrt(2/n_in)
        expected_std = np.sqrt(2.0 / 100)
        actual_std = np.std(layer.W)
        
        # Allow some variance due to randomness
        assert 0.1 < actual_std < 0.3

    def test_bias_initialization(self):
        """Biases should be initialized to zero."""
        layer = DenseLayer(n_in=10, n_out=5, activation=ACTIVATIONS["relu"])
        np.testing.assert_array_equal(layer.b, np.zeros(5))

    def test_output_layer_creation(self):
        """Output layer should have is_output=True."""
        layer = DenseLayer(n_in=10, n_out=1, activation=ACTIVATIONS["sigmoid"], is_output=True)
        assert layer.is_output is True


class TestDenseLayerForward:
    """Test DenseLayer forward pass."""

    def test_forward_shape(self, simple_layer):
        """Forward pass should preserve batch dimension."""
        x = np.random.randn(2)
        output = simple_layer.forward(x)
        assert output.shape == (4,)

    def test_forward_linear_transform(self):
        """Forward should compute W @ x + b."""
        np.random.seed(42)
        layer = DenseLayer(n_in=3, n_out=2, activation=ACTIVATIONS["relu"])

        x = np.array([1.0, 2.0, 3.0])
        z = layer.W @ x + layer.b
        expected = np.maximum(0, z)  # ReLU

        output = layer.forward(x)
        np.testing.assert_array_almost_equal(output, expected)

    def test_forward_caches_input(self, simple_layer):
        """Forward should cache input for backprop."""
        x = np.random.randn(2)
        simple_layer.forward(x)
        np.testing.assert_array_equal(simple_layer._x, x)

    def test_forward_caches_pre_activation(self, simple_layer):
        """Forward should cache pre-activation z."""
        x = np.random.randn(2)
        simple_layer.forward(x)
        expected_z = simple_layer.W @ x + simple_layer.b
        np.testing.assert_array_almost_equal(simple_layer._z, expected_z)

    def test_output_layer_sigmoid(self, output_layer):
        """Output layer should use sigmoid regardless of activation."""
        np.random.seed(42)
        x = np.random.randn(4)
        output = output_layer.forward(x)

        # Sigmoid output should be in (0, 1)
        assert np.all(output > 0) and np.all(output < 1)


class TestDenseLayerBackward:
    """Test DenseLayer backward pass."""

    def test_backward_output_layer(self, output_layer):
        """Backward for output layer should propagate delta directly."""
        x = np.random.randn(4)
        output_layer.forward(x)
        
        delta = np.random.randn(1)
        grad_prev = output_layer.backward(delta)
        
        assert grad_prev.shape == (4,)

    def test_backward_hidden_layer(self, simple_layer):
        """Backward should compute gradient w.r.t. input."""
        x = np.random.randn(2)
        simple_layer.forward(x)
        
        delta = np.random.randn(4)
        grad_prev = simple_layer.backward(delta)
        
        assert grad_prev.shape == (2,)

    def test_backward_computes_dW(self):
        """Backward should compute weight gradients correctly."""
        np.random.seed(42)
        layer = DenseLayer(n_in=2, n_out=4, activation=ACTIVATIONS["relu"])
        
        # Use positive pre-activations so ReLU derivative is 1
        x = np.array([1.0, 2.0])
        layer.forward(x)
        
        # Ensure all pre-activations are positive (ReLU active)
        layer._z = np.array([1.0, 1.0, 1.0, 1.0])
        layer._a = layer.activation.forward(layer._z)
        
        delta = np.array([0.1, 0.2, -0.1, 0.3])
        layer.backward(delta)
        
        # dW = outer(delta, x)
        expected_dW = np.outer(delta, x)
        np.testing.assert_array_almost_equal(layer._dW, expected_dW, decimal=5)

    def test_backward_computes_db(self):
        """Backward should compute bias gradients."""
        np.random.seed(42)
        layer = DenseLayer(n_in=2, n_out=4, activation=ACTIVATIONS["relu"])
        
        # Use positive pre-activations so ReLU derivative is 1
        x = np.array([1.0, 2.0])
        layer.forward(x)
        
        # Ensure all pre-activations are positive (ReLU active)
        layer._z = np.array([1.0, 1.0, 1.0, 1.0])
        layer._a = layer.activation.forward(layer._z)
        
        delta = np.array([0.1, 0.2, -0.1, 0.3])
        layer.backward(delta)
        
        # db = delta (for ReLU where activation is positive)
        np.testing.assert_array_almost_equal(layer._db, delta, decimal=5)


class TestDenseLayerUpdate:
    """Test DenseLayer weight updates."""

    def test_update_weights(self, simple_layer):
        """Update should modify weights using optimizer."""
        x = np.random.randn(2)
        simple_layer.forward(x)
        
        delta = np.random.randn(4)
        simple_layer.backward(delta)
        
        optimizer = Adam(lr=0.01)
        old_W = simple_layer.W.copy()
        old_b = simple_layer.b.copy()
        
        simple_layer.update(optimizer, layer_idx=0)
        
        # Weights should have changed
        assert not np.array_equal(simple_layer.W, old_W)
        assert not np.array_equal(simple_layer.b, old_b)


class TestDenseLayerSerialization:
    """Test DenseLayer serialization."""

    def test_to_dict(self, simple_layer):
        """to_dict should serialize layer config and weights."""
        d = simple_layer.to_dict()

        assert d["type"] == "dense"
        assert d["n_in"] == 2
        assert d["n_out"] == 4
        assert d["activation"] == "tanh"
        assert d["is_output"] is False
        assert "W" in d
        assert "b" in d

    def test_from_dict(self, simple_layer):
        """from_dict should reconstruct layer from dict."""
        d = simple_layer.to_dict()
        restored = DenseLayer.from_dict(d)

        assert restored.n_in == simple_layer.n_in
        assert restored.n_out == simple_layer.n_out
        assert restored.activation.name == simple_layer.activation.name
        np.testing.assert_array_almost_equal(restored.W, simple_layer.W)
        np.testing.assert_array_almost_equal(restored.b, simple_layer.b)

    def test_roundtrip(self):
        """to_dict -> from_dict should produce identical layer."""
        np.random.seed(42)
        original = DenseLayer(
            n_in=5, n_out=3,
            activation=ACTIVATIONS["relu"],
            is_output=False
        )
        
        restored = DenseLayer.from_dict(original.to_dict())
        
        assert original.n_in == restored.n_in
        assert original.n_out == restored.n_out
        np.testing.assert_array_almost_equal(original.W, restored.W)
        np.testing.assert_array_almost_equal(original.b, restored.b)


class TestDenseLayerProperties:
    """Test DenseLayer properties."""

    def test_param_count(self, simple_layer):
        """param_count should equal W.size + b.size."""
        expected = simple_layer.W.size + simple_layer.b.size
        assert simple_layer.param_count == expected

    def test_weight_snapshot(self, simple_layer):
        """weight_snapshot should return current state."""
        x = np.random.randn(2)
        simple_layer.forward(x)
        
        delta = np.random.randn(4)
        simple_layer.backward(delta)
        
        snapshot = simple_layer.weight_snapshot()
        
        assert "W" in snapshot
        assert "b" in snapshot
        assert "dW" in snapshot
        assert "db" in snapshot
        assert "activation" in snapshot
        
        assert snapshot["W"] == simple_layer.W.tolist()
        assert snapshot["dW"] == simple_layer._dW.tolist()


class TestDenseLayerIntegration:
    """Integration tests for DenseLayer."""

    def test_forward_backward_consistency(self):
        """Gradient check: numerical vs analytical."""
        np.random.seed(42)
        layer = DenseLayer(n_in=5, n_out=3, activation=ACTIVATIONS["tanh"])
        
        x = np.random.randn(5)
        target = np.random.randn(3)
        
        # Forward
        output = layer.forward(x)
        
        # Compute loss gradient (MSE)
        delta = 2 * (output - target) / 3
        
        # Backward
        layer.backward(delta)
        
        # Numerical gradient check for weights
        eps = 1e-5
        for i in range(min(3, layer.W.size)):
            idx = np.unravel_index(i, layer.W.shape)
            
            # f(x + eps)
            layer.W[idx] += eps
            out_plus = layer.forward(x)
            loss_plus = np.mean((out_plus - target) ** 2)
            
            # f(x - eps)
            layer.W[idx] -= 2 * eps
            out_minus = layer.forward(x)
            loss_minus = np.mean((out_minus - target) ** 2)
            
            # Restore
            layer.W[idx] += eps
            
            # Numerical gradient
            numerical_grad = (loss_plus - loss_minus) / (2 * eps)
            analytical_grad = layer._dW[idx]
            
            # Allow some tolerance
            assert abs(numerical_grad - analytical_grad) < 1e-4
