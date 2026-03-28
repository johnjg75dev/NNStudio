"""
tests/test_activations.py
Unit tests for activation functions.
"""
import pytest
import numpy as np
from app.core.activations import ACTIVATIONS, Activation


class TestActivations:
    """Test suite for activation functions."""

    def test_all_activations_registered(self):
        """Ensure all expected activations are in the registry."""
        expected = {"relu", "leakyrelu", "tanh", "sigmoid", "gelu", "swish"}
        assert set(ACTIVATIONS.keys()) == expected

    def test_activation_structure(self, activation_name):
        """Each activation should have required attributes."""
        act = ACTIVATIONS[activation_name]
        assert isinstance(act, Activation)
        assert act.name == activation_name
        assert act.label
        assert act.description
        assert callable(act.forward)
        assert callable(act.derivative)

    def test_relu_forward(self):
        """ReLU should output max(0, x)."""
        act = ACTIVATIONS["relu"]
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(act.forward(x), expected)

    def test_relu_derivative(self):
        """ReLU derivative should be 0 for x<0, 1 for x>0."""
        act = ACTIVATIONS["relu"]
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(act.derivative(x), expected)

    def test_sigmoid_forward(self):
        """Sigmoid should output values in (0, 1)."""
        act = ACTIVATIONS["sigmoid"]
        x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        result = act.forward(x)
        assert np.all(result > 0) and np.all(result < 1)
        assert result[2] == 0.5  # sigmoid(0) = 0.5
        assert result[0] < 0.01  # sigmoid(-10) ≈ 0
        assert result[4] > 0.99  # sigmoid(10) ≈ 1

    def test_sigmoid_derivative(self):
        """Sigmoid derivative = sigmoid(x) * (1 - sigmoid(x))."""
        act = ACTIVATIONS["sigmoid"]
        x = np.array([-1.0, 0.0, 1.0])
        s = act.forward(x)
        expected = s * (1 - s)
        np.testing.assert_array_almost_equal(act.derivative(x), expected)

    def test_tanh_forward(self):
        """Tanh should output values in (-1, 1)."""
        act = ACTIVATIONS["tanh"]
        x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        result = act.forward(x)
        assert np.all(result >= -1) and np.all(result <= 1)
        assert result[2] == 0.0  # tanh(0) = 0

    def test_tanh_derivative(self):
        """Tanh derivative = 1 - tanh(x)^2."""
        act = ACTIVATIONS["tanh"]
        x = np.array([-1.0, 0.0, 1.0])
        t = act.forward(x)
        expected = 1 - t ** 2
        np.testing.assert_array_almost_equal(act.derivative(x), expected)

    def test_leaky_relu_forward(self):
        """Leaky ReLU should have small negative slope."""
        act = ACTIVATIONS["leakyrelu"]
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = np.array([-0.02, -0.01, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(act.forward(x), expected)

    def test_leaky_relu_derivative(self):
        """Leaky ReLU derivative should be 0.01 for x<0, 1 for x>0."""
        act = ACTIVATIONS["leakyrelu"]
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = np.array([0.01, 0.01, 0.01, 1.0, 1.0])
        np.testing.assert_array_equal(act.derivative(x), expected)

    def test_gelu_forward(self):
        """GELU should be smooth and approximately linear for large x."""
        act = ACTIVATIONS["gelu"]
        x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        result = act.forward(x)
        assert result[2] == 0.0  # gelu(0) = 0
        assert result[0] <= 0  # negative input → negative output (or -0.0)
        assert result[4] > 9  # large positive → approximately linear

    def test_swish_forward(self):
        """Swish = x * sigmoid(x)."""
        act = ACTIVATIONS["swish"]
        sig = ACTIVATIONS["sigmoid"]
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        expected = x * sig.forward(x)
        np.testing.assert_array_almost_equal(act.forward(x), expected)

    def test_activations_preserve_shape(self, activation_name):
        """Activations should preserve input shape."""
        act = ACTIVATIONS[activation_name]
        shapes = [(1,), (10,), (5, 4), (2, 3, 4)]
        for shape in shapes:
            x = np.random.randn(*shape)
            result = act.forward(x)
            assert result.shape == shape

    def test_sigmoid_numerical_stability(self):
        """Sigmoid should handle extreme values without overflow."""
        act = ACTIVATIONS["sigmoid"]
        extreme = np.array([-1000.0, -500.0, 500.0, 1000.0])
        result = act.forward(extreme)
        assert np.all(np.isfinite(result))
        assert result[0] == 0.0  # Should not be NaN
        assert result[-1] == 1.0  # Should not be NaN

    def test_tanh_zero_centered(self):
        """Tanh outputs should be approximately zero-centered for symmetric input."""
        act = ACTIVATIONS["tanh"]
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = act.forward(x)
        # For symmetric input, mean should be close to 0
        assert abs(np.mean(result)) < 0.1


class TestActivationRegistry:
    """Test the activation registry itself."""

    def test_get_activation(self):
        """Should retrieve activation by name."""
        assert ACTIVATIONS["relu"].name == "relu"
        assert ACTIVATIONS["tanh"].name == "tanh"

    def test_get_invalid_activation(self):
        """Should raise KeyError for invalid activation."""
        with pytest.raises(KeyError):
            _ = ACTIVATIONS["invalid_activation"]
