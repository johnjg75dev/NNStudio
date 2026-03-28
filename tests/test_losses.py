"""
tests/test_losses.py
Unit tests for loss functions.
"""
import pytest
import numpy as np
from app.core.losses import LOSSES, LossFunction


class TestLossFunctions:
    """Test suite for loss functions."""

    def test_all_losses_registered(self):
        """Ensure all expected losses are in the registry."""
        expected = {"mse", "bce", "mae"}
        assert set(LOSSES.keys()) == expected

    def test_loss_structure(self, loss_name):
        """Each loss should have required attributes."""
        loss = LOSSES[loss_name]
        assert isinstance(loss, LossFunction)
        assert loss.name == loss_name
        assert loss.label
        assert loss.description
        assert callable(loss.compute)
        assert callable(loss.output_delta)

    def test_mse_compute(self):
        """MSE should compute mean squared error."""
        loss = LOSSES["mse"]
        p = np.array([0.1, 0.5, 0.9])
        y = np.array([0.0, 0.5, 1.0])
        expected = np.mean((p - y) ** 2)
        assert abs(loss.compute(p, y) - expected) < 1e-10

    def test_mse_zero_when_perfect(self):
        """MSE should be zero when predictions match targets."""
        loss = LOSSES["mse"]
        p = np.array([0.3, 0.7, 0.5])
        assert loss.compute(p, p) == 0.0

    def test_mse_output_delta(self):
        """MSE delta should be (p - y) * p * (1 - p) for sigmoid output."""
        loss = LOSSES["mse"]
        p = np.array([0.2, 0.5, 0.8])
        y = np.array([0.0, 0.5, 1.0])
        expected = (p - y) * p * (1.0 - p)
        np.testing.assert_array_almost_equal(loss.output_delta(p, y), expected)

    def test_bce_compute(self):
        """BCE should compute binary cross-entropy."""
        loss = LOSSES["bce"]
        p = np.array([0.1, 0.9, 0.5])
        y = np.array([0.0, 1.0, 0.5])
        # BCE = -mean(y * log(p) + (1-y) * log(1-p))
        expected = -np.mean(
            y * np.log(np.clip(p, 1e-9, 1)) +
            (1 - y) * np.log(np.clip(1 - p, 1e-9, 1))
        )
        assert abs(loss.compute(p, y) - expected) < 1e-10

    def test_bce_zero_when_perfect(self):
        """BCE should approach zero when predictions match targets."""
        loss = LOSSES["bce"]
        p = np.array([0.001, 0.999])
        y = np.array([0.0, 1.0])
        assert loss.compute(p, y) < 0.01

    def test_bce_output_delta(self):
        """BCE delta should be (p - y) for combined BCE + sigmoid."""
        loss = LOSSES["bce"]
        p = np.array([0.2, 0.5, 0.8])
        y = np.array([0.0, 0.5, 1.0])
        expected = p - y
        np.testing.assert_array_almost_equal(loss.output_delta(p, y), expected)

    def test_bce_high_penalty_for_confident_wrong(self):
        """BCE should heavily penalize confident wrong predictions."""
        loss = LOSSES["bce"]
        # Very confident but wrong
        p_wrong = np.array([0.999, 0.001])
        y = np.array([0.0, 1.0])
        loss_wrong = loss.compute(p_wrong, y)
        
        # Somewhat wrong
        p_less_wrong = np.array([0.6, 0.4])
        loss_less_wrong = loss.compute(p_less_wrong, y)
        
        assert loss_wrong > loss_less_wrong

    def test_mae_compute(self):
        """MAE should compute mean absolute error."""
        loss = LOSSES["mae"]
        p = np.array([0.1, 0.5, 0.9])
        y = np.array([0.0, 0.5, 1.0])
        expected = np.mean(np.abs(p - y))
        assert abs(loss.compute(p, y) - expected) < 1e-10

    def test_mae_zero_when_perfect(self):
        """MAE should be zero when predictions match targets."""
        loss = LOSSES["mae"]
        p = np.array([0.3, 0.7, 0.5])
        assert loss.compute(p, p) == 0.0

    def test_mae_output_delta(self):
        """MAE delta should be sign(p - y) * p * (1 - p)."""
        loss = LOSSES["mae"]
        p = np.array([0.2, 0.5, 0.8])
        y = np.array([0.0, 0.5, 1.0])
        expected = np.sign(p - y) * p * (1.0 - p)
        np.testing.assert_array_almost_equal(loss.output_delta(p, y), expected)

    def test_loss_preserves_dtype(self, loss_name):
        """Loss compute should return float."""
        loss = LOSSES[loss_name]
        p = np.array([0.5, 0.5], dtype=np.float64)
        y = np.array([0.5, 0.5], dtype=np.float64)
        result = loss.compute(p, y)
        assert isinstance(result, float)

    def test_bce_numerical_stability(self):
        """BCE should handle extreme predictions without NaN."""
        loss = LOSSES["bce"]
        p = np.array([1e-10, 1.0 - 1e-10, 0.5])
        y = np.array([0.0, 1.0, 0.5])
        result = loss.compute(p, y)
        assert np.isfinite(result)

    def test_loss_batch_independence(self, loss_name):
        """Loss should be the same whether computed in batch or individually."""
        loss = LOSSES[loss_name]
        p = np.array([0.2, 0.5, 0.8])
        y = np.array([0.0, 0.5, 1.0])
        
        batch_loss = loss.compute(p, y)
        individual_losses = [loss.compute(np.array([pi]), np.array([yi])) 
                           for pi, yi in zip(p, y)]
        expected = np.mean(individual_losses)
        
        assert abs(batch_loss - expected) < 1e-10


class TestLossRegistry:
    """Test the loss registry itself."""

    def test_get_loss(self):
        """Should retrieve loss by name."""
        assert LOSSES["bce"].name == "bce"
        assert LOSSES["mse"].name == "mse"

    def test_get_invalid_loss(self):
        """Should raise KeyError for invalid loss."""
        with pytest.raises(KeyError):
            _ = LOSSES["invalid_loss"]
