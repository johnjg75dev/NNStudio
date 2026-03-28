"""
tests/test_new_layers.py
Unit tests for new layer types: DropoutLayer and BatchNormLayer.
"""
import pytest
import numpy as np
from app.core.layers.dropout import DropoutLayer
from app.core.layers.batch_norm import BatchNormLayer
from app.core.optimizers import Adam


class TestDropoutLayer:
    """Test DropoutLayer."""

    def test_dropout_init(self):
        """Should create dropout layer with correct rate."""
        layer = DropoutLayer(rate=0.5)
        assert layer.rate == 0.5
        assert layer.keep_prob == 0.5

    def test_dropout_no_op_in_eval(self):
        """Dropout should be identity in eval mode."""
        layer = DropoutLayer(rate=0.5)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        output = layer.forward(x, training=False)
        np.testing.assert_array_equal(output, x)
        assert layer._mask is None

    def test_dropout_applies_mask_in_train(self):
        """Dropout should apply mask in training mode."""
        np.random.seed(42)
        layer = DropoutLayer(rate=0.5)
        x = np.ones(100)
        
        output = layer.forward(x, training=True)
        assert layer._mask is not None
        
        # Some values should be zeroed
        n_zero = np.sum(output == 0)
        assert n_zero > 0
        
        # Non-zero values should be scaled by 1/keep_prob
        non_zero = output[output != 0]
        expected_scale = 1.0 / 0.5  # 2.0
        np.testing.assert_array_almost_equal(non_zero, np.ones_like(non_zero) * expected_scale)

    def test_dropout_backward_with_mask(self):
        """Backward should apply mask to gradient."""
        np.random.seed(42)
        layer = DropoutLayer(rate=0.5)
        x = np.ones(100)
        
        layer.forward(x, training=True)
        mask = layer._mask.copy()
        
        delta = np.ones(100)
        grad_out = layer.backward(delta)
        
        # Gradient should be zero where mask was zero
        zero_mask = mask == 0
        if np.any(zero_mask):
            assert np.all(grad_out[zero_mask] == 0)

    def test_dropout_no_params(self):
        """Dropout should have no trainable parameters."""
        layer = DropoutLayer(rate=0.5)
        assert layer.param_count == 0
        assert layer.n_in == 0
        assert layer.n_out == 0

    def test_dropout_update_noop(self):
        """Dropout update should be no-op."""
        layer = DropoutLayer(rate=0.5)
        optimizer = Adam(lr=0.01)
        
        # Should not raise
        layer.update(optimizer, layer_idx=0)

    def test_dropout_serialization(self):
        """Dropout should serialize correctly."""
        layer = DropoutLayer(rate=0.3)
        d = layer.to_dict()
        
        assert d["type"] == "dropout"
        assert d["rate"] == 0.3
        
        restored = DropoutLayer.from_dict(d)
        assert restored.rate == layer.rate

    def test_dropout_different_rates(self):
        """Different dropout rates should produce different behavior."""
        np.random.seed(42)
        
        for rate in [0.1, 0.3, 0.5, 0.7]:
            layer = DropoutLayer(rate=rate)
            x = np.ones(1000)
            output = layer.forward(x, training=True)
            
            # Fraction of zeros should be approximately equal to rate
            frac_zero = np.mean(output == 0)
            assert abs(frac_zero - rate) < 0.1  # Allow some variance


class TestBatchNormLayer:
    """Test BatchNormLayer."""

    def test_batchnorm_init(self):
        """Should create batchnorm layer with correct parameters."""
        layer = BatchNormLayer(n_features=10)
        assert layer.n_features == 10
        assert layer.gamma.shape == (10,)
        assert layer.beta.shape == (10,)
        assert np.all(layer.gamma == 1)
        assert np.all(layer.beta == 0)

    def test_batchnorm_forward_train(self):
        """BatchNorm forward in training mode should normalize."""
        np.random.seed(42)
        layer = BatchNormLayer(n_features=5)
        
        # Create batch with known mean and variance
        x = np.random.randn(32, 5)
        
        output = layer.forward(x, training=True)
        
        # Output should be approximately normalized (mean≈0, std≈1)
        output_mean = np.mean(output, axis=0)
        output_std = np.std(output, axis=0)
        
        assert np.all(np.abs(output_mean) < 0.5)
        assert np.all(np.abs(output_std - 1) < 0.5)

    def test_batchnorm_forward_eval(self):
        """BatchNorm forward in eval mode should use running stats."""
        np.random.seed(42)
        layer = BatchNormLayer(n_features=5)
        
        # Train first to populate running stats
        x_train = np.random.randn(32, 5)
        layer.forward(x_train, training=True)
        
        # Eval mode
        x_eval = np.random.randn(10, 5)
        output1 = layer.forward(x_eval, training=False)
        output2 = layer.forward(x_eval, training=False)
        
        # Should be deterministic in eval mode
        np.testing.assert_array_equal(output1, output2)

    def test_batchnorm_backward(self):
        """BatchNorm backward should compute correct gradients."""
        np.random.seed(42)
        layer = BatchNormLayer(n_features=5)
        
        x = np.random.randn(32, 5)
        layer.forward(x, training=True)
        
        delta = np.random.randn(32, 5)
        grad_in = layer.backward(delta)
        
        assert grad_in.shape == x.shape
        assert layer._dgamma is not None
        assert layer._dbeta is not None
        assert layer._dgamma.shape == (5,)
        assert layer._dbeta.shape == (5,)

    def test_batchnorm_update(self):
        """BatchNorm update should modify gamma and beta."""
        np.random.seed(42)
        layer = BatchNormLayer(n_features=5)
        
        x = np.random.randn(32, 5)
        layer.forward(x, training=True)
        
        delta = np.random.randn(32, 5)
        layer.backward(delta)
        
        old_gamma = layer.gamma.copy()
        old_beta = layer.beta.copy()
        
        optimizer = Adam(lr=0.01)
        layer.update(optimizer, layer_idx=0)
        
        # Gamma and beta should have changed
        assert not np.array_equal(layer.gamma, old_gamma)
        assert not np.array_equal(layer.beta, old_beta)

    def test_batchnorm_param_count(self):
        """BatchNorm should have 2*n_features parameters."""
        layer = BatchNormLayer(n_features=10)
        assert layer.param_count == 20  # gamma (10) + beta (10)

    def test_batchnorm_serialization(self):
        """BatchNorm should serialize correctly."""
        np.random.seed(42)
        layer = BatchNormLayer(n_features=5, momentum=0.95, eps=1e-5)
        
        # Run forward to update running stats
        x = np.random.randn(32, 5)
        layer.forward(x, training=True)
        
        d = layer.to_dict()
        
        assert d["type"] == "batchnorm"
        assert d["n_features"] == 5
        assert d["momentum"] == 0.95
        assert d["eps"] == 1e-5
        assert len(d["gamma"]) == 5
        assert len(d["beta"]) == 5
        
        restored = BatchNormLayer.from_dict(d)
        assert restored.n_features == layer.n_features
        np.testing.assert_array_almost_equal(restored.gamma, layer.gamma)
        np.testing.assert_array_almost_equal(restored.beta, layer.beta)

    def test_batchnorm_running_stats(self):
        """BatchNorm should update running mean and variance."""
        np.random.seed(42)
        layer = BatchNormLayer(n_features=5, momentum=0.9)
        
        # Initial running stats should be 0 and 1
        assert np.all(layer.running_mean == 0)
        assert np.all(layer.running_var == 1)
        
        # Train with data that has known mean
        x = np.random.randn(32, 5) + 5  # Mean around 5
        layer.forward(x, training=True)
        
        # Running mean should have moved towards 5
        assert np.all(layer.running_mean > 0)
        
        # Run more iterations
        for _ in range(10):
            layer.forward(x, training=True)
        
        # Running mean should be closer to 5 now
        assert np.all(layer.running_mean > 2)

    def test_batchnorm_eps_stability(self):
        """BatchNorm should be numerically stable with small variance."""
        layer = BatchNormLayer(n_features=5, eps=1e-8)
        
        # Input with very small variance
        x = np.ones((32, 5)) * 0.5 + 1e-10 * np.random.randn(32, 5)
        
        # Should not raise or produce NaN
        output = layer.forward(x, training=True)
        assert np.all(np.isfinite(output))


class TestLayerIntegration:
    """Integration tests for new layer types."""

    def test_dense_dropout_sequence(self):
        """Dense followed by Dropout should work correctly."""
        from app.core.layers.dense import DenseLayer
        from app.core.activations import ACTIVATIONS
        
        np.random.seed(42)
        dense = DenseLayer(n_in=10, n_out=5, activation=ACTIVATIONS["relu"])
        dropout = DropoutLayer(rate=0.5)
        
        # Single sample (1D input as DenseLayer expects)
        x = np.random.randn(10)
        
        # Forward
        h = dense.forward(x)
        h_out = dropout.forward(h, training=True)
        
        assert h_out.shape == (5,)
        
        # Backward
        delta = np.random.randn(5)
        delta_h = dropout.backward(delta)
        assert delta_h.shape == (5,)

    def test_dense_batchnorm_sequence(self):
        """Dense followed by BatchNorm should work correctly."""
        from app.core.layers.dense import DenseLayer
        from app.core.activations import ACTIVATIONS
        
        np.random.seed(42)
        dense = DenseLayer(n_in=10, n_out=5, activation=ACTIVATIONS["relu"])
        bn = BatchNormLayer(n_features=5)
        
        # Single sample
        x = np.random.randn(10)
        
        # Forward
        h = dense.forward(x)
        h_out = bn.forward(h, training=True)
        
        assert h_out.shape == (5,)
        
        # Backward
        delta = np.random.randn(5)
        delta_h = bn.backward(delta)
        assert delta_h.shape == (5,)

    def test_full_block_dense_bn_dropout(self):
        """Full block: Dense -> BatchNorm -> Dropout should work."""
        from app.core.layers.dense import DenseLayer
        from app.core.activations import ACTIVATIONS
        
        np.random.seed(42)
        dense = DenseLayer(n_in=10, n_out=8, activation=ACTIVATIONS["relu"])
        bn = BatchNormLayer(n_features=8)
        dropout = DropoutLayer(rate=0.2)
        
        # Single sample
        x = np.random.randn(10)
        
        # Forward pass
        h = dense.forward(x)
        h = bn.forward(h, training=True)
        h = dropout.forward(h, training=True)
        
        assert h.shape == (8,)
        
        # Backward pass
        delta = np.random.randn(8)
        delta = dropout.backward(delta)
        delta = bn.backward(delta)
        
        assert delta.shape == (8,)
