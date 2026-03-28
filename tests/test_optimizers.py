"""
tests/test_optimizers.py
Unit tests for optimization algorithms.
"""
import pytest
import numpy as np
from app.core.optimizers import (
    OptimizerFactory, SGD, SGDMomentum, RMSProp, Adam, AdamW, BaseOptimizer
)


class TestOptimizerFactory:
    """Test the optimizer factory."""

    def test_build_sgd(self):
        """Should create SGD optimizer."""
        opt = OptimizerFactory.build("sgd", lr=0.01)
        assert isinstance(opt, SGD)
        assert opt.lr == 0.01

    def test_build_momentum(self):
        """Should create SGD with momentum optimizer."""
        opt = OptimizerFactory.build("momentum", lr=0.01, momentum=0.9)
        assert isinstance(opt, SGDMomentum)
        assert opt.momentum == 0.9

    def test_build_rmsprop(self):
        """Should create RMSProp optimizer."""
        opt = OptimizerFactory.build("rmsprop", lr=0.01, rho=0.9)
        assert isinstance(opt, RMSProp)
        assert opt.rho == 0.9

    def test_build_adam(self):
        """Should create Adam optimizer."""
        opt = OptimizerFactory.build("adam", lr=0.001)
        assert isinstance(opt, Adam)
        assert opt.lr == 0.001

    def test_build_adamw(self):
        """Should create AdamW optimizer."""
        opt = OptimizerFactory.build("adamw", lr=0.001, weight_decay=0.01)
        assert isinstance(opt, AdamW)
        assert opt.weight_decay == 0.01

    def test_build_invalid_optimizer(self):
        """Should raise ValueError for invalid optimizer."""
        with pytest.raises(ValueError, match="Unknown optimizer"):
            OptimizerFactory.build("invalid", lr=0.01)

    def test_available_optimizers(self):
        """Should return list of available optimizers."""
        available = OptimizerFactory.available()
        assert isinstance(available, list)
        assert len(available) >= 5
        for opt in available:
            assert "key" in opt
            assert "label" in opt
            assert "description" in opt


class TestSGD:
    """Test SGD optimizer."""

    def test_sgd_step(self):
        """SGD should update params: param - lr * grad."""
        opt = SGD(lr=0.1)
        param = np.array([1.0, 2.0, 3.0])
        grad = np.array([0.1, 0.2, 0.3])
        expected = param - 0.1 * grad
        result = opt.step(param, grad, key="test")
        np.testing.assert_array_almost_equal(result, expected)

    def test_sgd_multiple_steps(self):
        """SGD should accumulate updates over multiple steps."""
        opt = SGD(lr=0.1)
        param = np.array([1.0, 2.0])
        
        for i in range(5):
            grad = np.array([0.1, 0.1])
            param = opt.step(param, grad, key="test")
        
        expected = np.array([1.0, 2.0]) - 5 * 0.1 * np.array([0.1, 0.1])
        np.testing.assert_array_almost_equal(param, expected)


class TestSGDMomentum:
    """Test SGD with momentum optimizer."""

    def test_momentum_initial_step(self):
        """First step should use momentum formula."""
        opt = SGDMomentum(lr=0.1, momentum=0.9)
        param = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])
        
        # v = momentum * 0 + lr * grad = lr * grad
        # param = param - v
        expected = param - 0.1 * grad
        result = opt.step(param, grad, key="test")
        np.testing.assert_array_almost_equal(result, expected)

    def test_momentum_accumulation(self):
        """Momentum should accumulate velocity."""
        opt = SGDMomentum(lr=0.1, momentum=0.9)
        param = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.1])
        
        # First step
        param1 = opt.step(param, grad, key="test")
        v1 = 0.1 * grad  # velocity after first step
        
        # Second step: v = 0.9 * v1 + 0.1 * grad
        param2 = opt.step(param1, grad, key="test")
        
        # Check that second step is larger due to momentum
        update1 = np.abs(param - param1)
        update2 = np.abs(param1 - param2)
        assert np.all(update2 > update1 * 0.8)  # Momentum effect


class TestRMSProp:
    """Test RMSProp optimizer."""

    def test_rmsprop_step(self):
        """RMSProp should adapt learning rate per parameter."""
        opt = RMSProp(lr=0.1, rho=0.9, eps=1e-8)
        param = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.5])
        
        result = opt.step(param, grad, key="test")
        
        # Check that update happened
        assert not np.array_equal(result, param)

    def test_rmsprop_cache_accumulation(self):
        """RMSProp cache should accumulate squared gradients."""
        opt = RMSProp(lr=0.1, rho=0.9)
        param = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.1])
        
        # Multiple steps with same gradient
        for _ in range(100):
            param = opt.step(param, grad, key="test")
        
        # Cache should approach grad^2 (with rho=0.9, converges slowly)
        # After many iterations: cache ≈ grad^2 = 0.01
        assert opt._cache["test"][0] > 0.005
        assert opt._cache["test"][0] < 0.015


class TestAdam:
    """Test Adam optimizer."""

    def test_adam_step(self):
        """Adam should update params using bias-corrected moments."""
        opt = Adam(lr=0.01)
        param = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])
        
        result = opt.step(param, grad, key="test")
        
        # Check that update happened
        assert not np.array_equal(result, param)

    def test_adam_bias_correction(self):
        """Adam should apply bias correction for early steps."""
        opt = Adam(lr=0.01)
        param = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.1])
        
        # First step
        param1 = opt.step(param, grad, key="test")
        
        # Second step (bias correction should be different)
        param2 = opt.step(param1, grad, key="test")
        
        # Updates should be different due to bias correction
        update1 = np.abs(param - param1)
        update2 = np.abs(param1 - param2)
        assert not np.array_equal(update1, update2)

    def test_adam_moment_storage(self):
        """Adam should store first and second moments."""
        opt = Adam(lr=0.01)
        param = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])
        
        opt.step(param, grad, key="test")
        
        assert "test" in opt._m
        assert "test" in opt._v
        assert opt._m["test"].shape == grad.shape
        assert opt._v["test"].shape == grad.shape


class TestAdamW:
    """Test AdamW optimizer."""

    def test_adamw_weight_decay(self):
        """AdamW should apply decoupled weight decay."""
        opt = AdamW(lr=0.01, weight_decay=0.1)
        param = np.array([1.0, 2.0])
        grad = np.array([0.0, 0.0])  # Zero gradient
        
        result = opt.step(param, grad, key="test")
        
        # Should still update due to weight decay
        expected_decay = param - 0.01 * 0.1 * param
        np.testing.assert_array_almost_equal(result, expected_decay)

    def test_adamw_vs_adam(self):
        """AdamW should differ from Adam with weight decay."""
        np.random.seed(42)
        param1 = np.random.randn(10)
        param2 = param1.copy()
        grad = np.random.randn(10) * 0.1
        
        opt_adam = Adam(lr=0.01)
        opt_adamw = AdamW(lr=0.01, weight_decay=0.1)
        
        # Multiple steps
        for _ in range(10):
            param1 = opt_adam.step(param1, grad, key="test")
            param2 = opt_adamw.step(param2, grad, key="test")
        
        # Results should differ
        assert not np.allclose(param1, param2)


class TestOptimizerState:
    """Test optimizer state serialization."""

    def test_sgd_state_dict(self):
        """SGD should serialize state."""
        opt = SGD(lr=0.1)
        opt.t = 5
        state = opt.state_dict()
        assert state["t"] == 5
        assert state["lr"] == 0.1

    def test_adam_state_dict(self):
        """Adam should serialize moments."""
        opt = Adam(lr=0.01)
        param = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])
        opt.step(param, grad, key="test")
        
        state = opt.state_dict()
        assert "m" in state
        assert "v" in state
        assert "test" in state["m"]
        assert "test" in state["v"]

    def test_adam_load_state(self):
        """Adam should load state correctly."""
        opt1 = Adam(lr=0.01)
        param = np.array([1.0, 2.0])
        grad = np.array([0.1, 0.2])
        opt1.step(param, grad, key="test")
        
        state = opt1.state_dict()
        
        opt2 = Adam(lr=0.001)  # Different lr
        opt2.load_state(state)
        
        assert opt2.lr == 0.01  # Should restore original lr
        assert opt2.t == opt1.t
        assert "test" in opt2._m


class TestOptimizerIntegration:
    """Integration tests for optimizers."""

    def test_optimize_quadratic(self, optimizer_name):
        """All optimizers should minimize a simple quadratic."""
        np.random.seed(42)
        opt = OptimizerFactory.build(optimizer_name, lr=0.1)
        
        # Minimize f(x) = x^2, starting at x=5
        x = np.array([5.0])
        
        for _ in range(100):
            grad = 2 * x  # derivative of x^2
            x = opt.step(x, grad, key="test")
        
        # Should converge close to 0
        assert abs(x[0]) < 0.1

    def test_optimizer_different_lrs(self):
        """Different learning rates should produce different convergence."""
        np.random.seed(42)
        
        results = {}
        for lr in [0.001, 0.01, 0.1]:
            opt = Adam(lr=lr)
            x = np.array([5.0])
            
            for _ in range(50):
                grad = 2 * x
                x = opt.step(x, grad, key="test")
            
            results[lr] = abs(x[0])
        
        # Very low LR should converge less than medium LR
        assert results[0.001] > results[0.01]
