"""
tests/conftest.py
Pytest fixtures and configuration for NNStudio tests.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.activations import ACTIVATIONS, Activation
from app.core.losses import LOSSES, LossFunction
from app.core.optimizers import OptimizerFactory, SGD, Adam, AdamW, RMSProp, SGDMomentum
from app.core.layers.dense import DenseLayer
from app.core.network import NeuralNetwork, NetworkBuilder


@pytest.fixture
def sample_input():
    """Simple 2D input for testing."""
    return np.array([0.5, 0.8], dtype=np.float64)


@pytest.fixture
def sample_target():
    """Simple target for testing."""
    return np.array([1.0], dtype=np.float64)


@pytest.fixture
def xor_dataset():
    """XOR dataset for integration tests."""
    return [
        {"x": [0, 0], "y": [0]},
        {"x": [0, 1], "y": [1]},
        {"x": [1, 0], "y": [1]},
        {"x": [1, 1], "y": [0]},
    ]


@pytest.fixture
def simple_layer():
    """A simple dense layer for testing."""
    np.random.seed(42)
    return DenseLayer(
        n_in=2,
        n_out=4,
        activation=ACTIVATIONS["tanh"],
        dropout=0.0,
        is_output=False
    )


@pytest.fixture
def output_layer():
    """An output layer with sigmoid activation."""
    np.random.seed(42)
    return DenseLayer(
        n_in=4,
        n_out=1,
        activation=ACTIVATIONS["sigmoid"],
        dropout=0.0,
        is_output=True
    )


@pytest.fixture
def simple_network():
    """A simple 2-4-1 network for testing."""
    np.random.seed(42)
    layers = [
        DenseLayer(2, 4, ACTIVATIONS["tanh"]),
        DenseLayer(4, 1, ACTIVATIONS["sigmoid"], is_output=True),
    ]
    optimizer = Adam(lr=0.01)
    from app.core.losses import LOSSES
    return NeuralNetwork(layers, optimizer, LOSSES["bce"])


@pytest.fixture
def app_config():
    """Basic network configuration for building networks."""
    return {
        "inputs": 2,
        "outputs": 1,
        "layers": [
            {"neurons": 4, "activation": "tanh", "type": "dense"},
            {"neurons": 4, "activation": "tanh", "type": "dense"},
        ],
        "activation": "tanh",
        "optimizer": "adam",
        "loss": "bce",
        "lr": 0.01,
        "dropout": 0.0,
        "weight_decay": 0.0,
    }


@pytest.fixture
def flask_app():
    """Create a Flask app for API testing."""
    from app import create_app
    from app.core.session_manager import TrainingSessionManager
    
    app = create_app({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "WTF_CSRF_ENABLED": False,
        "SECRET_KEY": "test-secret-key",
    })
    
    # Set up session manager
    app.session_manager = TrainingSessionManager()
    
    return app


@pytest.fixture
def client(flask_app):
    """Test client for Flask app."""
    return flask_app.test_client()


# ═══════════════════════════════════════════════════════════════════════
# Parametrized fixtures
# ═══════════════════════════════════════════════════════════════════════
@pytest.fixture(params=list(ACTIVATIONS.keys()))
def activation_name(request):
    """Parametrized fixture for all activation functions."""
    return request.param


@pytest.fixture(params=list(LOSSES.keys()))
def loss_name(request):
    """Parametrized fixture for all loss functions."""
    return request.param


@pytest.fixture(params=["sgd", "momentum", "rmsprop", "adam", "adamw"])
def optimizer_name(request):
    """Parametrized fixture for all optimizers."""
    return request.param
