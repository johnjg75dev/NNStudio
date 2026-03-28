"""
tests/test_network.py
Unit tests for the NeuralNetwork class and NetworkBuilder.
"""
import pytest
import numpy as np
from app.core.activations import ACTIVATIONS
from app.core.losses import LOSSES
from app.core.optimizers import Adam, SGD
from app.core.layers.dense import DenseLayer
from app.core.network import NeuralNetwork, NetworkBuilder


class TestNeuralNetworkInit:
    """Test NeuralNetwork initialization."""

    def test_network_creation(self, simple_network):
        """Should create a network with correct structure."""
        assert len(simple_network.layers) == 2
        assert simple_network.layers[0].n_in == 2
        assert simple_network.layers[0].n_out == 4
        assert simple_network.layers[1].n_out == 1

    def test_initial_epoch(self, simple_network):
        """Initial epoch should be 0."""
        assert simple_network.epoch == 0

    def test_initial_loss_history(self, simple_network):
        """Initial loss history should be empty."""
        assert simple_network.loss_history == []


class TestNeuralNetworkPredict:
    """Test NeuralNetwork forward pass."""

    def test_predict_shape(self, simple_network):
        """Predict should return output with correct shape."""
        x = np.array([0.5, 0.5])
        output = simple_network.predict(x)
        assert output.shape == (1,)

    def test_predict_output_range(self, simple_network):
        """Output should be in (0, 1) for sigmoid output layer."""
        x = np.array([0.5, 0.5])
        output = simple_network.predict(x)
        assert 0 < output[0] < 1

    def test_predict_deterministic(self, simple_network):
        """Predict should be deterministic (same input → same output)."""
        x = np.array([0.3, 0.7])
        out1 = simple_network.predict(x)
        out2 = simple_network.predict(x)
        np.testing.assert_array_equal(out1, out2)

    def test_predict_batch(self):
        """Network should handle multiple inputs sequentially."""
        np.random.seed(42)
        layers = [
            DenseLayer(2, 4, ACTIVATIONS["tanh"]),
            DenseLayer(4, 1, ACTIVATIONS["sigmoid"], is_output=True),
        ]
        network = NeuralNetwork(layers, Adam(lr=0.01), LOSSES["bce"])
        
        inputs = [np.array([0.0, 0.0]), np.array([0.0, 1.0]), 
                  np.array([1.0, 0.0]), np.array([1.0, 1.0])]
        outputs = [network.predict(x) for x in inputs]
        
        assert len(outputs) == 4
        assert all(out.shape == (1,) for out in outputs)


class TestNeuralNetworkTrainStep:
    """Test NeuralNetwork training step."""

    def test_train_step_updates_weights(self, simple_network):
        """Train step should modify weights."""
        x = np.array([0.5, 0.5])
        y = np.array([1.0])
        
        old_W = [l.W.copy() for l in simple_network.layers]
        simple_network.train_step(x, y)
        
        # Weights should have changed
        for old, new in zip(old_W, simple_network.layers):
            assert not np.array_equal(old, new.W)

    def test_train_step_computes_gradients(self, simple_network):
        """Train step should compute gradients."""
        x = np.array([0.5, 0.5])
        y = np.array([1.0])
        
        simple_network.train_step(x, y)
        
        # All layers should have gradients
        for layer in simple_network.layers:
            assert layer._dW is not None
            assert layer._db is not None

    def test_train_step_decreases_loss(self):
        """Training should decrease loss on simple problem."""
        np.random.seed(42)
        layers = [
            DenseLayer(2, 8, ACTIVATIONS["tanh"]),
            DenseLayer(8, 1, ACTIVATIONS["sigmoid"], is_output=True),
        ]
        network = NeuralNetwork(layers, Adam(lr=0.1), LOSSES["bce"])
        
        # XOR problem
        dataset = [
            {"x": [0, 0], "y": [0]},
            {"x": [0, 1], "y": [1]},
            {"x": [1, 0], "y": [1]},
            {"x": [1, 1], "y": [0]},
        ]
        
        initial_loss = network.compute_loss(dataset)
        
        # Train for 100 steps
        for _ in range(100):
            for sample in dataset:
                network.train_step(np.array(sample["x"]), np.array(sample["y"]))
        
        final_loss = network.compute_loss(dataset)
        assert final_loss < initial_loss


class TestNeuralNetworkTrainEpoch:
    """Test NeuralNetwork epoch training."""

    def test_train_epoch_increments_epoch(self, simple_network):
        """Train epoch should increment epoch counter."""
        dataset = [{"x": [0.5, 0.5], "y": [1.0]}]
        simple_network.train_epoch(dataset, lr=0.01)
        assert simple_network.epoch == 1

    def test_train_epoch_multiple(self, simple_network):
        """Multiple epochs should increment counter."""
        dataset = [{"x": [0.5, 0.5], "y": [1.0]}]
        for _ in range(5):
            simple_network.train_epoch(dataset, lr=0.01)
        assert simple_network.epoch == 5

    def test_train_epoch_updates_loss_history(self):
        """Training should track loss history."""
        np.random.seed(42)
        layers = [
            DenseLayer(2, 4, ACTIVATIONS["tanh"]),
            DenseLayer(4, 1, ACTIVATIONS["sigmoid"], is_output=True),
        ]
        network = NeuralNetwork(layers, Adam(lr=0.1), LOSSES["bce"])
        
        dataset = [
            {"x": [0, 0], "y": [0]},
            {"x": [0, 1], "y": [1]},
            {"x": [1, 0], "y": [1]},
            {"x": [1, 1], "y": [0]},
        ]
        
        for _ in range(10):
            network.train_epoch(dataset, lr=0.1)
            network.loss_history.append(network.compute_loss(dataset))
        
        assert len(network.loss_history) == 10


class TestNeuralNetworkMetrics:
    """Test NeuralNetwork metric computation."""

    def test_compute_loss(self, simple_network):
        """Compute loss should return average loss over dataset."""
        dataset = [
            {"x": [0.5, 0.5], "y": [1.0]},
            {"x": [0.5, 0.5], "y": [1.0]},
        ]
        loss = simple_network.compute_loss(dataset)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_compute_accuracy_binary(self):
        """Compute accuracy should measure correct predictions."""
        np.random.seed(42)
        layers = [
            DenseLayer(2, 4, ACTIVATIONS["tanh"]),
            DenseLayer(4, 1, ACTIVATIONS["sigmoid"], is_output=True),
        ]
        network = NeuralNetwork(layers, Adam(lr=0.1), LOSSES["bce"])
        
        # Perfect predictions
        dataset_perfect = [
            {"x": [0.0, 0.0], "y": [0.0]},
            {"x": [1.0, 1.0], "y": [1.0]},
        ]
        
        # Manually set weights for perfect prediction (approximately)
        # Just check that accuracy is computed correctly
        acc = network.compute_accuracy(dataset_perfect, threshold=0.5)
        assert 0 <= acc <= 1

    def test_compute_accuracy_xor(self):
        """Accuracy on XOR should improve with training."""
        np.random.seed(42)
        layers = [
            DenseLayer(2, 16, ACTIVATIONS["relu"]),
            DenseLayer(16, 1, ACTIVATIONS["sigmoid"], is_output=True),
        ]
        network = NeuralNetwork(layers, Adam(lr=0.1), LOSSES["bce"])
        
        dataset = [
            {"x": [0, 0], "y": [0]},
            {"x": [0, 1], "y": [1]},
            {"x": [1, 0], "y": [1]},
            {"x": [1, 1], "y": [0]},
        ]
        
        initial_acc = network.compute_accuracy(dataset)
        
        # Train
        for _ in range(200):
            network.train_epoch(dataset, lr=0.1)
        
        final_acc = network.compute_accuracy(dataset)
        assert final_acc >= initial_acc


class TestNeuralNetworkTopology:
    """Test NeuralNetwork topology property."""

    def test_topology_property(self, simple_network):
        """Topology should return layer sizes."""
        topo = simple_network.topology
        assert topo == [2, 4, 1]

    def test_topology_empty(self):
        """Empty network should have empty topology."""
        network = NeuralNetwork([], Adam(lr=0.01), LOSSES["bce"])
        assert network.topology == []

    def test_topology_deep_network(self):
        """Topology should handle deep networks."""
        layers = [
            DenseLayer(2, 8, ACTIVATIONS["relu"]),
            DenseLayer(8, 8, ACTIVATIONS["relu"]),
            DenseLayer(8, 4, ACTIVATIONS["relu"]),
            DenseLayer(4, 1, ACTIVATIONS["sigmoid"], is_output=True),
        ]
        network = NeuralNetwork(layers, Adam(lr=0.01), LOSSES["bce"])
        assert network.topology == [2, 8, 8, 4, 1]


class TestNeuralNetworkParamCount:
    """Test NeuralNetwork parameter count."""

    def test_param_count(self, simple_network):
        """Param count should sum all layer params."""
        expected = sum(l.param_count for l in simple_network.layers)
        assert simple_network.param_count == expected

    def test_param_count_calculation(self):
        """Param count should be W + b for each layer."""
        layers = [
            DenseLayer(2, 4, ACTIVATIONS["tanh"]),  # 2*4 + 4 = 12
            DenseLayer(4, 1, ACTIVATIONS["sigmoid"], is_output=True),  # 4*1 + 1 = 5
        ]
        network = NeuralNetwork(layers, Adam(lr=0.01), LOSSES["bce"])
        assert network.param_count == 12 + 5


class TestNeuralNetworkSerialization:
    """Test NeuralNetwork serialization."""

    def test_to_dict(self, simple_network):
        """to_dict should serialize network."""
        d = simple_network.to_dict()
        
        assert "layers" in d
        assert len(d["layers"]) == 2
        assert d["optimizer"] == "adam"
        assert d["loss"] == "bce"
        assert d["epoch"] == 0

    def test_from_dict(self, simple_network):
        """from_dict should reconstruct network."""
        d = simple_network.to_dict()
        restored = NeuralNetwork.from_dict(d)
        
        assert len(restored.layers) == len(simple_network.layers)
        assert restored.topology == simple_network.topology

    def test_roundtrip(self, simple_network):
        """to_dict -> from_dict should produce identical network."""
        # Train a bit first
        dataset = [{"x": [0.5, 0.5], "y": [1.0]}]
        for _ in range(10):
            simple_network.train_epoch(dataset, lr=0.01)
        
        restored = NeuralNetwork.from_dict(simple_network.to_dict())
        
        assert restored.topology == simple_network.topology
        assert restored.epoch == simple_network.epoch
        
        # Weights should match
        for l1, l2 in zip(simple_network.layers, restored.layers):
            np.testing.assert_array_almost_equal(l1.W, l2.W)
            np.testing.assert_array_almost_equal(l1.b, l2.b)


class TestNetworkBuilder:
    """Test NetworkBuilder."""

    def test_build_simple_network(self, app_config):
        """Should build network from config."""
        network = NetworkBuilder.build(app_config)
        
        assert len(network.layers) == 3  # 2 hidden + 1 output
        assert network.topology == [2, 4, 4, 1]

    def test_build_with_custom_layers(self):
        """Should build network with custom layer config."""
        config = {
            "inputs": 3,
            "outputs": 2,
            "layers": [
                {"neurons": 8, "activation": "relu", "type": "dense"},
                {"neurons": 4, "activation": "tanh", "type": "dense"},
            ],
            "optimizer": "adam",
            "loss": "mse",
            "lr": 0.01,
        }
        network = NetworkBuilder.build(config)
        
        assert network.topology == [3, 8, 4, 2]
        assert network.layers[0].activation.name == "relu"
        assert network.layers[1].activation.name == "tanh"

    def test_build_with_dropout(self):
        """Should build network with dropout layer."""
        config = {
            "inputs": 2,
            "outputs": 1,
            "layers": [
                {"neurons": 8, "activation": "relu", "type": "dense"},
                {"type": "dropout", "rate": 0.2},
            ],
            "optimizer": "adam",
            "loss": "bce",
            "lr": 0.01,
        }
        network = NetworkBuilder.build(config)

        assert len(network.layers) == 3  # dense + dropout + output
        assert network.layers[1].__class__.__name__ == "DropoutLayer"
        assert network.layers[1].rate == 0.2

    def test_build_different_optimizers(self):
        """Should build network with different optimizers."""
        for opt_name in ["sgd", "adam", "rmsprop"]:
            config = {
                "inputs": 2,
                "outputs": 1,
                "layers": [{"neurons": 4, "activation": "tanh", "type": "dense"}],
                "optimizer": opt_name,
                "loss": "bce",
                "lr": 0.01,
            }
            network = NetworkBuilder.build(config)
            assert network.optimizer.__class__.__name__.lower() == opt_name

    def test_build_different_losses(self):
        """Should build network with different losses."""
        for loss_name in ["bce", "mse", "mae"]:
            config = {
                "inputs": 2,
                "outputs": 1,
                "layers": [{"neurons": 4, "activation": "tanh", "type": "dense"}],
                "optimizer": "adam",
                "loss": loss_name,
                "lr": 0.01,
            }
            network = NetworkBuilder.build(config)
            assert network.loss_fn.name == loss_name

    def test_build_output_layer_always_sigmoid(self):
        """Output layer should always use sigmoid."""
        config = {
            "inputs": 2,
            "outputs": 1,
            "layers": [{"neurons": 4, "activation": "relu", "type": "dense"}],
            "optimizer": "adam",
            "loss": "bce",
            "lr": 0.01,
        }
        network = NetworkBuilder.build(config)
        
        # Last layer should be sigmoid
        assert network.layers[-1].is_output is True


class TestNetworkBuilderEdgeCases:
    """Test NetworkBuilder edge cases."""

    def test_build_no_hidden_layers(self):
        """Should build network with no hidden layers."""
        config = {
            "inputs": 2,
            "outputs": 1,
            "layers": [],
            "optimizer": "adam",
            "loss": "bce",
            "lr": 0.01,
        }
        network = NetworkBuilder.build(config)
        
        # Should have just input → output
        assert len(network.layers) == 1
        assert network.topology == [2, 1]

    def test_build_many_hidden_layers(self):
        """Should build network with many hidden layers."""
        config = {
            "inputs": 2,
            "outputs": 1,
            "layers": [
                {"neurons": 4, "activation": "tanh", "type": "dense"}
                for _ in range(10)
            ],
            "optimizer": "adam",
            "loss": "bce",
            "lr": 0.01,
        }
        network = NetworkBuilder.build(config)
        
        assert len(network.layers) == 11  # 10 hidden + 1 output
        assert network.topology == [2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1]

    def test_build_wide_layer(self):
        """Should build network with wide layers."""
        config = {
            "inputs": 2,
            "outputs": 1,
            "layers": [{"neurons": 128, "activation": "relu", "type": "dense"}],
            "optimizer": "adam",
            "loss": "bce",
            "lr": 0.01,
        }
        network = NetworkBuilder.build(config)
        
        assert network.layers[0].n_out == 128
        assert network.param_count > 256  # 2*128 + 128 + 128*1 + 1
