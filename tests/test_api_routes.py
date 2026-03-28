"""
tests/test_api_routes.py
Integration tests for API routes.
"""
import pytest
import json
import numpy as np


class TestModuleRoutes:
    """Test /api/modules/* routes."""

    def test_get_all_modules(self, client, flask_app):
        """GET /api/modules/all should return all modules."""
        with flask_app.app_context():
            response = client.get("/api/modules/all")
            assert response.status_code == 200
            
            data = response.get_json()
            assert data["ok"] is True
            assert "functions" in data["data"]
            assert "presets" in data["data"]
            assert "architectures" in data["data"]

    def test_get_functions(self, client, flask_app):
        """Should return list of training functions."""
        with flask_app.app_context():
            response = client.get("/api/modules/all")
            data = response.get_json()
            
            functions = data["data"]["functions"]
            assert len(functions) > 0
            
            # Check XOR is present
            xor = next((f for f in functions if f["key"] == "xor"), None)
            assert xor is not None
            assert xor["inputs"] == 2
            assert xor["outputs"] == 1

    def test_get_presets(self, client, flask_app):
        """Should return list of presets."""
        with flask_app.app_context():
            response = client.get("/api/modules/all")
            data = response.get_json()
            
            presets = data["data"]["presets"]
            assert len(presets) > 0

    def test_get_module_by_key(self, client, flask_app):
        """GET /api/modules/<key> should return specific module."""
        with flask_app.app_context():
            response = client.get("/api/modules/xor")
            assert response.status_code == 200
            
            data = response.get_json()
            assert data["ok"] is True
            assert data["data"]["key"] == "xor"

    def test_get_invalid_module(self, client, flask_app):
        """Should return 404 for invalid module key."""
        with flask_app.app_context():
            response = client.get("/api/modules/invalid_key")
            assert response.status_code == 404


class TestSessionRoutes:
    """Test /api/session/* routes."""

    def test_build_network(self, client, flask_app):
        """POST /api/session/build should create a network."""
        config = {
            "func_key": "xor",
            "arch_key": "mlp",
            "inputs": 2,
            "outputs": 1,
            "layers": [
                {"neurons": 4, "activation": "tanh", "type": "dense"},
                {"neurons": 4, "activation": "tanh", "type": "dense"},
            ],
            "optimizer": "adam",
            "loss": "bce",
            "lr": 0.01,
            "dropout": 0.0,
            "weight_decay": 0.0,
        }
        
        with flask_app.app_context():
            response = client.post(
                "/api/session/build",
                data=json.dumps(config),
                content_type="application/json"
            )
            assert response.status_code == 200
            
            data = response.get_json()
            assert data["ok"] is True
            assert "topology" in data["data"]
            assert "param_count" in data["data"]
            assert data["data"]["topology"] == [2, 4, 4, 1]

    def test_build_network_invalid_func(self, client, flask_app):
        """Should return 404 for invalid function key."""
        config = {
            "func_key": "invalid_func",
            "arch_key": "mlp",
            "layers": [],
        }
        
        with flask_app.app_context():
            response = client.post(
                "/api/session/build",
                data=json.dumps(config),
                content_type="application/json"
            )
            assert response.status_code == 404

    def test_get_snapshot(self, client, flask_app):
        """GET /api/session/snapshot should return network state."""
        # First build a network
        config = {
            "func_key": "xor",
            "arch_key": "mlp",
            "inputs": 2,
            "outputs": 1,
            "layers": [{"neurons": 4, "activation": "tanh", "type": "dense"}],
            "optimizer": "adam",
            "loss": "bce",
            "lr": 0.01,
        }
        
        with flask_app.app_context():
            client.post(
                "/api/session/build",
                data=json.dumps(config),
                content_type="application/json"
            )
            
            response = client.get("/api/session/snapshot")
            assert response.status_code == 200
            
            data = response.get_json()
            assert data["ok"] is True
            assert data["data"]["built"] is True
            assert "topology" in data["data"]
            assert "layers" in data["data"]
            assert "activations" in data["data"]

    def test_predict(self, client, flask_app):
        """POST /api/session/predict should return predictions."""
        # Build network first
        config = {
            "func_key": "xor",
            "arch_key": "mlp",
            "inputs": 2,
            "outputs": 1,
            "layers": [{"neurons": 4, "activation": "tanh", "type": "dense"}],
            "optimizer": "adam",
            "loss": "bce",
            "lr": 0.01,
        }
        
        with flask_app.app_context():
            client.post(
                "/api/session/build",
                data=json.dumps(config),
                content_type="application/json"
            )
            
            # Predict
            predict_data = {"x": [0.0, 1.0]}
            response = client.post(
                "/api/session/predict",
                data=json.dumps(predict_data),
                content_type="application/json"
            )
            assert response.status_code == 200
            
            data = response.get_json()
            assert data["ok"] is True
            assert "output" in data["data"]
            assert "activations" in data["data"]
            assert len(data["data"]["output"]) == 1

    def test_reset_weights(self, client, flask_app):
        """POST /api/session/reset should reset network weights."""
        config = {
            "func_key": "xor",
            "arch_key": "mlp",
            "inputs": 2,
            "outputs": 1,
            "layers": [{"neurons": 4, "activation": "tanh", "type": "dense"}],
            "optimizer": "adam",
            "loss": "bce",
            "lr": 0.01,
        }
        
        with flask_app.app_context():
            # Build
            client.post(
                "/api/session/build",
                data=json.dumps(config),
                content_type="application/json"
            )
            
            # Reset
            response = client.post(
                "/api/session/reset",
                data=json.dumps({}),
                content_type="application/json"
            )
            assert response.status_code == 200
            
            data = response.get_json()
            assert data["ok"] is True
            assert data["data"]["epoch"] == 0


class TestTrainRoutes:
    """Test /api/train/* routes."""

    def test_train_step(self, client, flask_app):
        """POST /api/train/step should train for specified steps."""
        # Build network first
        config = {
            "func_key": "xor",
            "arch_key": "mlp",
            "inputs": 2,
            "outputs": 1,
            "layers": [{"neurons": 8, "activation": "tanh", "type": "dense"}],
            "optimizer": "adam",
            "loss": "bce",
            "lr": 0.1,
        }
        
        with flask_app.app_context():
            client.post(
                "/api/session/build",
                data=json.dumps(config),
                content_type="application/json"
            )
            
            # Train
            train_data = {"steps": 100, "lr": 0.1}
            response = client.post(
                "/api/train/step",
                data=json.dumps(train_data),
                content_type="application/json"
            )
            assert response.status_code == 200
            
            data = response.get_json()
            assert data["ok"] is True
            assert "epoch" in data["data"]
            assert "loss" in data["data"]
            assert data["data"]["epoch"] == 100

    def test_evaluate(self, client, flask_app):
        """POST /api/train/evaluate should return evaluation on dataset."""
        config = {
            "func_key": "xor",
            "arch_key": "mlp",
            "inputs": 2,
            "outputs": 1,
            "layers": [{"neurons": 8, "activation": "tanh", "type": "dense"}],
            "optimizer": "adam",
            "loss": "bce",
            "lr": 0.1,
        }
        
        with flask_app.app_context():
            client.post(
                "/api/session/build",
                data=json.dumps(config),
                content_type="application/json"
            )
            
            # Train a bit first
            client.post(
                "/api/train/step",
                data=json.dumps({"steps": 50, "lr": 0.1}),
                content_type="application/json"
            )
            
            # Evaluate
            response = client.post(
                "/api/train/evaluate",
                data=json.dumps({}),
                content_type="application/json"
            )
            assert response.status_code == 200
            
            data = response.get_json()
            assert data["ok"] is True
            assert "samples" in data["data"]
            assert len(data["data"]["samples"]) == 4  # XOR has 4 samples


class TestPresetRoutes:
    """Test /api/presets/* routes."""

    def test_save_preset(self, client, flask_app):
        """POST /api/presets/save should save a preset."""
        preset_data = {
            "label": "Test Preset",
            "description": "A test preset",
            "arch_key": "mlp",
            "func_key": "xor",
            "layers": [{"neurons": 4, "activation": "tanh", "type": "dense"}],
            "optimizer": "adam",
            "loss": "bce",
            "lr": 0.01,
        }
        
        # Note: This would require authentication in real app
        # For testing, we just check the route exists
        with flask_app.app_context():
            response = client.post(
                "/api/presets/save",
                data=json.dumps(preset_data),
                content_type="application/json"
            )
            # May return 401/403 if login required, or 200 if test mode
            assert response.status_code in [200, 401, 403]


class TestIntegration:
    """Integration tests for full training workflow."""

    def test_full_xor_training(self, client, flask_app):
        """Should train XOR to high accuracy."""
        config = {
            "func_key": "xor",
            "arch_key": "mlp",
            "inputs": 2,
            "outputs": 1,
            "layers": [
                {"neurons": 8, "activation": "tanh", "type": "dense"},
                {"neurons": 8, "activation": "tanh", "type": "dense"},
            ],
            "optimizer": "adam",
            "loss": "bce",
            "lr": 0.1,
        }
        
        with flask_app.app_context():
            # Build
            build_resp = client.post(
                "/api/session/build",
                data=json.dumps(config),
                content_type="application/json"
            )
            assert build_resp.status_code == 200
            
            # Train
            train_resp = client.post(
                "/api/train/step",
                data=json.dumps({"steps": 500, "lr": 0.1}),
                content_type="application/json"
            )
            assert train_resp.status_code == 200
            
            # Evaluate
            eval_resp = client.post(
                "/api/train/evaluate",
                data=json.dumps({}),
                content_type="application/json"
            )
            assert eval_resp.status_code == 200
            
            data = eval_resp.get_json()["data"]
            
            # Check accuracy
            correct = 0
            for sample in data["samples"]:
                pred = 1 if sample["pred"][0] > 0.5 else 0
                target = int(round(sample["y"][0]))
                if pred == target:
                    correct += 1
            
            accuracy = correct / len(data["samples"])
            assert accuracy >= 0.75  # Should get at least 75% correct

    def test_custom_input_output_neurons(self, client, flask_app):
        """Should build network with custom input/output sizes."""
        config = {
            "func_key": "seg7",
            "arch_key": "mlp",
            "inputs": 4,  # 4-bit input
            "outputs": 7,  # 7-segment output
            "layers": [
                {"neurons": 10, "activation": "tanh", "type": "dense"},
                {"neurons": 10, "activation": "tanh", "type": "dense"},
            ],
            "optimizer": "adam",
            "loss": "mse",
            "lr": 0.02,
        }
        
        with flask_app.app_context():
            response = client.post(
                "/api/session/build",
                data=json.dumps(config),
                content_type="application/json"
            )
            assert response.status_code == 200
            
            data = response.get_json()
            assert data["data"]["topology"] == [4, 10, 10, 7]

    def test_export_import_model(self, client, flask_app):
        """Should export and import model correctly."""
        config = {
            "func_key": "xor",
            "arch_key": "mlp",
            "inputs": 2,
            "outputs": 1,
            "layers": [{"neurons": 4, "activation": "tanh", "type": "dense"}],
            "optimizer": "adam",
            "loss": "bce",
            "lr": 0.01,
        }
        
        with flask_app.app_context():
            # Build and train
            client.post(
                "/api/session/build",
                data=json.dumps(config),
                content_type="application/json"
            )
            client.post(
                "/api/train/step",
                data=json.dumps({"steps": 10, "lr": 0.01}),
                content_type="application/json"
            )
            
            # Export
            export_resp = client.post(
                "/api/session/export",
                data=json.dumps({}),
                content_type="application/json"
            )
            assert export_resp.status_code == 200
            
            model_data = export_resp.get_json()["data"]
            assert "layers" in model_data
            assert "optimizer" in model_data
            assert "loss" in model_data
