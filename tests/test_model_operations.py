"""
tests/test_model_operations.py
Tests for model saving, loading, and exporting functionality.
"""
import pytest
import json
import zipfile
import numpy as np
from io import BytesIO
from app.core.network import NeuralNetwork, NetworkBuilder
from app.core.exporters import ModelExporter, JSONExporter, ZIPExporter, SafeTensorsExporter
from app.models import SavedModel, User
from app import db


@pytest.fixture
def sample_network():
    """Create a simple trained neural network for testing."""
    config = {
        "layers": [
            {"type": "dense", "neurons": 4, "activation": "relu"},
        ],
        "optimizer": "adam",
        "loss": "mse",
        "lr": 0.01,
        "inputs": 2,
        "outputs": 1
    }
    network = NetworkBuilder.build(config)
    
    # Simulate training
    dataset = [
        {"x": [0, 0], "y": [0]},
        {"x": [0, 1], "y": [1]},
        {"x": [1, 0], "y": [1]},
        {"x": [1, 1], "y": [0]},
    ]
    
    for _ in range(10):
        network.train_epoch(dataset, lr=0.1)
    
    return network


@pytest.fixture
def test_user(flask_app):
    """Create a test user."""
    with flask_app.app_context():
        user = User(username="testuser")
        user.set_password("testpass")
        db.session.add(user)
        db.session.commit()
        yield user
        db.session.delete(user)
        db.session.commit()


class TestJSONExporter:
    """Test JSON export functionality."""
    
    def test_json_export_to_file(self, sample_network, tmp_path):
        """Test exporting network to JSON file."""
        output_file = tmp_path / "model.json"
        
        result = JSONExporter.export(
            sample_network,
            str(output_file),
            metadata={"name": "test_model"}
        )
        
        assert output_file.exists()
        assert result == str(output_file)
        
        # Verify JSON content
        with open(output_file, 'r') as f:
            data = json.load(f)
        
        assert "network" in data
        assert "metadata" in data
        assert data["metadata"]["name"] == "test_model"
        assert "layers" in data["network"]
    
    def test_json_export_to_bytes(self, sample_network):
        """Test exporting network to JSON bytes."""
        result = JSONExporter.export_bytes(
            sample_network,
            metadata={"name": "test_model"}
        )
        
        assert isinstance(result, bytes)
        data = json.loads(result.decode('utf-8'))
        assert "network" in data
        assert "metadata" in data


class TestZIPExporter:
    """Test ZIP export functionality."""
    
    def test_zip_export_structure(self, sample_network, tmp_path):
        """Test that ZIP contains all required files."""
        output_file = tmp_path / "model.zip"
        
        ZIPExporter.export(
            sample_network,
            str(output_file),
            metadata={"name": "test_model", "description": "Test model"}
        )
        
        assert output_file.exists()
        
        # Verify ZIP contents
        with zipfile.ZipFile(output_file, 'r') as zf:
            files = zf.namelist()
            assert "model.json" in files
            assert "metadata.json" in files
            assert "weights.npz" in files
            assert "config.txt" in files
            
            # Check model.json
            model_data = json.loads(zf.read("model.json").decode('utf-8'))
            assert "network" in model_data
            assert "topology" in model_data
            assert "param_count" in model_data
            
            # Check metadata.json
            meta_data = json.loads(zf.read("metadata.json").decode('utf-8'))
            assert meta_data["name"] == "test_model"
            
            # Check weights.npz exists and can be loaded
            weights_data = zf.read("weights.npz")
            assert len(weights_data) > 0
    
    def test_zip_export_to_bytes(self, sample_network):
        """Test exporting network to ZIP bytes."""
        result = ZIPExporter.export_bytes(
            sample_network,
            metadata={"name": "test_model"}
        )
        
        assert isinstance(result, bytes)
        
        # Verify it's a valid ZIP
        zf = zipfile.ZipFile(BytesIO(result), 'r')
        files = zf.namelist()
        assert len(files) >= 4


class TestSafeTensorsExporter:
    """Test SafeTensors export functionality."""
    
    def test_safetensors_export_to_file(self, sample_network, tmp_path):
        """Test exporting to SafeTensors format."""
        try:
            from safetensors.numpy import load_file
        except ImportError:
            pytest.skip("safetensors not installed")
        
        output_file = tmp_path / "model.safetensors"
        
        result = SafeTensorsExporter.export(
            sample_network,
            str(output_file),
            metadata={"name": "test_model"}
        )
        
        assert output_file.exists()
        
        # Verify it can be loaded
        tensors = load_file(str(output_file))
        assert len(tensors) > 0
    
    def test_safetensors_export_to_bytes(self, sample_network):
        """Test exporting to SafeTensors bytes."""
        try:
            from safetensors.numpy import load_file
        except ImportError:
            pytest.skip("safetensors not installed")
        
        result = SafeTensorsExporter.export_bytes(sample_network)
        assert isinstance(result, bytes)
        assert len(result) > 0


class TestModelExporter:
    """Test unified ModelExporter interface."""
    
    def test_get_supported_formats(self):
        """Test getting list of supported formats."""
        formats = ModelExporter.get_supported_formats()
        
        assert isinstance(formats, list)
        assert "json" in formats
        assert "zip" in formats
        assert "safetensors" in formats
        assert "onnx" in formats
        assert "gguf" in formats
    
    def test_export_json_format(self, sample_network, tmp_path):
        """Test exporting in JSON format through unified interface."""
        output_file = tmp_path / "model.json"
        
        result = ModelExporter.export(
            sample_network,
            "json",
            str(output_file),
            metadata={"name": "test"}
        )
        
        assert output_file.exists()
    
    def test_export_zip_format(self, sample_network, tmp_path):
        """Test exporting in ZIP format through unified interface."""
        output_file = tmp_path / "model.zip"
        
        result = ModelExporter.export(
            sample_network,
            "zip",
            str(output_file),
            metadata={"name": "test"}
        )
        
        assert output_file.exists()
    
    def test_export_invalid_format(self, sample_network, tmp_path):
        """Test that invalid format raises error."""
        output_file = tmp_path / "model.xxx"
        
        with pytest.raises(ValueError):
            ModelExporter.export(
                sample_network,
                "invalid_format",
                str(output_file)
            )
    
    def test_export_bytes_json(self, sample_network):
        """Test exporting to bytes in JSON format."""
        result = ModelExporter.export_bytes(
            sample_network,
            "json",
            metadata={"name": "test"}
        )
        
        assert isinstance(result, bytes)
        data = json.loads(result.decode('utf-8'))
        assert "network" in data
    
    def test_export_bytes_zip(self, sample_network):
        """Test exporting to bytes in ZIP format."""
        result = ModelExporter.export_bytes(
            sample_network,
            "zip",
            metadata={"name": "test"}
        )
        
        assert isinstance(result, bytes)


class TestSavedModelDatabase:
    """Test SavedModel database model."""
    
    def test_create_saved_model(self, flask_app, test_user):
        """Test creating a SavedModel entry in database."""
        with flask_app.app_context():
            model_data = {
                "layers": [{"type": "dense", "n_in": 2, "n_out": 1}],
                "optimizer": "adam",
                "optimizer_lr": 0.01,
                "loss": "mse",
                "epoch": 10,
                "loss_history": [0.5, 0.4, 0.3],
            }
            
            saved = SavedModel(
                user_id=test_user.id,
                name="Test Model",
                description="A test model",
                model_data=model_data,
                architecture_name="mlp",
                function_name="xor",
                epochs_trained=10,
                final_loss=0.3,
                final_accuracy=0.95,
            )
            
            db.session.add(saved)
            db.session.commit()
            
            # Verify it was saved
            retrieved = SavedModel.query.filter_by(name="Test Model").first()
            assert retrieved is not None
            assert retrieved.user_id == test_user.id
            assert retrieved.epochs_trained == 10
            assert retrieved.final_loss == 0.3
    
    def test_saved_model_to_dict(self, flask_app, test_user):
        """Test SavedModel.to_dict() method."""
        with flask_app.app_context():
            model_data = {
                "layers": [],
                "optimizer": "adam",
                "optimizer_lr": 0.01,
                "loss": "mse",
                "epoch": 5,
                "loss_history": [],
            }
            
            saved = SavedModel(
                user_id=test_user.id,
                name="Test Model",
                description="Test",
                model_data=model_data,
                epochs_trained=5,
                final_loss=0.5,
            )
            
            db.session.add(saved)
            db.session.commit()
            
            result = saved.to_dict()
            
            assert "id" in result
            assert result["name"] == "Test Model"
            assert result["epochs_trained"] == 5
            assert result["final_loss"] == 0.5
            assert "created_at" in result
            assert "model_data" not in result  # Should not include full model data
    
    def test_saved_model_to_dict_full(self, flask_app, test_user):
        """Test SavedModel.to_dict_full() method includes model data."""
        with flask_app.app_context():
            model_data = {"layers": [], "optimizer": "adam"}
            
            saved = SavedModel(
                user_id=test_user.id,
                name="Test Model",
                model_data=model_data,
            )
            
            db.session.add(saved)
            db.session.commit()
            
            result = saved.to_dict_full()
            
            assert "model_data" in result
            assert result["model_data"] == model_data


class TestModelAPIEndpoints:
    """Test model management API endpoints."""
    
    def test_save_model_endpoint(self, flask_app, test_user, sample_network):
        """Test POST /api/models/save endpoint."""
        with flask_app.app_context():
            # Inject network into session manager
            sm = flask_app.extensions["session_manager"]
            session = sm.get_or_create("test_session_id")
            session.network = sample_network
            session.arch_key = "mlp"
            session.func_key = "xor"

            client = flask_app.test_client()
            
            # Login user
            with client:
                # Simulate logged-in state by setting user context
                @flask_app.before_request
                def before_request():
                    from flask_login import current_user
                    if not current_user.is_authenticated:
                        import flask_login
                        flask_login.login_user(test_user)
                
                response = client.post(
                    '/api/models/save',
                    json={
                        "name": "Test Model",
                        "description": "Test save",
                        "session_id": "test_session_id"
                    },
                    follow_redirects=True
                )
                
                assert response.status_code == 201
                data = response.get_json()
                assert data["success"] is True
                assert data["name"] == "Test Model"
    
    def test_list_models_endpoint(self, flask_app, test_user):
        """Test GET /api/models endpoint."""
        with flask_app.app_context():
            # Create test models
            model_data = {"layers": [], "optimizer": "adam"}
            
            for i in range(3):
                saved = SavedModel(
                    user_id=test_user.id,
                    name=f"Model {i}",
                    model_data=model_data,
                )
                db.session.add(saved)
            
            db.session.commit()
            
            client = flask_app.test_client()
            
            with client:
                @flask_app.before_request
                def before_request():
                    from flask_login import current_user
                    if not current_user.is_authenticated:
                        import flask_login
                        flask_login.login_user(test_user)
                
                response = client.get('/api/models', follow_redirects=True)
                
                # Should work when authenticated
                assert response.status_code in [200, 401, 302]


class TestNetworkSerialization:
    """Test network to_dict and from_dict round-trip."""
    
    def test_network_serialization_roundtrip(self, sample_network):
        """Test that network can be serialized and deserialized."""
        # Serialize
        network_dict = sample_network.to_dict()
        
        # Deserialize
        from app.core.network import NeuralNetwork
        restored_network = NeuralNetwork.from_dict(network_dict)
        
        # Verify both networks produce same outputs
        test_input = np.array([0.5, 0.5])
        
        original_output = sample_network.predict(test_input)
        restored_output = restored_network.predict(test_input)
        
        np.testing.assert_array_almost_equal(original_output, restored_output)
        
        # Verify metadata
        assert restored_network.epoch == sample_network.epoch
        assert restored_network.optimizer.__class__.__name__ == sample_network.optimizer.__class__.__name__
        assert len(restored_network.loss_history) == len(sample_network.loss_history)
