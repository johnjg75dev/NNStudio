"""
app/core/exporters.py
Export trained neural networks to various formats:
- JSON: Plain JSON serialization
- SafeTensors: Hugging Face's efficient format
- GGUF: GGML's quantized format
- ONNX: Cross-platform model interchange format
- ZIP: Packaged archive with metadata
"""
import json
import os
import zipfile
import tempfile
import numpy as np
from io import BytesIO
from pathlib import Path
from typing import Optional, BinaryIO


class BaseExporter:
    """Base class for all exporters."""
    
    @staticmethod
    def export(network, output_path: str, metadata: Optional[dict] = None):
        """Export network to specified format."""
        raise NotImplementedError


class JSONExporter(BaseExporter):
    """Export to JSON format."""
    
    @staticmethod
    def export(network, output_path: str, metadata: Optional[dict] = None) -> str:
        """
        Export network to JSON format.
        
        Args:
            network: NeuralNetwork instance
            output_path: Path to write JSON file
            metadata: Optional metadata dict to include
            
        Returns:
            Path to exported file
        """
        export_data = {
            "network": network.to_dict(),
            "metadata": metadata or {},
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return output_path

    @staticmethod
    def export_bytes(network, metadata: Optional[dict] = None) -> bytes:
        """Export to JSON bytes."""
        export_data = {
            "network": network.to_dict(),
            "metadata": metadata or {},
        }
        return json.dumps(export_data, indent=2).encode('utf-8')


class SafeTensorsExporter(BaseExporter):
    """Export to SafeTensors format (Hugging Face)."""
    
    @staticmethod
    def export(network, output_path: str, metadata: Optional[dict] = None) -> str:
        """
        Export network to SafeTensors format.
        
        Args:
            network: NeuralNetwork instance
            output_path: Path to write safetensors file
            metadata: Optional metadata dict
            
        Returns:
            Path to exported file
        """
        try:
            from safetensors.numpy import save_file
        except ImportError:
            raise ImportError("Please install safetensors: pip install safetensors")
        
        tensors = {}
        metadata_dict = metadata or {}
        
        # Serialize layers' weights as tensors
        for i, layer in enumerate(network.layers):
            layer_dict = layer.to_dict()
            layer_name = layer.__class__.__name__
            
            # Add weight and bias tensors
            if "W" in layer_dict and layer_dict["W"] is not None:
                tensors[f"layer_{i}_{layer_name}_W"] = np.array(layer_dict["W"])
            
            if "b" in layer_dict and layer_dict["b"] is not None:
                tensors[f"layer_{i}_{layer_name}_b"] = np.array(layer_dict["b"])
        
        # Add optimizer and loss info to metadata
        metadata_dict["optimizer"] = network.optimizer.__class__.__name__
        metadata_dict["optimizer_lr"] = str(network.optimizer.lr)
        metadata_dict["loss"] = network.loss_fn.name
        metadata_dict["epoch"] = str(network.epoch)
        metadata_dict["param_count"] = str(network.param_count)
        
        save_file(tensors, output_path, metadata=metadata_dict)
        return output_path

    @staticmethod
    def export_bytes(network, metadata: Optional[dict] = None) -> bytes:
        """Export to SafeTensors bytes."""
        try:
            from safetensors.numpy import save_file
        except ImportError:
            raise ImportError("Please install safetensors: pip install safetensors")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".safetensors") as tmp:
            tmp_path = tmp.name
        
        try:
            SafeTensorsExporter.export(network, tmp_path, metadata)
            with open(tmp_path, 'rb') as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class GGUFExporter(BaseExporter):
    """Export to GGUF format (GGML)."""
    
    @staticmethod
    def export(network, output_path: str, metadata: Optional[dict] = None) -> str:
        """
        Export network to GGUF format.
        Note: This is a simplified GGUF export. Full GGUF support requires ggml library.
        
        Args:
            network: NeuralNetwork instance
            output_path: Path to write GGUF file
            metadata: Optional metadata dict
            
        Returns:
            Path to exported file
        """
        try:
            from gguf import GGUFWriter
        except ImportError:
            raise ImportError(
                "Please install gguf: pip install gguf\n"
                "Note: GGUF export requires Python 3.10+ and proper ggml setup"
            )
        
        metadata_dict = metadata or {}
        
        # Create GGUF writer
        writer = GGUFWriter(output_path, "nn-studio")
        
        # Add model metadata
        writer.add_string("model.name", metadata_dict.get("name", "trained_model"))
        writer.add_string("model.architecture", str(network.topology))
        writer.add_uint32("model.layers", len(network.layers))
        writer.add_uint32("model.parameters", network.param_count)
        writer.add_float32("training.loss", float(network.loss_history[-1] if network.loss_history else 0.0))
        writer.add_uint32("training.epochs", network.epoch)
        writer.add_string("training.optimizer", network.optimizer.__class__.__name__)
        writer.add_float32("training.learning_rate", float(network.optimizer.lr))
        writer.add_string("training.loss_function", network.loss_fn.name)
        
        # Add layer weights as tensors
        for i, layer in enumerate(network.layers):
            layer_dict = layer.to_dict()
            layer_name = layer.__class__.__name__
            
            # Add weights
            if "W" in layer_dict and layer_dict["W"] is not None:
                W = np.array(layer_dict["W"], dtype=np.float32)
                writer.add_array(f"layer_{i}_{layer_name}_weights", W)
            
            # Add biases
            if "b" in layer_dict and layer_dict["b"] is not None:
                b = np.array(layer_dict["b"], dtype=np.float32)
                writer.add_array(f"layer_{i}_{layer_name}_bias", b)
        
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.close()
        
        return output_path

    @staticmethod
    def export_bytes(network, metadata: Optional[dict] = None) -> bytes:
        """Export to GGUF bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as tmp:
            tmp_path = tmp.name
        
        try:
            GGUFExporter.export(network, tmp_path, metadata)
            with open(tmp_path, 'rb') as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class ONNXExporter(BaseExporter):
    """Export to ONNX format (Open Neural Network Exchange)."""
    
    @staticmethod
    def export(network, output_path: str, metadata: Optional[dict] = None) -> str:
        """
        Export network to ONNX format.
        Note: Simplified ONNX export. Full support requires executing actual inference.
        
        Args:
            network: NeuralNetwork instance
            output_path: Path to write ONNX file
            metadata: Optional metadata dict
            
        Returns:
            Path to exported file
        """
        try:
            import onnx
            import onnx.helper as helper
        except ImportError:
            raise ImportError("Please install onnx: pip install onnx")
        
        metadata_dict = metadata or {}
        
        # Create input info (assuming 1D input for now)
        input_size = network.layers[0].n_in if hasattr(network.layers[0], 'n_in') else 2
        output_size = network.layers[-1].n_out if hasattr(network.layers[-1], 'n_out') else 1
        
        # Define input and output
        X = helper.make_tensor_value_info('X', onnx.TensorProto.FLOAT, [None, input_size])
        Y = helper.make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [None, output_size])
        
        # Create initializers for weights
        initializers = []
        node_inputs = ['X']
        
        for i, layer in enumerate(network.layers):
            layer_dict = layer.to_dict()
            layer_name = layer.__class__.__name__
            
            # Create weight tensor
            if "W" in layer_dict and layer_dict["W"] is not None:
                W = np.array(layer_dict["W"], dtype=np.float32).T  # Transpose for ONNX
                W_init = helper.make_tensor(
                    name=f"W_{i}",
                    data_type=onnx.TensorProto.FLOAT,
                    dims=W.shape,
                    vals=W.tobytes(),
                    raw=True
                )
                initializers.append(W_init)
            
            # Create bias tensor if exists
            if "b" in layer_dict and layer_dict["b"] is not None:
                b = np.array(layer_dict["b"], dtype=np.float32)
                b_init = helper.make_tensor(
                    name=f"b_{i}",
                    data_type=onnx.TensorProto.FLOAT,
                    dims=b.shape,
                    vals=b.tobytes(),
                    raw=True
                )
                initializers.append(b_init)
        
        # Create a simple graph (linear model for demo)
        nodes = [
            helper.make_node('Gemm', inputs=['X', 'W_0', 'b_0'], outputs=['Y'])
        ]
        
        graph_def = helper.make_graph(
            nodes,
            'NNStudio_Model',
            [X],
            [Y],
            initializers
        )
        
        # Create model
        model_def = helper.make_model(graph_def, producer_name='NNStudio')
        model_def.opset_import[0].version = 12
        
        # Save model
        onnx.save(model_def, output_path)
        return output_path

    @staticmethod
    def export_bytes(network, metadata: Optional[dict] = None) -> bytes:
        """Export to ONNX bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".onnx") as tmp:
            tmp_path = tmp.name
        
        try:
            ONNXExporter.export(network, tmp_path, metadata)
            with open(tmp_path, 'rb') as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class ZIPExporter(BaseExporter):
    """Export to ZIP archive with model and metadata."""
    
    @staticmethod
    def export(network, output_path: str, metadata: Optional[dict] = None) -> str:
        """
        Export network to ZIP archive containing:
        - model.json: Full network serialization
        - metadata.json: Training metadata
        - weights.npy: NumPy archive of all weights
        - config.txt: Human-readable config
        
        Args:
            network: NeuralNetwork instance
            output_path: Path to write ZIP file
            metadata: Optional metadata dict
            
        Returns:
            Path to exported file
        """
        metadata_dict = metadata or {}
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add model JSON
            model_data = {
                "network": network.to_dict(),
                "topology": network.topology,
                "param_count": network.param_count,
            }
            zf.writestr('model.json', json.dumps(model_data, indent=2))
            
            # Add metadata
            meta = {
                **metadata_dict,
                "epoch": network.epoch,
                "loss": network.loss_history[-1] if network.loss_history else None,
                "optimizer": network.optimizer.__class__.__name__,
                "learning_rate": network.optimizer.lr,
            }
            zf.writestr('metadata.json', json.dumps(meta, indent=2))
            
            # Add weights as NPZ (NumPy compressed)
            weights_dict = {}
            for i, layer in enumerate(network.layers):
                layer_dict = layer.to_dict()
                if "W" in layer_dict and layer_dict["W"] is not None:
                    weights_dict[f"layer_{i}_W"] = np.array(layer_dict["W"])
                if "b" in layer_dict and layer_dict["b"] is not None:
                    weights_dict[f"layer_{i}_b"] = np.array(layer_dict["b"])
            
            # Save to BytesIO first, then add to zip
            npz_buffer = BytesIO()
            np.savez_compressed(npz_buffer, **weights_dict)
            npz_buffer.seek(0)
            zf.writestr('weights.npz', npz_buffer.read())
            
            # Add human-readable config
            config_text = f"""NNStudio Model Export
=====================
Name: {metadata_dict.get('name', 'Unnamed Model')}
Description: {metadata_dict.get('description', 'No description')}

Architecture
============
Topology: {network.topology}
Total Parameters: {network.param_count}
Optimizer: {network.optimizer.__class__.__name__}
Learning Rate: {network.optimizer.lr}
Loss Function: {network.loss_fn.name}

Training Stats
==============
Epochs: {network.epoch}
Current Loss: {network.loss_history[-1] if network.loss_history else 'N/A'}
Loss History Length: {len(network.loss_history)}

Layers
======
"""
            for i, layer in enumerate(network.layers):
                layer_dict = layer.to_dict()
                config_text += f"\nLayer {i} ({layer.__class__.__name__}):\n"
                for k, v in layer_dict.items():
                    if k not in ['W', 'b', '_x', '_a', '_dW', '_db']:
                        config_text += f"  {k}: {v}\n"
            
            zf.writestr('config.txt', config_text)
        
        return output_path

    @staticmethod
    def export_bytes(network, metadata: Optional[dict] = None) -> bytes:
        """Export to ZIP bytes."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp_path = tmp.name
        
        try:
            ZIPExporter.export(network, tmp_path, metadata)
            with open(tmp_path, 'rb') as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class ModelExporter:
    """Unified exporter interface."""
    
    EXPORTERS = {
        'json': JSONExporter,
        'safetensors': SafeTensorsExporter,
        'gguf': GGUFExporter,
        'onnx': ONNXExporter,
        'zip': ZIPExporter,
    }
    
    @classmethod
    def export(cls, network, format: str, output_path: str, 
               metadata: Optional[dict] = None) -> str:
        """
        Export network to specified format.
        
        Args:
            network: NeuralNetwork instance
            format: Export format ('json', 'safetensors', 'gguf', 'onnx', 'zip')
            output_path: Output file path
            metadata: Optional metadata dict
            
        Returns:
            Path to exported file
            
        Raises:
            ValueError: If format is unsupported
        """
        if format not in cls.EXPORTERS:
            raise ValueError(f"Unsupported format: {format}. Available: {list(cls.EXPORTERS.keys())}")
        
        exporter = cls.EXPORTERS[format]
        return exporter.export(network, output_path, metadata)
    
    @classmethod
    def export_bytes(cls, network, format: str, 
                     metadata: Optional[dict] = None) -> bytes:
        """Export network to bytes (in-memory)."""
        if format not in cls.EXPORTERS:
            raise ValueError(f"Unsupported format: {format}. Available: {list(cls.EXPORTERS.keys())}")
        
        exporter = cls.EXPORTERS[format]
        return exporter.export_bytes(network, metadata)
    
    @classmethod
    def get_supported_formats(cls) -> list[str]:
        """Get list of supported export formats."""
        return list(cls.EXPORTERS.keys())


def load_from_dict(model_dict: dict):
    """Load a NeuralNetwork from a serialized dict."""
    from .network import NeuralNetwork
    return NeuralNetwork.from_dict(model_dict)
