"""
app/api/model_routes.py
API endpoints for saving, loading, and exporting trained models.

Routes:
  POST   /api/models/save              - Save current session model to database
  GET    /api/models                   - List all saved models for user
  GET    /api/models/<id>              - Get model details
  DELETE /api/models/<id>              - Delete a saved model
  POST   /api/models/<id>/export       - Export model to various formats
  POST   /api/models/<id>/load-session - Load model into training session
  GET    /api/models/<id>/download/<format> - Download exported model
"""
from flask import Blueprint, request, jsonify, send_file
from flask_login import login_required, current_user
from io import BytesIO
import json
import os

from .. import db
from ..models import SavedModel
from ..core.exporters import ModelExporter
from ..core.session_manager import SessionManager
from .helpers import get_session_manager

model_bp = Blueprint('models', __name__)


# ════════════════════════════════════════════════════════════════════════
# Save Model
# ════════════════════════════════════════════════════════════════════════
@model_bp.route('/save', methods=['POST'])
@login_required
def save_model():
    """
    Save the current training session's network to the database.
    
    Request JSON:
    {
        "name": "str (required) - Model name",
        "description": "str (optional) - Model description",
        "session_id": "str (optional) - Session ID (from cookie if not provided)"
    }
    
    Response:
    {
        "success": bool,
        "model_id": int,
        "name": str,
        "message": str
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        if not data or 'name' not in data:
            return jsonify({"success": False, "error": "Missing 'name' field"}), 400
        
        # Get session
        session_manager = get_session_manager()
        session_id = data.get('session_id', request.cookies.get('session_id'))
        
        if not session_id:
            return jsonify({"success": False, "error": "No active session"}), 400
        
        training_session = session_manager.get(session_id)
        if not training_session or training_session.network is None:
            return jsonify({"success": False, "error": "No trained network in session"}), 400
        
        # Create SavedModel entry
        network = training_session.network
        saved_model = SavedModel(
            user_id=current_user.id,
            name=data['name'],
            description=data.get('description'),
            model_data=network.to_dict(),
            architecture_name=training_session.arch_key,
            function_name=training_session.func_key,
            epochs_trained=network.epoch,
            final_loss=network.loss_history[-1] if network.loss_history else None,
            final_accuracy=None,  # Can be computed if needed
        )
        
        db.session.add(saved_model)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "model_id": saved_model.id,
            "name": saved_model.name,
            "message": f"Model '{saved_model.name}' saved successfully"
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════
# List Models
# ════════════════════════════════════════════════════════════════════════
@model_bp.route('', methods=['GET'])
@login_required
def list_models():
    """
    Get all saved models for the current user.
    
    Query Parameters:
    - limit: int (default 50) - Max results
    - offset: int (default 0) - Pagination offset
    
    Response:
    {
        "success": bool,
        "models": [
            {
                "id": int,
                "name": str,
                "description": str,
                "architecture_name": str,
                "function_name": str,
                "epochs_trained": int,
                "final_loss": float,
                "final_accuracy": float,
                "created_at": str (ISO),
                "updated_at": str (ISO)
            }
        ],
        "total": int
    }
    """
    try:
        limit = min(request.args.get('limit', 50, type=int), 100)
        offset = request.args.get('offset', 0, type=int)
        
        query = SavedModel.query.filter_by(user_id=current_user.id)
        total = query.count()
        
        models = query.order_by(SavedModel.created_at.desc())\
                     .limit(limit)\
                     .offset(offset)\
                     .all()
        
        return jsonify({
            "success": True,
            "models": [m.to_dict() for m in models],
            "total": total,
            "limit": limit,
            "offset": offset
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════
# Get Model Details
# ════════════════════════════════════════════════════════════════════════
@model_bp.route('/<int:model_id>', methods=['GET'])
@login_required
def get_model(model_id):
    """
    Get details of a specific saved model (with weights).
    
    Response:
    {
        "success": bool,
        "model": {
            ...to_dict_full() output...
        }
    }
    """
    try:
        model = SavedModel.query.filter_by(
            id=model_id,
            user_id=current_user.id
        ).first()
        
        if not model:
            return jsonify({"success": False, "error": "Model not found"}), 404
        
        return jsonify({
            "success": True,
            "model": model.to_dict_full()
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════
# Delete Model
# ════════════════════════════════════════════════════════════════════════
@model_bp.route('/<int:model_id>', methods=['DELETE'])
@login_required
def delete_model(model_id):
    """Delete a saved model."""
    try:
        model = SavedModel.query.filter_by(
            id=model_id,
            user_id=current_user.id
        ).first()
        
        if not model:
            return jsonify({"success": False, "error": "Model not found"}), 404
        
        db.session.delete(model)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": f"Model '{model.name}' deleted"
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════
# Export Model
# ════════════════════════════════════════════════════════════════════════
@model_bp.route('/<int:model_id>/export', methods=['POST'])
@login_required
def export_model(model_id):
    """
    Export a model to various formats.
    
    Request JSON:
    {
        "format": "str (required) - 'json', 'safetensors', 'gguf', 'onnx', or 'zip'"
    }
    
    Response:
    {
        "success": bool,
        "format": str,
        "size_bytes": int,
        "message": str,
        "download_url": str
    }
    """
    try:
        model = SavedModel.query.filter_by(
            id=model_id,
            user_id=current_user.id
        ).first()
        
        if not model:
            return jsonify({"success": False, "error": "Model not found"}), 404
        
        data = request.get_json()
        if not data or 'format' not in data:
            return jsonify({"success": False, "error": "Missing 'format' field"}), 400
        
        export_format = data['format'].lower()
        supported = ModelExporter.get_supported_formats()
        
        if export_format not in supported:
            return jsonify({
                "success": False,
                "error": f"Unsupported format '{export_format}'. Supported: {supported}"
            }), 400
        
        # Reconstruct network from stored data
        from ..core.network import NeuralNetwork
        network = NeuralNetwork.from_dict(model.model_data)
        
        # Export to bytes
        metadata = {
            "name": model.name,
            "description": model.description,
            "architecture": model.architecture_name,
            "function": model.function_name,
            "epochs": model.epochs_trained,
            "final_loss": model.final_loss,
        }
        
        export_bytes = ModelExporter.export_bytes(network, export_format, metadata)
        size = len(export_bytes)
        
        return jsonify({
            "success": True,
            "format": export_format,
            "size_bytes": size,
            "message": f"Model exported as {export_format} ({size} bytes)",
            "download_url": f"/api/models/{model_id}/download/{export_format}"
        }), 200
        
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except ImportError as e:
        return jsonify({
            "success": False,
            "error": f"Export format not available: {str(e)}"
        }), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════
# Download Exported Model
# ════════════════════════════════════════════════════════════════════════
@model_bp.route('/<int:model_id>/download/<format>', methods=['GET'])
@login_required
def download_model(model_id, format):
    """
    Download an exported model file.
    
    Path Parameters:
    - model_id: int
    - format: str - 'json', 'safetensors', 'gguf', 'onnx', or 'zip'
    """
    try:
        model = SavedModel.query.filter_by(
            id=model_id,
            user_id=current_user.id
        ).first()
        
        if not model:
            return jsonify({"success": False, "error": "Model not found"}), 404
        
        format = format.lower()
        supported = ModelExporter.get_supported_formats()
        
        if format not in supported:
            return jsonify({
                "success": False,
                "error": f"Unsupported format '{format}'"
            }), 400
        
        # Reconstruct and export
        from ..core.network import NeuralNetwork
        network = NeuralNetwork.from_dict(model.model_data)
        
        metadata = {
            "name": model.name,
            "description": model.description,
            "architecture": model.architecture_name,
            "function": model.function_name,
            "epochs": model.epochs_trained,
        }
        
        export_bytes = ModelExporter.export_bytes(network, format, metadata)
        
        # Determine MIME type
        mime_types = {
            'json': 'application/json',
            'safetensors': 'application/octet-stream',
            'gguf': 'application/octet-stream',
            'onnx': 'application/octet-stream',
            'zip': 'application/zip',
        }
        
        # Filename
        filenames = {
            'json': '.json',
            'safetensors': '.safetensors',
            'gguf': '.gguf',
            'onnx': '.onnx',
            'zip': '.zip',
        }
        
        filename = f"{model.name.replace(' ', '_')}{filenames[format]}"
        
        return send_file(
            BytesIO(export_bytes),
            mimetype=mime_types[format],
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════
# Load Model into Session
# ════════════════════════════════════════════════════════════════════════
@model_bp.route('/<int:model_id>/load-session', methods=['POST'])
@login_required
def load_model_session(model_id):
    """
    Load a saved model into the training session for further training.
    
    Request JSON:
    {
        "session_id": "str (optional) - Session ID from cookie if not provided"
    }
    
    Response:
    {
        "success": bool,
        "message": str,
        "model_loaded": {
            "name": str,
            "epochs": int,
            "loss": float,
            "topology": list[int]
        }
    }
    """
    try:
        model = SavedModel.query.filter_by(
            id=model_id,
            user_id=current_user.id
        ).first()
        
        if not model:
            return jsonify({"success": False, "error": "Model not found"}), 404
        
        # Get or create session
        session_manager = get_session_manager()
        session_id = request.get_json().get('session_id') if request.is_json else None
        
        if not session_id:
            session_id = request.cookies.get('session_id')
        
        if not session_id:
            return jsonify({"success": False, "error": "No session ID"}), 400
        
        training_session = session_manager.get_or_create(session_id)
        
        # Reconstruct network
        from ..core.network import NeuralNetwork
        network = NeuralNetwork.from_dict(model.model_data)
        
        # Load into session
        training_session.network = network
        training_session.arch_key = model.architecture_name or "unknown"
        training_session.func_key = model.function_name or "unknown"
        training_session.touch()
        
        return jsonify({
            "success": True,
            "message": f"Model '{model.name}' loaded into session",
            "model_loaded": {
                "name": model.name,
                "epochs": network.epoch,
                "loss": network.loss_history[-1] if network.loss_history else None,
                "topology": network.topology,
                "param_count": network.param_count,
            }
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════
# Get Supported Formats
# ════════════════════════════════════════════════════════════════════════
@model_bp.route('/formats', methods=['GET'])
@login_required
def get_supported_formats():
    """Get list of supported export formats."""
    return jsonify({
        "success": True,
        "formats": ModelExporter.get_supported_formats(),
        "descriptions": {
            "json": "Plain JSON format with full model serialization",
            "safetensors": "Hugging Face SafeTensors efficient format",
            "gguf": "GGML quantized format for fast inference",
            "onnx": "Open Neural Network Exchange format (cross-platform)",
            "zip": "Compressed ZIP archive with model, weights, and metadata"
        }
    }), 200
