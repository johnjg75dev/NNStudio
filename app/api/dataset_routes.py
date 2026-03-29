"""
app/api/dataset_routes.py
API endpoints for dataset management.
"""
import os
import json
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from .. import db
from ..models import Dataset

dataset_bp = Blueprint('datasets', __name__)

@dataset_bp.route('', methods=['GET'])
@login_required
def list_datasets():  
    """List datasets for the current user including predefined ones."""
    # User's custom datasets
    custom = Dataset.query.filter_by(user_id=current_user.id).all()
a    
    # Predefined datasets (global or specific user)
    # For now, let's just show user's ones and some placeholder predefined
    return jsonify({
        "success": True,
        "datasets": [d.to_dict() for d in custom]
    }), 200

@dataset_bp.route('', methods=['POST'])
@login_required
def create_dataset():
    """Create a new dataset."""
    try:
        data = request.get_json()
        required = ['name', 'ds_type', 'num_inputs']
        for field in required:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing field: {field}"}), 400
        
        new_ds = Dataset(
            user_id=current_user.id,
            name=data['name'],
            description=data.get('description'),
            ds_type=data['ds_type'],
            is_input_only=data.get('is_input_only', False),
            num_inputs=data['num_inputs'],
            num_outputs=data.get('num_outputs'),
            input_labels=data.get('input_labels', []),
            output_labels=data.get('output_labels', []),
            width=data.get('width'),
            height=data.get('height'),
            channels=data.get('channels', 1),
            data=data.get('data'),
            is_predefined=False
        )
        
        db.session.add(new_ds)
        db.session.commit()
        
        return jsonify({"success": True, "dataset": new_ds.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500

@dataset_bp.route('/<int:ds_id>', methods=['GET'])
@login_required
def get_dataset(ds_id):
    """Get full dataset details."""
    ds = Dataset.query.filter_by(id=ds_id, user_id=current_user.id).first()
    if not ds:
        return jsonify({"success": False, "error": "Dataset not found"}), 404
    return jsonify({"success": True, "dataset": ds.to_dict_full()}), 200

@dataset_bp.route('/<int:ds_id>', methods=['PUT'])
@login_required
def update_dataset(ds_id):
    """Update dataset."""
    try:
        ds = Dataset.query.filter_by(id=ds_id, user_id=current_user.id).first()
        if not ds:
            return jsonify({"success": False, "error": "Dataset not found"}), 404
              
        data = request.get_json()
        for field in ['name', 'description', 'data', 'is_input_only', 'num_inputs', 'num_outputs', 'width', 'height']:
            if field in data:
                setattr(ds, field, data[field])
              
        db.session.commit()
        return jsonify({"success": True, "dataset": ds.to_dict()}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500

@dataset_bp.route('/<int:ds_id>', methods=['DELETE'])
@login_required
def delete_dataset(ds_id):
    """Delete dataset."""
    ds = Dataset.query.filter_by(id=ds_id, user_id=current_user.id).first()
    if not ds:
        return jsonify({"success": False, "error": "Dataset not found"}), 404
    
    db.session.delete(ds)
    db.session.commit()
    return jsonify({"success": True, "message": "Dataset deleted"}), 200

@dataset_bp.route('/<int:ds_id>/download', methods=['POST'])
@login_required
def download_predefined(ds_id):
    """Trigger download for a predefined dataset."""
    # Mock implementation for now
    ds = Dataset.query.filter_by(id=ds_id, user_id=current_user.id, is_predefined=True).first()
    if not ds:
        return jsonify({"success": False, "error": "Predefined dataset not found"}), 404
    
    # In reality, this would start a background thread to download
    ds.downloaded = True
    db.session.commit()
    
    return jsonify({"success": True, "message": "Download started"}), 200
