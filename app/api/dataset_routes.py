"""
app/api/dataset_routes.py
API endpoints for dataset management.
Implements a dual-registry: Global (system, read-only) + User (custom, editable).
"""
import os
import json
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from .. import db
from ..models import Dataset

dataset_bp = Blueprint('datasets', __name__)

# ═══════════════════════════════════════════════════════════════
# GLOBAL DATASET REGISTRY
# Predefined datasets that are auto-seeded for every user.
# Users can use them but cannot modify or delete them.
# ═══════════════════════════════════════════════════════════════
GLOBAL_DATASETS = [
    {
        "name": "XOR Gate",
        "description": "The classic exclusive-OR logic gate. 2 binary inputs → 1 binary output. Not linearly separable — requires at least one hidden layer.",
        "ds_type": "tabular",
        "num_inputs": 2, "num_outputs": 1,
        "input_labels": ["A", "B"], "output_labels": ["A XOR B"],
        "data": [
            {"x": [0, 0], "y": [0]},
            {"x": [0, 1], "y": [1]},
            {"x": [1, 0], "y": [1]},
            {"x": [1, 1], "y": [0]},
        ],
        "auto_download": True,
    },
    {
        "name": "Iris (3-class)",
        "description": "Fisher's Iris dataset. 4 measurements (sepal/petal length & width) → 3 species. A classic classification benchmark.",
        "ds_type": "tabular",
        "num_inputs": 4, "num_outputs": 3,
        "input_labels": ["Sepal L", "Sepal W", "Petal L", "Petal W"],
        "output_labels": ["Setosa", "Versicolor", "Virginica"],
        "data": [],  # seeded with sample data below
        "auto_download": True,
    },
    {
        "name": "Circles (2-class)",
        "description": "Two concentric circles of points. Inner circle = class 0, outer = class 1. Tests non-linear decision boundaries.",
        "ds_type": "tabular",
        "num_inputs": 2, "num_outputs": 1,
        "input_labels": ["X", "Y"], "output_labels": ["Class"],
        "data": [],
        "auto_download": True,
    },
    {
        "name": "MNIST (Handwritten Digits)",
        "description": "The classic 28×28 grayscale image dataset of handwritten digits (0-9). 784 inputs → 10 outputs.",
        "ds_type": "mnist",
        "num_inputs": 784, "num_outputs": 10,
        "width": 28, "height": 28, "channels": 1,
        "data": [],
        "auto_download": False,
    },
    {
        "name": "Fashion-MNIST",
        "description": "28×28 grayscale images of 10 fashion categories (T-shirt, trouser, pullover, etc). Drop-in MNIST replacement.",
        "ds_type": "fashion_mnist",
        "num_inputs": 784, "num_outputs": 10,
        "width": 28, "height": 28, "channels": 1,
        "data": [],
        "auto_download": False,
    },
]


def _generate_iris_samples():
    """Generate a representative subset of Iris-like data (normalised to [0,1])."""
    import random
    random.seed(42)
    samples = []
    # Setosa cluster
    for _ in range(10):
        samples.append({"x": [round(random.uniform(0.15, 0.35), 3), round(random.uniform(0.55, 0.85), 3),
                               round(random.uniform(0.03, 0.15), 3), round(random.uniform(0.01, 0.08), 3)], "y": [1, 0, 0]})
    # Versicolor cluster
    for _ in range(10):
        samples.append({"x": [round(random.uniform(0.35, 0.60), 3), round(random.uniform(0.20, 0.50), 3),
                               round(random.uniform(0.35, 0.55), 3), round(random.uniform(0.25, 0.50), 3)], "y": [0, 1, 0]})
    # Virginica cluster
    for _ in range(10):
        samples.append({"x": [round(random.uniform(0.45, 0.80), 3), round(random.uniform(0.20, 0.55), 3),
                               round(random.uniform(0.55, 0.85), 3), round(random.uniform(0.50, 0.85), 3)], "y": [0, 0, 1]})
    random.shuffle(samples)
    return samples


def _generate_circles_samples():
    """Generate two concentric circles of 2D points."""
    import random, math
    random.seed(42)
    samples = []
    for _ in range(20):
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(0.0, 0.25)
        samples.append({"x": [round(0.5 + r * math.cos(angle), 3), round(0.5 + r * math.sin(angle), 3)], "y": [0]})
    for _ in range(20):
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(0.35, 0.48)
        samples.append({"x": [round(0.5 + r * math.cos(angle), 3), round(0.5 + r * math.sin(angle), 3)], "y": [1]})
    random.shuffle(samples)
    return samples


def _seed_global_datasets(user_id, existing):
    """Ensure all GLOBAL_DATASETS exist for the given user. Returns newly created records."""
    existing_names = {d.name for d in existing if d.is_predefined}
    created = []
    for gds in GLOBAL_DATASETS:
        if gds["name"] in existing_names:
            continue
        # Generate data for datasets that auto-download
        data = gds.get("data", [])
        if gds["name"] == "Iris (3-class)" and not data:
            data = _generate_iris_samples()
        elif gds["name"] == "Circles (2-class)" and not data:
            data = _generate_circles_samples()

        ds = Dataset(
            user_id=user_id,
            name=gds["name"],
            description=gds["description"],
            ds_type=gds["ds_type"],
            is_input_only=False,
            num_inputs=gds["num_inputs"],
            num_outputs=gds["num_outputs"],
            input_labels=gds.get("input_labels", []),
            output_labels=gds.get("output_labels", []),
            width=gds.get("width"),
            height=gds.get("height"),
            channels=gds.get("channels", 1),
            is_predefined=True,
            downloaded=gds.get("auto_download", False),
            data=data,
        )
        db.session.add(ds)
        created.append(ds)
    if created:
        db.session.commit()
    return created


@dataset_bp.route('', methods=['GET'])
@login_required
def list_datasets():
    """List datasets for the current user including predefined global ones."""
    user_datasets = Dataset.query.filter_by(user_id=current_user.id).all()

    # Auto-seed any missing global datasets for this user
    newly_created = _seed_global_datasets(current_user.id, user_datasets)
    all_datasets = user_datasets + newly_created

    # Sort: predefined (global) first, then user datasets, alphabetically within each
    all_datasets.sort(key=lambda d: (0 if d.is_predefined else 1, d.name))

    return jsonify({
        "ok": True,
        "data": {
            "datasets": [d.to_dict() for d in all_datasets]
        }
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
                return jsonify({"ok": False, "error": f"Missing field: {field}"}), 400
        
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
            is_predefined=False,
            downloaded=True
        )
        
        db.session.add(new_ds)
        db.session.commit()
        
        return jsonify({"ok": True, "data": {"dataset": new_ds.to_dict()}}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500

@dataset_bp.route('/<int:ds_id>', methods=['GET'])
@login_required
def get_dataset(ds_id):
    """Get full dataset details."""
    ds = Dataset.query.filter_by(id=ds_id, user_id=current_user.id).first()
    if not ds:
        return jsonify({"ok": False, "error": "Dataset not found"}), 404
    return jsonify({"ok": True, "data": {"dataset": ds.to_dict_full()}}), 200

@dataset_bp.route('/<int:ds_id>', methods=['PUT'])
@login_required
def update_dataset(ds_id):
    """Update dataset."""
    try:
        ds = Dataset.query.filter_by(id=ds_id, user_id=current_user.id).first()
        if not ds:
            return jsonify({"ok": False, "error": "Dataset not found"}), 404
        if ds.is_predefined:
            return jsonify({"ok": False, "error": "Cannot modify predefined datasets."}), 403
              
        data = request.get_json()
        for field in ['name', 'description', 'data', 'is_input_only', 'num_inputs', 'num_outputs', 'width', 'height']:
            if field in data:
                setattr(ds, field, data[field])
              
        db.session.commit()
        return jsonify({"ok": True, "data": {"dataset": ds.to_dict()}}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500

@dataset_bp.route('/<int:ds_id>', methods=['DELETE'])
@login_required
def delete_dataset(ds_id):
    """Delete dataset."""
    ds = Dataset.query.filter_by(id=ds_id, user_id=current_user.id).first()
    if not ds:
        return jsonify({"ok": False, "error": "Dataset not found"}), 404
    if ds.is_predefined:
        return jsonify({"ok": False, "error": "Cannot delete predefined datasets from global registry."}), 403
    
    db.session.delete(ds)
    db.session.commit()
    return jsonify({"ok": True, "data": {"message": "Dataset deleted"}}), 200

@dataset_bp.route('/<int:ds_id>/download', methods=['POST'])
@login_required
def download_predefined(ds_id):
    """Trigger download for a predefined dataset."""
    ds = Dataset.query.filter_by(id=ds_id, user_id=current_user.id, is_predefined=True).first()
    if not ds:
        return jsonify({"ok": False, "error": "Predefined dataset not found"}), 404
    
    # Simulate downloading the dataset
    import random
    mock_data = []
    num_samples = 10
    for _ in range(num_samples):
        x = [round(random.random(), 3) for _ in range(ds.num_inputs)]
        y = [0] * ds.num_outputs
        y[random.randint(0, ds.num_outputs - 1)] = 1
        mock_data.append({"x": x, "y": y})
        
    ds.data = mock_data
    ds.downloaded = True
    db.session.commit()
    
    return jsonify({"ok": True, "data": {"message": "Dataset downloaded successfully!"}}), 200

