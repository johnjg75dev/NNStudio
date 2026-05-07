"""
app/api/studio_routes.py
/api/studio/*  — API endpoints for the new studio surfaces
(data pipeline, training, evaluation, export/quantization).
"""
import json
from io import BytesIO
from flask import Blueprint, request, send_file
from flask_login import login_required
from .helpers import ok, err, api_route, get_training_session
from app.core.exporters import ModelExporter

studio_bp = Blueprint("studio", __name__)


@studio_bp.get("/status")
@login_required
@api_route
def studio_status():
    """Return the status of the studio surfaces."""
    ts = get_training_session()
    has_model = ts.network is not None
    return ok({
        "surfaces": {
            "data":     {"ready": True},
            "train":    {"ready": has_model},
            "evaluate": {"ready": has_model},
            "export":   {"ready": has_model},
        }
    })


@studio_bp.post("/session/export")
@login_required
@api_route
def session_export():
    """
    Export the current session model to a specified format.
    Body: { format: "json"|"safetensors"|"gguf"|"onnx"|"zip" }
    Returns the file as a download.
    """
    ts = get_training_session()
    if ts.network is None:
        return err("No model in session. Build or load a model first.", 400)

    body = request.get_json(force=True)
    fmt = body.get("format", "json").lower()
    supported = ModelExporter.get_supported_formats()

    if fmt not in supported:
        return err(f"Unsupported format '{fmt}'. Supported: {supported}", 400)

    metadata = {
        "name": body.get("name", "nnstudio-model"),
        "description": body.get("description", ""),
        "architecture": ts.arch_key,
        "function": ts.func_key,
        "epochs": ts.network.epoch,
        "final_loss": ts.network.loss_history[-1] if ts.network.loss_history else None,
    }

    export_bytes = ModelExporter.export_bytes(ts.network, fmt, metadata)

    mime_map = {
        "json": "application/json",
        "safetensors": "application/octet-stream",
        "gguf": "application/octet-stream",
        "onnx": "application/octet-stream",
        "zip": "application/zip",
    }
    ext_map = {
        "json": ".json",
        "safetensors": ".safetensors",
        "gguf": ".gguf",
        "onnx": ".onnx",
        "zip": ".zip",
    }

    filename = f"{body.get('name', 'nnstudio-model').replace(' ', '_')}{ext_map[fmt]}"
    return send_file(
        BytesIO(export_bytes),
        mimetype=mime_map.get(fmt, "application/octet-stream"),
        as_attachment=True,
        download_name=filename,
    )
