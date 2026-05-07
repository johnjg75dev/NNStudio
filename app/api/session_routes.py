"""
app/api/session_routes.py
/api/session/*  — manage the per-browser training session.
"""

from __future__ import annotations
import json
import numpy as np
from flask import Blueprint, request
from flask_login import current_user

from .helpers import ok, err, api_route, get_training_session, get_registry
from app.core.network import NeuralNetwork

session_bp = Blueprint("session", __name__)


@session_bp.post("/build")
@api_route
def build():
    """
    Build a new network for this session.
    Body: { func_key, ds_id, arch_key, layers, inputs, outputs, activation, optimizer, lr, loss, weight_decay }

    Supports both flat layer format (type + params at same level) and
    playground format where params are nested under a "params" key.
    When func_key is omitted and layers are provided, auto-detects
    inputs/outputs from the layer config and uses a generic "xor" default.
    """
    import math
    body = request.get_json(force=True)
    registry = get_registry()

    from ..models import Dataset
    from ..modules.functions.custom_function_wrapper import DynamicCustomFunction

    ds_id = body.get("ds_id")
    func_key = body.get("func_key")

    dataset = []
    fn_mod = None

    # Normalize layers: flatten "params" key if present (playground format)
    # and map display names to internal builder type names
    TYPE_ALIASES = {
        "dense": "dense", "fc": "dense", "fullyconnected": "dense",
        "conv1d": "conv2d", "conv2d": "conv2d", "conv3d": "conv2d",
        "maxpool1d": "maxpool2d", "maxpool2d": "maxpool2d", "maxpool3d": "maxpool2d",
        "avgpool1d": "maxpool2d", "avgpool2d": "maxpool2d", "avgpool3d": "maxpool2d",
        "dropout": "dropout", "batchnorm": "batchnorm", "layernorm": "layernorm",
        "flatten": "flatten", "reshape": "flatten",
        "rnn": "simple_rnn", "lstm": "lstm", "gru": "simple_rnn",
        "embedding": "embedding", "attention": "multihead_attention",
    }

    raw_layers = body.get("layers", [])
    layers = []
    for lc in raw_layers:
        raw_type = lc.get("type", "dense").lower().replace(" ", "").replace("-", "").replace("_", "")
        mapped = TYPE_ALIASES.get(raw_type, "dense")
        entry = {"type": mapped}
        params = lc.get("params", {})
        if isinstance(params, dict) and len(params) > 0:
            entry.update(params)
        else:
            for k, v in lc.items():
                if k != "type":
                    entry[k] = v
        layers.append(entry)

    # Auto-detect inputs/outputs from layers when not explicitly provided.
    # Inputs are a property of the data, not the architecture, so only infer
    # if we also have a dataset or function. Outputs can be inferred from the
    # last layer's neuron/filter count.
    inputs = body.get("inputs")
    outputs = body.get("outputs")
    if outputs is None and layers:
        last = layers[-1]
        t = last.get("type", "dense")
        if t == "dense":
            outputs = last.get("neurons", 1)
        elif t in ("conv2d",):
            outputs = last.get("filters", 1)
        else:
            outputs = 1
    # inputs remain None — will be resolved by fn_mod or defaulted later

    if ds_id:
        user_id = getattr(current_user, "id", None)
        ds = Dataset.query.filter_by(id=ds_id, user_id=user_id).first()
        if ds:
            dataset = ds.data or []
            class DatasetMock:
                inputs = ds.num_inputs
                outputs = ds.num_outputs or 1
                def to_dict(self):
                    return {
                        "key": f"ds_{ds.id}",
                        "label": ds.name,
                        "inputs": self.inputs,
                        "outputs": self.outputs,
                    }
            fn_mod = DatasetMock()
            if inputs is None:
                inputs = fn_mod.inputs
            if outputs is None:
                outputs = fn_mod.outputs

    if not fn_mod and func_key:
        user_id = getattr(current_user, "id", None)
        fn_mod = registry.get_with_custom(func_key, user_id)
        if fn_mod:
            dataset = fn_mod.generate_dataset()
            if inputs is None:
                inputs = fn_mod.inputs
            if outputs is None:
                outputs = fn_mod.outputs
        else:
            return err(f"Unknown function key: {func_key!r}", 404)

    if not fn_mod:
        if inputs is None:
            inputs = 2
        if outputs is None:
            outputs = 1
        fn_mod = registry.get("xor")
        if fn_mod:
            dataset = fn_mod.generate_dataset()

    config = {
        "inputs": inputs or fn_mod.inputs,
        "outputs": outputs or fn_mod.outputs,
        "layers": layers,
        "activation": body.get("activation", "tanh"),
        "optimizer": body.get("optimizer", "adam"),
        "lr": float(body.get("lr", 0.01)),
        "loss": body.get("loss", "bce"),
        "weight_decay": float(body.get("weight_decay", 0.0)),
        "func_key": func_key or "xor",
        "arch_key": body.get("arch_key", "mlp"),
    }

    ts = get_training_session()
    ts.build_network(config, dataset)

    net = ts.network
    return ok(
        {
            "topology": net.topology,
            "param_count": net.param_count,
            "total_params": net.param_count,
            "epoch": net.epoch,
            "func": fn_mod.to_dict() if fn_mod else {},
        }
    )


@session_bp.post("/reset")
@api_route
def reset_weights():
    """Re-initialise weights without rebuilding the topology."""
    ts = get_training_session()
    if ts.network is None:
        return err("No network built yet.")
    ts.network._build_layers() if hasattr(ts.network, "_build_layers") else None
    # Rebuild layers in-place
    from app.core.network import NetworkBuilder
    from app.core.layers import (
        DropoutLayer,
        BatchNormLayer,
        Conv2DLayer,
        MaxPool2DLayer,
        FlattenLayer,
    )

    # Build layers config from existing network
    layers_config = []
    for i, layer in enumerate(ts.network.layers[:-1]):  # Exclude output layer
        layer_type = layer.__class__.__name__

        if isinstance(layer, DropoutLayer):
            layers_config.append(
                {
                    "type": "dropout",
                    "rate": layer.rate,
                }
            )
        elif isinstance(layer, BatchNormLayer):
            layers_config.append(
                {
                    "type": "batchnorm",
                }
            )
        elif isinstance(layer, Conv2DLayer):
            layers_config.append(
                {
                    "type": "conv2d",
                    "out_channels": layer.out_channels,
                    "kernel_size": layer.kernel_size,
                    "padding": layer.padding,
                    "activation": layer.activation.name,
                }
            )
        elif isinstance(layer, MaxPool2DLayer):
            layers_config.append(
                {
                    "type": "maxpool2d",
                    "pool_size": layer.pool_size,
                    "stride": layer.stride,
                }
            )
        elif isinstance(layer, FlattenLayer):
            layers_config.append(
                {
                    "type": "flatten",
                }
            )
        elif hasattr(layer, "activation"):
            layers_config.append(
                {
                    "type": "dense",
                    "neurons": layer.n_out,
                    "activation": layer.activation.name,
                }
            )

    cfg = {
        "inputs": ts.network.topology[0],
        "outputs": ts.network.topology[-1],
        "layers": layers_config,
        "activation": ts.network.layers[0].activation.name
        if hasattr(ts.network.layers[0], "activation")
        else "tanh",
        "optimizer": ts.network.optimizer.__class__.__name__.lower(),
        "lr": ts.network.optimizer.lr,
        "loss": ts.network.loss_fn.name,
        "weight_decay": getattr(ts.network.optimizer, "weight_decay", 0.0),
        "func_key": ts.func_key,
        "arch_key": ts.arch_key,
    }
    ts.build_network(cfg, ts.dataset)
    return ok({"message": "Weights reset.", "epoch": 0})


@session_bp.post("/predict")
@api_route
def predict():
    """
    Run a single forward pass.
    Body: { x: [float, ...], start_layer: int, end_layer: int|null, node_overrides: dict|null }
    """
    body = request.get_json(force=True)
    x = body.get("x", [])
    sl = body.get("start_layer", 0)
    el = body.get("end_layer", None)
    overrides = body.get("node_overrides")
    ts = get_training_session()

    output = ts.predict(
        x,
        start_layer=int(sl) if sl is not None else 0,
        end_layer=int(el) if el is not None else None,
        node_overrides=overrides,
    )
    activations = ts.activation_snapshot(x)

    return ok(
        {
            "output": output,
            "activations": activations,
        }
    )


@session_bp.post("/latent-sweep")
@api_route
def latent_sweep():
    """
    Sweep a single latent node range and return output variations.
    """
    body = request.get_json(force=True)
    x = body.get("x", [])
    layer = body.get("layer")
    node = body.get("node")
    rng = body.get("range", [-2, 2, 0.2])

    ts = get_training_session()
    data = ts.latent_sweep(x, layer, node, *rng)
    return ok({"sweep_data": data})


@session_bp.get("/snapshot")
@api_route
def snapshot():
    """
    Return a full visual snapshot of the current network state:
    topology, all weights, activations on first training sample.
    """
    ts = get_training_session()
    if ts.network is None:
        return ok({"built": False})

    net = ts.network
    registry = get_registry()
    fn_mod = registry.get(ts.func_key)

    # Activations on first sample
    first_x = ts.dataset[0]["x"] if ts.dataset else [0] * net.topology[0]
    activations = ts.activation_snapshot(first_x)

    layer_data = []
    for i, layer in enumerate(net.layers):
        snap = layer.weight_snapshot()

        # Get proper dimensions for each layer type
        layer_type = layer.__class__.__name__
        if layer_type == "Conv2DLayer":
            n_in = layer.in_channels * 64  # Assume 8x8 input
            n_out = layer.out_channels * 16  # Assume 4x4 output
        elif layer_type == "MaxPool2DLayer":
            n_in = 0  # Will be inferred from previous layer
            n_out = 0
        elif layer_type == "FlattenLayer":
            n_in = 0
            n_out = 0
        elif hasattr(layer, "n_in") and hasattr(layer, "n_out"):
            n_in = layer.n_in
            n_out = layer.n_out
        else:
            n_in = 0
            n_out = 0

        layer_info = {
            "index": i,
            "n_in": n_in,
            "n_out": n_out,
            "is_output": layer.is_output,
            "W": snap["W"],
            "b": snap["b"],
            "dW": snap["dW"],
            "activation": snap["activation"],
            "type": layer_type.replace("Layer", "").lower(),
        }
        # Add type-specific info
        if hasattr(layer, "rate"):  # Dropout
            layer_info["rate"] = layer.rate
        if hasattr(layer, "kernel_size"):  # Conv2D
            layer_info["kernel_size"] = layer.kernel_size
            layer_info["out_channels"] = layer.out_channels
        if hasattr(layer, "pool_size"):  # MaxPool
            layer_info["pool_size"] = layer.pool_size
        if hasattr(layer, "hidden_size"):  # LSTM/RNN
            layer_info["hidden_size"] = layer.hidden_size
        if hasattr(layer, "vocab_size"):  # Embedding
            layer_info["vocab_size"] = layer.vocab_size
            layer_info["embed_dim"] = layer.embed_dim
        if hasattr(layer, "num_heads"):  # Attention
            layer_info["num_heads"] = layer.num_heads
        layer_data.append(layer_info)

    return ok(
        {
            "built": True,
            "topology": net.topology,
            "param_count": net.param_count,
            "epoch": net.epoch,
            "loss_history": net.loss_history[-300:],
            "layers": layer_data,
            "activations": activations,
            "func_key": ts.func_key,
            "arch_key": ts.arch_key,
            "func": fn_mod.to_dict() if fn_mod else {},
        }
    )


@session_bp.post("/export")
@api_route
def export_model():
    """Return the full serialised model as JSON."""
    ts = get_training_session()
    if ts.network is None:
        return err("No network to export.")
    return ok(ts.serialise())


@session_bp.post("/import")
@api_route
def import_model():
    """
    Load a previously exported model.
    Body: the JSON object returned by /export.
    """
    body = request.get_json(force=True)
    net = NeuralNetwork.from_dict(body)

    ts = get_training_session()
    ts.network = net
    ts.func_key = body.get("func_key", "xor")
    ts.arch_key = body.get("arch_key", "mlp")

    registry = get_registry()
    fn_mod = registry.get(ts.func_key)
    if fn_mod:
        ts.dataset = fn_mod.generate_dataset()

    ts.touch()
    return ok(
        {
            "topology": net.topology,
            "param_count": net.param_count,
            "epoch": net.epoch,
        }
    )
