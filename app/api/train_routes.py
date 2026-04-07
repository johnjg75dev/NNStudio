"""
app/api/train_routes.py
/api/train/*  — drive training steps and return metrics.
"""
from flask import Blueprint, request
from .helpers import ok, err, api_route, get_training_session

train_bp = Blueprint("train", __name__)


@train_bp.post("/step")
@api_route
def train_step():
    """
    Run N training steps and return updated metrics + visual snapshot.
    Body: { steps: int, lr: float }
    """
    body  = request.get_json(force=True)
    steps = max(1, int(body.get("steps", 10)))
    lr    = float(body.get("lr", 0.01))

    ts      = get_training_session()
    metrics = ts.train_steps(steps, lr)

    # Return lightweight snapshot for live visualisation
    net = ts.network
    layer_data = []
    for i, layer in enumerate(net.layers):
        snap = layer.weight_snapshot()
        layer_info = {
            "index":      i,
            "W":          snap["W"],
            "b":          snap["b"],
            "dW":         snap["dW"],
            "activation": snap["activation"],
            "type":       layer.__class__.__name__.replace("Layer", "").lower(),
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

    # Activations on first training sample
    if ts.dataset:
        activations = ts.activation_snapshot(ts.dataset[0]["x"])
    else:
        activations = []

    return ok({
        **metrics,
        "loss_history": net.loss_history[-300:],
        "layers":       layer_data,
        "activations":  activations,
    })


@train_bp.post("/evaluate")
@api_route
def evaluate():
    """
    Evaluate training samples or a range sweep.
    Body: {} (uses current session dataset)
    Body: { "ranges": [{"min": 0, "max": 1, "step": 0.2}, ...] } (grid sweep of input ranges)
    """
    import itertools
    ts  = get_training_session()
    if ts.network is None:
        return err("No network built.")

    data = request.get_json() or {}
    ranges = data.get("ranges")
    sl = data.get("start_layer", 0)
    sl = int(sl) if sl is not None else 0
    el = data.get("end_layer", None)
    el = int(el) if el is not None else None
    
    results = []
    
    if ranges:
        # Grid sweep: evaluate all combinations across input ranges
        # step is the increment between points (e.g., step=0.2 with min=0, max=1 → 0, 0.2, 0.4, 0.6, 0.8, 1.0)
        range_lists = []
        
        for r in ranges:
            min_val = r.get("min", 0)
            max_val = r.get("max", 1)
            step_val = r.get("step", 0.2)
            
            # Generate points from min to max with given step
            points = []
            current = min_val
            while current <= max_val + 1e-9:  # Small epsilon for floating point
                points.append(round(current, 10))  # Round to avoid floating point errors
                current += step_val
            
            range_lists.append(points)
        
        # Generate all combinations
        for combo in itertools.product(*range_lists):
            x = list(combo)
            pred = ts.predict(x, start_layer=sl, end_layer=el)
            results.append({
                "x":    x,
                "y":    [0] * len(pred),  # No ground truth for range sweep
                "pred": pred,
            })
    else:
        # Standard: evaluate all training samples
        for sample in ts.dataset:
            pred = ts.predict(sample["x"], start_layer=sl, end_layer=el)
            results.append({
                "x":    sample["x"],
                "y":    sample["y"],
                "pred": pred,
            })

    if not ranges and sl == 0 and el is None:
        loss = ts.network.compute_loss(ts.dataset)
        acc  = ts.network.compute_accuracy(ts.dataset)
    else:
        loss = 0.0
        acc  = 0.0

    return ok({
        "samples":  results,
        "loss":     round(loss, 6),
        "accuracy": round(acc, 4),
    })
