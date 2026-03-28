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
        layer_data.append({
            "index":      i,
            "W":          snap["W"],
            "b":          snap["b"],
            "dW":         snap["dW"],
            "activation": snap["activation"],
        })

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
    Evaluate all training samples and return per-sample predictions.
    Body: {} (uses current session dataset)
    """
    ts  = get_training_session()
    if ts.network is None:
        return err("No network built.")

    results = []
    for sample in ts.dataset:
        pred = ts.predict(sample["x"])
        results.append({
            "x":    sample["x"],
            "y":    sample["y"],
            "pred": pred,
        })

    loss = ts.network.compute_loss(ts.dataset)
    acc  = ts.network.compute_accuracy(ts.dataset)

    return ok({
        "samples":  results,
        "loss":     round(loss, 6),
        "accuracy": round(acc, 4),
    })
