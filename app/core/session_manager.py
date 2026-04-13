"""
app/core/session_manager.py
In-memory session store keyed by Flask session ID.
Each entry is a TrainingSession object that owns a NeuralNetwork instance.
"""
from __future__ import annotations

import time
import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .network import NeuralNetwork, NetworkBuilder
from ..modules.registry import get_registry


# ── per-session state ────────────────────────────────────────────────
@dataclass
class TrainingSession:
    session_id:  str
    network:     Optional[NeuralNetwork] = None
    func_key:    str = "xor"
    arch_key:    str = "mlp"
    dataset:     list[dict] = field(default_factory=list)
    created_at:  float = field(default_factory=time.time)
    updated_at:  float = field(default_factory=time.time)
    evaluation_history: list[dict] = field(default_factory=list)
    modification_history: list[dict] = field(default_factory=list)
    eval_sample_indices: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    snapshots:  dict = field(default_factory=dict)

    def touch(self):
        self.updated_at = time.time()

    def build_network(self, config: dict, dataset: list[dict]):
        self.network  = NetworkBuilder.build(config)
        self.dataset  = dataset
        self.func_key = config.get("func_key", "xor")
        self.arch_key = config.get("arch_key", "mlp")
        self.touch()

    def train_steps(self, steps: int, lr: float) -> dict:
        if self.network is None:
            raise RuntimeError("Network not initialised.")

        # Resolve dataset labels if input-only
        active_dataset = self.dataset
        if not active_dataset:
            # Generate temporary live dataset from active function if no dataset provided
            registry = get_registry()
            fn_mod = registry.get_with_custom(self.func_key)
            if fn_mod:
                active_dataset = fn_mod.generate_dataset()
            else:
                raise RuntimeError("No dataset or function available for training.")

        # Check if first sample is input-only (missing 'y')
        if active_dataset and "y" not in active_dataset[0]:
            registry = get_registry()
            fn_mod = registry.get_with_custom(self.func_key)
            if fn_mod:
                # Generate labels on-the-fly for the current batch
                if hasattr(fn_mod, 'f'):
                    for sample in active_dataset:
                        sample["y"] = fn_mod.f(np.array(sample["x"]))
                else:
                    # Fallback to re-generating a full dummy and matching by input
                    # This is better than assigning first sample's y to all
                    dummy = fn_mod.generate_dataset()
                    dummy_map = {tuple(s["x"]): s["y"] for s in dummy}
                    for sample in active_dataset:
                        sample["y"] = dummy_map.get(tuple(sample["x"]), dummy[0]["y"])

        for _ in range(steps):
            self.network.train_epoch(active_dataset, lr=lr)

        loss = self.network.compute_loss(active_dataset)
        acc  = self.network.compute_accuracy(active_dataset)
        self.network.loss_history.append(loss)
        if len(self.network.loss_history) > 500:
            self.network.loss_history = self.network.loss_history[-500:]
        
        # Periodic Evaluation Log
        if self.network.epoch % 50 == 0 or self.network.epoch == 1:
            eval_preds = []
            for idx in self.eval_sample_indices:
                if idx < len(active_dataset):
                    s = active_dataset[idx]
                    p = self.predict(s["x"])
                    eval_preds.append({"x": s["x"], "y": s["y"], "pred": p})
            self.evaluation_history.append({
                "epoch": self.network.epoch,
                "preds": eval_preds,
                "loss": loss,
                "acc": acc
            })
            if len(self.evaluation_history) > 50: self.evaluation_history.pop(0)

        self.touch()

        return {
            "epoch":   self.network.epoch,
            "loss":    round(loss, 6),
            "accuracy": round(acc, 4),
            "eval_history": self.evaluation_history[-1:] # Return latest for live update
        }

    def predict(self, x: list[float], start_layer: int = 0, end_layer: Optional[int] = None, node_overrides: Optional[dict] = None) -> list[float]:
        if self.network is None:
            raise RuntimeError("No network built.")
        return self.network.predict(np.array(x), start_layer=start_layer, end_layer=end_layer, node_overrides=node_overrides).tolist()

    def latent_sweep(self, x: list[float], layer: int, node: int, r_min: float = -2, r_max: float = 2, step: float = 0.2):
        if self.network is None:
            return []
        
        sweep_data = []
        for val in np.arange(r_min, r_max + step, step):
            p = self.predict(x, node_overrides={"layer": layer, "node": node, "val": float(val)})
            # We take the first output for simpler plotting if there are many
            sweep_data.append({"val": float(val), "result": float(p[0])})
        return sweep_data

    def activation_snapshot(self, x: list[float]) -> list[list[float]]:
        if self.network is None:
            raise RuntimeError("No network built.")
        return self.network.activation_snapshot(np.array(x))

    def serialise(self) -> dict:
        if self.network is None:
            return {}
        return {
            **self.network.to_dict(),
            "func_key": self.func_key,
            "arch_key": self.arch_key,
        }


# ── global in-memory store ────────────────────────────────────────────
class SessionManager:
    """
    Singleton-style store.  Flask doesn't support true singletons cleanly
    so we keep one instance on app.extensions.
    TTL-based eviction runs on every write.
    """
    TTL = 3600  # seconds

    def __init__(self):
        self._store: dict[str, TrainingSession] = {}

    def get_or_create(self, session_id: str) -> TrainingSession:
        if session_id not in self._store:
            self._store[session_id] = TrainingSession(session_id=session_id)
        self._evict()
        return self._store[session_id]

    def get(self, session_id: str) -> Optional[TrainingSession]:
        return self._store.get(session_id)

    def delete(self, session_id: str):
        self._store.pop(session_id, None)

    def _evict(self):
        now = time.time()
        stale = [k for k, v in self._store.items()
                 if now - v.updated_at > self.TTL]
        for k in stale:
            del self._store[k]
