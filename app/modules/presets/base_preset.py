"""
app/modules/presets/base_preset.py
A Preset bundles a complete training configuration.
The registry scans this folder — add a new .py file with a PresetModule
subclass and it appears in the UI automatically with no other changes needed.
"""
from ..base import BaseModule


class PresetModule(BaseModule):
    """
    Describes a recommended configuration bundle.
    """
    category = "presets"

    # All config fields with safe defaults
    arch_key:      str   = "mlp"
    func_key:      str   = "xor"
    layers:        list  = None  # List of layer configs: [{"neurons": 4, "activation": "tanh", "type": "dense"}]
    activation:    str   = "tanh"
    optimizer:     str   = "adam"
    loss:          str   = "bce"
    lr:            float = 0.01
    weight_decay:  float = 0.0

    def __init__(self):
        if self.layers is None:
            self.layers = []

    def to_dict(self) -> dict:
        return {
            "key":           self.key,
            "label":         self.label,
            "description":   self.description,
            "category":      self.category,
            "arch_key":      self.arch_key,
            "func_key":      self.func_key,
            "layers":        self.layers,
            "activation":    self.activation,
            "optimizer":     self.optimizer,
            "loss":          self.loss,
            "lr":            self.lr,
            "weight_decay":  self.weight_decay,
        }
