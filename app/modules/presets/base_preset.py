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
    hidden_layers: int   = 1
    neurons:       int   = 4
    activation:    str   = "tanh"
    optimizer:     str   = "adam"
    loss:          str   = "bce"
    lr:            float = 0.01
    dropout:       float = 0.0
    weight_decay:  float = 0.0

    def to_dict(self) -> dict:
        return {
            "key":           self.key,
            "label":         self.label,
            "description":   self.description,
            "category":      self.category,
            "arch_key":      self.arch_key,
            "func_key":      self.func_key,
            "hidden_layers": self.hidden_layers,
            "neurons":       self.neurons,
            "activation":    self.activation,
            "optimizer":     self.optimizer,
            "loss":          self.loss,
            "lr":            self.lr,
            "dropout":       self.dropout,
            "weight_decay":  self.weight_decay,
        }
