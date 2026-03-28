"""
app/modules/presets/builtin_presets.py
Built-in preset configurations.  Each is a standalone class.
Add more here or in any new file — the registry finds them all.
"""
from .base_preset import PresetModule


class TinyXORPreset(PresetModule):
    key           = "preset_tiny_xor"
    label         = "Tiny XOR"
    description   = "Minimal 2-3-1 network solving XOR. Great first experiment."
    arch_key      = "mlp"
    func_key      = "xor"
    layers        = [{"neurons": 3, "activation": "tanh", "type": "dense"}]
    optimizer     = "adam"
    loss          = "bce"
    lr            = 0.1


class ANDGatePreset(PresetModule):
    key           = "preset_and"
    label         = "AND Gate"
    description   = "Linearly separable — converges fast with a single hidden layer."
    arch_key      = "mlp"
    func_key      = "and"
    layers        = [{"neurons": 2, "activation": "sigmoid", "type": "dense"}]
    optimizer     = "adam"
    loss          = "bce"
    lr            = 0.05


class SevenSegmentPreset(PresetModule):
    key           = "preset_seg7"
    label         = "7-Segment"
    description   = "Teach the network all 10 digit patterns for a 7-segment display."
    arch_key      = "mlp"
    func_key      = "seg7"
    layers        = [
        {"neurons": 10, "activation": "tanh", "type": "dense"},
        {"neurons": 10, "activation": "tanh", "type": "dense"}
    ]
    optimizer     = "adam"
    loss          = "mse"
    lr            = 0.02


class ParityPreset(PresetModule):
    key           = "preset_parity"
    label         = "Parity"
    description   = "4-bit parity — needs 3+ hidden layers to learn the complex XOR pattern."
    arch_key      = "mlp"
    func_key      = "parity"
    layers        = [
        {"neurons": 8, "activation": "relu", "type": "dense"},
        {"neurons": 8, "activation": "relu", "type": "dense"},
        {"neurons": 8, "activation": "relu", "type": "dense"}
    ]
    optimizer     = "adam"
    loss          = "bce"
    lr            = 0.01


class SineFitPreset(PresetModule):
    key           = "preset_sine"
    label         = "Sine Fit"
    description   = "Regression task: approximate sin(2πx). Tests smooth function fitting."
    arch_key      = "mlp"
    func_key      = "sine"
    layers        = [
        {"neurons": 8, "activation": "tanh", "type": "dense"},
        {"neurons": 8, "activation": "tanh", "type": "dense"}
    ]
    optimizer     = "adam"
    loss          = "mse"
    lr            = 0.005


class HalfAdderPreset(PresetModule):
    key           = "preset_adder"
    label         = "Half Adder"
    description   = "Learn both Sum (XOR) and Carry (AND) simultaneously."
    arch_key      = "mlp"
    func_key      = "adder"
    layers        = [{"neurons": 4, "activation": "tanh", "type": "dense"}]
    optimizer     = "adam"
    loss          = "bce"
    lr            = 0.05


class SpiralPreset(PresetModule):
    key           = "preset_spiral"
    label         = "Spiral Deep"
    description   = "Interleaved spirals — needs a deep network and dropout regularisation."
    arch_key      = "mlp"
    func_key      = "spiral"
    layers        = [
        {"neurons": 12, "activation": "leakyrelu", "type": "dense"},
        {"type": "dropout", "rate": 0.1},
        {"neurons": 12, "activation": "leakyrelu", "type": "dense"},
        {"type": "dropout", "rate": 0.1},
        {"neurons": 12, "activation": "leakyrelu", "type": "dense"},
        {"type": "dropout", "rate": 0.1},
        {"neurons": 12, "activation": "leakyrelu", "type": "dense"}
    ]
    optimizer     = "adam"
    loss          = "bce"
    lr            = 0.005


class AutoencoderPreset(PresetModule):
    key           = "preset_autoenc"
    label         = "Autoencoder"
    description   = "Compress 8-bit one-hot vectors to 3 neurons and reconstruct."
    arch_key      = "autoencoder"
    func_key      = "autoenc"
    layers        = [
        {"neurons": 3, "activation": "sigmoid", "type": "dense"},
        {"neurons": 3, "activation": "sigmoid", "type": "dense"}
    ]
    optimizer     = "adam"
    loss          = "mse"
    lr            = 0.01


class CirclePreset(PresetModule):
    key           = "preset_circle"
    label         = "Circle Boundary"
    description   = "Classify points inside vs outside a circle — non-linear boundary."
    arch_key      = "mlp"
    func_key      = "circle"
    layers        = [
        {"neurons": 6, "activation": "relu", "type": "dense"},
        {"neurons": 6, "activation": "relu", "type": "dense"}
    ]
    optimizer     = "adam"
    loss          = "bce"
    lr            = 0.02


class RegularisedPreset(PresetModule):
    key           = "preset_regularised"
    label         = "Regularised Deep"
    description   = "Wide deep spiral solver with AdamW + dropout to demonstrate regularisation."
    arch_key      = "mlp"
    func_key      = "spiral"
    layers        = [
        {"neurons": 16, "activation": "gelu", "type": "dense"},
        {"type": "dropout", "rate": 0.2},
        {"neurons": 16, "activation": "gelu", "type": "dense"},
        {"type": "dropout", "rate": 0.2},
        {"neurons": 16, "activation": "gelu", "type": "dense"},
        {"type": "dropout", "rate": 0.2},
        {"neurons": 16, "activation": "gelu", "type": "dense"},
        {"type": "dropout", "rate": 0.2},
        {"neurons": 16, "activation": "gelu", "type": "dense"}
    ]
    optimizer     = "adamw"
    loss          = "bce"
    lr            = 0.003
    weight_decay  = 0.01
