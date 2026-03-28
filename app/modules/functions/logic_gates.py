"""app/modules/functions/logic_gates.py — AND, OR, XNOR"""
from .base_function import TrainingFunction


class ANDFunction(TrainingFunction):
    key         = "and"
    label       = "AND Gate"
    description = "<b>AND</b>: Output 1 only when both inputs are 1. Linearly separable."
    inputs, outputs = 2, 1
    input_labels = ["A", "B"]
    output_labels = ["Out"]
    recommended = {"layers": [{"neurons": 2, "activation": "sigmoid", "type": "dense"}],
                   "optimizer": "adam", "loss": "bce", "dropout": 0.0, "lr": 0.05}

    def generate_dataset(self):
        return [{"x": [a, b], "y": [a & b]} for a in (0, 1) for b in (0, 1)]


class ORFunction(TrainingFunction):
    key         = "or"
    label       = "OR Gate"
    description = "<b>OR</b>: Output 1 if at least one input is 1. Linearly separable."
    inputs, outputs = 2, 1
    input_labels = ["A", "B"]
    output_labels = ["Out"]
    recommended = {"layers": [{"neurons": 2, "activation": "sigmoid", "type": "dense"}],
                   "optimizer": "adam", "loss": "bce", "dropout": 0.0, "lr": 0.05}

    def generate_dataset(self):
        return [{"x": [a, b], "y": [min(1, a + b)]} for a in (0, 1) for b in (0, 1)]


class XNORFunction(TrainingFunction):
    key         = "xnor"
    label       = "XNOR Gate"
    description = "<b>XNOR</b>: Inverse of XOR. Output 1 when inputs are equal. Non-linearly separable."
    inputs, outputs = 2, 1
    input_labels = ["A", "B"]
    output_labels = ["Out"]
    recommended = {"layers": [{"neurons": 4, "activation": "tanh", "type": "dense"}],
                   "optimizer": "adam", "loss": "bce", "dropout": 0.0, "lr": 0.1}

    def generate_dataset(self):
        return [{"x": [a, b], "y": [1 - (a ^ b)]} for a in (0, 1) for b in (0, 1)]
