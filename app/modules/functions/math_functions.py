"""app/modules/functions/math_functions.py — Parity, Sine, Half-Adder"""
import math
from .base_function import TrainingFunction


class ParityFunction(TrainingFunction):
    key         = "parity"
    label       = "4-bit Parity"
    description = ("<b>Parity</b>: Output 1 if an odd number of the 4 input "
                   "bits are set. Complex non-linearity — needs depth.")
    inputs, outputs = 4, 1
    input_labels  = ["b3", "b2", "b1", "b0"]
    output_labels = ["Odd?"]
    is_classification = True
    recommended = {"hidden_layers": 3, "neurons": 8, "activation": "relu",
                   "optimizer": "adam", "loss": "bce", "dropout": 0.0, "lr": 0.01}

    def generate_dataset(self):
        data = []
        for i in range(16):
            bits = [(i >> b) & 1 for b in (3, 2, 1, 0)]
            data.append({"x": bits, "y": [sum(bits) % 2]})
        return data


class SineFunction(TrainingFunction):
    key         = "sine"
    label       = "Sine Approximation"
    description = ("<b>Sine</b>: Approximate sin(2πx) from a single input "
                   "in [0,1]. Regression task — tests smooth function fitting.")
    inputs, outputs = 1, 1
    input_labels  = ["x"]
    output_labels = ["sin(x)"]
    is_classification = False
    recommended = {"hidden_layers": 2, "neurons": 8, "activation": "tanh",
                   "optimizer": "adam", "loss": "mse", "dropout": 0.0, "lr": 0.005}

    def generate_dataset(self):
        return [
            {"x": [i / 19], "y": [(math.sin(2 * math.pi * i / 19) + 1) / 2]}
            for i in range(20)
        ]


class HalfAdderFunction(TrainingFunction):
    key         = "adder"
    label       = "Half Adder"
    description = ("<b>Half Adder</b>: Two bits → Sum (XOR) and Carry (AND). "
                   "Dual output — tests multi-task learning.")
    inputs, outputs = 2, 2
    input_labels  = ["A", "B"]
    output_labels = ["Sum", "Carry"]
    is_classification = True
    recommended = {"hidden_layers": 1, "neurons": 4, "activation": "tanh",
                   "optimizer": "adam", "loss": "bce", "dropout": 0.0, "lr": 0.05}

    def generate_dataset(self):
        return [{"x": [a, b], "y": [a ^ b, a & b]}
                for a in (0, 1) for b in (0, 1)]
