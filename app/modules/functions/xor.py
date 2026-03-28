"""app/modules/functions/xor.py"""
from .base_function import TrainingFunction


class XORFunction(TrainingFunction):
    key         = "xor"
    label       = "XOR Gate"
    description = ("<b>XOR</b>: Output 1 when inputs differ. "
                   "Non-linearly separable — the classic test of hidden layers.")
    inputs      = 2
    outputs     = 1
    input_labels  = ["A", "B"]
    output_labels = ["Out"]
    is_classification = True
    recommended = {"layers": [{"neurons": 4, "activation": "tanh", "type": "dense"}],
                   "optimizer": "adam", "loss": "bce", "dropout": 0.0, "lr": 0.1}

    def generate_dataset(self):
        return [{"x": [a, b], "y": [a ^ b]}
                for a in (0, 1) for b in (0, 1)]
