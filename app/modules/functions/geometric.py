"""app/modules/functions/geometric.py — Circle, Spiral, Autoencoder"""
import math, random
from .base_function import TrainingFunction


class CircleFunction(TrainingFunction):
    key         = "circle"
    label       = "Circle Boundary"
    description = ("<b>Circle</b>: Classify if point (x,y) is inside a "
                   "circle centred at (0.5, 0.5). Non-linear decision boundary.")
    inputs, outputs = 2, 1
    input_labels  = ["x", "y"]
    output_labels = ["Inside"]
    is_classification = True
    recommended = {"hidden_layers": 2, "neurons": 6, "activation": "relu",
                   "optimizer": "adam", "loss": "bce", "dropout": 0.0, "lr": 0.02}

    _POINTS = [
        (.1,.5),(.2,.2),(.5,.9),(.9,.5),(.8,.8),
        (.5,.5),(.4,.4),(.6,.6),(.5,.1),(.3,.7),
        (.7,.3),(.2,.6),(.6,.2),(.4,.6),(.6,.4),
        (.3,.3),(.7,.7),(.5,.6),(.6,.5),(.4,.5),
    ]

    def generate_dataset(self):
        return [
            {"x": [x, y],
             "y": [1 if math.sqrt((x-.5)**2+(y-.5)**2) < .32 else 0]}
            for x, y in self._POINTS
        ]


class SpiralFunction(TrainingFunction):
    key         = "spiral"
    label       = "Spiral Classes"
    description = ("<b>Spiral</b>: Two interleaved spirals — classic hard "
                   "problem needing deep non-linear decision boundary.")
    inputs, outputs = 2, 1
    input_labels  = ["x", "y"]
    output_labels = ["Class"]
    is_classification = True
    recommended = {"hidden_layers": 4, "neurons": 12, "activation": "leakyrelu",
                   "optimizer": "adam", "loss": "bce", "dropout": 0.1, "lr": 0.005}

    def generate_dataset(self):
        data = []
        for i in range(20):
            t = i / 20 * 1.8 * math.pi + .3
            r = i / 20 * .4
            data.append({"x": [.5 + r*math.cos(t),   .5 + r*math.sin(t)],   "y": [0]})
            data.append({"x": [.5 + r*math.cos(t+math.pi), .5 + r*math.sin(t+math.pi)], "y": [1]})
        return data


class AutoencoderFunction(TrainingFunction):
    key         = "autoenc"
    label       = "Autoencoder (8→3→8)"
    description = ("<b>Autoencoder</b>: Compress 8-bit identity matrix "
                   "through a 3-neuron bottleneck and reconstruct. "
                   "Tests representational learning.")
    inputs  = 8
    outputs = 8
    input_labels  = [f"i{i}" for i in range(8)]
    output_labels = [f"o{i}" for i in range(8)]
    is_classification = False
    recommended = {"hidden_layers": 2, "neurons": 3, "activation": "sigmoid",
                   "optimizer": "adam", "loss": "mse", "dropout": 0.0, "lr": 0.01}

    def generate_dataset(self):
        return [
            {"x": [1 if j == i else 0 for j in range(8)],
             "y": [1 if j == i else 0 for j in range(8)]}
            for i in range(8)
        ]
