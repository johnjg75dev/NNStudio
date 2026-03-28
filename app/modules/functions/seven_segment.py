"""app/modules/functions/seven_segment.py"""
from .base_function import TrainingFunction


class SevenSegmentFunction(TrainingFunction):
    key         = "seg7"
    label       = "7-Segment Display"
    description = ("<b>7-Segment</b>: 4-bit binary input (digits 0–9) → "
                   "7 segment outputs a–g. Tests multi-output classification.")
    inputs      = 4
    outputs     = 7
    input_labels  = ["b3", "b2", "b1", "b0"]
    output_labels = ["a", "b", "c", "d", "e", "f", "g"]
    is_classification = True
    recommended = {"hidden_layers": 2, "neurons": 10, "activation": "tanh",
                   "optimizer": "adam", "loss": "mse", "dropout": 0.0, "lr": 0.02}

    # segments[digit] = [a,b,c,d,e,f,g]
    _SEGMENTS = [
        [1,1,1,1,1,1,0],  # 0
        [0,1,1,0,0,0,0],  # 1
        [1,1,0,1,1,0,1],  # 2
        [1,1,1,1,0,0,1],  # 3
        [0,1,1,0,0,1,1],  # 4
        [1,0,1,1,0,1,1],  # 5
        [1,0,1,1,1,1,1],  # 6
        [1,1,1,0,0,0,0],  # 7
        [1,1,1,1,1,1,1],  # 8
        [1,1,1,1,0,1,1],  # 9
    ]

    def generate_dataset(self):
        return [
            {
                "x": [(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1],
                "y": self._SEGMENTS[i],
            }
            for i in range(10)
        ]
