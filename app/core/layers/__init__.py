from .base import Layer
from .dense import DenseLayer
from .dropout import DropoutLayer
from .batch_norm import BatchNormLayer

LAYER_TYPES = {
    "dense": DenseLayer,
    "dropout": DropoutLayer,
    "batchnorm": BatchNormLayer,
}
