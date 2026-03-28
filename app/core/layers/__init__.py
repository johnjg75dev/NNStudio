from .base import Layer
from .dense import DenseLayer
from .dropout import DropoutLayer
from .batch_norm import BatchNormLayer
from .conv import Conv2DLayer, MaxPool2DLayer, FlattenLayer
from .rnn import SimpleRNNLayer, LSTMLayer
from .transformer import EmbeddingLayer, LayerNorm, MultiHeadAttention, PositionalEncoding

LAYER_TYPES = {
    "dense": DenseLayer,
    "dropout": DropoutLayer,
    "batchnorm": BatchNormLayer,
    "conv2d": Conv2DLayer,
    "maxpool2d": MaxPool2DLayer,
    "flatten": FlattenLayer,
    "simple_rnn": SimpleRNNLayer,
    "lstm": LSTMLayer,
    "embedding": EmbeddingLayer,
    "layernorm": LayerNorm,
    "multihead_attention": MultiHeadAttention,
    "positional_encoding": PositionalEncoding,
}
