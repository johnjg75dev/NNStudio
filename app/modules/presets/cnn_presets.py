"""
app/modules/presets/cnn_presets.py
CNN architecture presets based on real-world architectures.
"""
from .base_preset import PresetModule


class LeNet5Preset(PresetModule):
    """
    LeNet-5 inspired architecture.
    Classic CNN for digit recognition (Y. LeCun, 1998).
    """
    key = "lenet5"
    label = "LeNet-5 Classic"
    description = (
        "<b>LeNet-5 (1998)</b>: Pioneering CNN architecture for digit recognition.<br>"
        "<code>Conv(6) → Pool → Conv(16) → Pool → FC(120) → FC(84) → Output</code><br>"
        "Historical significance: First successful CNN application. "
        "Used for postal ZIP code reading."
    )
    arch_key = "cnn"
    func_key = "mnist_like"
    layers = [
        {"type": "conv2d", "out_channels": 6, "kernel_size": 5, "activation": "relu"},
        {"type": "maxpool2d", "pool_size": 2},
        {"type": "conv2d", "out_channels": 16, "kernel_size": 5, "activation": "relu"},
        {"type": "maxpool2d", "pool_size": 2},
        {"type": "flatten"},
        {"type": "dense", "neurons": 120, "activation": "relu"},
        {"type": "dense", "neurons": 84, "activation": "relu"},
    ]
    optimizer = "adam"
    loss = "mse"
    lr = 0.001


class SimpleCNNDataset(PresetModule):
    """
    Simple CNN for quick experimentation.
    Minimal architecture for learning CNN basics.
    """
    key = "simple_cnn"
    label = "Simple CNN"
    description = (
        "<b>Simple CNN</b>: Minimal convolutional network for learning.<br>"
        "<code>Conv(16) → Pool → Conv(32) → Pool → FC → Output</code><br>"
        "Good starting point for image classification tasks. "
        "Fast training, easy to understand."
    )
    arch_key = "cnn"
    func_key = "mnist_like"  # Use MNIST-like which has 64 (8x8) spatial input
    layers = [
        {"type": "conv2d", "out_channels": 16, "kernel_size": 3, "padding": 1, "activation": "relu"},
        {"type": "maxpool2d", "pool_size": 2},
        {"type": "conv2d", "out_channels": 32, "kernel_size": 3, "padding": 1, "activation": "relu"},
        {"type": "maxpool2d", "pool_size": 2},
        {"type": "flatten"},
        {"type": "dense", "neurons": 64, "activation": "relu"},
    ]
    optimizer = "adam"
    loss = "mse"
    lr = 0.001


class VGGBlockPreset(PresetModule):
    """
    VGG-style block architecture.
    Deep network with small 3×3 filters (Simonyan & Zisserman, 2014).
    """
    key = "vgg_block"
    label = "VGG-Style Block"
    description = (
        "<b>VGG-Style (2014)</b>: Deep network with uniform 3×3 convolutions.<br>"
        "<code>[Conv(64)×2 → Pool] → [Conv(128)×2 → Pool] → FC</code><br>"
        "Key insight: Depth matters more than filter size. "
        "Won ImageNet 2014 localization task."
    )
    arch_key = "cnn"
    func_key = "mnist_like"
    layers = [
        {"type": "conv2d", "out_channels": 64, "kernel_size": 3, "activation": "relu"},
        {"type": "conv2d", "out_channels": 64, "kernel_size": 3, "activation": "relu"},
        {"type": "maxpool2d", "pool_size": 2},
        {"type": "conv2d", "out_channels": 128, "kernel_size": 3, "activation": "relu"},
        {"type": "conv2d", "out_channels": 128, "kernel_size": 3, "activation": "relu"},
        {"type": "maxpool2d", "pool_size": 2},
        {"type": "flatten"},
        {"type": "dense", "neurons": 256, "activation": "relu"},
        {"type": "dropout", "rate": 0.5},
        {"type": "dense", "neurons": 128, "activation": "relu"},
    ]
    optimizer = "adam"
    loss = "mse"
    lr = 0.0005


class EdgeDetectorPreset(PresetModule):
    """
    CNN for edge detection task.
    Learns Sobel-like filters automatically.
    """
    key = "edge_cnn"
    label = "Edge Detector CNN"
    description = (
        "<b>Edge Detection CNN</b>: Learns edge filters from data.<br>"
        "<code>Conv(8) → Conv(4) → Conv(1)</code><br>"
        "Demonstrates CNN's ability to learn traditional CV filters. "
        "Alternative to hand-crafted Sobel/Canny operators."
    )
    arch_key = "cnn"
    func_key = "edge_detect"
    layers = [
        {"type": "conv2d", "out_channels": 8, "kernel_size": 3, "activation": "relu"},
        {"type": "conv2d", "out_channels": 4, "kernel_size": 3, "activation": "relu"},
        {"type": "conv2d", "out_channels": 1, "kernel_size": 3, "activation": "sigmoid"},
    ]
    optimizer = "adam"
    loss = "mse"
    lr = 0.001


class ResidualBlockPreset(PresetModule):
    """
    Simplified ResNet-style residual block.
    Skip connections for training deeper networks (He et al., 2015).
    """
    key = "resnet_block"
    label = "ResNet-Style Block"
    description = (
        "<b>ResNet-Style (2015)</b>: Residual learning with skip connections.<br>"
        "<code>Conv → BN → ReLU → Conv → BN → (+input) → ReLU</code><br>"
        "Revolutionary: Enabled training of 100+ layer networks. "
        "Won ImageNet 2015 classification."
    )
    arch_key = "cnn"
    func_key = "mnist_like"
    layers = [
        {"type": "conv2d", "out_channels": 32, "kernel_size": 3, "activation": "relu"},
        {"type": "batchnorm"},
        {"type": "conv2d", "out_channels": 32, "kernel_size": 3, "activation": "relu"},
        {"type": "batchnorm"},
        {"type": "maxpool2d", "pool_size": 2},
        {"type": "flatten"},
        {"type": "dense", "neurons": 128, "activation": "relu"},
        {"type": "dropout", "rate": 0.3},
    ]
    optimizer = "adam"
    loss = "mse"
    lr = 0.001
