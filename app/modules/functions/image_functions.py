"""
app/modules/functions/image_functions.py
Image-like training functions for CNN architectures.
"""
import numpy as np
from .base_function import TrainingFunction


class MNISTLikeFunction(TrainingFunction):
    """
    Simulated MNIST-like digit classification.
    Generates simple 8x8 grayscale digit-like patterns.
    """
    key = "mnist_like"
    label = "MNIST-like Digits"
    description = (
        "<b>Synthetic MNIST</b>: 8×8 grayscale digit-like patterns → 10 classes.<br>"
        "Tests CNN ability to learn spatial hierarchies. "
        "Each 'digit' is represented by characteristic pixel patterns."
    )
    inputs = 64  # 8x8 flattened
    outputs = 10  # 10 digit classes
    input_labels = [f"p{i}" for i in range(64)]
    output_labels = [f"d{i}" for i in range(10)]
    is_classification = True
    recommended = {
        "layers": [
            {"type": "conv2d", "out_channels": 16, "kernel_size": 3, "activation": "relu"},
            {"type": "maxpool2d", "pool_size": 2},
            {"type": "conv2d", "out_channels": 32, "kernel_size": 3, "activation": "relu"},
            {"type": "maxpool2d", "pool_size": 2},
            {"type": "flatten"},
            {"type": "dense", "neurons": 64, "activation": "relu"},
            {"type": "dense", "neurons": 10, "activation": "sigmoid"},
        ],
        "optimizer": "adam",
        "loss": "mse",
        "lr": 0.001,
    }

    def generate_dataset(self):
        """Generate synthetic digit-like patterns."""
        np.random.seed(42)
        data = []
        
        # Generate patterns for each digit (0-9)
        for digit in range(10):
            for _ in range(20):  # 20 samples per digit
                # Create base 8x8 pattern
                img = np.zeros((8, 8))
                
                # Add digit-specific features
                if digit == 0:  # Circle-like
                    img[2:6, 2:6] = 0.8
                    img[3:5, 3:5] = 0.2
                elif digit == 1:  # Vertical line
                    img[:, 4] = 0.9
                    img[3:5, 3:5] = 0.7
                elif digit == 2:  # Z-shape
                    img[2, :] = 0.8
                    img[5, :] = 0.8
                    img[2:6, 5] = 0.8
                    img[4:6, 2] = 0.8
                elif digit == 3:  # Two horizontal bars
                    img[2, :] = 0.8
                    img[4, :] = 0.8
                    img[6, :] = 0.8
                    img[2:4, 5] = 0.8
                    img[4:6, 5] = 0.8
                elif digit == 4:  # L-shape with vertical
                    img[:, 2] = 0.8
                    img[5, 2:] = 0.8
                    img[2:5, 5] = 0.8
                elif digit == 5:  # S-shape
                    img[2, :] = 0.8
                    img[4, :] = 0.8
                    img[6, :] = 0.8
                    img[2:4, 2] = 0.8
                    img[4:6, 5] = 0.8
                elif digit == 6:  # Loop at bottom
                    img[2:6, 2] = 0.8
                    img[4:6, 2:6] = 0.8
                    img[6, 2:6] = 0.8
                    img[2, 2:4] = 0.8
                elif digit == 7:  # Top bar with diagonal
                    img[2, :] = 0.8
                    img[2:6, 5] = 0.8
                    img[4:6, 3:5] = 0.6
                elif digit == 8:  # Two loops
                    img[2:6, 2] = 0.8
                    img[2:6, 5] = 0.8
                    img[2, 2:6] = 0.8
                    img[4, 2:6] = 0.8
                    img[6, 2:6] = 0.8
                elif digit == 9:  # Loop at top
                    img[2:4, 2:6] = 0.8
                    img[2:6, 2] = 0.8
                    img[2:6, 5] = 0.8
                    img[4, 2:6] = 0.8
                
                # Add noise
                img += np.random.randn(8, 8) * 0.1
                img = np.clip(img, 0, 1)
                
                # One-hot label
                label = np.zeros(10)
                label[digit] = 1
                
                data.append({
                    "x": img.flatten().tolist(),
                    "y": label.tolist()
                })
        
        return data


class EdgeDetectionFunction(TrainingFunction):
    """
    Edge detection task.
    Input: image with shapes, Output: edge map.
    """
    key = "edge_detect"
    label = "Edge Detection"
    description = (
        "<b>Edge Detection</b>: 8×8 image → edge map.<br>"
        "Tests CNN's ability to learn convolutional filters. "
        "Classic computer vision task demonstrating feature extraction."
    )
    inputs = 64  # 8x8 input
    outputs = 64  # 8x8 edge map
    input_labels = [f"p{i}" for i in range(64)]
    output_labels = [f"e{i}" for i in range(64)]
    is_classification = False
    recommended = {
        "layers": [
            {"type": "conv2d", "out_channels": 8, "kernel_size": 3, "activation": "relu"},
            {"type": "conv2d", "out_channels": 4, "kernel_size": 3, "activation": "relu"},
            {"type": "conv2d", "out_channels": 1, "kernel_size": 3, "activation": "sigmoid"},
        ],
        "optimizer": "adam",
        "loss": "mse",
        "lr": 0.001,
    }

    def generate_dataset(self):
        """Generate images with edges."""
        np.random.seed(42)
        data = []
        
        for _ in range(50):
            # Create random shapes
            img = np.zeros((8, 8))
            
            # Add random rectangles
            for _ in range(np.random.randint(1, 3)):
                x1, y1 = np.random.randint(0, 6, 2)
                x2, y2 = x1 + np.random.randint(2, 4), y1 + np.random.randint(2, 4)
                x2, y2 = min(x2, 8), min(y2, 8)
                img[x1:x2, y1:y2] = 0.8
            
            # Compute edges (simple gradient)
            edges = np.zeros((8, 8))
            for i in range(7):
                for j in range(7):
                    edges[i, j] = abs(img[i+1, j] - img[i, j]) + abs(img[i, j+1] - img[i, j])
            edges = np.clip(edges / 1.6, 0, 1)
            
            data.append({
                "x": img.flatten().tolist(),
                "y": edges.flatten().tolist()
            })
        
        return data


class PatternClassificationFunction(TrainingFunction):
    """
    Simple pattern classification for CNN.
    Classifies geometric patterns in 4x4 grid.
    """
    key = "pattern_cls"
    label = "Pattern Classification"
    description = (
        "<b>Pattern Classification</b>: 4×4 grid → 4 pattern classes.<br>"
        "Lightweight CNN task. Patterns: horizontal, vertical, diagonal, checkerboard."
    )
    inputs = 16  # 4x4 flattened
    outputs = 4  # 4 pattern classes
    input_labels = [f"p{i}" for i in range(16)]
    output_labels = ["horiz", "vert", "diag", "check"]
    is_classification = True
    recommended = {
        "layers": [
            {"type": "conv2d", "out_channels": 8, "kernel_size": 2, "activation": "relu"},
            {"type": "flatten"},
            {"type": "dense", "neurons": 16, "activation": "relu"},
            {"type": "dense", "neurons": 4, "activation": "sigmoid"},
        ],
        "optimizer": "adam",
        "loss": "bce",
        "lr": 0.01,
    }

    def generate_dataset(self):
        """Generate pattern classification dataset."""
        data = []
        
        # Horizontal stripes
        for _ in range(15):
            img = np.zeros((4, 4))
            img[::2, :] = 0.8 + np.random.randn(2, 4) * 0.1
            data.append({"x": np.clip(img, 0, 1).flatten().tolist(), "y": [1, 0, 0, 0]})
        
        # Vertical stripes
        for _ in range(15):
            img = np.zeros((4, 4))
            img[:, ::2] = 0.8 + np.random.randn(4, 2) * 0.1
            data.append({"x": np.clip(img, 0, 1).flatten().tolist(), "y": [0, 1, 0, 0]})
        
        # Diagonal
        for _ in range(15):
            img = np.zeros((4, 4))
            for i in range(4):
                img[i, i] = 0.8 + np.random.randn() * 0.1
            data.append({"x": np.clip(img, 0, 1).flatten().tolist(), "y": [0, 0, 1, 0]})
        
        # Checkerboard
        for _ in range(15):
            img = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    if (i + j) % 2 == 0:
                        img[i, j] = 0.8 + np.random.randn() * 0.1
            data.append({"x": np.clip(img, 0, 1).flatten().tolist(), "y": [0, 0, 0, 1]})
        
        return data
