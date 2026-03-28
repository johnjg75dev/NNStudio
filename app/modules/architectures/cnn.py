from .base_architecture import ArchitectureModule

class CNNArchitecture(ArchitectureModule):
    key          = "cnn"
    label        = "CNN — Convolutional"
    accent_color = "#f0883e"
    diagram_type = "cnn"
    trainable    = True
    description  = (
        "<h3>Convolutional Neural Network</h3>"
        "<code>Input → Conv2D → MaxPool → Flatten → Dense → Output</code><br>"
        "Sliding kernel filters detect local spatial patterns. "
        "Parameter sharing makes CNNs efficient for image data. "
        "Foundation of ResNet, VGG, EfficientNet.<br><b>Live training enabled.</b>"
    )
