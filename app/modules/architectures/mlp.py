from .base_architecture import ArchitectureModule

class MLPArchitecture(ArchitectureModule):
    key          = "mlp"
    label        = "MLP — Fully Connected"
    accent_color = "#58a6ff"
    diagram_type = "mlp"
    trainable    = True
    description  = (
        "<h3>Multi-Layer Perceptron</h3>"
        "<code>Input → [Dense → Activation] × N → Output</code><br>"
        "Every neuron connects to every neuron in the next layer (fully connected). "
        "Trains via backpropagation. Best for tabular data, logic gates, "
        "function approximation. <b>Live training enabled.</b>"
    )
