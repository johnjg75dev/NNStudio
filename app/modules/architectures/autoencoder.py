from .base_architecture import ArchitectureModule

class AutoencoderArchitecture(ArchitectureModule):
    key           = "autoencoder"
    label         = "Autoencoder"
    accent_color  = "#bc8cff"
    diagram_type  = "autoencoder"
    trainable     = True
    is_autoencoder = True
    description   = (
        "<h3>Autoencoder</h3>"
        "<code>Input → Encoder → Latent → Decoder → Reconstruction</code><br>"
        "Bottleneck forces the network to learn a compressed representation. "
        "Trained on reconstruction error (MSE). "
        "Used for compression, denoising, anomaly detection. <b>Live training enabled.</b>"
    )
