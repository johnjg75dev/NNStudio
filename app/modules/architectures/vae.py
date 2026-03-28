from .base_architecture import ArchitectureModule

class VAEArchitecture(ArchitectureModule):
    key          = "vae"
    label        = "VAE — Variational Autoencoder"
    accent_color = "#f85149"
    diagram_type = "vae"
    description  = (
        "<h3>Variational Autoencoder</h3>"
        "<code>x → Encoder → (μ, σ²) → Sample z → Decoder → x̂</code><br>"
        "Encoder outputs a distribution, not a point. "
        "KL divergence + reconstruction loss keep the latent space smooth. "
        "Enables interpolation and generation by sampling from the prior."
    )
