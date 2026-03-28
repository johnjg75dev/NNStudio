from .base_architecture import ArchitectureModule

class DiffusionArchitecture(ArchitectureModule):
    key          = "diffusion"
    label        = "Diffusion / Stable Diffusion"
    accent_color = "#bc8cff"
    diagram_type = "diffusion"
    trainable    = True
    description  = (
        "<h3>Diffusion Model</h3>"
        "<code>x₀ → [+noise×T] → xT  then  xT → [U-Net×T] → x₀</code><br>"
        "Forward process: gradually add Gaussian noise over T timesteps. "
        "Reverse: a U-Net learns to predict and remove noise at each step. "
        "Stable Diffusion adds VAE latent space + CLIP text conditioning.<br><b>Live training enabled.</b>"
    )
