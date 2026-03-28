from .base_architecture import ArchitectureModule

class ViTArchitecture(ArchitectureModule):
    key          = "vit"
    label        = "ViT — Vision Transformer"
    accent_color = "#d29922"
    diagram_type = "vit"
    trainable    = True
    description  = (
        "<h3>Vision Transformer (ViT)</h3>"
        "<code>Image → Patches → Linear Embed → [CLS] → Transformer → Head</code><br>"
        "Splits an image into fixed-size patches (e.g. 16×16), "
        "treats each as a token and runs a standard Transformer encoder. "
        "Outperforms CNNs at scale (Dosovitskiy et al., 2020).<br><b>Live training enabled.</b>"
    )
