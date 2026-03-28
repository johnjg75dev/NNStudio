from .base_architecture import ArchitectureModule

class GANArchitecture(ArchitectureModule):
    key          = "gan"
    label        = "GAN — Generative Adversarial"
    accent_color = "#f0883e"
    diagram_type = "gan"
    trainable    = True
    description  = (
        "<h3>Generative Adversarial Network</h3>"
        "<code>z → Generator → Fake ←→ Discriminator ← Real</code><br>"
        "Adversarial minimax game: Generator fools Discriminator; "
        "Discriminator gets sharper, forcing Generator to improve. "
        "Produces very sharp outputs but training is notoriously unstable.<br><b>Live training enabled.</b>"
    )
