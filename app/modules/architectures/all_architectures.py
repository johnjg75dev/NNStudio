"""app/modules/architectures/all_architectures.py
One file, one class per architecture.  Add new architectures here or in
any new .py file — the registry will find them automatically.
"""
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


class CNNArchitecture(ArchitectureModule):
    key          = "cnn"
    label        = "CNN — Convolutional"
    accent_color = "#f0883e"
    diagram_type = "cnn"
    description  = (
        "<h3>Convolutional Neural Network</h3>"
        "<code>Input → Conv → Pool → Conv → Pool → Flatten → Dense → Output</code><br>"
        "Sliding kernel filters detect local spatial patterns. "
        "Parameter sharing makes CNNs efficient for image data. "
        "Foundation of ResNet, VGG, EfficientNet."
    )


class TransformerArchitecture(ArchitectureModule):
    key          = "transformer"
    label        = "Transformer"
    accent_color = "#39d353"
    diagram_type = "transformer"
    description  = (
        "<h3>Transformer</h3>"
        "<code>Tokens → Embed+PosEnc → [MultiHeadAttn + FFN] × L → Output</code><br>"
        "Self-attention lets every token attend to every other token in parallel. "
        "No recurrence — fully parallelisable. "
        "Foundation of GPT, BERT, T5 and virtually all modern AI."
    )


class ViTArchitecture(ArchitectureModule):
    key          = "vit"
    label        = "ViT — Vision Transformer"
    accent_color = "#d29922"
    diagram_type = "vit"
    description  = (
        "<h3>Vision Transformer (ViT)</h3>"
        "<code>Image → Patches → Linear Embed → [CLS] → Transformer → Head</code><br>"
        "Splits an image into fixed-size patches (e.g. 16×16), "
        "treats each as a token and runs a standard Transformer encoder. "
        "Outperforms CNNs at scale (Dosovitskiy et al., 2020)."
    )


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


class DiffusionArchitecture(ArchitectureModule):
    key          = "diffusion"
    label        = "Diffusion / Stable Diffusion"
    accent_color = "#bc8cff"
    diagram_type = "diffusion"
    description  = (
        "<h3>Diffusion Model</h3>"
        "<code>x₀ → [+noise×T] → xT  then  xT → [U-Net×T] → x₀</code><br>"
        "Forward process: gradually add Gaussian noise over T timesteps. "
        "Reverse: a U-Net learns to predict and remove noise at each step. "
        "Stable Diffusion adds VAE latent space + CLIP text conditioning."
    )


class GANArchitecture(ArchitectureModule):
    key          = "gan"
    label        = "GAN — Generative Adversarial"
    accent_color = "#f0883e"
    diagram_type = "gan"
    description  = (
        "<h3>Generative Adversarial Network</h3>"
        "<code>z → Generator → Fake ←→ Discriminator ← Real</code><br>"
        "Adversarial minimax game: Generator fools Discriminator; "
        "Discriminator gets sharper, forcing Generator to improve. "
        "Produces very sharp outputs but training is notoriously unstable."
    )


class RNNArchitecture(ArchitectureModule):
    key          = "rnn"
    label        = "RNN / LSTM"
    accent_color = "#39d353"
    diagram_type = "rnn"
    description  = (
        "<h3>Recurrent / LSTM Network</h3>"
        "<code>x₁ → [h₁] → [h₂] → … → [hT] → Output</code><br>"
        "Hidden state carries memory across timesteps. "
        "LSTM gates (forget / input / output) solve the vanishing gradient problem "
        "of vanilla RNNs. Best for sequences, time series, language."
    )
