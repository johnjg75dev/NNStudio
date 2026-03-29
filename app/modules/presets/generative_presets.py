"""
app/modules/presets/generative_presets.py
VAE, GAN, and Diffusion architecture presets.
"""
from .base_preset import PresetModule


class SimpleVAEPreset(PresetModule):
    """
    Basic Variational Autoencoder.
    Encoder-decoder with latent space regularization (Kingma & Welling, 2013).
    """
    key = "simple_vae"
    label = "Simple VAE"
    description = (
        "<b>VAE (2013)</b>: Variational Autoencoder for generation.<br>"
        "<code>Input → Enc → μ,σ → Sample z → Dec → Output</code><br>"
        "Probabilistic: Learns smooth latent space. "
        "Enables interpolation and sampling. Loss = Reconstruction + KL."
    )
    arch_key = "vae"
    func_key = "autoenc"
    layers = [
        {"type": "dense", "neurons": 16, "activation": "relu"},  # Encoder
        {"type": "dense", "neurons": 8, "activation": "relu"},   # μ
        {"type": "dense", "neurons": 8, "activation": "relu"},   # σ
        {"type": "dense", "neurons": 16, "activation": "relu"},  # Decoder
        {"type": "dense", "neurons": 8, "activation": "sigmoid"},
    ]
    optimizer = "adam"
    loss = "mse"
    lr = 0.001


class DeepVAEPreset(PresetModule):
    """
    Deep Variational Autoencoder.
    Multi-layer VAE for complex data.
    """
    key = "deep_vae"
    label = "Deep VAE"
    description = (
        "<b>Deep VAE</b>: Multi-layer variational autoencoder.<br>"
        "<code>Input → [FC×3] → Latent(16) → [FC×3] → Output</code><br>"
        "Deeper architecture for complex patterns. "
        "Better reconstruction quality, smoother latent space."
    )
    arch_key = "vae"
    func_key = "autoenc"
    layers = [
        {"type": "dense", "neurons": 32, "activation": "relu"},
        {"type": "dense", "neurons": 24, "activation": "relu"},
        {"type": "dense", "neurons": 16, "activation": "relu"},  # Latent
        {"type": "dense", "neurons": 24, "activation": "relu"},
        {"type": "dense", "neurons": 32, "activation": "relu"},
        {"type": "dense", "neurons": 8, "activation": "sigmoid"},
    ]
    optimizer = "adam"
    loss = "mse"
    lr = 0.0005


class SimpleGANPreset(PresetModule):
    """
    Basic GAN architecture.
    Generator vs Discriminator adversarial training (Goodfellow et al., 2014).
    """
    key = "simple_gan"
    label = "Simple GAN"
    description = (
        "<b>GAN (2014)</b>: Generative Adversarial Network.<br>"
        "<code>z → Generator → Fake | Real → Discriminator → Real/Fake</code><br>"
        "Adversarial: Generator creates, Discriminator judges. "
        "Nash equilibrium produces realistic samples."
    )
    arch_key = "gan"
    func_key = "autoenc"
    layers = [
        # Generator
        {"type": "dense", "neurons": 32, "activation": "relu"},
        {"type": "dense", "neurons": 16, "activation": "relu"},
        {"type": "dense", "neurons": 8, "activation": "sigmoid"},
    ]
    optimizer = "adam"
    loss = "bce"
    lr = 0.0002


class DCGANPreset(PresetModule):
    """
    DCGAN-style architecture.
    Deep Convolutional GAN (Radford et al., 2015).
    """
    key = "dcgan"
    label = "DCGAN Style"
    description = (
        "<b>DCGAN (2015)</b>: Deep Convolutional GAN.<br>"
        "<code>z → ConvTranspose×3 → Image | Image → Conv×3 → Real/Fake</code><br>"
        "Convolutional: Stable GAN training with convolutions. "
        "Foundation for modern image generation."
    )
    arch_key = "gan"
    func_key = "mnist_like"
    layers = [
        # Generator (simplified)
        {"type": "dense", "neurons": 64, "activation": "relu"},
        {"type": "dense", "neurons": 128, "activation": "relu"},
        {"type": "dense", "neurons": 64, "activation": "relu"},
        {"type": "dense", "neurons": 64, "activation": "sigmoid"},
    ]
    optimizer = "adam"
    loss = "bce"
    lr = 0.0002
    weight_decay = 0.0001


class WGANPreset(PresetModule):
    """
    WGAN-style architecture.
    Wasserstein GAN with improved training stability (Arjovsky et al., 2017).
    """
    key = "wgan"
    label = "WGAN Style"
    description = (
        "<b>WGAN (2017)</b>: Wasserstein GAN.<br>"
        "<code>Uses Earth-Mover distance instead of JS divergence</code><br>"
        "Stable: Solves mode collapse and vanishing gradients. "
        "Requires weight clipping or gradient penalty."
    )
    arch_key = "gan"
    func_key = "autoenc"
    layers = [
        {"type": "dense", "neurons": 64, "activation": "leakyrelu"},
        {"type": "dense", "neurons": 32, "activation": "leakyrelu"},
        {"type": "dense", "neurons": 16, "activation": "leakyrelu"},
        {"type": "dense", "neurons": 8, "activation": "linear"},
    ]
    optimizer = "rmsprop"
    loss = "mse"
    lr = 0.00005


class DiffusionPreset(PresetModule):
    """
    Simplified diffusion model.
    Denoising diffusion probabilistic model (Ho et al., 2020).
    """
    key = "diffusion_simple"
    label = "Simple Diffusion"
    description = (
        "<b>Diffusion (2020)</b>: Denoising Diffusion Probabilistic Model.<br>"
        "<code>Noisy Input → U-Net → Predict Noise → Denoise</code><br>"
        "Iterative: Gradually removes noise over timesteps. "
        "State-of-the-art image generation (DALL-E 2, Stable Diffusion)."
    )
    arch_key = "diffusion"
    func_key = "mnist_like"
    layers = [
        {"type": "dense", "neurons": 128, "activation": "relu"},
        {"type": "dense", "neurons": 128, "activation": "relu"},
        {"type": "dense", "neurons": 64, "activation": "relu"},
        {"type": "dense", "neurons": 64, "activation": "relu"},
        {"type": "dense", "neurons": 10, "activation": "sigmoid"},
    ]
    optimizer = "adam"
    loss = "mse"
    lr = 0.0001


class AutoencoderDenoisingPreset(PresetModule):
    """
    Denoising Autoencoder.
    Learns robust representations by reconstructing clean from noisy input.
    """
    key = "denoising_ae"
    label = "Denoising Autoencoder"
    description = (
        "<b>Denoising AE</b>: Robust representation learning.<br>"
        "<code>Noisy Input → Encoder → Latent → Decoder → Clean Output</code><br>"
        "Robust: Forces network to learn meaningful features. "
        "Used for image denoising and pre-training."
    )
    arch_key = "autoencoder"
    func_key = "autoenc"
    layers = [
        {"type": "dense", "neurons": 16, "activation": "relu"},
        {"type": "dense", "neurons": 8, "activation": "relu"},
        {"type": "dense", "neurons": 4, "activation": "relu"},  # Bottleneck
        {"type": "dense", "neurons": 8, "activation": "relu"},
        {"type": "dense", "neurons": 16, "activation": "relu"},
        {"type": "dense", "neurons": 8, "activation": "sigmoid"},
    ]
    optimizer = "adam"
    loss = "mse"
    lr = 0.001


class CompressionAEPreset(PresetModule):
    """
    Compression-focused Autoencoder.
    Extreme bottleneck for dimensionality reduction.
    """
    key = "compress_ae"
    label = "Compression AE"
    description = (
        "<b>Compression AE</b>: Extreme dimensionality reduction.<br>"
        "<code>8 → 4 → 2 → 1 → 2 → 4 → 8</code><br>"
        "Efficient: Maximum compression with minimal information loss. "
        "Tests network's ability to find optimal encoding."
    )
    arch_key = "autoencoder"
    func_key = "autoenc"
    layers = [
        {"type": "dense", "neurons": 4, "activation": "relu"},
        {"type": "dense", "neurons": 2, "activation": "relu"},
        {"type": "dense", "neurons": 1, "activation": "relu"},  # 8:1 compression!
        {"type": "dense", "neurons": 2, "activation": "relu"},
        {"type": "dense", "neurons": 4, "activation": "relu"},
        {"type": "dense", "neurons": 8, "activation": "sigmoid"},
    ]
    optimizer = "adam"
    loss = "mse"
    lr = 0.001
