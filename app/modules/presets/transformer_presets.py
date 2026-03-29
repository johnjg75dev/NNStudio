"""
app/modules/presets/transformer_presets.py
Transformer and ViT architecture presets based on real-world architectures.
"""
from .base_preset import PresetModule


class MiniTransformerPreset(PresetModule):
    """
    Miniature Transformer for learning.
    Simplified version of "Attention Is All You Need" (Vaswani et al., 2017).
    """
    key = "mini_transformer"
    label = "Mini Transformer"
    description = (
        "<b>Mini Transformer (2017)</b>: Simplified attention-based architecture.<br>"
        "<code>Embed → PosEnc → Attention → LayerNorm → FC → Output</code><br>"
        "Educational: Core attention mechanism without complexity. "
        "Foundation for GPT, BERT, and modern LLMs."
    )
    arch_key = "transformer"
    func_key = "sentiment"
    layers = [
        {"type": "embedding", "vocab_size": 20, "embed_dim": 32},
        {"type": "positional_encoding", "max_seq_len": 8, "embed_dim": 32},
        {"type": "multihead_attention", "embed_dim": 32, "num_heads": 4},
        {"type": "layernorm"},
        {"type": "dense", "neurons": 64, "activation": "relu"},
        {"type": "layernorm"},
    ]
    optimizer = "adam"
    loss = "bce"
    lr = 0.0005


class TransformerEncoderPreset(PresetModule):
    """
    Transformer encoder block.
    Multi-layer self-attention encoder.
    """
    key = "transformer_enc"
    label = "Transformer Encoder"
    description = (
        "<b>Transformer Encoder</b>: Multi-layer self-attention.<br>"
        "<code>Embed → [Attention + FFN] × 2 → Pool → Output</code><br>"
        "BERT-style encoder. Parallel processing of all tokens. "
        "Bidirectional context understanding."
    )
    arch_key = "transformer"
    func_key = "sentiment"
    layers = [
        {"type": "embedding", "vocab_size": 20, "embed_dim": 64},
        {"type": "positional_encoding", "max_seq_len": 8, "embed_dim": 64},
        {"type": "multihead_attention", "embed_dim": 64, "num_heads": 8},
        {"type": "layernorm"},
        {"type": "dense", "neurons": 128, "activation": "relu"},
        {"type": "dense", "neurons": 64, "activation": "relu"},
        {"type": "layernorm"},
        {"type": "multihead_attention", "embed_dim": 64, "num_heads": 8},
        {"type": "layernorm"},
    ]
    optimizer = "adamw"
    loss = "bce"
    lr = 0.0003


class TinyViTPreset(PresetModule):
    """
    Tiny Vision Transformer.
    Simplified ViT for image classification (Dosovitskiy et al., 2020).
    """
    key = "tiny_vit"
    label = "Tiny ViT"
    description = (
        "<b>Tiny ViT (2020)</b>: Vision Transformer for images.<br>"
        "<code>Patches → Linear → PosEnc → Attention → MLP → Class</code><br>"
        "Revolutionary: Pure attention for vision, no convolutions. "
        "Scales exceptionally well with data and compute."
    )
    arch_key = "vit"
    func_key = "pattern_cls"
    layers = [
        {"type": "dense", "neurons": 64, "activation": "relu"},  # Patch embedding
        {"type": "positional_encoding", "max_seq_len": 4, "embed_dim": 64},
        {"type": "multihead_attention", "embed_dim": 64, "num_heads": 4},
        {"type": "layernorm"},
        {"type": "dense", "neurons": 128, "activation": "gelu"},
        {"type": "layernorm"},
    ]
    optimizer = "adamw"
    loss = "bce"
    lr = 0.0003


class ViTBasePreset(PresetModule):
    """
    Base Vision Transformer architecture.
    Closer to original ViT-B/16 configuration.
    """
    key = "vit_base"
    label = "ViT Base"
    description = (
        "<b>ViT Base</b>: Standard Vision Transformer configuration.<br>"
        "<code>Patch Embed → [MSA + MLP] × 4 → Head</code><br>"
        "Production-ready ViT. Multiple attention heads capture different features. "
        "Requires substantial data for best results."
    )
    arch_key = "vit"
    func_key = "mnist_like"
    layers = [
        {"type": "dense", "neurons": 128, "activation": "gelu"},  # Patch embedding
        {"type": "positional_encoding", "max_seq_len": 16, "embed_dim": 128},
        {"type": "multihead_attention", "embed_dim": 128, "num_heads": 8},
        {"type": "layernorm"},
        {"type": "dense", "neurons": 256, "activation": "gelu"},
        {"type": "dense", "neurons": 128, "activation": "gelu"},
        {"type": "layernorm"},
        {"type": "multihead_attention", "embed_dim": 128, "num_heads": 8},
        {"type": "layernorm"},
        {"type": "dense", "neurons": 256, "activation": "gelu"},
    ]
    optimizer = "adamw"
    loss = "mse"
    lr = 0.0002
    weight_decay = 0.01


class AttentionOnlyPreset(PresetModule):
    """
    Attention-only network (no feed-forward).
    Demonstrates power of self-attention alone.
    """
    key = "attn_only"
    label = "Attention Only"
    description = (
        "<b>Attention-Only</b>: Pure self-attention, no FFN.<br>"
        "<code>Embed → Attention → Attention → Output</code><br>"
        "Research: Shows attention can work without feed-forward layers. "
        "Simpler but less expressive than full Transformer."
    )
    arch_key = "transformer"
    func_key = "next_word"
    layers = [
        {"type": "embedding", "vocab_size": 16, "embed_dim": 32},
        {"type": "positional_encoding", "max_seq_len": 4, "embed_dim": 32},
        {"type": "multihead_attention", "embed_dim": 32, "num_heads": 4},
        {"type": "layernorm"},
        {"type": "multihead_attention", "embed_dim": 32, "num_heads": 4},
        {"type": "layernorm"},
    ]
    optimizer = "adam"
    loss = "mse"
    lr = 0.001


class GPTStylePreset(PresetModule):
    """
    GPT-style decoder architecture.
    Causal language modeling structure.
    """
    key = "gpt_style"
    label = "GPT-Style Decoder"
    description = (
        "<b>GPT-Style</b>: Decoder-only Transformer.<br>"
        "<code>Embed + PosEnc → [Masked Attn + FFN] × 2 → Vocab</code><br>"
        "Autoregressive generation. Foundation of GPT series. "
        "Causal attention prevents looking ahead."
    )
    arch_key = "transformer"
    func_key = "next_word"
    layers = [
        {"type": "embedding", "vocab_size": 16, "embed_dim": 64},
        {"type": "positional_encoding", "max_seq_len": 4, "embed_dim": 64},
        {"type": "multihead_attention", "embed_dim": 64, "num_heads": 4},
        {"type": "layernorm"},
        {"type": "dense", "neurons": 128, "activation": "gelu"},
        {"type": "dense", "neurons": 64, "activation": "gelu"},
        {"type": "layernorm"},
        {"type": "dense", "neurons": 32, "activation": "relu"},
    ]
    optimizer = "adamw"
    loss = "mse"
    lr = 0.0003
    weight_decay = 0.001
