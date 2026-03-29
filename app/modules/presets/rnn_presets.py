"""
app/modules/presets/rnn_presets.py
RNN/LSTM architecture presets based on real-world architectures.
"""
from .base_preset import PresetModule


class SimpleRNNPreset(PresetModule):
    """
    Simple RNN for sequence tasks.
    Basic recurrent architecture for learning temporal patterns.
    """
    key = "simple_rnn"
    label = "Simple RNN"
    description = (
        "<b>Simple RNN</b>: Basic recurrent neural network.<br>"
        "<code>Input → RNN(32) → FC → Output</code><br>"
        "Educational: Demonstrates recurrent connections. "
        "Limited by vanishing gradients on long sequences."
    )
    arch_key = "rnn"
    func_key = "seq_predict"
    layers = [
        {"type": "simple_rnn", "hidden_size": 32},
        {"type": "dense", "neurons": 16, "activation": "relu"},
    ]
    optimizer = "adam"
    loss = "mse"
    lr = 0.001


class LSTMPreset(PresetModule):
    """
    LSTM for sequence modeling.
    Long Short-Term Memory networks (Hochreiter & Schmidhuber, 1997).
    """
    key = "lstm_standard"
    label = "Standard LSTM"
    description = (
        "<b>LSTM (1997)</b>: Long Short-Term Memory network.<br>"
        "<code>Input → LSTM(64) → FC → Output</code><br>"
        "Solves vanishing gradient problem with gating mechanism. "
        "Industry standard for sequence tasks before Transformers."
    )
    arch_key = "rnn"
    func_key = "seq_predict"
    layers = [
        {"type": "lstm", "hidden_size": 64},
        {"type": "dense", "neurons": 32, "activation": "relu"},
        {"type": "dropout", "rate": 0.2},
    ]
    optimizer = "adam"
    loss = "mse"
    lr = 0.001


class BidirectionalLSTMPreset(PresetModule):
    """
    Bidirectional LSTM for context-aware processing.
    Processes sequence in both directions.
    """
    key = "bilstm"
    label = "Bidirectional LSTM"
    description = (
        "<b>Bidirectional LSTM</b>: Processes sequence forwards and backwards.<br>"
        "<code>Input → LSTM(32 fwd) + LSTM(32 bwd) → FC → Output</code><br>"
        "Captures both past and future context. "
        "Common in NLP tasks like NER and sentiment analysis."
    )
    arch_key = "rnn"
    func_key = "sentiment"
    layers = [
        {"type": "embedding", "vocab_size": 20, "embed_dim": 16},
        {"type": "lstm", "hidden_size": 32},
        {"type": "lstm", "hidden_size": 32},
        {"type": "dense", "neurons": 16, "activation": "relu"},
    ]
    optimizer = "adam"
    loss = "bce"
    lr = 0.001


class DeepLSTMPreset(PresetModule):
    """
    Deep stacked LSTM architecture.
    Multiple LSTM layers for hierarchical feature learning.
    """
    key = "deep_lstm"
    label = "Deep Stacked LSTM"
    description = (
        "<b>Deep LSTM</b>: Multiple stacked LSTM layers.<br>"
        "<code>Input → LSTM(64) → LSTM(64) → LSTM(32) → FC → Output</code><br>"
        "Hierarchical temporal feature extraction. "
        "Used in speech recognition and machine translation."
    )
    arch_key = "rnn"
    func_key = "ts_classify"
    layers = [
        {"type": "lstm", "hidden_size": 64, "return_sequences": True},
        {"type": "lstm", "hidden_size": 64, "return_sequences": True},
        {"type": "lstm", "hidden_size": 32},
        {"type": "dense", "neurons": 64, "activation": "relu"},
        {"type": "dropout", "rate": 0.3},
        {"type": "dense", "neurons": 32, "activation": "relu"},
    ]
    optimizer = "adam"
    loss = "bce"
    lr = 0.0005


class LanguageModelPreset(PresetModule):
    """
    LSTM language model for next word prediction.
    Classic neural language modeling architecture.
    """
    key = "lstm_lm"
    label = "LSTM Language Model"
    description = (
        "<b>LSTM Language Model</b>: Predicts next word in sequence.<br>"
        "<code>Embedding → LSTM(128) → FC → Vocab</code><br>"
        "Foundation of modern text generation. "
        "Predecessor to Transformer-based models like GPT."
    )
    arch_key = "rnn"
    func_key = "next_word"
    layers = [
        {"type": "embedding", "vocab_size": 16, "embed_dim": 32},
        {"type": "lstm", "hidden_size": 128},
        {"type": "dense", "neurons": 64, "activation": "relu"},
        {"type": "dropout", "rate": 0.3},
        {"type": "dense", "neurons": 32, "activation": "relu"},
    ]
    optimizer = "adam"
    loss = "mse"
    lr = 0.001


class SequenceToSequencePreset(PresetModule):
    """
    Simplified seq2seq architecture.
    Encoder-decoder structure for sequence transformation.
    """
    key = "seq2seq"
    label = "Seq2Seq Encoder"
    description = (
        "<b>Seq2Seq Encoder</b>: Encodes input sequence to fixed representation.<br>"
        "<code>Embedding → LSTM(64) → LSTM(64) → Context</code><br>"
        "Building block for translation and summarization. "
        "Decoder mirrors encoder structure."
    )
    arch_key = "rnn"
    func_key = "sentiment"
    layers = [
        {"type": "embedding", "vocab_size": 20, "embed_dim": 24},
        {"type": "lstm", "hidden_size": 64},
        {"type": "lstm", "hidden_size": 64},
        {"type": "dense", "neurons": 32, "activation": "relu"},
    ]
    optimizer = "adam"
    loss = "bce"
    lr = 0.001
