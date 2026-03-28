"""
tests/test_advanced_layers.py
Unit tests for advanced layer types: Conv2D, Pooling, RNN, LSTM, Transformer layers.
"""
import pytest
import numpy as np
from app.core.layers.conv import Conv2DLayer, MaxPool2DLayer, FlattenLayer
from app.core.layers.rnn import SimpleRNNLayer, LSTMLayer
from app.core.layers.transformer import EmbeddingLayer, LayerNorm, MultiHeadAttention, PositionalEncoding
from app.core.optimizers import Adam
from app.core.activations import ACTIVATIONS


# ═══════════════════════════════════════════════════════════════════════
# Conv2D Layer Tests
# ═══════════════════════════════════════════════════════════════════════
class TestConv2DLayer:
    """Test Conv2DLayer."""

    def test_conv2d_init(self):
        """Should create Conv2D layer with correct parameters."""
        layer = Conv2DLayer(in_channels=3, out_channels=16, kernel_size=3)
        assert layer.in_channels == 3
        assert layer.out_channels == 16
        assert layer.kernel_size == 3
        assert layer.W.shape == (16, 3, 3, 3)

    def test_conv2d_forward_shape(self):
        """Conv2D forward should produce correct output shape."""
        layer = Conv2DLayer(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        
        # 4x4 input, padding=1, kernel=3, stride=1 → 4x4 output
        x = np.random.randn(1, 4, 4).flatten()
        output = layer.forward(x)
        
        expected_size = 4 * 4 * 4  # out_channels * out_h * out_w
        assert output.shape == (expected_size,)

    def test_conv2d_backward_shape(self):
        """Conv2D backward should produce correct gradient shape."""
        layer = Conv2DLayer(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        
        x = np.random.randn(1, 4, 4).flatten()
        layer.forward(x)
        
        delta = np.random.randn(4 * 4 * 4)
        grad = layer.backward(delta)
        
        assert grad.shape == x.shape

    def test_conv2d_update(self):
        """Conv2D weights should update."""
        layer = Conv2DLayer(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        
        x = np.random.randn(1, 4, 4).flatten()
        layer.forward(x)
        
        # Delta should match output shape (4 channels, 4x4 output with padding=1)
        delta = np.random.randn(4, 4, 4)
        layer.backward(delta)
        
        old_W = layer.W.copy()
        optimizer = Adam(lr=0.01)
        layer.update(optimizer, layer_idx=0)
        
        assert not np.array_equal(layer.W, old_W)

    def test_conv2d_serialization(self):
        """Conv2D should serialize correctly."""
        layer = Conv2DLayer(in_channels=3, out_channels=8, kernel_size=5, activation=ACTIVATIONS["relu"])
        d = layer.to_dict()
        
        assert d["type"] == "conv2d"
        assert d["in_channels"] == 3
        assert d["out_channels"] == 8
        assert d["kernel_size"] == 5
        assert d["activation"] == "relu"
        
        restored = Conv2DLayer.from_dict(d)
        assert restored.W.shape == layer.W.shape


# ═══════════════════════════════════════════════════════════════════════
# MaxPool2D Layer Tests
# ═══════════════════════════════════════════════════════════════════════
class TestMaxPool2DLayer:
    """Test MaxPool2DLayer."""

    def test_maxpool_init(self):
        """Should create MaxPool layer with correct parameters."""
        layer = MaxPool2DLayer(pool_size=2, stride=2)
        assert layer.pool_size == 2
        assert layer.stride == 2

    def test_maxpool_forward_shape(self):
        """MaxPool forward should downsample correctly."""
        layer = MaxPool2DLayer(pool_size=2, stride=2)
        
        # 4x4 input → 2x2 output
        x = np.random.randn(1, 4, 4).flatten()
        output = layer.forward(x)
        
        expected_size = 1 * 2 * 2  # channels * out_h * out_w
        assert output.shape == (expected_size,)

    def test_maxpool_backward_shape(self):
        """MaxPool backward should route gradients to max positions."""
        layer = MaxPool2DLayer(pool_size=2, stride=2)
        
        # Provide 3D input directly
        x = np.random.randn(1, 4, 4)
        layer.forward(x)
        
        delta = np.random.randn(1 * 2 * 2)
        grad = layer.backward(delta)
        
        assert grad.shape == x.flatten().shape

    def test_maxpool_no_params(self):
        """MaxPool should have no trainable parameters."""
        layer = MaxPool2DLayer(pool_size=2)
        assert layer.param_count == 0


# ═══════════════════════════════════════════════════════════════════════
# Flatten Layer Tests
# ═══════════════════════════════════════════════════════════════════════
class TestFlattenLayer:
    """Test FlattenLayer."""

    def test_flatten_forward(self):
        """Flatten should convert multi-dim to 1D."""
        layer = FlattenLayer()
        
        x = np.random.randn(3, 4, 4)
        output = layer.forward(x)
        
        assert output.shape == (48,)
        np.testing.assert_array_equal(output, x.flatten())

    def test_flatten_backward(self):
        """Flatten backward should restore original shape."""
        layer = FlattenLayer()
        
        x = np.random.randn(3, 4, 4)
        layer.forward(x)
        
        delta = np.random.randn(48)
        grad = layer.backward(delta)
        
        assert grad.shape == x.shape

    def test_flatten_no_params(self):
        """Flatten should have no parameters."""
        layer = FlattenLayer()
        assert layer.param_count == 0


# ═══════════════════════════════════════════════════════════════════════
# SimpleRNN Layer Tests
# ═══════════════════════════════════════════════════════════════════════
class TestSimpleRNNLayer:
    """Test SimpleRNNLayer."""

    def test_rnn_init(self):
        """Should create RNN layer with correct parameters."""
        layer = SimpleRNNLayer(input_size=10, hidden_size=32)
        assert layer.input_size == 10
        assert layer.hidden_size == 32
        assert layer.W_xh.shape == (32, 10)
        assert layer.W_hh.shape == (32, 32)

    def test_rnn_forward_single_step(self):
        """RNN forward should handle single step input."""
        layer = SimpleRNNLayer(input_size=5, hidden_size=16, return_sequences=False)
        
        x = np.random.randn(5)
        output = layer.forward(x)
        
        assert output.shape == (16,)

    def test_rnn_forward_sequence(self):
        """RNN forward should handle sequence input."""
        layer = SimpleRNNLayer(input_size=5, hidden_size=16, return_sequences=True)
        
        x = np.random.randn(10, 5)  # 10 timesteps
        output = layer.forward(x)
        
        expected_size = 10 * 16
        assert output.shape == (expected_size,)

    def test_rnn_backward(self):
        """RNN backward should compute gradients."""
        layer = SimpleRNNLayer(input_size=5, hidden_size=16, return_sequences=True)
        
        x = np.random.randn(10, 5)
        layer.forward(x)
        
        # Delta should match output shape (10 timesteps * 16 hidden)
        delta = np.random.randn(10 * 16)
        grad = layer.backward(delta)
        
        assert grad.shape == x.flatten().shape

    def test_rnn_serialization(self):
        """RNN should serialize correctly."""
        layer = SimpleRNNLayer(input_size=8, hidden_size=32, activation=ACTIVATIONS["tanh"])
        d = layer.to_dict()
        
        assert d["type"] == "simple_rnn"
        assert d["input_size"] == 8
        assert d["hidden_size"] == 32
        
        restored = SimpleRNNLayer.from_dict(d)
        assert restored.W_xh.shape == layer.W_xh.shape


# ═══════════════════════════════════════════════════════════════════════
# LSTM Layer Tests
# ═══════════════════════════════════════════════════════════════════════
class TestLSTMLayer:
    """Test LSTMLayer."""

    def test_lstm_init(self):
        """Should create LSTM layer with correct parameters."""
        layer = LSTMLayer(input_size=10, hidden_size=32)
        assert layer.input_size == 10
        assert layer.hidden_size == 32
        assert layer.W_f.shape == (32, 42)  # input_size + hidden_size

    def test_lstm_forward_single_step(self):
        """LSTM forward should handle single step."""
        layer = LSTMLayer(input_size=5, hidden_size=16, return_sequences=False)
        
        x = np.random.randn(5)
        output = layer.forward(x)
        
        assert output.shape == (16,)

    def test_lstm_forward_sequence(self):
        """LSTM forward should handle sequence."""
        layer = LSTMLayer(input_size=5, hidden_size=16, return_sequences=True)
        
        x = np.random.randn(10, 5)
        output = layer.forward(x)
        
        expected_size = 10 * 16
        assert output.shape == (expected_size,)

    def test_lstm_param_count(self):
        """LSTM should have 4x the parameters of simple RNN."""
        rnn_layer = SimpleRNNLayer(input_size=10, hidden_size=32)
        lstm_layer = LSTMLayer(input_size=10, hidden_size=32)
        
        # LSTM has 4 gate matrices vs 1 for RNN
        assert lstm_layer.param_count > rnn_layer.param_count

    def test_lstm_serialization(self):
        """LSTM should serialize correctly."""
        layer = LSTMLayer(input_size=8, hidden_size=32)
        d = layer.to_dict()
        
        assert d["type"] == "lstm"
        assert "W_f" in d
        assert "W_i" in d
        assert "W_c" in d
        assert "W_o" in d
        
        restored = LSTMLayer.from_dict(d)
        assert restored.W_f.shape == layer.W_f.shape


# ═══════════════════════════════════════════════════════════════════════
# Embedding Layer Tests
# ═══════════════════════════════════════════════════════════════════════
class TestEmbeddingLayer:
    """Test EmbeddingLayer."""

    def test_embedding_init(self):
        """Should create Embedding layer with correct parameters."""
        layer = EmbeddingLayer(vocab_size=1000, embed_dim=128)
        assert layer.vocab_size == 1000
        assert layer.embed_dim == 128
        assert layer.W.shape == (1000, 128)

    def test_embedding_single_index(self):
        """Embedding should lookup single index."""
        layer = EmbeddingLayer(vocab_size=100, embed_dim=32)
        
        idx = 5
        output = layer.forward(idx)
        
        assert output.shape == (32,)
        np.testing.assert_array_equal(output, layer.W[5])

    def test_embedding_sequence(self):
        """Embedding should handle sequence of indices."""
        layer = EmbeddingLayer(vocab_size=100, embed_dim=32)
        
        indices = [1, 5, 10, 15]
        output = layer.forward(indices)
        
        expected = layer.W[indices].flatten()
        np.testing.assert_array_almost_equal(output, expected)

    def test_embedding_serialization(self):
        """Embedding should serialize correctly."""
        layer = EmbeddingLayer(vocab_size=500, embed_dim=64)
        d = layer.to_dict()
        
        assert d["type"] == "embedding"
        assert d["vocab_size"] == 500
        assert d["embed_dim"] == 64
        
        restored = EmbeddingLayer.from_dict(d)
        np.testing.assert_array_almost_equal(restored.W, layer.W)


# ═══════════════════════════════════════════════════════════════════════
# LayerNorm Tests
# ═══════════════════════════════════════════════════════════════════════
class TestLayerNorm:
    """Test LayerNorm."""

    def test_layernorm_init(self):
        """Should create LayerNorm with correct parameters."""
        layer = LayerNorm(normalized_shape=64)
        assert layer.normalized_shape == 64
        assert layer.gamma.shape == (64,)
        assert layer.beta.shape == (64,)
        assert np.all(layer.gamma == 1)
        assert np.all(layer.beta == 0)

    def test_layernorm_forward_normalizes(self):
        """LayerNorm forward should normalize input."""
        layer = LayerNorm(normalized_shape=32)
        
        # Input with non-zero mean and non-unit variance
        x = np.random.randn(32) * 5 + 10
        output = layer.forward(x)
        
        # Output should be normalized (approximately)
        assert abs(np.mean(output)) < 0.5
        assert abs(np.std(output) - 1) < 0.5

    def test_layernorm_backward(self):
        """LayerNorm backward should compute gradients."""
        layer = LayerNorm(normalized_shape=32)
        
        x = np.random.randn(32)
        layer.forward(x)
        
        delta = np.random.randn(32)
        grad = layer.backward(delta)
        
        assert grad.shape == x.shape
        assert layer._dW is not None
        assert layer._db is not None

    def test_layernorm_serialization(self):
        """LayerNorm should serialize correctly."""
        layer = LayerNorm(normalized_shape=64, eps=1e-5)
        d = layer.to_dict()
        
        assert d["type"] == "layernorm"
        assert d["normalized_shape"] == 64
        assert d["eps"] == 1e-5
        
        restored = LayerNorm.from_dict(d)
        np.testing.assert_array_almost_equal(restored.gamma, layer.gamma)


# ═══════════════════════════════════════════════════════════════════════
# MultiHeadAttention Tests
# ═══════════════════════════════════════════════════════════════════════
class TestMultiHeadAttention:
    """Test MultiHeadAttention."""

    def test_attention_init(self):
        """Should create MultiHeadAttention with correct parameters."""
        layer = MultiHeadAttention(embed_dim=128, num_heads=4)
        assert layer.embed_dim == 128
        assert layer.num_heads == 4
        assert layer.head_dim == 32

    def test_attention_forward_shape(self):
        """Attention forward should preserve embedding dimension."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=4)
        
        x = np.random.randn(64)
        output = layer.forward(x)
        
        assert output.shape == (64,)

    def test_attention_param_count(self):
        """Attention should have 4 projection matrices."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=4)
        
        # 4 matrices of size embed_dim x embed_dim
        expected = 4 * 64 * 64
        assert layer.param_count == expected

    def test_attention_serialization(self):
        """Attention should serialize correctly."""
        layer = MultiHeadAttention(embed_dim=64, num_heads=8)
        d = layer.to_dict()
        
        assert d["type"] == "multihead_attention"
        assert d["embed_dim"] == 64
        assert d["num_heads"] == 8
        assert "W_q" in d
        assert "W_k" in d
        assert "W_v" in d
        assert "W_o" in d


# ═══════════════════════════════════════════════════════════════════════
# PositionalEncoding Tests
# ═══════════════════════════════════════════════════════════════════════
class TestPositionalEncoding:
    """Test PositionalEncoding."""

    def test_posenc_init(self):
        """Should create PositionalEncoding with correct parameters."""
        layer = PositionalEncoding(max_seq_len=512, embed_dim=128)
        assert layer.max_seq_len == 512
        assert layer.embed_dim == 128
        assert layer.pe.shape == (512, 128)

    def test_posenc_forward(self):
        """PositionalEncoding should add position info."""
        layer = PositionalEncoding(max_seq_len=100, embed_dim=64)
        
        x = np.random.randn(64)
        output = layer.forward(x)
        
        assert output.shape == x.shape
        # Output should differ from input (position added)
        assert not np.array_equal(output, x)

    def test_posenc_no_params(self):
        """PositionalEncoding should have no trainable parameters."""
        layer = PositionalEncoding(max_seq_len=512, embed_dim=128)
        assert layer.param_count == 0

    def test_posenc_serialization(self):
        """PositionalEncoding should serialize correctly."""
        layer = PositionalEncoding(max_seq_len=256, embed_dim=64)
        d = layer.to_dict()
        
        assert d["type"] == "positional_encoding"
        assert d["max_seq_len"] == 256
        assert d["embed_dim"] == 64
        
        restored = PositionalEncoding.from_dict(d)
        np.testing.assert_array_almost_equal(restored.pe, layer.pe)


# ═══════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════
class TestLayerIntegration:
    """Integration tests for layer combinations."""

    def test_conv_pool_flatten_sequence(self):
        """Conv → Pool → Flatten should work together."""
        conv = Conv2DLayer(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        pool = MaxPool2DLayer(pool_size=2, stride=2)
        flatten = FlattenLayer()
        
        x = np.random.randn(1, 8, 8).flatten()
        
        h = conv.forward(x)
        h = pool.forward(h)
        output = flatten.forward(h)
        
        assert output.ndim == 1

    def test_embedding_rnn_sequence(self):
        """Embedding → RNN should work together."""
        embed = EmbeddingLayer(vocab_size=100, embed_dim=32)
        rnn = SimpleRNNLayer(input_size=32, hidden_size=64, return_sequences=False)
        
        # Single token
        idx = 5
        h = embed.forward(idx)
        output = rnn.forward(h)
        
        assert output.shape == (64,)

    def test_transformer_block(self):
        """Embedding → PositionalEncoding → LayerNorm → Attention should work."""
        embed = EmbeddingLayer(vocab_size=1000, embed_dim=128)
        posenc = PositionalEncoding(max_seq_len=512, embed_dim=128)
        norm = LayerNorm(normalized_shape=128)
        attention = MultiHeadAttention(embed_dim=128, num_heads=4)
        
        idx = 10
        h = embed.forward(idx)
        h = posenc.forward(h)
        h = norm.forward(h)
        output = attention.forward(h)
        
        assert output.shape == (128,)

    def test_lstm_sequence_classification(self):
        """LSTM for sequence classification."""
        lstm = LSTMLayer(input_size=10, hidden_size=32, return_sequences=False)
        
        # Sequence of 20 timesteps
        x = np.random.randn(20, 10)
        output = lstm.forward(x)
        
        # Single output vector (last hidden state)
        assert output.shape == (32,)