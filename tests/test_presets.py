"""
tests/test_presets.py
Unit tests for preset modules.
"""
import pytest
from app.modules.presets.builtin_presets import (
    TinyXORPreset, ANDGatePreset, SevenSegmentPreset, ParityPreset,
    SineFitPreset, HalfAdderPreset, SpiralPreset, AutoencoderPreset,
    CirclePreset, RegularisedPreset
)
from app.modules.presets.base_preset import PresetModule


class TestPresetModuleBase:
    """Test base preset module structure."""

    def test_base_category(self):
        """Base preset should have 'presets' category."""
        fn = PresetModule()
        assert fn.category == "presets"

    def test_base_defaults(self):
        """Base preset should have sensible defaults."""
        fn = PresetModule()
        assert fn.arch_key == "mlp"
        assert fn.func_key == "xor"
        assert isinstance(fn.layers, list)
        assert fn.optimizer == "adam"
        assert fn.loss == "bce"

    def test_base_to_dict(self):
        """Base preset to_dict should return complete config."""
        fn = PresetModule()
        d = fn.to_dict()
        
        assert "key" in d
        assert "label" in d
        assert "description" in d
        assert "category" in d
        assert "arch_key" in d
        assert "func_key" in d
        assert "layers" in d
        assert "optimizer" in d
        assert "loss" in d


class TestTinyXORPreset:
    """Test Tiny XOR preset."""

    def test_tiny_xor_key(self):
        """Should have correct key."""
        p = TinyXORPreset()
        assert p.key == "preset_tiny_xor"

    def test_tiny_xor_func(self):
        """Should target XOR function."""
        p = TinyXORPreset()
        assert p.func_key == "xor"

    def test_tiny_xor_layers(self):
        """Should have minimal architecture."""
        p = TinyXORPreset()
        assert len(p.layers) == 1
        assert p.layers[0]["neurons"] == 3

    def test_tiny_xor_high_lr(self):
        """Should have high learning rate for quick experiments."""
        p = TinyXORPreset()
        assert p.lr == 0.1


class TestANDGatePreset:
    """Test AND gate preset."""

    def test_and_key(self):
        """Should have correct key."""
        p = ANDGatePreset()
        assert p.key == "preset_and"

    def test_and_func(self):
        """Should target AND function."""
        p = ANDGatePreset()
        assert p.func_key == "and"

    def test_and_layers(self):
        """Should have simple architecture."""
        p = ANDGatePreset()
        assert len(p.layers) == 1
        assert p.layers[0]["neurons"] == 2


class TestSevenSegmentPreset:
    """Test 7-segment preset."""

    def test_seg7_key(self):
        """Should have correct key."""
        p = SevenSegmentPreset()
        assert p.key == "preset_seg7"

    def test_seg7_func(self):
        """Should target 7-segment function."""
        p = SevenSegmentPreset()
        assert p.func_key == "seg7"

    def test_seg7_layers(self):
        """Should have 2 hidden layers."""
        p = SevenSegmentPreset()
        assert len(p.layers) == 2
        assert p.layers[0]["neurons"] == 10
        assert p.layers[1]["neurons"] == 10


class TestParityPreset:
    """Test parity preset."""

    def test_parity_key(self):
        """Should have correct key."""
        p = ParityPreset()
        assert p.key == "preset_parity"

    def test_parity_func(self):
        """Should target parity function."""
        p = ParityPreset()
        assert p.func_key == "parity"

    def test_parity_deep(self):
        """Should have deep architecture for complex pattern."""
        p = ParityPreset()
        assert len(p.layers) == 3
        assert all(l["neurons"] == 8 for l in p.layers)
        assert all(l["activation"] == "relu" for l in p.layers)


class TestSineFitPreset:
    """Test sine fit preset."""

    def test_sine_key(self):
        """Should have correct key."""
        p = SineFitPreset()
        assert p.key == "preset_sine"

    def test_sine_func(self):
        """Should target sine function."""
        p = SineFitPreset()
        assert p.func_key == "sine"

    def test_sine_regression_loss(self):
        """Should use MSE loss for regression."""
        p = SineFitPreset()
        assert p.loss == "mse"


class TestHalfAdderPreset:
    """Test half adder preset."""

    def test_adder_key(self):
        """Should have correct key."""
        p = HalfAdderPreset()
        assert p.key == "preset_adder"

    def test_adder_func(self):
        """Should target half adder function."""
        p = HalfAdderPreset()
        assert p.func_key == "adder"


class TestSpiralPreset:
    """Test spiral preset."""

    def test_spiral_key(self):
        """Should have correct key."""
        p = SpiralPreset()
        assert p.key == "preset_spiral"

    def test_spiral_deep(self):
        """Should have deep architecture."""
        p = SpiralPreset()
        assert len(p.layers) == 4

    def test_spiral_regularization(self):
        """Should have dropout for regularization."""
        p = SpiralPreset()
        assert p.dropout == 0.1

    def test_spiral_leakyrelu(self):
        """Should use LeakyReLU activation."""
        p = SpiralPreset()
        assert all(l["activation"] == "leakyrelu" for l in p.layers)


class TestAutoencoderPreset:
    """Test autoencoder preset."""

    def test_autoenc_key(self):
        """Should have correct key."""
        p = AutoencoderPreset()
        assert p.key == "preset_autoenc"

    def test_autoenc_arch(self):
        """Should use autoencoder architecture."""
        p = AutoencoderPreset()
        assert p.arch_key == "autoencoder"
        assert p.func_key == "autoenc"

    def test_autoenc_bottleneck(self):
        """Should have 3-neuron bottleneck."""
        p = AutoencoderPreset()
        assert len(p.layers) == 2
        assert all(l["neurons"] == 3 for l in p.layers)


class TestCirclePreset:
    """Test circle boundary preset."""

    def test_circle_key(self):
        """Should have correct key."""
        p = CirclePreset()
        assert p.key == "preset_circle"

    def test_circle_func(self):
        """Should target circle function."""
        p = CirclePreset()
        assert p.func_key == "circle"


class TestRegularisedPreset:
    """Test regularised deep preset."""

    def test_regularised_key(self):
        """Should have correct key."""
        p = RegularisedPreset()
        assert p.key == "preset_regularised"

    def test_regularised_very_deep(self):
        """Should have very deep architecture."""
        p = RegularisedPreset()
        assert len(p.layers) == 5
        assert all(l["neurons"] == 16 for l in p.layers)

    def test_regularised_adamw(self):
        """Should use AdamW optimizer."""
        p = RegularisedPreset()
        assert p.optimizer == "adamw"

    def test_regularised_gelu(self):
        """Should use GELU activation."""
        p = RegularisedPreset()
        assert all(l["activation"] == "gelu" for l in p.layers)

    def test_regularised_strong_regularization(self):
        """Should have strong regularization."""
        p = RegularisedPreset()
        assert p.dropout == 0.2
        assert p.weight_decay == 0.01


class TestAllPresets:
    """Test all preset modules."""

    @pytest.mark.parametrize("preset_class", [
        TinyXORPreset, ANDGatePreset, SevenSegmentPreset, ParityPreset,
        SineFitPreset, HalfAdderPreset, SpiralPreset, AutoencoderPreset,
        CirclePreset, RegularisedPreset
    ])
    def test_preset_to_dict(self, preset_class):
        """to_dict should return complete config."""
        p = preset_class()
        d = p.to_dict()
        
        assert "key" in d
        assert "label" in d
        assert "description" in d
        assert "arch_key" in d
        assert "func_key" in d
        assert "layers" in d
        assert "optimizer" in d
        assert "loss" in d
        assert "lr" in d

    @pytest.mark.parametrize("preset_class", [
        TinyXORPreset, ANDGatePreset, SevenSegmentPreset, ParityPreset,
        SineFitPreset, HalfAdderPreset, SpiralPreset, AutoencoderPreset,
        CirclePreset, RegularisedPreset
    ])
    def test_preset_layers_structure(self, preset_class):
        """Layers should have correct structure."""
        p = preset_class()
        for layer in p.layers:
            assert "neurons" in layer
            assert "activation" in layer
            assert "type" in layer
            assert layer["type"] == "dense"
