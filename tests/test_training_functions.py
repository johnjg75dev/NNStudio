"""
tests/test_training_functions.py
Unit tests for training function modules (XOR, 7-segment, etc.).
"""
import pytest
import numpy as np
from app.modules.functions.xor import XORFunction
from app.modules.functions.seven_segment import SevenSegmentFunction
from app.modules.functions.math_functions import ParityFunction, SineFunction, HalfAdderFunction
from app.modules.functions.logic_gates import ANDFunction, ORFunction, XNORFunction
from app.modules.functions.geometric import CircleFunction, SpiralFunction, AutoencoderFunction


class TestXORFunction:
    """Test XOR training function."""

    def test_xor_key(self):
        """XOR function should have correct key."""
        fn = XORFunction()
        assert fn.key == "xor"

    def test_xor_inputs_outputs(self):
        """XOR should have 2 inputs and 1 output."""
        fn = XORFunction()
        assert fn.inputs == 2
        assert fn.outputs == 1

    def test_xor_dataset_size(self):
        """XOR dataset should have 4 samples."""
        fn = XORFunction()
        dataset = fn.generate_dataset()
        assert len(dataset) == 4

    def test_xor_dataset_correctness(self):
        """XOR dataset should have correct labels."""
        fn = XORFunction()
        dataset = fn.generate_dataset()
        
        expected = {
            (0, 0): [0],
            (0, 1): [1],
            (1, 0): [1],
            (1, 1): [0],
        }
        
        for sample in dataset:
            x = tuple(sample["x"])
            assert sample["y"] == expected[x]

    def test_xor_recommended_config(self):
        """XOR should have recommended config."""
        fn = XORFunction()
        rec = fn.recommended
        assert "layers" in rec
        assert "optimizer" in rec
        assert "loss" in rec


class TestSevenSegmentFunction:
    """Test 7-segment display training function."""

    def test_seg7_key(self):
        """7-segment function should have correct key."""
        fn = SevenSegmentFunction()
        assert fn.key == "seg7"

    def test_seg7_inputs_outputs(self):
        """7-segment should have 4 inputs and 7 outputs."""
        fn = SevenSegmentFunction()
        assert fn.inputs == 4
        assert fn.outputs == 7

    def test_seg7_dataset_size(self):
        """7-segment dataset should have 16 samples (hex digits 0-F)."""
        fn = SevenSegmentFunction()
        dataset = fn.generate_dataset()
        assert len(dataset) == 16

    def test_seg7_input_range(self):
        """7-segment inputs should be 4-bit binary."""
        fn = SevenSegmentFunction()
        dataset = fn.generate_dataset()
        
        for sample in dataset:
            assert len(sample["x"]) == 4
            assert all(b in [0, 1] for b in sample["x"])

    def test_seg7_output_range(self):
        """7-segment outputs should be 7-bit binary."""
        fn = SevenSegmentFunction()
        dataset = fn.generate_dataset()
        
        for sample in dataset:
            assert len(sample["y"]) == 7
            assert all(b in [0, 1] for b in sample["y"])


class TestParityFunction:
    """Test parity training function."""

    def test_parity_key(self):
        """Parity function should have correct key."""
        fn = ParityFunction()
        assert fn.key == "parity"

    def test_parity_inputs_outputs(self):
        """Parity should have 4 inputs and 1 output."""
        fn = ParityFunction()
        assert fn.inputs == 4
        assert fn.outputs == 1

    def test_parity_dataset_size(self):
        """Parity dataset should have 16 samples (2^4)."""
        fn = ParityFunction()
        dataset = fn.generate_dataset()
        assert len(dataset) == 16

    def test_parity_correctness(self):
        """Parity output should be 1 for odd number of 1s."""
        fn = ParityFunction()
        dataset = fn.generate_dataset()
        
        for sample in dataset:
            expected = sum(sample["x"]) % 2
            assert sample["y"][0] == expected


class TestSineFunction:
    """Test sine approximation training function."""

    def test_sine_key(self):
        """Sine function should have correct key."""
        fn = SineFunction()
        assert fn.key == "sine"

    def test_sine_inputs_outputs(self):
        """Sine should have 1 input and 1 output."""
        fn = SineFunction()
        assert fn.inputs == 1
        assert fn.outputs == 1

    def test_sine_dataset_size(self):
        """Sine dataset should have 20 samples."""
        fn = SineFunction()
        dataset = fn.generate_dataset()
        assert len(dataset) == 20

    def test_sine_output_range(self):
        """Sine outputs should be in [0, 1] (scaled)."""
        fn = SineFunction()
        dataset = fn.generate_dataset()
        
        for sample in dataset:
            assert 0 <= sample["y"][0] <= 1

    def test_sine_is_regression(self):
        """Sine should be marked as regression task."""
        fn = SineFunction()
        assert fn.is_classification is False


class TestANDFunction:
    """Test AND gate training function."""

    def test_and_key(self):
        """AND function should have correct key."""
        fn = ANDFunction()
        assert fn.key == "and"

    def test_and_dataset_correctness(self):
        """AND dataset should have correct labels."""
        fn = ANDFunction()
        dataset = fn.generate_dataset()
        
        expected = {
            (0, 0): [0],
            (0, 1): [0],
            (1, 0): [0],
            (1, 1): [1],
        }
        
        for sample in dataset:
            x = tuple(sample["x"])
            assert sample["y"] == expected[x]


class TestORFunction:
    """Test OR gate training function."""

    def test_or_key(self):
        """OR function should have correct key."""
        fn = ORFunction()
        assert fn.key == "or"

    def test_or_dataset_correctness(self):
        """OR dataset should have correct labels."""
        fn = ORFunction()
        dataset = fn.generate_dataset()
        
        expected = {
            (0, 0): [0],
            (0, 1): [1],
            (1, 0): [1],
            (1, 1): [1],
        }
        
        for sample in dataset:
            x = tuple(sample["x"])
            assert sample["y"] == expected[x]


class TestXNORFunction:
    """Test XNOR gate training function."""

    def test_xnor_key(self):
        """XNOR function should have correct key."""
        fn = XNORFunction()
        assert fn.key == "xnor"

    def test_xnor_dataset_correctness(self):
        """XNOR dataset should have correct labels."""
        fn = XNORFunction()
        dataset = fn.generate_dataset()
        
        expected = {
            (0, 0): [1],
            (0, 1): [0],
            (1, 0): [0],
            (1, 1): [1],
        }
        
        for sample in dataset:
            x = tuple(sample["x"])
            assert sample["y"] == expected[x]


class TestHalfAdderFunction:
    """Test half adder training function."""

    def test_adder_key(self):
        """Half adder function should have correct key."""
        fn = HalfAdderFunction()
        assert fn.key == "adder"

    def test_adder_inputs_outputs(self):
        """Half adder should have 2 inputs and 2 outputs."""
        fn = HalfAdderFunction()
        assert fn.inputs == 2
        assert fn.outputs == 2

    def test_adder_dataset_correctness(self):
        """Half adder should compute Sum (XOR) and Carry (AND)."""
        fn = HalfAdderFunction()
        dataset = fn.generate_dataset()
        
        expected = {
            (0, 0): [0, 0],  # 0+0=0, carry=0
            (0, 1): [1, 0],  # 0+1=1, carry=0
            (1, 0): [1, 0],  # 1+0=1, carry=0
            (1, 1): [0, 1],  # 1+1=0, carry=1
        }
        
        for sample in dataset:
            x = tuple(sample["x"])
            assert sample["y"] == expected[x]


class TestCircleFunction:
    """Test circle boundary training function."""

    def test_circle_key(self):
        """Circle function should have correct key."""
        fn = CircleFunction()
        assert fn.key == "circle"

    def test_circle_inputs_outputs(self):
        """Circle should have 2 inputs and 1 output."""
        fn = CircleFunction()
        assert fn.inputs == 2
        assert fn.outputs == 1

    def test_circle_dataset_size(self):
        """Circle dataset should have 20 samples."""
        fn = CircleFunction()
        dataset = fn.generate_dataset()
        assert len(dataset) == 20

    def test_circle_is_classification(self):
        """Circle should be classification task."""
        fn = CircleFunction()
        assert fn.is_classification is True


class TestSpiralFunction:
    """Test spiral classification training function."""

    def test_spiral_key(self):
        """Spiral function should have correct key."""
        fn = SpiralFunction()
        assert fn.key == "spiral"

    def test_spiral_dataset_size(self):
        """Spiral dataset should have 40 samples (20 per class)."""
        fn = SpiralFunction()
        dataset = fn.generate_dataset()
        assert len(dataset) == 40

    def test_spiral_classes_balanced(self):
        """Spiral should have balanced classes."""
        fn = SpiralFunction()
        dataset = fn.generate_dataset()
        
        class_0 = sum(1 for s in dataset if s["y"][0] == 0)
        class_1 = sum(1 for s in dataset if s["y"][0] == 1)
        
        assert class_0 == class_1 == 20


class TestAutoencoderFunction:
    """Test autoencoder training function."""

    def test_autoenc_key(self):
        """Autoencoder function should have correct key."""
        fn = AutoencoderFunction()
        assert fn.key == "autoenc"

    def test_autoenc_inputs_outputs(self):
        """Autoencoder should have 8 inputs and 8 outputs."""
        fn = AutoencoderFunction()
        assert fn.inputs == 8
        assert fn.outputs == 8

    def test_autoenc_dataset_size(self):
        """Autoencoder dataset should have 8 samples (identity matrix)."""
        fn = AutoencoderFunction()
        dataset = fn.generate_dataset()
        assert len(dataset) == 8

    def test_autoenc_dataset_structure(self):
        """Autoencoder should have identity matrix patterns."""
        fn = AutoencoderFunction()
        dataset = fn.generate_dataset()
        
        for i, sample in enumerate(dataset):
            expected = [1 if j == i else 0 for j in range(8)]
            assert sample["x"] == expected
            assert sample["y"] == expected

    def test_autoenc_is_regression(self):
        """Autoencoder should be regression task."""
        fn = AutoencoderFunction()
        assert fn.is_classification is False


class TestTrainingFunctionRegistry:
    """Test training function module structure."""

    @pytest.mark.parametrize("fn_class", [
        XORFunction, ANDFunction, ORFunction, XNORFunction,
        ParityFunction, SineFunction, HalfAdderFunction,
        SevenSegmentFunction, CircleFunction, SpiralFunction, AutoencoderFunction
    ])
    def test_function_structure(self, fn_class):
        """All functions should have required attributes."""
        fn = fn_class()
        
        assert hasattr(fn, "key")
        assert hasattr(fn, "label")
        assert hasattr(fn, "description")
        assert hasattr(fn, "inputs")
        assert hasattr(fn, "outputs")
        assert hasattr(fn, "generate_dataset")
        assert hasattr(fn, "to_dict")

    @pytest.mark.parametrize("fn_class", [
        XORFunction, ANDFunction, ORFunction, XNORFunction,
        ParityFunction, SineFunction, HalfAdderFunction,
        SevenSegmentFunction, CircleFunction, SpiralFunction, AutoencoderFunction
    ])
    def test_function_to_dict(self, fn_class):
        """to_dict should return complete metadata."""
        fn = fn_class()
        d = fn.to_dict()
        
        assert "key" in d
        assert "label" in d
        assert "description" in d
        assert "inputs" in d
        assert "outputs" in d
        assert "recommended" in d
