"""
app/modules/functions/sequence_functions.py
Sequence and time-series training functions for RNN/LSTM architectures.
"""
import numpy as np
from .base_function import TrainingFunction


class SequencePredictionFunction(TrainingFunction):
    """
    Next value in sequence prediction.
    Classic RNN task: predict next number in a sequence.
    """
    key = "seq_predict"
    label = "Sequence Prediction"
    description = (
        "<b>Sequence Prediction</b>: Predict next value in numeric sequence.<br>"
        "Tests RNN/LSTM ability to learn temporal patterns. "
        "Sequences follow arithmetic or geometric progressions."
    )
    inputs = 5  # 5 previous values
    outputs = 1  # Next value
    input_labels = [f"t-{i}" for i in range(5, 0, -1)]
    output_labels = ["t"]
    is_classification = False
    recommended = {
        "layers": [
            {"type": "lstm", "hidden_size": 32},
            {"type": "dense", "neurons": 16, "activation": "relu"},
            {"type": "dense", "neurons": 1, "activation": "sigmoid"},
        ],
        "optimizer": "adam",
        "loss": "mse",
        "lr": 0.001,
    }

    def generate_dataset(self):
        """Generate sequence prediction dataset."""
        np.random.seed(42)
        data = []
        
        for _ in range(100):
            # Generate random sequence type
            seq_type = np.random.choice(['arithmetic', 'geometric', 'alternating', 'random_walk'])
            
            # Start with random values
            seq = [np.random.uniform(0.2, 0.8)]
            
            for _ in range(10):
                if seq_type == 'arithmetic':
                    diff = np.random.uniform(-0.1, 0.1)
                    seq.append(np.clip(seq[-1] + diff, 0, 1))
                elif seq_type == 'geometric':
                    ratio = np.random.uniform(0.8, 1.2)
                    seq.append(np.clip(seq[-1] * ratio, 0, 1))
                elif seq_type == 'alternating':
                    if len(seq) % 2 == 0:
                        seq.append(np.clip(seq[-1] + 0.15, 0, 1))
                    else:
                        seq.append(np.clip(seq[-1] - 0.15, 0, 1))
                else:  # random_walk
                    seq.append(np.clip(seq[-1] + np.random.randn() * 0.1, 0, 1))
            
            # Create sliding window samples
            for i in range(len(seq) - 5):
                data.append({
                    "x": seq[i:i+5],
                    "y": [seq[i+5]]
                })
        
        return data


class SentimentClassificationFunction(TrainingFunction):
    """
    Simplified sentiment classification.
    Binary classification based on word patterns.
    """
    key = "sentiment"
    label = "Sentiment Analysis"
    description = (
        "<b>Sentiment Analysis</b>: Classify text as positive/negative.<br>"
        "Simplified with 8-word vocabulary. Tests LSTM ability to capture context."
    )
    inputs = 8  # 8 time steps (words)
    outputs = 1  # Positive/negative
    input_labels = [f"w{i}" for i in range(8)]
    output_labels = ["sentiment"]
    is_classification = True
    recommended = {
        "layers": [
            {"type": "embedding", "vocab_size": 20, "embed_dim": 16},
            {"type": "lstm", "hidden_size": 32},
            {"type": "dense", "neurons": 16, "activation": "relu"},
            {"type": "dense", "neurons": 1, "activation": "sigmoid"},
        ],
        "optimizer": "adam",
        "loss": "bce",
        "lr": 0.001,
    }

    def generate_dataset(self):
        """Generate sentiment classification dataset."""
        np.random.seed(42)
        data = []
        
        # Positive words (indices 0-9)
        # Negative words (indices 10-19)
        # Neutral words (indices 20-29)
        
        # Positive sentences (more positive words)
        for _ in range(50):
            sentence = np.random.choice(20, 8)
            # Bias towards positive words
            for i in range(8):
                if np.random.random() > 0.3:
                    sentence[i] = np.random.randint(0, 10)
            data.append({
                "x": sentence.tolist(),
                "y": [1.0]
            })
        
        # Negative sentences (more negative words)
        for _ in range(50):
            sentence = np.random.choice(20, 8)
            # Bias towards negative words
            for i in range(8):
                if np.random.random() > 0.3:
                    sentence[i] = np.random.randint(10, 20)
            data.append({
                "x": sentence.tolist(),
                "y": [0.0]
            })
        
        return data


class TimeSeriesClassificationFunction(TrainingFunction):
    """
    Time series pattern classification.
    Classify time series into trend categories.
    """
    key = "ts_classify"
    label = "Time Series Classification"
    description = (
        "<b>Time Series Classification</b>: Classify trend as rising/falling/stable/volatile.<br>"
        "Tests RNN ability to recognize temporal patterns over time."
    )
    inputs = 10  # 10 time steps
    outputs = 4  # 4 trend classes
    input_labels = [f"t-{i}" for i in range(10, 0, -1)]
    output_labels = ["rising", "falling", "stable", "volatile"]
    is_classification = True
    recommended = {
        "layers": [
            {"type": "lstm", "hidden_size": 32, "return_sequences": False},
            {"type": "dense", "neurons": 32, "activation": "relu"},
            {"type": "dense", "neurons": 4, "activation": "sigmoid"},
        ],
        "optimizer": "adam",
        "loss": "bce",
        "lr": 0.001,
    }

    def generate_dataset(self):
        """Generate time series classification dataset."""
        np.random.seed(42)
        data = []
        
        # Rising trend
        for _ in range(25):
            base = np.random.uniform(0.1, 0.3)
            slope = np.random.uniform(0.05, 0.1)
            ts = [np.clip(base + slope * i + np.random.randn() * 0.05, 0, 1) for i in range(10)]
            data.append({"x": ts, "y": [1, 0, 0, 0]})
        
        # Falling trend
        for _ in range(25):
            base = np.random.uniform(0.7, 0.9)
            slope = np.random.uniform(0.05, 0.1)
            ts = [np.clip(base - slope * i + np.random.randn() * 0.05, 0, 1) for i in range(10)]
            data.append({"x": ts, "y": [0, 1, 0, 0]})
        
        # Stable
        for _ in range(25):
            base = np.random.uniform(0.4, 0.6)
            ts = [np.clip(base + np.random.randn() * 0.03, 0, 1) for _ in range(10)]
            data.append({"x": ts, "y": [0, 0, 1, 0]})
        
        # Volatile
        for _ in range(25):
            base = 0.5
            ts = [np.clip(base + np.random.randn() * 0.2, 0, 1) for _ in range(10)]
            data.append({"x": ts, "y": [0, 0, 0, 1]})
        
        return data


class NextWordPredictionFunction(TrainingFunction):
    """
    Simplified next word prediction.
    Predict next word index from vocabulary.
    """
    key = "next_word"
    label = "Next Word Prediction"
    description = (
        "<b>Next Word Prediction</b>: Predict next word from vocabulary.<br>"
        "Core language modeling task. Tests LSTM ability to learn word sequences."
    )
    inputs = 4  # 4 context words
    outputs = 8  # 8 possible next words
    input_labels = [f"c{i}" for i in range(4)]
    output_labels = [f"word{i}" for i in range(8)]
    is_classification = True
    recommended = {
        "layers": [
            {"type": "embedding", "vocab_size": 16, "embed_dim": 16},
            {"type": "lstm", "hidden_size": 64},
            {"type": "dense", "neurons": 32, "activation": "relu"},
            {"type": "dense", "neurons": 8, "activation": "sigmoid"},
        ],
        "optimizer": "adam",
        "loss": "mse",
        "lr": 0.001,
    }

    def generate_dataset(self):
        """Generate next word prediction dataset."""
        np.random.seed(42)
        data = []
        
        # Define simple grammar patterns
        # Pattern 1: article -> adjective -> noun -> verb
        # Pattern 2: pronoun -> verb -> adverb
        
        patterns = [
            [0, 1, 2, 3],  # the big dog runs
            [0, 4, 5, 6],  # the small cat jumps
            [7, 3, 8],     # I run fast
            [7, 6, 8],     # I jump fast
            [0, 1, 5, 3],  # the big cat runs
            [0, 4, 2, 6],  # the small dog jumps
        ]
        
        # Generate sequences from patterns
        for pattern in patterns:
            for i in range(len(pattern) - 1):
                context = pattern[max(0, i-3):i]
                while len(context) < 4:
                    context.insert(0, 15)  # PAD token
                target = np.zeros(8)
                target[pattern[i+1] % 8] = 1.0
                data.append({
                    "x": context[-4:],
                    "y": target.tolist()
                })
        
        # Add variations
        for _ in range(50):
            context = np.random.randint(0, 16, 4)
            target = np.zeros(8)
            target[np.random.randint(0, 8)] = 1.0
            data.append({
                "x": context.tolist(),
                "y": target.tolist()
            })
        
        return data
