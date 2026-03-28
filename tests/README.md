# NNStudio Test Suite

Comprehensive unit and integration tests for the NNStudio neural network visualization and training application.

## Running Tests

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Files

```bash
# Core components
pytest tests/test_activations.py
pytest tests/test_losses.py
pytest tests/test_optimizers.py
pytest tests/test_layers.py
pytest tests/test_network.py

# Domain modules
pytest tests/test_training_functions.py
pytest tests/test_presets.py

# API routes (integration tests)
pytest tests/test_api_routes.py
```

### Run with Coverage

```bash
pytest --cov=app --cov-report=html
```

### Run Specific Test Classes or Functions

```bash
# Specific class
pytest tests/test_activations.py::TestActivations

# Specific test
pytest tests/test_activations.py::TestActivations::test_relu_forward

# Pattern matching
pytest -k "sigmoid"
pytest -k "backward"
```

### Verbose Output

```bash
pytest -v
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── test_activations.py      # Activation function tests (ReLU, Sigmoid, Tanh, GELU, Swish)
├── test_losses.py           # Loss function tests (MSE, BCE, MAE)
├── test_optimizers.py       # Optimizer tests (SGD, Momentum, RMSProp, Adam, AdamW)
├── test_layers.py           # Dense layer forward/backward/update tests
├── test_network.py          # NeuralNetwork and NetworkBuilder tests
├── test_training_functions.py  # Training function tests (XOR, 7-segment, etc.)
├── test_presets.py          # Preset module tests
└── test_api_routes.py       # API route integration tests
```

## Test Coverage

### Core Components (251 tests)

| Module | Tests | Description |
|--------|-------|-------------|
| Activations | 24 | Forward/backward pass, numerical stability, shape preservation |
| Losses | 20 | Loss computation, gradient calculation, edge cases |
| Optimizers | 24 | Weight updates, momentum, adaptive learning rates, state serialization |
| Layers | 24 | Forward/backward pass, dropout, serialization, gradient computation |
| Network | 30 | Architecture building, training, metrics, serialization |
| Training Functions | 44 | Dataset generation, I/O labels, recommended configs |
| Presets | 35 | Preset configurations, layer structures, metadata |
| API Routes | 50+ | HTTP endpoints, build/train/predict workflows |

## Fixtures

Key fixtures available in `conftest.py`:

- `sample_input` - Simple 2D input array
- `sample_target` - Simple target array
- `xor_dataset` - Complete XOR dataset
- `simple_layer` - 2→4 dense layer with tanh
- `output_layer` - 4→1 output layer with sigmoid
- `simple_network` - 2→4→1 network
- `app_config` - Network configuration dict
- `client` - Flask test client
- `flask_app` - Flask app instance

## Writing New Tests

### Test Naming Convention

```python
def test_<component>_<behavior>():
    """<Description of what is being tested>."""
    # Arrange
    # Act
    # Assert
```

### Example Test

```python
def test_relu_forward():
    """ReLU should output max(0, x)."""
    act = ACTIVATIONS["relu"]
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    np.testing.assert_array_equal(act.forward(x), expected)
```

### Using Fixtures

```python
def test_forward_shape(simple_layer):
    """Forward pass should preserve batch dimension."""
    x = np.random.randn(2)
    output = simple_layer.forward(x)
    assert output.shape == (4,)
```

### Parametrized Tests

```python
@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid"])
def test_activation_preserves_shape(activation):
    """Activations should preserve input shape."""
    act = ACTIVATIONS[activation]
    x = np.random.randn(5, 4)
    result = act.forward(x)
    assert result.shape == x.shape
```

## CI/CD Integration

Add to your CI pipeline:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest --cov=app --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
```

## Troubleshooting

### Import Errors

Ensure you're running from the project root:
```bash
cd "C:\Users\John\Desktop\AI Gens\NNStudio"
pytest
```

### Flask App Issues

API tests require the Flask app context:
```python
with flask_app.app_context():
    response = client.get("/api/modules/all")
```

### Random Seed

For reproducible tests, set seeds:
```python
np.random.seed(42)
```

## Coverage Goals

- Core math (activations, losses, optimizers): 100%
- Layer implementations: 95%+
- Network training: 90%+
- API routes: 85%+
