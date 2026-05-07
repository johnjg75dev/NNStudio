# NNStudio — Project Context

## Project Overview

**NNStudio** (Neural Network Studio) is a fully object-oriented, modular Flask web application for training, visualizing, and exploring neural networks in the browser. The entire training engine is written from scratch in **pure NumPy** — no PyTorch, no TensorFlow — making every line of backpropagation readable and hackable.

### Key Features
- **Live network graph visualization** — nodes colored by activation value, edges colored by weight sign, thickness proportional to weight magnitude
- **Real-time loss curve** — scrolling chart updated every animation frame
- **Pure NumPy training engine** with 6 activation functions, 5 optimizers, 3 loss functions, dropout, and weight decay
- **11 built-in training tasks** — XOR, logic gates, 7-segment display, parity, sine approximation, circle/spiral classification, autoencoder
- **9 architecture diagrams** — MLP, CNN, Autoencoder, Transformer, ViT, VAE, Diffusion, GAN, RNN/LSTM
- **Plugin-based module system** — auto-discovers training functions, presets, architectures, and optimizer descriptors via filesystem scan
- **Save/Load** — export full model (weights + config) as JSON
- **10 one-click presets** — from "Tiny XOR" to "Regularised Deep"

## Technologies & Dependencies

| Category | Technology |
|----------|------------|
| Backend | Flask 3.0+, Flask-SQLAlchemy, Flask-Login |
| ML Engine | NumPy 1.26+ (pure, no PyTorch/TensorFlow) |
| Frontend | Vanilla JavaScript (no framework), Canvas API |
| Serialization | safetensors, ONNX |
| Testing | pytest, pytest-cov |
| Python | 3.10+ |

## Project Structure

```
NNStudio/
├── run.py                              # CLI entry point (server, test commands)
├── requirements.txt
├── README.md
├── QWEN.md
│
├── app/
│   ├── __init__.py                     # Flask app factory (create_app)
│   │
│   ├── core/                           # Pure-Python ML engine — no Flask imports
│   │   ├── activations.py              # Activation dataclass + ACTIVATIONS dict
│   │   ├── losses.py                   # LossFunction dataclass + LOSSES dict
│   │   ├── optimizers.py               # BaseOptimizer hierarchy + OptimizerFactory
│   │   ├── network.py                  # Layer → DenseLayer → NeuralNetwork + NetworkBuilder
│   │   ├── session_manager.py          # TrainingSession + SessionManager (TTL store)
│   │   ├── exporters.py                # Model export utilities
│   │   ├── function_executor.py        # Training function execution
│   │   └── layers/                     # Layer implementations
│   │
│   ├── modules/                        # Auto-discovered plugin system
│   │   ├── base.py                     # BaseModule ABC
│   │   ├── registry.py                 # ModuleRegistry — scans folders, no manual lists
│   │   ├── functions/                  # Training tasks (drop .py to add)
│   │   ├── architectures/              # Architecture diagrams (drop .py to add)
│   │   ├── presets/                    # Preset configurations (drop .py to add)
│   │   └── optimizers/                 # UI descriptor metadata
│   │
│   ├── api/                            # Flask blueprints
│   │   ├── helpers.py                  # ok()/err(), @api_route, session helpers
│   │   ├── session_routes.py           # /api/session/* — build/reset/predict/export/import
│   │   ├── train_routes.py             # /api/train/step, /api/train/evaluate
│   │   ├── module_routes.py            # /api/modules/* — registry queries
│   │   ├── page_routes.py              # GET / → renders index.html
│   │   ├── auth_routes.py              # Authentication endpoints
│   │   ├── preset_routes.py            # Preset management
│   │   ├── model_routes.py             # Model save/load operations
│   │   ├── custom_function_routes.py   # User-defined training functions
│   │   ├── dataset_routes.py           # Dataset management
│   │   └── admin_routes.py             # Admin operations
│   │
│   ├── models/                         # SQLAlchemy ORM models
│   │   ├── user.py                     # User model
│   │   ├── preset.py                   # Preset model
│   │   ├── architecture.py             # Architecture definitions
│   │   ├── architecture_definition.py  # Architecture schema
│   │   ├── custom_function.py          # Custom training functions
│   │   ├── dataset.py                  # Dataset definitions
│   │   ├── layer_definition.py         # Layer configurations
│   │   └── saved_model.py              # Saved model storage
│   │
│   ├── static/
│   │   ├── css/main.css
│   │   └── js/
│   │       ├── api.js                  # All fetch() calls
│   │       ├── renderer.js             # Canvas renderers (network, arch diagrams, loss chart)
│   │       ├── trainer.js              # TrainingController (rAF loop, EventTarget)
│   │       ├── ui.js                   # UIController — all DOM, zero API calls
│   │       └── app.js                  # Top-level orchestrator
│   │
│   └── templates/
│       └── pages/index.html            # Jinja2 SPA shell (injects window.REGISTRY)
│
├── tests/                              # Comprehensive test suite
│   ├── conftest.py                     # Shared fixtures
│   ├── test_activations.py
│   ├── test_losses.py
│   ├── test_optimizers.py
│   ├── test_layers.py
│   ├── test_network.py
│   ├── test_training_functions.py
│   ├── test_presets.py
│   ├── test_api_routes.py
│   ├── test_datasets.py
│   ├── test_custom_functions.py
│   ├── test_model_operations.py
│   ├── test_advanced_layers.py
│   └── test_new_layers.py
│
├── scripts/                            # Utility scripts
│   ├── download_datasets.py
│   ├── manage_users.py
│   └── migrate_architectures.py
│
├── instance/                           # Runtime data (DB, datasets)
└── docs/                               # Documentation
```

## Building and Running

### Installation

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
# Start the Flask development server
python run.py server
# or simply
python run.py
```

The app starts at **http://localhost:5000** with the "Tiny XOR" preset loaded.

### Running Tests

```bash
# Run all tests
python run.py test all
# or
pytest tests/

# Run specific test file
python run.py test activations
python run.py test network

# Run with coverage
pytest --cov=app --cov-report=html

# Run with HTML report
python run.py test all -o report.html
```

### Utility Scripts

```bash
python scripts/download_datasets.py
python scripts/manage_users.py
python scripts/migrate_architectures.py
```

## Architecture Overview

### Backend Architecture

The server follows **strict layer separation**:
- `core/` — Pure Python/NumPy ML code, zero Flask imports
- `api/` — Flask blueprints, zero NumPy/ML code
- `modules/` — Plugin system with auto-discovery via `ModuleRegistry`

**Request flow:**
```
Browser → Flask Blueprint → SessionManager (in-memory, TTL) → TrainingSession → NeuralNetwork
```

The server is **stateless per request** — all model state lives in `SessionManager`, keyed by a browser `session_id` cookie. Multiple browser tabs each get their own independent training session.

### Module System (Plugin Architecture)

`ModuleRegistry` in `app/modules/registry.py` is the heart of the plugin architecture. On startup (`registry.discover()`), it:

1. Iterates over four scan packages: `functions`, `architectures`, `presets`, `optimizers`
2. Imports every `.py` file in each folder using `pkgutil.iter_modules`
3. Inspects every attribute for subclasses of `BaseModule`
4. Registers any class with a non-empty `key` attribute

**No registration list, no `__all__`, no decorator needed.** Dropping a file into the right folder is the entire registration process.

### Frontend Architecture

The JavaScript follows strict separation of concerns with no framework:

```
App (app.js)                    ← orchestrator; owns nothing visual
 ├── UIController (ui.js)       ← all DOM reads/writes; zero API calls
 ├── TrainingController (trainer.js)  ← rAF loop; fires EventTarget events
 ├── NetworkRenderer (renderer.js)    ← draws MLP on <canvas>
 ├── ArchDiagramRenderer (renderer.js)← draws arch diagrams on <canvas>
 └── LossChartRenderer (renderer.js)  ← draws loss curve strip

API (api.js)                    ← all fetch() calls; no DOM or canvas
```

**Data flow:**
1. User interacts with DOM → `UIController` emits named event
2. `App` handles event → calls `API.*` or `TrainingController.*`
3. `TrainingController` fires `"step"` event after each server response
4. `App` receives event → updates `UIController` stats + calls `Renderer.draw()`

## Development Conventions

### Design Principles

1. **Zero magic registration** — `ModuleRegistry` finds every module by filesystem scan. No decorator, no `__all__`, no import needed in `__init__.py`.
2. **Strict layer separation** — `core/` has zero Flask imports; `api/` has zero NumPy/ML code; `ui.js` has zero `fetch()` calls; `api.js` has zero DOM calls.
3. **Every public class has a single responsibility** — DenseLayer, NeuralNetwork, NetworkBuilder, SessionManager, etc.
4. **Serialization is first-class** — every Layer, NeuralNetwork, and BaseOptimizer implements `to_dict()` / `from_dict()` / `state_dict()`.
5. **Adding content never requires editing existing files** — new functions, presets, and architectures are self-contained.

### Code Style

- **Backend**: Python 3.10+, type hints where practical, dataclass pattern for config objects
- **Frontend**: Vanilla ES6+, no frameworks, strict separation of DOM/API/canvas concerns
- **Testing**: pytest with fixtures in `conftest.py`, Arrange-Act-Assert pattern, parametrized tests where applicable

### Adding New Modules

**Training Function** — create `app/modules/functions/my_task.py`:
```python
from .base_function import TrainingFunction

class MyTaskFunction(TrainingFunction):
    key = "my_task"
    label = "My Custom Task"
    description = "..."
    inputs = 2
    outputs = 1
    input_labels = ["x", "y"]
    output_labels = ["result"]
    is_classification = True
    recommended = {"hidden_layers": 1, "neurons": 4, "activation": "relu", "optimizer": "adam", "loss": "bce", "dropout": 0.0, "lr": 0.01}

    def generate_dataset(self):
        return [{"x": [...], "y": [...]}]
```

**Preset** — create `app/modules/presets/my_preset.py`:
```python
from .base_preset import PresetModule

class MyPreset(PresetModule):
    key = "preset_my_task"
    label = "My Task Preset"
    # ... config fields ...
```

**Architecture Diagram** — create `app/modules/architectures/my_arch.py`:
```python
from .base_architecture import ArchitectureModule

class MyArchitecture(ArchitectureModule):
    key = "my_arch"
    label = "My Architecture"
    diagram_type = "my_arch"
    trainable = False
    description = "..."
```

## API Reference

All endpoints return `{ "ok": true, "data": { ... } }` on success, or `{ "ok": false, "error": "message" }` on failure.

### Module Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/modules/all` | Full registry grouped by category |
| GET | `/api/modules/category/<cat>` | All modules in one category |
| GET | `/api/modules/<key>` | Single module metadata |
| GET | `/api/modules/functions/<key>/dataset` | Raw dataset for a function |

### Session Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/session/build` | Build a new network |
| POST | `/api/session/reset` | Re-initialise weights, keep topology |
| POST | `/api/session/predict` | Forward pass: `{ "x": [float, ...] }` |
| GET | `/api/session/snapshot` | Full visual state |
| POST | `/api/session/export` | Serialise model to JSON |
| POST | `/api/session/import` | Load a previously exported model |

### Training Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/train/step` | Run N steps: `{ "steps": 10, "lr": 0.01 }` |
| POST | `/api/train/evaluate` | Evaluate all samples |

## Key Fixtures (Tests)

Available in `tests/conftest.py`:
- `sample_input` — Simple 2D input array
- `sample_target` — Simple target array
- `xor_dataset` — Complete XOR dataset
- `simple_layer` — 2→4 dense layer with tanh
- `output_layer` — 4→1 output layer with sigmoid
- `simple_network` — 2→4→1 network
- `app_config` — Network configuration dict
- `client` — Flask test client
- `flask_app` — Flask app instance
- `activation_name`, `loss_name`, `optimizer_name` — Parametrized fixtures
