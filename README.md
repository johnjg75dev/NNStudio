# ⬡ Neural Network Trainer

A fully object-oriented, modular Flask web application for training, visualising,
and exploring neural networks in the browser.  No PyTorch, no TensorFlow — the
entire training engine is written from scratch in pure NumPy so every line of
backpropagation is readable and hackable.

---

## Table of Contents

1. [Features](#features)
2. [Quick Start](#quick-start)
3. [Project Structure](#project-structure)
4. [Architecture Overview](#architecture-overview)
5. [The Module System](#the-module-system)
6. [Extending the App](#extending-the-app)
   - [Adding a Training Function](#adding-a-training-function)
   - [Adding a Preset](#adding-a-preset)
   - [Adding an Architecture Diagram](#adding-an-architecture-diagram)
   - [Adding an Optimizer](#adding-an-optimizer)
7. [API Reference](#api-reference)
8. [Training Tasks](#training-tasks)
9. [Hyperparameter Guide](#hyperparameter-guide)
10. [Frontend Architecture](#frontend-architecture)
11. [Design Principles](#design-principles)
12. [Requirements](#requirements)

---

## Features

### Training & Visualisation
- **Live network graph** — nodes coloured by activation value, edges coloured
  green/red by weight sign, thickness proportional to `|weight|`
- **Real-time loss curve** — scrolling chart updated every animation frame
- **Gradient overlay** — optional visualisation of per-edge gradient magnitude
- **Bias arrows** — dashed lines showing learned bias contributions
- **Node inspector** — click any neuron to see its activation, bias, and all
  incoming weights
- **Weight matrix panel** — colour-coded heatmap of every weight matrix

### Training Engine (pure NumPy)
- **6 activation functions** — ReLU, Leaky ReLU, Tanh, Sigmoid, GELU, Swish
- **5 optimizers** — SGD, SGD + Momentum, RMSProp, Adam, AdamW
- **3 loss functions** — MSE, Binary Cross-Entropy, MAE
- **Dropout** — applied per hidden layer during training, disabled at inference
- **Weight decay** — L2 regularisation (decoupled in AdamW)
- **He initialisation** — correct scaling for deep networks

### Training Tasks (11 built-in)
- Logic gates: XOR, AND, OR, XNOR
- 7-Segment Display (4-bit → 7 outputs)
- 4-bit Parity, Half Adder
- Sine Approximation (regression)
- Circle Boundary, Spiral Classes (geometric classification)
- Autoencoder (8→3→8 identity compression)

### Architecture Zoo (9 diagrams)
Interactive educational diagrams for: MLP, CNN, Autoencoder, Transformer,
Vision Transformer (ViT), VAE, Diffusion / Stable Diffusion, GAN, RNN/LSTM

### Other
- **Test Inputs tab** — enter any input vector and run a live forward pass
- **Sweep** — evaluate every training sample at once, see per-sample error
- **Save / Load** — export the full model (weights + config) as JSON, re-import
  to resume training
- **10 one-click presets** — from "Tiny XOR" to "Regularised Deep"
- **Tooltip system** — every hyperparameter control has a `?` icon explaining
  pros, cons, and typical ranges
- **Keyboard shortcuts** — `Space` train/pause, `R` reset weights, `B` rebuild

---

## Quick Start

```bash
# 1. Clone / unzip the project
cd nn_trainer

# 2. Install dependencies (Flask + NumPy only)
pip install -r requirements.txt

# 3. Run
python run.py
```

Open **http://localhost:5000** in your browser.

The app starts with the "Tiny XOR" preset already loaded.
Press **▶ Train** or hit **Space** to begin.

---

## Project Structure

```
nn_trainer/
├── run.py                              # Entry point
├── requirements.txt
├── README.md
└── app/
    ├── __init__.py                     # Flask app factory (create_app)
    │
    ├── core/                           # Pure-Python ML engine — no Flask here
    │   ├── activations.py              # Activation dataclass + ACTIVATIONS dict
    │   ├── losses.py                   # LossFunction dataclass + LOSSES dict
    │   ├── optimizers.py               # BaseOptimizer hierarchy + OptimizerFactory
    │   ├── network.py                  # Layer → DenseLayer → NeuralNetwork + NetworkBuilder
    │   └── session_manager.py          # TrainingSession + SessionManager (TTL store)
    │
    ├── modules/                        # Auto-discovered plugin system
    │   ├── base.py                     # BaseModule ABC
    │   ├── registry.py                 # ModuleRegistry — scans folders, no manual lists
    │   │
    │   ├── functions/                  # ← drop a new .py here to add a training task
    │   │   ├── base_function.py        # TrainingFunction ABC
    │   │   ├── xor.py                  # XOR gate
    │   │   ├── logic_gates.py          # AND, OR, XNOR
    │   │   ├── seven_segment.py        # 7-segment display
    │   │   ├── math_functions.py       # Parity, Sine, Half Adder
    │   │   └── geometric.py            # Circle, Spiral, Autoencoder
    │   │
    │   ├── architectures/              # ← drop a new .py here to add an arch diagram
    │   │   ├── base_architecture.py    # ArchitectureModule ABC
    │   │   └── all_architectures.py    # MLP, CNN, AE, Transformer, ViT, VAE, Diffusion, GAN, RNN
    │   │
    │   ├── presets/                    # ← drop a new .py here to add a preset
    │   │   ├── base_preset.py          # PresetModule ABC
    │   │   └── builtin_presets.py      # 10 built-in presets
    │   │
    │   └── optimizers/                 # UI descriptor metadata for each optimizer
    │       └── optimizer_descriptors.py
    │
    ├── api/                            # Flask blueprints
    │   ├── helpers.py                  # ok()/err(), @api_route, session helpers
    │   ├── session_routes.py           # /api/session/* — build/reset/predict/export/import
    │   ├── train_routes.py             # /api/train/step, /api/train/evaluate
    │   ├── module_routes.py            # /api/modules/* — registry queries
    │   └── page_routes.py             # GET / → renders index.html
    │
    ├── static/
    │   ├── css/main.css
    │   └── js/
    │       ├── api.js                  # All fetch() calls — single source of truth
    │       ├── renderer.js             # NetworkRenderer, ArchDiagramRenderer, LossChartRenderer
    │       ├── trainer.js              # TrainingController (rAF loop, EventTarget)
    │       ├── ui.js                   # UIController — all DOM, zero API calls
    │       └── app.js                  # App — top-level orchestrator, wires everything
    │
    └── templates/
        └── pages/index.html            # Jinja2 SPA shell (injects registry as window.REGISTRY)
```

**Total: ~4,440 lines across 39 source files.**

---

## Architecture Overview

```
Browser                          Flask Server
───────                          ────────────
App (orchestrator)
 ├── UIController          ←──── Jinja2 renders index.html
 │    └── emits events           with window.REGISTRY injected
 ├── TrainingController
 │    └── rAF loop ──POST /api/train/step──────► train_routes.py
 ├── NetworkRenderer                                    │
 │    └── draws <canvas>   ◄── JSON snapshot ───────────┘
 ├── ArchDiagramRenderer                         session_routes.py
 └── LossChartRenderer          ◄── /api/session/snapshot
                                          │
                           SessionManager (in-memory, TTL)
                                          │
                                   TrainingSession
                                          │
                                   NeuralNetwork
                                    ├── [DenseLayer, ...]
                                    ├── BaseOptimizer
                                    └── LossFunction
```

The server is **stateless per request** — all model state lives in
`SessionManager`, keyed by a browser `session_id` cookie.  Multiple browser
tabs each get their own independent training session.

---

## The Module System

`ModuleRegistry` in `app/modules/registry.py` is the heart of the plugin
architecture.  On startup (`registry.discover()`), it:

1. Iterates over the four scan packages:
   `app.modules.functions`, `app.modules.architectures`,
   `app.modules.presets`, `app.modules.optimizers`
2. Imports every `.py` file found in each folder using `pkgutil.iter_modules`
3. Inspects every attribute in each imported module
4. Registers any class that:
   - Is a subclass of `BaseModule`
   - Is **not** `BaseModule` itself
   - Has a non-empty `key` attribute

No registration list, no `__init__.py` imports, no decorators needed.
**Dropping a file into the right folder is the entire registration process.**

The full registry is serialised and injected into the HTML page as
`window.REGISTRY` by Jinja2, so the JavaScript frontend has immediate access
to all module metadata without a separate API call on boot.

---

## Extending the App

### Adding a Training Function

Create a new file anywhere inside `app/modules/functions/`:

```python
# app/modules/functions/my_task.py
from .base_function import TrainingFunction

class MyTaskFunction(TrainingFunction):
    key           = "my_task"          # unique slug — used in API calls
    label         = "My Custom Task"   # shown in the dropdown
    description   = "<b>My Task</b>: What it tests and why it's interesting."
    inputs        = 3
    outputs       = 1
    input_labels  = ["x", "y", "z"]
    output_labels = ["result"]
    is_classification = True
    recommended   = {
        "hidden_layers": 2,
        "neurons":       8,
        "activation":    "relu",
        "optimizer":     "adam",
        "loss":          "bce",
        "dropout":       0.0,
        "lr":            0.01,
    }

    def generate_dataset(self):
        # Return a list of {"x": [...], "y": [...]} dicts
        return [
            {"x": [0, 0, 0], "y": [0]},
            {"x": [1, 0, 1], "y": [1]},
            # ...
        ]
```

Restart the server — your task appears in the Training Task dropdown immediately.

---

### Adding a Preset

```python
# app/modules/presets/my_preset.py
from .base_preset import PresetModule

class MyPreset(PresetModule):
    key           = "preset_my_task"
    label         = "My Task Preset"
    description   = "Optimal settings for my custom task."
    arch_key      = "mlp"
    func_key      = "my_task"
    hidden_layers = 2
    neurons       = 8
    activation    = "relu"
    optimizer     = "adam"
    loss          = "bce"
    lr            = 0.01
    dropout       = 0.0
    weight_decay  = 0.0
```

Restart — the preset button appears in the grid automatically.

---

### Adding an Architecture Diagram

```python
# app/modules/architectures/my_arch.py
from .base_architecture import ArchitectureModule

class MyArchitecture(ArchitectureModule):
    key          = "my_arch"
    label        = "My Architecture"
    accent_color = "#bc8cff"
    diagram_type = "my_arch"      # matched by JS renderer
    trainable    = False           # True only if training is implemented
    description  = "<h3>My Architecture</h3>Description here."
```

To add the canvas diagram, add a `_drawMyArch(W, H, cx, cy)` method to
`ArchDiagramRenderer` in `app/static/js/renderer.js` and register it in the
`drawFn` dispatch map inside `ArchDiagramRenderer.draw()`.

---

### Adding an Optimizer

**Step 1** — implement the optimizer class in `app/core/optimizers.py`:

```python
class MyOptimizer(BaseOptimizer):
    label       = "My Optimizer"
    description = "What makes it special."

    def step(self, param, grad, key=""):
        # update param using grad, return updated param
        return param - self.lr * grad
```

Register it in `_REGISTRY` at the bottom of `optimizers.py`:
```python
_REGISTRY["myopt"] = MyOptimizer
```

**Step 2** — add the UI descriptor:

```python
# app/modules/optimizers/optimizer_descriptors.py (add to existing file)
class MyOptimizerDescriptor(OptimizerDescriptor):
    key         = "myopt"
    label       = "My Optimizer"
    description = "What makes it special."
    lr_range    = "0.001 – 0.1"
    pros        = "..."
    cons        = "..."
```

**Step 3** — add the `<option>` to `app/templates/pages/index.html`:
```html
<option value="myopt">My Optimizer</option>
```

---

## API Reference

All endpoints return `{ "ok": true, "data": { ... } }` on success,
or `{ "ok": false, "error": "message" }` on failure.

### Module Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/api/modules/all` | Full registry grouped by category |
| `GET`  | `/api/modules/category/<cat>` | All modules in one category |
| `GET`  | `/api/modules/<key>` | Single module metadata |
| `GET`  | `/api/modules/functions/<key>/dataset` | Raw dataset for a function |

### Session Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/session/build` | Build a new network (see body below) |
| `POST` | `/api/session/reset` | Re-initialise weights, keep topology |
| `POST` | `/api/session/predict` | Forward pass: `{ "x": [float, ...] }` |
| `GET`  | `/api/session/snapshot` | Full visual state (topology, weights, activations) |
| `POST` | `/api/session/export` | Serialise current model to JSON |
| `POST` | `/api/session/import` | Load a previously exported model |

**Build body:**
```json
{
  "func_key":      "xor",
  "arch_key":      "mlp",
  "hidden_layers": 1,
  "neurons":       4,
  "activation":    "tanh",
  "optimizer":     "adam",
  "lr":            0.01,
  "loss":          "bce",
  "dropout":       0.0,
  "weight_decay":  0.0
}
```

### Training Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/train/step` | Run N steps: `{ "steps": 10, "lr": 0.01 }` |
| `POST` | `/api/train/evaluate` | Evaluate all samples, return per-sample predictions |

---

## Training Tasks

| Key | Name | Inputs | Outputs | Type | Challenge |
|-----|------|--------|---------|------|-----------|
| `xor` | XOR Gate | 2 | 1 | Classification | Non-linearly separable — needs hidden layer |
| `and` | AND Gate | 2 | 1 | Classification | Linearly separable |
| `or` | OR Gate | 2 | 1 | Classification | Linearly separable |
| `xnor` | XNOR Gate | 2 | 1 | Classification | Non-linearly separable |
| `seg7` | 7-Segment Display | 4 | 7 | Classification | Multi-output, 10 patterns |
| `parity` | 4-bit Parity | 4 | 1 | Classification | Complex XOR-like, needs depth |
| `sine` | Sine Approximation | 1 | 1 | Regression | Smooth function fitting |
| `adder` | Half Adder | 2 | 2 | Classification | Dual-output (Sum + Carry) |
| `circle` | Circle Boundary | 2 | 1 | Classification | Non-linear decision boundary |
| `spiral` | Spiral Classes | 2 | 1 | Classification | Hard — interleaved spirals |
| `autoenc` | Autoencoder | 8 | 8 | Regression | Bottleneck compression (8→3→8) |

---

## Hyperparameter Guide

### Learning Rate
The most important hyperparameter.  Too high → loss oscillates or diverges.
Too low → training stalls.

| Optimizer | Typical Range |
|-----------|--------------|
| SGD | 0.01 – 0.1 |
| Momentum | 0.001 – 0.1 |
| RMSProp | 0.0001 – 0.01 |
| Adam | 0.0001 – 0.01 |
| AdamW | 0.0001 – 0.01 |

### Activation Functions

| Name | Best For | Watch Out For |
|------|----------|---------------|
| ReLU | Deep nets, fast training | Dead neurons at high LR |
| Leaky ReLU | Same as ReLU | Slightly more compute |
| Tanh | Shallow nets, zero-centred | Saturates at extremes |
| Sigmoid | Output layer only (with BCE) | Vanishing gradients in depth |
| GELU | Transformers, BERT-style | More compute than ReLU |
| Swish | Modern deep nets | More compute than ReLU |

### Dropout
Randomly zeros activations during training, disabled at inference.
Recommended range: 0.1 – 0.3.  Use 0 for tiny networks (they don't overfit).

### Weight Decay
L2 regularisation.  Decoupled from the gradient in AdamW (more principled).
Recommended: 0 to 0.01.  Higher values simplify the model but risk underfitting.

### Steps per Frame
Controls the tradeoff between training speed and visualisation smoothness.
- `1` — watch individual weight updates happen in real time
- `10` — balanced default
- `50–200` — fast convergence, choppy visuals

---

## Frontend Architecture

The JavaScript follows a strict separation of concerns with no framework:

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
5. Renderers are pure functions of their input data — no internal state beyond
   the last drawn snapshot

---

## Design Principles

**1. Zero magic registration** — the `ModuleRegistry` finds every module by
filesystem scan.  No decorator, no `__all__`, no import needed in `__init__.py`.

**2. Strict layer separation**
- `core/` has zero Flask imports
- `api/` has zero NumPy/ML code
- `static/js/ui.js` has zero `fetch()` calls
- `static/js/api.js` has zero DOM calls

**3. Every public class has a single responsibility**
- `DenseLayer` — one layer's maths
- `NeuralNetwork` — orchestrates layers
- `NetworkBuilder` — constructs from config dict
- `SessionManager` — owns the per-user session store
- `TrainingSession` — owns one user's model + dataset
- `ModuleRegistry` — discovers and indexes modules

**4. Serialisation is a first-class concern** — every `Layer`, `NeuralNetwork`,
and `BaseOptimizer` implements `to_dict()` / `from_dict()` / `state_dict()` so
save/load works without pickling.

**5. Adding content never requires editing existing files** — new functions,
presets and architectures are self-contained in their own file and discovered
automatically.

---

## Requirements

```
flask>=3.0
numpy>=1.26
python>=3.10
```

No GPU, no CUDA, no heavy ML framework.  Runs on any machine that can run Python.
