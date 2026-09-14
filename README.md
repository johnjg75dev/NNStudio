# ⬡ NNStudio

**Neural Network Trainer & Visualiser**

A fully object-oriented, modular Flask web application for training, visualising,
and exploring neural networks in the browser.  No PyTorch, no TensorFlow — the
entire training engine is written from scratch in pure NumPy so every line of
backpropagation is readable and hackable.

The front-end is a **React 18 single-page app** (Vite + React Router) served by
Flask from `frontend/dist`.  Six routes — Studio, Datasets, Playground, Models,
Functions and Learn — share one live training session, so a network trained on
one page can be probed, saved or exported from any other.

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

### Training Tasks (18 built-in, plus your own)
- Logic & arithmetic — XOR, AND, OR, XNOR, Half Adder, 4-bit Parity,
  7-Segment Display
- Regression & geometry — Sine Approximation, Circle Boundary, Spiral Classes
- Images (8×8, 4×4) — Edge Detection, MNIST-like Digits, Pattern Classification
- Sequences & text — Next Word Prediction, Sentiment Analysis, Sequence
  Prediction, Time Series Classification
- Autoencoder (8→3→8 identity compression)
- Anything you write on the **Functions** page — Python or JavaScript, tested in
  a bench before you train on it

### Architecture Zoo (9 diagrams)
Interactive educational diagrams for: MLP, CNN, Autoencoder, Transformer,
Vision Transformer (ViT), VAE, Diffusion / Stable Diffusion, GAN, RNN/LSTM

### Pages
- **Studio (`/train`)** — setup, editable layer stack, live network graph, loss curve,
  weight-matrix heatmaps, node inspector, sample table, sweep and latent probes
- **Playground (`/playground`)** — drive the network by hand: per-input sliders, a
  pixel drawing pad for image tasks, partial forward passes
  (`start_layer` → `end_layer`), node overrides, and a grid sweep that plots the
  response surface
- **Datasets (`/datasets`)** — browse the built-in library or create your own;
  edit samples in a table, a pixel editor with histogram (image datasets), a
  class-coloured scatter plot (2-input datasets) or raw JSON, then send the
  dataset straight to the studio
- **Models (`/models`)** — save the live session to a library, reload it, and
  export to JSON, SafeTensors, GGUF, ONNX or ZIP
- **Functions (`/functions`)** — write a custom training task, test it against a
  bench, preview the generated dataset, then train on it
- **Learn (`/learn`)** — in-app reference: the training loop, all nine
  architecture blueprints, activation curves, optimizer and loss trade-offs,
  layer types, glossary

### Other
- **35 one-click presets** — architecture + layer stack + hyperparameters, plus
  your own saved setups
- **Tooltip system** — every hyperparameter control has a `?` icon explaining
  pros, cons, and typical ranges
- **Keyboard shortcuts** — `Space` train/pause, `B` rebuild, `R` reset weights,
  `1`–`6` jump between pages, `T` theme, `?` the shortcut list, `Esc` close
- **Accounts** — sign-up seeds your presets, architectures, layer definitions and
  the MNIST dataset; training sessions are per-user
- **Light & dark themes** — CSS custom properties, persisted in `localStorage`,
  toggled with `T`

---

## Quick Start

```bash
# 1. Clone the project
cd NNStudio

# 2. Back-end dependencies (Flask + NumPy)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Build the front-end once
cd frontend && npm install && npm run build && cd ..

# 4. Run
python run.py server            # or: python run.py test all
```

Open **http://localhost:5000**, create an account, pick a task in the **Setup**
column and press **Train** — or hit `Space`.

**Hot-reload development:** run `python run.py server` in one terminal and
`cd frontend && npm run dev` in another.  Vite serves on `:5173` and proxies
`/api`, `/login`, `/signup`, `/logout`, `/check-username` and `/admin` to Flask
(`FLASK_ORIGIN` overrides the target), so session cookies keep working.

If `frontend/dist` is missing, Flask renders a short "front-end not built" page
with the exact commands instead of a 404.

---

## Project Structure

```
NNStudio/
├── run.py                              # Entry point
├── requirements.txt
├── README.md
├── instance/nnstudio.db                # SQLite (created on first run)
├── tests/                              # 386 pytest cases
│
├── app/                                # ─── BACK-END ────────────────────────
│   ├── __init__.py                     # create_app + db + login_manager + registry
│   ├── auth_tokens.py                  # signed X-Session-Token (cookie-free login)
│   │
│   ├── models/                         # SQLAlchemy models (one file per table)
│   │   ├── user.py  preset.py  dataset.py  saved_model.py
│   │   ├── architecture.py  architecture_definition.py
│   │   └── layer_definition.py  custom_function.py
│   │
│   ├── core/                           # Pure-Python ML engine — no Flask here
│   │   ├── activations.py              # Activation dataclass + ACTIVATIONS dict
│   │   ├── losses.py                   # LossFunction dataclass + LOSSES dict
│   │   ├── optimizers.py               # BaseOptimizer hierarchy + OptimizerFactory
│   │   ├── layers/                     # base, dense, conv, batch_norm, dropout,
│   │   │                               # rnn, transformer layer implementations
│   │   ├── network.py                  # NeuralNetwork + NetworkBuilder
│   │   ├── session_manager.py          # TrainingSession + SessionManager (TTL store)
│   │   ├── function_executor.py        # sandboxed exec of user Python/JS tasks
│   │   └── exporters.py                # JSON, SafeTensors, GGUF, ONNX, ZIP
│   │
│   ├── modules/                        # Auto-discovered plugin system
│   │   ├── base.py                     # BaseModule ABC
│   │   ├── registry.py                 # ModuleRegistry — scans folders, no manual lists
│   │   ├── functions/                  # ← drop a .py here to add a training task
│   │   │   ├── base_function.py        # TrainingFunction ABC
│   │   │   ├── xor.py  logic_gates.py  seven_segment.py  math_functions.py
│   │   │   ├── geometric.py  image_functions.py  sequence_functions.py
│   │   │   └── custom_function_wrapper.py  # bridges DB functions into the registry
│   │   ├── architectures/              # ← drop a .py here to add an arch diagram
│   │   │   ├── base_architecture.py    # ArchitectureModule ABC
│   │   │   ├── mlp.py  cnn.py  autoencoder.py  transformer.py  vit.py
│   │   │   ├── vae.py  diffusion.py  gan.py  rnn.py
│   │   │   └── database_architecture.py
│   │   ├── presets/                    # ← drop a .py here to add a preset (35 total)
│   │   │   ├── base_preset.py          # PresetModule ABC
│   │   │   └── builtin_presets.py  cnn_presets.py  rnn_presets.py
│   │   │       transformer_presets.py  generative_presets.py
│   │   └── optimizers/                 # UI descriptor metadata for each optimizer
│   │
│   └── api/                            # Flask blueprints
│       ├── helpers.py                  # ok()/err(), @api_route, session helpers
│       ├── session_routes.py           # /api/session/* — build, reset, predict,
│       │                               # snapshot, export, import, latent-sweep
│       ├── train_routes.py             # /api/train/step, /api/train/evaluate
│       ├── module_routes.py            # /api/modules/* — registry queries
│       ├── auth_routes.py              # signup/login/logout (JSON, no redirects)
│       ├── preset_routes.py            # saved setups (CRUD)
│       ├── dataset_routes.py           # dataset library (CRUD)
│       ├── model_routes.py             # model library (CRUD + 5 export formats)
│       ├── custom_function_routes.py   # user-written tasks + bench + preview
│       ├── admin_routes.py             # built-in architecture CRUD (admin only)
│       └── page_routes.py              # serves frontend/dist; SPA catch-all route
│
└── frontend/                           # ─── FRONT-END (React 18 + Vite) ─────
    ├── index.html                      # SPA shell
    ├── vite.config.js                  # base '/', build → dist, dev proxy → :5000
    ├── package.json                    # dev · build · check · smoke · verify
    ├── scripts/
    │   ├── check-undefined.mjs         # fails on used-but-never-imported names
    │   └── smoke-render.mjs            # SSR-renders all 8 pages, fails on a throw
    ├── .gitignore                      # re-includes index.html + src/lib (the root
    │                                   # .gitignore blanket-ignores *.html and lib/)
    └── src/
        ├── main.jsx                    # providers → router → App
        ├── App.jsx                     # route table
        ├── api/client.js               # every fetch() call — single source of truth
        │
        ├── state/
        │   ├── sessionStore.js         # live-model state machine (framework-free)
        │   ├── SessionContext.jsx      # provider + useSession(selector)
        │   ├── CatalogContext.jsx      # registry, datasets, presets, models, functions
        │   ├── ToastContext.jsx        # toasts + promise-based useConfirm()
        │   └── ThemeContext.jsx        # light/dark theme, persisted
        │
        ├── lib/                        # pure maths + canvas painters (no React)
        │   ├── activations.js          # ACTIVATION_INFO, evaluate, derivative, curve painter
        │   ├── archDiagrams.js         # one painter per diagram_type
        │   ├── networkGraph.js         # MLP graph layout + painter
        │   ├── lossChart.js  plot2d.js  influence.js  samples.js  grid.js
        │   ├── colors.js               # palettes, weight/activation/heat colour ramps
        │   ├── layers.js               # ACTIVATIONS, OPTIMIZERS, LOSSES, LAYER_CATALOG
        │   ├── ioShape.js              # dataset → [w,h,c] / flat input shape
        │   ├── format.js               # number/date formatting, download, file read
        │   └── hooks.js                # useCanvas, useHotkeys, useElementSize,
        │                               # useLocalStorage, useDebounced, useInterval …
        │
        ├── components/
        │   ├── ui.jsx                  # 25 primitives: Button, Card, Field, Select,
        │   │                           # Slider, Switch, Segmented, Tabs, Badge,
        │   │                           # Metric, Tooltip, Modal, Popover, Accordion …
        │   ├── Icon.jsx                # inline SVG icon set
        │   ├── layout/                 # AppShell (top bar + side nav), AuthLayout,
        │   │                           # ShortcutsDialog, PageFallback
        │   ├── canvas/                 # NetworkCanvas, ArchDiagramCanvas, LossChartCanvas,
        │   │                           # ActivationChart, PixelCanvas, Plot2DCanvas, SweepChart
        │   ├── train/                  # SetupPanel, TaskPicker, LayerStack, AddLayerDialog,
        │   │                           # PresetGallery, InspectorPanel, NodePanel, WeightsPanel,
        │   │                           # WeightHeatmap, StagePanel, IOPanel, LatentPanel,
        │   │                           # HistoryPanel, SavePresetDialog
        │   ├── playground/             # InputControls, OutputDisplay, SweepPanel
        │   ├── datasets/               # SampleEditor, ScatterPlot, DatasetDialog
        │   ├── models/                 # ModelCard
        │   ├── functions/              # FunctionEditor
        │   └── shared/                 # PixelPreview, SevenSegment
        │
        ├── pages/                      # TrainPage, DatasetsPage, PlaygroundPage,
        │                               # ModelsPage, FunctionsPage, LearnPage,
        │                               # LoginPage, SignupPage
        └── styles/                     # tokens.css, global.css,
                                        # components.css, pages.css
```

**Back-end:** ~8,700 lines of Python across 73 files, plus ~4,400 lines of
tests.  **Front-end:** ~12,400 lines of JS/JSX across 66 files, plus ~4,300 lines
of CSS.

`frontend/dist` is a build artefact and is git-ignored (as is `node_modules`), so
run `npm install && npm run build` once after cloning.  Until it exists Flask
serves a short page with those exact commands instead of a 404.

---

## Architecture Overview

```
Browser (React SPA)                            Flask Server
───────────────────                            ────────────
main.jsx → Theme/Toast/Catalog/Session providers → router
 │
 ├── CatalogContext ──GET /api/modules/all───────────► module_routes.py
 │                    GET /api/datasets|presets|             │
 │                        models|functions             ModuleRegistry (scan)
 │                                                            │
 ├── sessionStore  (the live-model state machine)        models.py → SQLite
 │    │                                                   (per-user libraries)
 │    ├─POST /api/session/build ─────────────────────► session_routes.py
 │    ├─POST /api/train/step ────────────────────────► train_routes.py
 │    ├─POST /api/train/evaluate ────────────────────►       │
 │    └─GET  /api/session/snapshot ──────────────────► SessionManager (in-memory)
 │                                                             │
 ├── components subscribe to slices                     TrainingSession
 │    NetworkCanvas · LayerStack · LossChart                   │
 │    SampleTable · PixelPad · PlaygroundGrid           NeuralNetwork
 │                                                      ├─ [DenseLayer, …]
 └── AppShell: status pill, theme, account             ├─ BaseOptimizer
                                                       └─ LossFunction
```

The server is **stateless per request** — all model state lives in
`SessionManager`, keyed by the signed-in user (and a `session_id` cookie for
guests).  Multiple browser tabs each get their own independent training session.

Flask serves `frontend/dist/index.html` for `/` and for every non-`/api` path
(catch-all), so deep links like `/datasets` survive a refresh.  In development,
`npm run dev` proxies the API back to Flask so hot reload and session cookies
both work.

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

The React app fetches the whole registry once with a single
`GET /api/modules/all` (wrapped by `CatalogContext`) and indexes it into maps —
`functionByKey`, `architectureByKey`, `datasetById` — so components never
re-fetch.  User-created rows (datasets, presets, architectures, layer
definitions, custom functions) are merged into the same categories and, unlike
the built-ins, are editable and deletable in the UI.

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
        "layers": [
            {"type": "dense", "neurons": 8, "activation": "relu"},
            {"type": "dense", "neurons": 8, "activation": "relu"},
        ],
        "activation": "relu",
        "optimizer":  "adam",
        "loss":       "bce",
        "dropout":    0.0,
        "lr":         0.01,
    }

    def generate_dataset(self):
        # Return a list of {"x": [...], "y": [...]} dicts
        return [
            {"x": [0, 0, 0], "y": [0]},
            {"x": [1, 0, 1], "y": [1]},
            # ...
        ]
```

Restart the server — the task appears in the **Setup → Task** picker and in the
Datasets page immediately (the registry is re-scanned on boot).

---

### Adding a Preset

```python
# app/modules/presets/my_preset.py
from .base_preset import PresetModule

class MyPreset(PresetModule):
    key          = "preset_my_task"
    label        = "My Task Preset"
    description  = "Optimal settings for my custom task."
    arch_key     = "mlp"
    func_key     = "my_task"
    layers       = [
        {"type": "dense", "neurons": 8, "activation": "relu"},
        {"type": "dense", "neurons": 8, "activation": "relu"},
    ]
    activation   = "relu"
    optimizer    = "adam"
    loss         = "bce"
    lr           = 0.01
    dropout      = 0.0
    weight_decay = 0.0
```

Restart — the preset card appears in the gallery automatically.  Presets saved
from the UI land in the same gallery as editable, deletable database rows.

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

To add the canvas diagram, register a painter in
`frontend/src/lib/archDiagrams.js`:

```js
PAINTERS.my_arch = (ctx, W, H, theme) => { /* draw */ }
```

`ArchDiagramCanvas` dispatches on `diagram_type` and falls back to the generic
MLP painter for unknown keys, so a new architecture renders sensibly before you
write a bespoke diagram.

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

**Step 3** — nothing.  The optimizer `<select>` is rendered straight from
`GET /api/modules/optimizers`; `_REGISTRY` in `app/core/optimizers.py` is the
only place the key has to exist.  The descriptor's `pros` / `cons` / `lr_range`
also feed the Learn page and the `?` tooltips automatically.

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
| `POST` | `/api/session/build` | Build a network from an explicit layer stack (see body); returns `topology`, `param_count`, `epoch`, `func` |
| `POST` | `/api/session/reset` | Re-initialise weights, keep topology |
| `POST` | `/api/session/predict` | Forward pass: `{ x, start_layer?, end_layer?, node_overrides? }` |
| `GET`  | `/api/session/snapshot` | Full visual state (topology, weights, activations, metrics) |
| `POST` | `/api/session/export` | Serialise current model to JSON |
| `POST` | `/api/session/import` | Load a previously exported model |

**Build body:**

```json
{
  "func_key":  "xor",
  "ds_id":     null,
  "arch_key":  "mlp",
  "optimizer": "adam",
  "lr":        0.05,
  "loss":      "bce",
  "weight_decay": 0.0,
  "activation": "tanh",
  "layers": [
    { "type": "dense", "neurons": 4, "activation": "relu" },
    { "type": "dropout", "rate": 0.1 }
  ]
}
```

- **Data source** — `ds_id` (a row from the dataset library) wins; otherwise
  `func_key` picks a registry task (`custom_<id>` for user-written ones).
  `inputs` / `outputs` can override the task's own dimensions.
- **`layers`** is the hidden stack only — `NetworkBuilder` always appends a
  `DenseLayer` sized to the task's outputs with a sigmoid activation and
  `is_output=True`.  An empty `layers` list therefore builds a single-layer
  logistic model.
- **`arch_key`** selects the diagram and metadata shown in the UI; it does not
  constrain the stack.  **`activation`** is only the default offered when you add
  a layer in the editor.

Layer types (`LAYER_TYPES` in `app/core/layers/__init__.py`, mirrored by
`LAYER_CATALOG` in `frontend/src/lib/layers.js`):

| Group | Type | Per-layer keys |
|-------|------|----------------|
| Core | `dense` | `neurons`, `activation` |
| Core | `dropout` | `rate` |
| Core | `batchnorm` · `layernorm` | — (`layernorm` takes `eps`) |
| Vision | `conv2d` | `out_channels`, `kernel_size`, `stride`, `padding`, `activation` |
| Vision | `maxpool2d` | `pool_size`, `stride` |
| Vision | `flatten` | — |
| Sequence | `simple_rnn` · `lstm` | `hidden_size`, `activation`, `return_sequences` |
| Sequence | `embedding` | `vocab_size`, `embed_dim` |
| Sequence | `multihead_attention` | `embed_dim`, `num_heads` |
| Sequence | `positional_encoding` | `max_seq_len`, `embed_dim` |

Input and output widths are derived from the dataset and the layer stack, never
passed by the client; an unknown `type` falls back to `dense`.

### Training Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/train/step` | Run N steps: `{ steps, lr }` |
| `POST` | `/api/train/evaluate` | Accuracy, loss and per-sample predictions. `{}` evaluates the session dataset; `{ ranges: [{min,max,step}], start_layer?, end_layer? }` evaluates the cartesian product of the ranges as a grid sweep |
| `POST` | `/api/session/latent-sweep` | Sample latent points and decode them (autoencoders / VAEs) |

### Library & Account Endpoints

| Group | Paths |
|-------|-------|
| Auth | `POST /signup` · `POST /login` · `GET /logout` · `GET /check-username` · `GET /api/me` |
| Presets | `POST /api/presets/save` · `DELETE /api/presets/<id>` (the list comes from `/api/modules/all`) |
| Datasets | `GET/POST /api/datasets` · `GET/PUT/DELETE /api/datasets/<id>` · `POST /api/datasets/<id>/download` |
| Models | `GET /api/models` · `POST /api/models/save` · `GET/DELETE /api/models/<id>` · `POST /api/models/<id>/export` · `GET /api/models/<id>/download/<format>` · `POST /api/models/<id>/load-session` · `GET /api/models/formats` |
| Custom functions | `GET/POST /api/functions/custom` · `GET/PUT/DELETE /api/functions/custom/<id>` · `POST /api/functions/custom/<id>/test` · `POST /api/functions/custom/<id>/preview` · `GET /api/functions/custom/templates` |
| Admin | `GET/POST /api/admin/architectures` · `PUT /api/admin/architectures/<key>` (admin only) |

The auth routes accept form-encoded bodies and answer with JSON when the request
sends `Accept: application/json` (which the SPA always does), so the same routes
also work for a classic browser form post.  Every library endpoint is user-scoped;
built-in rows are readable by everyone but writable only by an admin.
Unauthenticated API calls get a `401` (never a `302` to HTML, which `fetch()`
would silently follow) and the SPA redirects to `/login?next=<path>`.

**Two ways to be signed in.** A successful `/login` or `/signup` returns
`{ ok, id, username, is_admin, token }`.  The `token` is the user id signed with
`SECRET_KEY` (`app/auth_tokens.py`, 30-day expiry, no server-side state):

| Path | How it works |
|------|--------------|
| Cookie | `login_user()` writes the Flask session; `user_loader` reads it. Cookies now get `SameSite=None; Secure` when the request arrives over HTTPS. |
| Token | The SPA stores `token` in `localStorage` and sends `X-Session-Token` (or `?session_token=`) on every call; `login_manager.request_loader` resolves it. The cookie wins when both are present. |

The token path exists because browsers increasingly refuse to *store* a
third-party cookie when the app is embedded in an iframe on another origin — a
live preview, say.  Without it, `POST /login` succeeds and the very next request
is anonymous again, trapping the SPA in a redirect loop.

The in-memory training session is keyed `user:<id>` for a signed-in visitor (and
by a random id in the session cookie for an anonymous one), so a network survives
refreshes, extra tabs, and browsers that dropped the cookie.

---

## Training Tasks

| Key | Name | In | Out | Samples | Type | Challenge |
|-----|------|----|-----|---------|------|-----------|
| `xor` | XOR Gate | 2 | 1 | 4 | Classification | Non-linearly separable — needs a hidden layer |
| `and` | AND Gate | 2 | 1 | 4 | Classification | Linearly separable |
| `or` | OR Gate | 2 | 1 | 4 | Classification | Linearly separable |
| `xnor` | XNOR Gate | 2 | 1 | 4 | Classification | Non-linearly separable |
| `seg7` | 7-Segment Display | 4 | 7 | 16 | Classification | Multi-output, all 16 hex digits |
| `parity` | 4-bit Parity | 4 | 1 | 16 | Classification | XOR-like, needs depth |
| `adder` | Half Adder | 2 | 2 | 4 | Classification | Dual output (Sum + Carry) |
| `sine` | Sine Approximation | 1 | 1 | 20 | Regression | Smooth function fitting |
| `circle` | Circle Boundary | 2 | 1 | 20 | Classification | Non-linear decision boundary |
| `spiral` | Spiral Classes | 2 | 1 | 40 | Classification | Hard — interleaved spirals |
| `autoenc` | Autoencoder | 8 | 8 | 8 | Regression | Bottleneck compression (8→3→8) |
| `edge_detect` | Edge Detection | 64 | 64 | 50 | Image → image | Learns a convolution-like operator |
| `mnist_like` | MNIST-like Digits | 64 | 10 | 200 | Image classification | 8×8 grayscale, 10 classes |
| `pattern_cls` | Pattern Classification | 16 | 4 | 60 | Classification | 4×4 grid → 4 pattern classes |
| `next_word` | Next Word Prediction | 4 | 8 | 66 | Sequence | Token vectors → next-token logits |
| `sentiment` | Sentiment Analysis | 8 | 1 | 100 | Sequence | Bag of features → polarity |
| `seq_predict` | Sequence Prediction | 5 | 1 | 600 | Regression | Next value from a 5-step window |
| `ts_classify` | Time Series Classification | 10 | 4 | 100 | Sequence | 10-step window → 4 classes |

Custom tasks created on the **Functions** page show up in the same picker with
`func_key = custom_<id>`.

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

React 18 + Vite, no UI kit — all 25 primitives are hand-rolled in
`components/ui.jsx` and styled by a small design-token system.

```
main.jsx
 ├── ThemeProvider      ← light/dark via CSS custom properties, persisted
 ├── ToastProvider      ← toast queue + promise-based useConfirm()
 ├── CatalogProvider    ← registry, datasets, presets, models, functions
 └── SessionProvider    ← one live training session shared by every route
      └── App (react-router, every page React.lazy + Suspense)
           ├── /train       TrainPage      Studio: setup, stack, canvas, inspector
           ├── /datasets    DatasetsPage   library + 4 sample editors
           ├── /playground  PlaygroundPage manual inputs, partial forward passes
           ├── /models      ModelsPage     save / load / export
           ├── /functions   FunctionsPage  custom tasks + bench
           ├── /learn       LearnPage      in-app reference
           ├── /login · /signup            AuthLayout, no catalog/session providers
           └── / and *      → redirect to /train
```

**`state/sessionStore.js` is the heart of it** — a framework-free store class (no
Redux, no Zustand) owning the whole live-model lifecycle: config and layer stack,
build/reset, the training loop, snapshots, evaluation, sweeps, latent probes,
export/import and undo history.  `SessionContext` exposes it through a
`useSyncExternalStore` selector hook, so a component re-renders only when the
slice it asked for actually changes:

```jsx
const running = useSession((s) => s.running)          // re-renders on change only
const layers  = useSession((s) => s.snapshot?.layers, shallowEqual)
useSessionFrames((state) => paint(state.frame))       // canvas: no React re-render
```

**Training loop:** `start()` kicks a `requestAnimationFrame` chain; each frame
issues `POST /api/train/step` with `config.steps` (steps per frame, default 10) and
the current learning rate, then folds the returned weights, activations and loss
history into `frame`.  An in-flight guard means a slow frame never stacks
requests, `stop()` keeps the trained model, and `reset()` re-initialises the
weights.  Changing the task or the stack marks the config dirty, and the next
`start()` rebuilds transparently first.

**Data flow:**
1. User interacts → component calls a store method (`build`, `setConfig`, …)
2. The store calls `api/client.js` — the only file in the app with `fetch()`
3. The response folds into store state → subscribed components re-render
4. Canvas painters (`lib/networkGraph.js`, `archDiagrams.js`, `lossChart.js`,
   `plot2d.js`) are pure functions of `(ctx, data, opts)` — no React state
5. `lib/hooks.js` supplies the shared plumbing: `useCanvas` (DPR-aware canvas
   with a repaint callback), `useHotkeys`, `useElementSize`, `useLocalStorage`,
   `useDebounced`, `useInterval`, `useClickOutside`

**Guards:** `npm run check` walks every source file with Babel's scope analysis
and fails on identifiers that are used but never imported — Rollup happily
bundles those as globals, so a missing import otherwise surfaces as a runtime
`ReferenceError` in the browser (it runs automatically as a `prebuild` hook).
`npm run smoke` server-renders all eight pages inside the real provider stack and
fails if any of them throws, which catches broken destructuring and components
that assume the catalogue has already arrived. `npm run verify` runs both plus the
build.

**Styling:** `styles/tokens.css` defines the design tokens (`--bg`, `--panel`,
`--line`, `--text`, `--muted`, `--accent`, radii, shadows, spacing scales) with a
`[data-theme='light']` override block — a theme swap is one attribute on
`<html>`.  `global.css` holds the base layout and scrollbars, `components.css` the
shared primitives, `pages.css` the page-level layouts.

---

## Design Principles

**1. Zero magic registration** — the `ModuleRegistry` finds every module by
filesystem scan.  No decorator, no `__all__`, no import needed in `__init__.py`.

**2. Strict layer separation**
- `app/core/` has zero Flask imports
- `app/api/` holds no ML maths — it validates, calls `app/core/`, and shapes JSON
- `frontend/src/api/client.js` is the only file with `fetch()` — no DOM, no React
- `frontend/src/state/sessionStore.js` holds all model state — components never
  keep a private copy
- `frontend/src/lib/*` painters and maths helpers are pure — no React, no store

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

**Run the app** (back-end + the committed pre-built front-end):
```
python>=3.10
flask>=3.0
flask-sqlalchemy>=3.1
flask-login>=0.6
numpy>=1.26
safetensors>=0.4
onnx>=1.14                # optional — ONNX export
pytest>=7.4               # optional — test suite
```

**Rebuild the front-end:**
```
node>=18
react 18 · react-dom 18 · react-router-dom 6
vite 5 · @vitejs/plugin-react
```

**Tests:** `python -m pytest tests -q` → 386 passing.

No GPU, no CUDA, no heavy ML framework.  Node is needed once to build the
front-end (`frontend/dist` is git-ignored); after that
`pip install -r requirements.txt && python run.py server` runs the whole app.
