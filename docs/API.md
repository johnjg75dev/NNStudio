# NNStudio API Documentation

This document outlines all API endpoints in NNStudio, including input and output structures.

---

## Authentication Routes (`/api/auth`)

### Check Username Availability
**GET** `/api/auth/check-username?username=<string>`

Check if a username is available for registration.

**Query Parameters:**
- `username` (string, required) - Username to check

**Response:**
```json
{
  "ok": true,
  "data": {
    "available": true,
    "message": "Username is available"
  }
}
```

---

### Login
**GET/POST** `/api/auth/login`

Login page. On POST, accepts form data with username and password.

**Form Parameters:**
- `username` (string, required)
- `password` (string, required)

**Response:** Redirects to home page on success, renders login template on GET.

---

### Signup
**GET/POST** `/api/auth/signup`

Registration page. On POST, creates a new user account.

**Form Parameters:**
- `username` (string, required, min 3 chars)
- `password` (string, required)

**Response:** Redirects to home page on success, renders signup template on GET.

---

### Logout
**GET** `/api/auth/logout`

Logs out the current user.

**Response:** Redirects to login page.

---

## Session Routes (`/api/session`)

### Build Network
**POST** `/api/session/build`

Build a new neural network for the training session.

**Input:**
```json
{
  "func_key": "xor",
  "ds_id": 1,
  "arch_key": "mlp",
  "layers": [{"type": "dense", "neurons": 4, "activation": "tanh"}],
  "inputs": 2,
  "outputs": 1,
  "activation": "tanh",
  "optimizer": "adam",
  "lr": 0.01,
  "loss": "bce",
  "weight_decay": 0.0
}
```

**Output:**
```json
{
  "ok": true,
  "data": {
    "topology": [2, 4, 1],
    "param_count": 13,
    "epoch": 0,
    "func": {
      "key": "xor",
      "label": "XOR Gate",
      "inputs": 2,
      "outputs": 1
    }
  }
}
```

---

### Reset Weights
**POST** `/api/session/reset`

Re-initialize weights without rebuilding the topology.

**Input:** Empty body

**Output:**
```json
{
  "ok": true,
  "data": {
    "message": "Weights reset.",
    "epoch": 0
  }
}
```

---

### Predict
**POST** `/api/session/predict`

Run a single forward pass through the network.

**Input:**
```json
{
  "x": [0.0, 1.0],
  "start_layer": 0,
  "end_layer": null,
  "node_overrides": null
}
```

**Output:**
```json
{
  "ok": true,
  "data": {
    "output": [0.87],
    "activations": [[...], [...]]
  }
}
```

---

### Latent Sweep
**POST** `/api/session/latent-sweep`

Sweep a single latent node range and return output variations.

**Input:**
```json
{
  "x": [0.5, 0.5],
  "layer": 1,
  "node": 0,
  "range": [-2, 2, 0.2]
}
```

**Output:**
```json
{
  "ok": true,
  "data": {
    "sweep_data": [[x_value, y_value], ...]
  }
}
```

---

### Snapshot
**GET** `/api/session/snapshot`

Return a full visual snapshot of the current network state.

**Output:**
```json
{
  "ok": true,
  "data": {
    "built": true,
    "topology": [2, 4, 1],
    "param_count": 13,
    "epoch": 100,
    "loss_history": [...],
    "layers": [
      {
        "index": 0,
        "n_in": 2,
        "n_out": 4,
        "is_output": false,
        "W": [[...]],
        "b": [...],
        "dW": [[...]],
        "activation": "tanh",
        "type": "dense"
      }
    ],
    "activations": [...],
    "func_key": "xor",
    "arch_key": "mlp",
    "func": {...}
  }
}
```

---

### Export Model
**POST** `/api/session/export`

Return the full serialized model as JSON.

**Input:** Empty body

**Output:**
```json
{
  "ok": true,
  "data": {
    "topology": [...],
    "layers": [...],
    "weights": {...},
    ...
  }
}
```

---

### Import Model
**POST** `/api/session/import`

Load a previously exported model.

**Input:** JSON object returned by `/export`

**Output:**
```json
{
  "ok": true,
  "data": {
    "topology": [2, 4, 1],
    "param_count": 13,
    "epoch": 100
  }
}
```

---

## Training Routes (`/api/train`)

### Train Step
**POST** `/api/train/step`

Run N training steps and return updated metrics.

**Input:**
```json
{
  "steps": 10,
  "lr": 0.01
}
```

**Output:**
```json
{
  "ok": true,
  "data": {
    "epoch": 110,
    "loss": 0.123,
    "loss_history": [...],
    "layers": [...],
    "activations": [...]
  }
}
```

---

### Evaluate
**POST** `/api/train/evaluate`

Evaluate training samples or perform a range sweep.

**Input (standard evaluation):**
```json
{}
```

**Input (grid sweep):**
```json
{
  "ranges": [
    {"min": 0, "max": 1, "step": 0.2}
  ],
  "start_layer": 0,
  "end_layer": null
}
```

**Output:**
```json
{
  "ok": true,
  "data": {
    "samples": [
      {
        "x": [0, 0],
        "y": [0],
        "pred": [0.1]
      }
    ],
    "loss": 0.123,
    "accuracy": 0.95
  }
}
```

---

## Dataset Routes (`/api/datasets`)

### List Datasets
**GET** `/api/datasets`

List all datasets for the current user including predefined global ones.

**Output:**
```json
{
  "ok": true,
  "data": {
    "datasets": [
      {
        "id": 1,
        "name": "XOR Gate",
        "description": "The classic exclusive-OR logic gate...",
        "ds_type": "tabular",
        "num_inputs": 2,
        "num_outputs": 1,
        "input_labels": ["A", "B"],
        "output_labels": ["A XOR B"],
        "is_predefined": true
      }
    ]
  }
}
```

---

### Create Dataset
**POST** `/api/datasets`

Create a new dataset.

**Input:**
```json
{
  "name": "My Dataset",
  "description": "Custom dataset",
  "ds_type": "tabular",
  "num_inputs": 2,
  "num_outputs": 1,
  "input_labels": ["X", "Y"],
  "output_labels": ["Z"],
  "data": [{"x": [0, 0], "y": [0]}, ...]
}
```

**Output:**
```json
{
  "ok": true,
  "data": {
    "dataset": {...}
  }
}
```

---

### Get Dataset
**GET** `/api/datasets/<int:ds_id>`

Get full dataset details including all data.

**Output:**
```json
{
  "ok": true,
  "data": {
    "dataset": {
      "id": 1,
      "name": "XOR Gate",
      "data": [{"x": [0, 0], "y": [0]}, ...],
      ...
    }
  }
}
```

---

### Update Dataset
**PUT** `/api/datasets/<int:ds_id>`

Update dataset metadata or data.

**Input:**
```json
{
  "name": "New Name",
  "data": [{"x": [0, 0], "y": [0]}, ...]
}
```

**Output:**
```json
{
  "ok": true,
  "data": {
    "dataset": {...}
  }
}
```

---

### Delete Dataset
**DELETE** `/api/datasets/<int:ds_id>`

Delete a user-created dataset.

**Output:**
```json
{
  "ok": true,
  "data": {
    "message": "Dataset deleted"
  }
}
```

---

### Download Predefined Dataset
**POST** `/api/datasets/<int:ds_id>/download`

Trigger download for a predefined dataset.

**Output:**
```json
{
  "ok": true,
  "data": {
    "message": "Dataset downloaded successfully!"
  }
}
```

---

## Model Routes (`/api/models`)

### Save Model
**POST** `/api/models/save`

Save the current training session's network to the database.

**Input:**
```json
{
  "name": "My Trained Model",
  "description": "XOR network trained for 100 epochs",
  "session_id": "abc123"
}
```

**Output:**
```json
{
  "success": true,
  "model_id": 1,
  "name": "My Trained Model",
  "message": "Model 'My Trained Model' saved successfully"
}
```

---

### List Models
**GET** `/api/models?limit=50&offset=0`

Get all saved models for the current user.

**Query Parameters:**
- `limit` (int, default 50)
- `offset` (int, default 0)

**Output:**
```json
{
  "success": true,
  "models": [
    {
      "id": 1,
      "name": "My Trained Model",
      "description": "...",
      "architecture_name": "mlp",
      "function_name": "xor",
      "epochs_trained": 100,
      "final_loss": 0.05,
      "created_at": "2024-01-01T00:00:00",
      "updated_at": "2024-01-01T00:00:00"
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

---

### Get Model
**GET** `/api/models/<int:model_id>`

Get details of a specific saved model (with weights).

**Output:**
```json
{
  "success": true,
  "model": {
    "id": 1,
    "name": "My Trained Model",
    "model_data": {...},
    ...
  }
}
```

---

### Delete Model
**DELETE** `/api/models/<int:model_id>`

Delete a saved model.

**Output:**
```json
{
  "success": true,
  "message": "Model 'My Trained Model' deleted"
}
```

---

### Export Model
**POST** `/api/models/<int:model_id>/export`

Export a model to various formats.

**Input:**
```json
{
  "format": "json"
}
```

**Output:**
```json
{
  "success": true,
  "format": "json",
  "size_bytes": 1234,
  "message": "Model exported as json (1234 bytes)",
  "download_url": "/api/models/1/download/json"
}
```

---

### Download Model
**GET** `/api/models/<int:model_id>/download/<format>`

Download an exported model file.

**Path Parameters:**
- `model_id` (int)
- `format` - `json`, `safetensors`, `gguf`, `onnx`, or `zip`

**Response:** Binary file download.

---

### Load Model into Session
**POST** `/api/models/<int:model_id>/load-session`

Load a saved model into the training session for further training.

**Input:**
```json
{
  "session_id": "abc123"
}
```

**Output:**
```json
{
  "success": true,
  "message": "Model 'My Trained Model' loaded into session",
  "model_loaded": {
    "name": "My Trained Model",
    "epochs": 100,
    "loss": 0.05,
    "topology": [2, 4, 1],
    "param_count": 13
  }
}
```

---

### Get Supported Formats
**GET** `/api/models/formats`

Get list of supported export formats.

**Output:**
```json
{
  "success": true,
  "formats": ["json", "safetensors", "gguf", "onnx", "zip"],
  "descriptions": {
    "json": "Plain JSON format with full model serialization",
    "safetensors": "Hugging Face SafeTensors efficient format",
    "gguf": "GGML quantized format for fast inference",
    "onnx": "Open Neural Network Exchange format (cross-platform)",
    "zip": "Compressed ZIP archive with model, weights, and metadata"
  }
}
```

---

## Module Routes (`/api/modules`)

### Get All Modules
**GET** `/api/modules/all`

Return the full registry grouped by category (architectures, functions, optimizers, losses, layers, presets).

**Output:**
```json
{
  "ok": true,
  "data": {
    "architectures": [
      {"key": "mlp", "label": "Multi-Layer Perceptron", ...}
    ],
    "functions": [
      {"key": "xor", "label": "XOR Gate", ...}
    ],
    "optimizers": [...],
    "losses": [...],
    "layers": [...],
    "presets": [...]
  }
}
```

---

### Get Modules by Category
**GET** `/api/modules/category/<category>`

Get modules of a specific category.

**Path Parameters:**
- `category` - `architectures`, `functions`, `optimizers`, `losses`, `layers`, `presets`

**Output:**
```json
{
  "ok": true,
  "data": [
    {"key": "mlp", "label": "Multi-Layer Perceptron", ...}
  ]
}
```

---

### Get Single Module
**GET** `/api/modules/<key>`

Get a specific module by key.

**Output:**
```json
{
  "ok": true,
  "data": {
    "key": "xor",
    "label": "XOR Gate",
    ...
  }
}
```

---

### Get Function Dataset
**GET** `/api/modules/functions/<key>/dataset`

Return the raw dataset for a training function.

**Output:**
```json
{
  "ok": true,
  "data": [
    {"x": [0, 0], "y": [0]},
    {"x": [0, 1], "y": [1]},
    ...
  ]
}
```

---

## Preset Routes (`/api/presets`)

### Save Preset
**POST** `/api/presets/save`

Create a new preset from the provided config.

**Input:**
```json
{
  "label": "My Preset",
  "description": "Custom preset config",
  "arch_key": "mlp",
  "func_key": "xor",
  "layers": [{"type": "dense", "neurons": 4, "activation": "tanh"}],
  "activation": "tanh",
  "optimizer": "adam",
  "loss": "bce",
  "lr": 0.01,
  "dropout": 0.0,
  "weight_decay": 0.0
}
```

**Output:**
```json
{
  "ok": true,
  "data": {
    "id": 1,
    "label": "My Preset",
    ...
  }
}
```

---

### Delete Preset
**DELETE** `/api/presets/<int:preset_id>`

Delete a preset if it belongs to the current user.

**Output:**
```json
{
  "ok": true,
  "data": {
    "message": "Preset deleted"
  }
}
```

---

## Custom Function Routes (`/api/functions/custom`)

### Create Custom Function
**POST** `/api/functions/custom`

Create a new custom training function.

**Input:**
```json
{
  "name": "My Function",
  "description": "Custom sine wave",
  "language": "python",
  "code": "def f(x):\n    import math\n    return [math.sin(x[0] * 2 * math.pi)]",
  "num_inputs": 1,
  "num_outputs": 1,
  "input_labels": ["x"],
  "output_labels": ["sin(x)"],
  "is_classification": false,
  "sample_strategy": "linspace"
}
```

**Output:**
```json
{
  "success": true,
  "function": {
    "id": 1,
    "name": "My Function",
    ...
  },
  "message": "Custom function 'My Function' created"
}
```

---

### List Custom Functions
**GET** `/api/functions/custom?limit=50&offset=0`

List all custom functions for the current user.

**Output:**
```json
{
  "success": true,
  "functions": [...],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

---

### Get Custom Function
**GET** `/api/functions/custom/<int:func_id>`

Get custom function details including code.

**Output:**
```json
{
  "success": true,
  "function": {
    "id": 1,
    "name": "My Function",
    "code": "def f(x):\n    ...",
    ...
  }
}
```

---

### Update Custom Function
**PUT** `/api/functions/custom/<int:func_id>`

Update custom function code and metadata.

**Input:**
```json
{
  "name": "New Name",
  "code": "def f(x):\n    return x",
  "input_labels": ["X"]
}
```

**Output:**
```json
{
  "success": true,
  "function": {...},
  "message": "Function updated"
}
```

---

### Delete Custom Function
**DELETE** `/api/functions/custom/<int:func_id>`

Delete a custom function.

**Output:**
```json
{
  "success": true,
  "message": "Function 'My Function' deleted"
}
```

---

### Test Custom Function
**POST** `/api/functions/custom/<int:func_id>/test`

Test custom function with sample input.

**Input:**
```json
{
  "input": [0.5]
}
```

**Output:**
```json
{
  "success": true,
  "output": [0.0],
  "exec_time": 0.001,
  "message": "Function executed successfully"
}
```

---

### Preview Dataset
**POST** `/api/functions/custom/<int:func_id>/preview`

Generate and preview dataset for custom function.

**Input:**
```json
{
  "samples_per_input": 5,
  "strategy": "linspace"
}
```

**Output:**
```json
{
  "success": true,
  "preview": [...],
  "total_samples": 25,
  "first_sample": {"x": [0], "y": [0]},
  "last_sample": {"x": [1], "y": [0]}
}
```

---

### Get Templates
**GET** `/api/functions/custom/templates`

Get code templates for Python and JavaScript.

**Output:**
```json
{
  "success": true,
  "templates": {
    "python": {...},
    "javascript": {...}
  },
  "examples": {
    "python": [...],
    "javascript": [...]
  }
}
```

---

## Admin Routes (`/api/admin`)

### List Architectures
**GET** `/api/admin/architectures`

List all built-in architectures. Requires admin privileges.

**Output:**
```json
{
  "architectures": [
    {"key": "mlp", "label": "Multi-Layer Perceptron", ...}
  ]
}
```

---

### Update Architecture
**PUT** `/api/admin/architectures/<key>`

Update a built-in architecture. Requires admin privileges.

**Input:**
```json
{
  "label": "New Label",
  "description": "Updated description",
  "accent_color": "#ff0000",
  "diagram_type": "custom"
}
```

**Output:**
```json
{
  "message": "Architecture 'mlp' updated",
  "architecture": {...}
}
```

---

### Create Architecture
**POST** `/api/admin/architectures`

Create a new built-in architecture. Requires admin privileges.

**Input:**
```json
{
  "key": "new_arch",
  "label": "New Architecture",
  "description": "Custom architecture",
  "accent_color": "#58a6ff",
  "diagram_type": "generic"
}
```

**Output:**
```json
{
  "message": "Architecture 'new_arch' created",
  "architecture": {...}
}
```
