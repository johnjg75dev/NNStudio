# Model Save/Load/Export System Documentation

## Overview

This system provides complete model persistence and export capabilities for NNStudio. You can now:

- **Save** trained models to the database
- **Load** saved models back into training sessions
- **Export** models to multiple formats for sharing and Integration
- **Download** exported models as files

## Features

### Save/Load to Database

Save trained models with metadata to SQLite database:
- Model architecture and weights
- Training history (epochs, loss, accuracy)
- Custom names and descriptions
- Timestamps for organization

### Export Formats

Models can be exported to 5 different formats:

1. **JSON** - Plain text JSON serialization
   - Human-readable
   - Easy to interface with web applications
   - Includes full model definition and weights

2. **SafeTensors** - Hugging Face efficient format
   - Fast loading and saving
   - Memory-efficient
   - Safe by design (no code execution)
   - Compatible with transformer ecosystem

3. **GGUF** - GGML quantized format
   - Optimized for inference
   - Supports quantization
   - Small file size
   - Requires `gguf` library

4. **ONNX** - Open Neural Network Exchange
   - Cross-platform model interchange
   - Framework-agnostic
   - Compatible with many deployment targets
   - Requires `onnx` library

5. **ZIP** - Packaged archive
   - Complete model bundle
   - Includes model.json, metadata.json, weights.npz, and config.txt
   - Best for sharing and backup
   - Self-contained delivery

## Database Schema

### SavedModel Model

```python
class SavedModel(db.Model):
    id                  # Primary key
    user_id            # Foreign key to User
    name               # Model name (required)
    description        # Optional description
    model_data         # JSON - Full network serialization
    architecture_name  # e.g., "mlp", "cnn"
    function_name      # e.g., "xor", "iris"
    epochs_trained     # Number of training epochs
    final_loss         # Final loss value
    final_accuracy     # Final accuracy (optional)
    created_at         # Timestamp
    updated_at         # Timestamp
```

## API Endpoints

### 1. Save Model to Database

**POST** `/api/models/save`

Save the current training session's network to database.

```bash
curl -X POST http://localhost:5000/api/models/save \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Trained Model",
    "description": "XOR network with 2 hidden layers",
    "session_id": "optional-session-id"
  }'
```

**Response:**
```json
{
  "success": true,
  "model_id": 1,
  "name": "My Trained Model",
  "message": "Model 'My Trained Model' saved successfully"
}
```

### 2. List Saved Models

**GET** `/api/models`

Get all models for current user with pagination.

```bash
curl http://localhost:5000/api/models?limit=10&offset=0
```

**Query Parameters:**
- `limit` (int, default 50): Max results per page
- `offset` (int, default 0): Pagination offset

**Response:**
```json
{
  "success": true,
  "models": [
    {
      "id": 1,
      "name": "My Trained Model",
      "description": "XOR network",
      "architecture_name": "mlp",
      "function_name": "xor",
      "epochs_trained": 100,
      "final_loss": 0.001,
      "final_accuracy": null,
      "created_at": "2026-03-28T12:34:56",
      "updated_at": "2026-03-28T12:34:56"
    }
  ],
  "total": 1,
  "limit": 10,
  "offset": 0
}
```

### 3. Get Model Details (with Weights)

**GET** `/api/models/<model_id>`

Get complete model data including weights.

```bash
curl http://localhost:5000/api/models/1
```

**Response:**
```json
{
  "success": true,
  "model": {
    "id": 1,
    "name": "My Trained Model",
    "...metadata fields...",
    "model_data": {
      "layers": [...],
      "optimizer": "adam",
      "optimizer_lr": 0.01,
      "loss": "mse",
      "epoch": 100,
      "loss_history": [...]
    }
  }
}
```

### 4. Delete Model

**DELETE** `/api/models/<model_id>`

Delete a saved model.

```bash
curl -X DELETE http://localhost:5000/api/models/1
```

**Response:**
```json
{
  "success": true,
  "message": "Model 'My Trained Model' deleted"
}
```

### 5. Export Model to Various Formats

**POST** `/api/models/<model_id>/export`

Export model to specified format (generates download URL).

```bash
curl -X POST http://localhost:5000/api/models/1/export \
  -H "Content-Type: application/json" \
  -d '{"format": "json"}'
```

**Supported Formats:**
- `json` - Plain JSON
- `zip` - Packaged archive
- `safetensors` - SafeTensors format
- `onnx` - ONNX format
- `gguf` - GGUF format

**Response:**
```json
{
  "success": true,
  "format": "json",
  "size_bytes": 125896,
  "message": "Model exported as json (125896 bytes)",
  "download_url": "/api/models/1/download/json"
}
```

### 6. Download Exported Model File

**GET** `/api/models/<model_id>/download/<format>`

Download the exported model file.

```bash
curl http://localhost:5000/api/models/1/download/json > model.json
curl http://localhost:5000/api/models/1/download/zip > model.zip
curl http://localhost:5000/api/models/1/download/safetensors > model.safetensors
```

Returns the file as binary attachment with appropriate MIME type.

### 7. Load Model into Training Session

**POST** `/api/models/<model_id>/load-session`

Load a saved model back into the training session for further training.

```bash
curl -X POST http://localhost:5000/api/models/1/load-session \
  -H "Content-Type: application/json" \
  -d '{"session_id": "optional-session-id"}'
```

**Response:**
```json
{
  "success": true,
  "message": "Model 'My Trained Model' loaded into session",
  "model_loaded": {
    "name": "My Trained Model",
    "epochs": 100,
    "loss": 0.001,
    "topology": [2, 4, 1],
    "param_count": 25
  }
}
```

### 8. Get Supported Export Formats

**GET** `/api/models/formats`

Get list of available export formats with descriptions.

```bash
curl http://localhost:5000/api/models/formats
```

**Response:**
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

## Usage Examples

### Python Example: Using the Exporter Directly

```python
from app.core.exporters import ModelExporter
from app.core.network import NeuralNetwork

# Load a network (from training or database)
network = NeuralNetwork.from_dict(model_dict)

# Export to different formats
metadata = {
    "name": "My Model",
    "description": "A trained network",
    "epochs": 100
}

# JSON
ModelExporter.export(network, "json", "model.json", metadata)

# SafeTensors
ModelExporter.export(network, "safetensors", "model.safetensors", metadata)

# ZIP (recommended for complete backup)
ModelExporter.export(network, "zip", "model.zip", metadata)

# Get as bytes (for HTTP response)
json_bytes = ModelExporter.export_bytes(network, "json", metadata)
zip_bytes = ModelExporter.export_bytes(network, "zip", metadata)
```

### Flask Endpoint Example: Custom Export Logic

```python
from flask import Blueprint, request, jsonify, send_file
from app.models import SavedModel
from app.core.exporters import ModelExporter
from app.core.network import NeuralNetwork
from io import BytesIO

model_bp = Blueprint('models', __name__)

@model_bp.route('/models/<model_id>/export/<format>', methods=['GET'])
def export_and_download(model_id, format):
    model = SavedModel.query.get(model_id)
    network = NeuralNetwork.from_dict(model.model_data)
    
    # Export to bytes
    export_bytes = ModelExporter.export_bytes(network, format)
    
    return send_file(
        BytesIO(export_bytes),
        mimetype='application/json' if format == 'json' else 'application/octet-stream',
        as_attachment=True,
        download_name=f"{model.name}.{format}"
    )
```

### JavaScript/Frontend Example: Save and Export

```javascript
// Save current model
async function saveModel() {
    const response = await fetch('/api/models/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            name: 'My Trained Model',
            description: 'XOR network'
        })
    });
    
    const data = await response.json();
    const modelId = data.model_id;
    return modelId;
}

// List all models
async function listModels() {
    const response = await fetch('/api/models?limit=20');
    const data = await response.json();
    console.log(data.models);
}

// Export and download
async function downloadModel(modelId, format) {
    // First, create export
    const exportResponse = await fetch(`/api/models/${modelId}/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format: format })
    });
    
    const exportData = await exportResponse.json();
    
    // Then download
    const downloadUrl = exportData.download_url;
    window.location.href = downloadUrl;
}

// Load model into session
async function loadModel(modelId) {
    const response = await fetch(`/api/models/${modelId}/load-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    });
    
    const data = await response.json();
    console.log('Model loaded:', data.model_loaded);
}
```

## ZIP Format Structure

When exporting to ZIP format, the archive contains:

```
model.zip
├── model.json          # Full network serialization (layers, weights)
├── metadata.json       # Training metadata and info
├── weights.npz         # NumPy compressed weights archive
└── config.txt          # Human-readable configuration
```

### Contents:

**model.json:**
```json
{
  "network": {
    "layers": [...],
    "optimizer": "adam",
    "loss": "mse",
    "epoch": 100,
    "loss_history": [...]
  },
  "topology": [2, 4, 1],
  "param_count": 25
}
```

**metadata.json:**
```json
{
  "name": "My Model",
  "description": "...",
  "architecture": "mlp",
  "function": "xor",
  "epochs": 100,
  "final_loss": 0.001,
  "optimizer": "Adam",
  "learning_rate": 0.01
}
```

**weights.npz:**
- Contains NumPy arrays for each layer's weights and biases
- Can be loaded with: `numpy.load('weights.npz')`

## Installation Requirements

### Core Dependencies
Already in requirements.txt:
- `flask>=3.0`
- `numpy>=1.26`
- `flask-sqlalchemy`

### Optional Export Format Dependencies

Add to requirements.txt as needed:

```bash
# For SafeTensors export
pip install safetensors>=0.4.0

# For ONNX export
pip install onnx>=1.14.0

# For GGUF export (requires more setup)
pip install gguf
```

Current requirements.txt already includes:
- `safetensors>=0.4.0`
- `onnx>=1.14.0`

## Testing

Run the test suite:

```bash
pytest tests/test_model_operations.py -v
```

Test coverage includes:
- JSON export/import
- ZIP export structure
- SafeTensors export
- Database operations
- API endpoint functionality
- Network serialization round-trip

## Error Handling

### Common Errors and Solutions

1. **"No trained network in session"**
   - Train a network first before saving
   - Ensure session_id is correct

2. **"Model not found"**
   - Check model_id exists and belongs to current user
   - Verify authentication

3. **"Unsupported format"**
   - Use one of: json, zip, safetensors, gguf, onnx
   - Install required packages for unavailable formats

4. **ImportError: "safetensors not installed"**
   ```bash
   pip install safetensors
   ```

5. **ImportError: "onnx not installed"**
   ```bash
   pip install onnx
   ```

## Performance Considerations

- **JSON** - Fast but larger files (best for small models)
- **SafeTensors** - Very fast, memory-efficient (recommended)
- **ZIP** - Good compression, slower than SafeTensors
- **ONNX** - Cross-platform, slight overhead
- **GGUF** - Optimized for inference, requires ggml setup

## Security

- Models are stored per-user in database
- Authentication required for all endpoints
- No arbitrary code execution in SafeTensors format
- ZIP archives validated before extraction
- All user inputs validated

## Future Enhancements

Potential improvements:
- Model versioning and history
- Model sharing/permissions between users
- Automatic format conversion
- Model compression/quantization options
- Training resumption from checkpoint
- Batch export operations
- Model comparison tools
- Integration with cloud storage (S3, etc.)

## Troubleshooting

### Models not appearing in list
- Verify you're querying as the correct user
- Check database is initialized: `db.create_all()`

### Export fails with missing library
- Install the required package
- Check library version compatibility

### Large models slow to export
- ZIP exports large networks can be slow
- Use SafeTensors format for faster export
- Consider model compression

### Can't load model into session
- Verify session_id is valid
- Check model exists and is accessible
- Ensure no active training in progress

## Files Modified/Created

### New Files:
- `app/models/saved_model.py` - SavedModel database model
- `app/core/exporters.py` - Export functionality for all formats
- `app/api/model_routes.py` - API endpoints
- `tests/test_model_operations.py` - Comprehensive tests

### Modified Files:
- `app/models/__init__.py` - Added SavedModel import
- `app/__init__.py` - Registered model_bp blueprint
- `app/api/helpers.py` - Added get_session_manager() helper
- `requirements.txt` - Added safetensors and onnx

## Support

For issues or questions:
1. Check this documentation
2. Run tests: `pytest tests/test_model_operations.py`
3. Review error messages
4. Check database is properly initialized
