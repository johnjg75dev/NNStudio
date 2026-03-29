# Custom Training Functions - Complete Guide

## Overview

This system allows you to create and train on custom training functions written in Python or JavaScript. Instead of using predefined functions like XOR or AND gates, you can now write arbitrary functions that define what your neural network should learn.

## How It Works

### Basic Concept

1. **Write a function** — Define a function that takes a fixed-length input array and returns an output array
2. **Save to database** — Store your function with metadata about inputs/outputs
3. **Use for training** — Select your custom function to generate training data
4. **Train your network** — The network learns to approximate your function

### Example: Simple Sum Function

```python
def f(x):
    """Returns the sum of inputs."""
    import numpy as np
    return [np.sum(x)]
```

When training:
- Input: `[2.0, 3.0]` → Output: `[5.0]`
- Input: `[1.0, 1.0]` → Output: `[2.0]`

## Python Function Format

### Syntax

```python
def f(x):
    """
    Your function.
    
    Args:
        x: numpy array of input values
        
    Returns:
        A single number or list of output numbers
    """
    # Your code here
    return output_value_or_array
```

### Requirements

- Function **must** be named `f`
- Must accept single parameter `x` (the input array)
- Must return either:
  - Single number → Auto-wrapped in list
  - List/array of numbers → Used directly
  - Output is automatically padded/truncated to output dimensions

### Available Libraries

Safe execution environment includes:

```python
import numpy as np      # Full numpy support
import math            # Standard math functions

# Available functions:
abs, min, max, round, int, float, list, sum, len, range
```

### Python Examples

#### 1. Simple Arithmetic
```python
def f(x):
    """Sum of all inputs."""
    return sum(x)
```

#### 2. Sine Wave
```python
def f(x):
    """Sine of first input."""
    import math
    return math.sin(x[0] * 2 * math.pi)
```

#### 3. Nonlinear Function
```python
def f(x):
    """Nonlinear: x[0]^2 + x[1]^2."""
    return x[0] ** 2 + x[1] ** 2
```

#### 4. Conditional Logic
```python
def f(x):
    """Returns max of inputs."""
    return max(x[0], x[1])
```

#### 5. Multiple Outputs
```python
def f(x):
    """Returns multiple outputs."""
    import numpy as np
    return [np.sum(x), np.mean(x), np.max(x)]
```

#### 6. Classification
```python
def f(x):
    """Classify: 1 if sum > 1, else 0."""
    return [1.0 if sum(x) > 1.0 else 0.0]
```

## JavaScript Function Format

### Syntax

```javascript
function f(x) {
    // Your function
    // Args:
    //   x: array of input values
    // Returns:
    //   A single number or array of numbers
    
    return output_value_or_array;
}
```

### Requirements

- Function **must** be named `f`
- Must accept single parameter `x` (input array)
- Must return number or array
- Available in JavaScript: standard Math object

### JavaScript Examples

#### 1. Sum
```javascript
function f(x) {
    return x.reduce((a, b) => a + b, 0);
}
```

#### 2. Maximum
```javascript
function f(x) {
    return Math.max(...x);
}
```

#### 3. Sine Function
```javascript
function f(x) {
    return Math.sin(x[0] * 2 * Math.PI);
}
```

#### 4. Multiple Outputs
```javascript
function f(x) {
    return [
        x[0] + x[1],           // sum
        Math.max(...x),        // max
        Math.min(...x)         // min
    ];
}
```

#### 5. XOR Logic
```javascript
function f(x) {
    const xor = (x[0] > 0.5) !== (x[1] > 0.5) ? 1 : 0;
    return [xor];
}
```

## Database Schema

### CustomTrainingFunction Model

```python
class CustomTrainingFunction(db.Model):
    id                    # Primary key
    user_id              # User who created it
    name                 # Display name
    description          # Optional description
    language             # 'python' or 'javascript'
    code                 # Function source code
    num_inputs           # Fixed input array length
    num_outputs          # Fixed output array length
    input_labels         # ["A", "B"] etc
    output_labels        # ["Result"] etc
    is_classification    # Boolean flag
    sample_strategy      # 'linspace', 'random', or 'custom'
    custom_dataset       # Optional manual dataset
    is_valid             # Passes validation
    last_test_result     # Last execution result
    created_at           # Timestamp
    updated_at           # Timestamp
```

## API Endpoints

### 1. Create Custom Function

**POST** `/api/functions/custom`

Create a new custom training function.

```bash
curl -X POST http://localhost:5000/api/functions/custom \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Sum Function",
    "description": "Returns sum of inputs",
    "language": "python",
    "code": "def f(x):\n    return [sum(x)]",
    "num_inputs": 2,
    "num_outputs": 1,
    "input_labels": ["A", "B"],
    "output_labels": ["Sum"],
    "is_classification": false,
    "sample_strategy": "linspace"
  }'
```

**Response:**
```json
{
  "success": true,
  "function": {
    "id": 1,
    "name": "Sum Function",
    "num_inputs": 2,
    "num_outputs": 1,
    "is_valid": true,
    "created_at": "2026-03-28T12:34:56"
  }
}
```

### 2. List Custom Functions

**GET** `/api/functions/custom`

Get all custom functions for current user.

```bash
curl http://localhost:5000/api/functions/custom?limit=10&offset=0
```

**Response:**
```json
{
  "success": true,
  "functions": [
    {
      "id": 1,
      "name": "Sum Function",
      "language": "python",
      "num_inputs": 2,
      "num_outputs": 1,
      "is_valid": true,
      "created_at": "2026-03-28T12:34:56"
    }
  ],
  "total": 1
}
```

### 3. Get Function Details

**GET** `/api/functions/custom/<id>`

Get full function details including source code.

```bash
curl http://localhost:5000/api/functions/custom/1
```

**Response:**
```json
{
  "success": true,
  "function": {
    "id": 1,
    "name": "Sum Function",
    "code": "def f(x):\n    return [sum(x)]",
    "language": "python",
    "num_inputs": 2,
    "num_outputs": 1,
    "is_classification": false,
    "sample_strategy": "linspace",
    "last_test_result": {...}
  }
}
```

### 4. Update Function

**PUT** `/api/functions/custom/<id>`

Update function code or metadata.

```bash
curl -X PUT http://localhost:5000/api/functions/custom/1 \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def f(x):\n    import numpy as np\n    return [np.sum(x)]",
    "is_classification": false
  }'
```

### 5. Delete Function

**DELETE** `/api/functions/custom/<id>`

```bash
curl -X DELETE http://localhost:5000/api/functions/custom/1
```

### 6. Test Function

**POST** `/api/functions/custom/<id>/test`

Execute function with sample input.

```bash
curl -X POST http://localhost:5000/api/functions/custom/1/test \
  -H "Content-Type: application/json" \
  -d '{"input": [2.0, 3.0]}'
```

**Response:**
```json
{
  "success": true,
  "output": [5.0],
  "exec_time": 0.001,
  "message": "Function executed successfully"
}
```

### 7. Preview Dataset

**POST** `/api/functions/custom/<id>/preview`

Generate and preview training dataset.

```bash
curl -X POST http://localhost:5000/api/functions/custom/1/preview \
  -H "Content-Type: application/json" \
  -d '{"samples_per_input": 5, "strategy": "linspace"}'
```

**Response:**
```json
{
  "success": true,
  "preview": [
    {"x": [0.0, 0.0], "y": [0.0]},
    {"x": [0.25, 0.0], "y": [0.25]},
    {"x": [0.5, 0.0], "y": [0.5]},
    {"x": [0.75, 0.0], "y": [0.75]},
    {"x": [1.0, 0.0], "y": [1.0]}
  ],
  "total_samples": 25,
  "first_sample": {"x": [0.0, 0.0], "y": [0.0]},
  "last_sample": {"x": [1.0, 1.0], "y": [2.0]}
}
```

### 8. Get Code Templates

**GET** `/api/functions/custom/templates`

Get starter templates for Python and JavaScript.

```bash
curl http://localhost:5000/api/functions/custom/templates
```

## Sample Strategies

When creating a dataset, choose a strategy:

### 1. Linspace (Grid Sampling)
- Generates uniform grid across [0, 1] for each input
- For 2D: Creates NxN samples
- Best for: Classification, discrete outputs
- Example: 5 samples per input → 25 total samples for 2D

### 2. Random
- Generates random samples in [0, 1] range
- Fixed number of samples
- Best for: Continuous functions, high dimensions
- Example: 100 random samples

### 3. Custom
- Use manually provided dataset
- Complete control over training data
- Best for: Specific edge cases, non-uniform distributions

## Integration with Training

### Using Custom Functions in Training

1. **Create your function** via API or UI
2. **Select it** when configuring a training session
   - Function will appear in module list as `custom_<id>_<name>`
3. **Configure network** based on input/output dimensions
4. **Start training** — Network trains on generated dataset

### Example Workflow (Python)

```python
# 1. Create custom function
import requests

function_data = {
    "name": "Sine Wave",
    "language": "python",
    "code": """
def f(x):
    import math
    return [math.sin(x[0] * 2 * math.pi)]
""",
    "num_inputs": 1,
    "num_outputs": 1,
    "is_classification": False,
    "sample_strategy": "linspace"
}

response = requests.post(
    'http://localhost:5000/api/functions/custom',
    json=function_data,
    cookies={'session_id': 'your_session_id'}
)

func_id = response.json()['function']['id']

# 2. Test the function
test_response = requests.post(
    f'http://localhost:5000/api/functions/custom/{func_id}/test',
    json={"input": [0.5]},
    cookies={'session_id': 'your_session_id'}
)

print(test_response.json()['output'])  # Should be ~0.951

# 3. Preview dataset
preview_response = requests.post(
    f'http://localhost:5000/api/functions/custom/{func_id}/preview',
    json={"samples_per_input": 10},
    cookies={'session_id': 'your_session_id'}
)

# 4. Use in training session (select custom function)
```

## Code Validation

### Python Validation

Checks for:
- ✅ Valid Python syntax
- ✅ Function named `f`
- ✅ Takes single parameter `x`
- ✅ Executable code

### JavaScript Validation

Checks for:
- ✅ Function named `f`
- ✅ Matched braces and parentheses
- ✅ Basic syntax

### Testing

Always test your function before training:

```bash
curl -X POST http://localhost:5000/api/functions/custom/<id>/test \
  -H "Content-Type: application/json" \
  -d '{"input": [0.5, 0.3]}'
```

## Common Patterns

### Pattern 1: Binary Classification
```python
def f(x):
    """XOR gate: 1 if inputs differ."""
    return [1.0 if (x[0] > 0.5) != (x[1] > 0.5) else 0.0]
```

### Pattern 2: Regression
```python
def f(x):
    """Linear regression: y = 2x + 1."""
    return [2 * x[0] + 1]
```

### Pattern 3: Multi-Output
```python
def f(x):
    """Returns sum and product."""
    import numpy as np
    return [np.sum(x), np.prod(x)]
```

### Pattern 4: Nonlinear
```python
def f(x):
    """Polynomial: x^2 + 2xy + y^2."""
    return [(x[0]**2) + (2*x[0]*x[1]) + (x[1]**2)]
```

### Pattern 5: Conditional
```python
def f(x):
    """Step function."""
    return [1.0 if sum(x) > 0.5 else 0.0]
```

## Troubleshooting

### Function Won't Execute

**Error:** "Syntax error in code"
**Solution:** Check Python/JavaScript syntax is valid

**Error:** "Function must be named 'f'"
**Solution:** Ensure function is named exactly `f`

**Error:** "Code compilation failed"
**Solution:** Test code locally, check for import statements

### Dataset Generation Failed

**Error:** "Failed to generate dataset"
**Solution:** 
- Verify function executes correctly (use /test endpoint)
- Check num_inputs and num_outputs match your code

### Output Size Mismatch

**Error:** "Input size mismatch"  
**Solution:** Ensure input array size matches `num_inputs`

**Behavior:** Output is automatically padded/truncated
- Too small → Padded with 0.0
- Too large → Truncated to expected size

### Performance Issues

**Issue:** Dataset generation is slow
**Solution:**
- Use `random` strategy instead of `linspace` for high dimensions
- Reduce `samples_per_input`
- Simplify function logic

## Security Notes

- **Python execution** uses restricted namespace (no file I/O, no imports besides numpy/math)
- **Each function is per-user** — Can't access other users' code
- **Code is validated** before execution
- **Timeouts** prevent infinite loops

## Frontend Integration

### React Example

```javascript
// Create function
async function createCustomFunction() {
    const response = await fetch('/api/functions/custom', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            name: 'My Function',
            language: 'python',
            code: 'def f(x):\n    return [sum(x)]',
            num_inputs: 2,
            num_outputs: 1
        })
    });
    return response.json();
}

// List functions
async function listCustomFunctions() {
    const response = await fetch('/api/functions/custom');
    return response.json();
}

// Test function
async function testFunction(funcId, input) {
    const response = await fetch(`/api/functions/custom/${funcId}/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input })
    });
    return response.json();
}
```

## Files Created/Modified

### New Files
- `app/models/custom_function.py` - Database model
- `app/core/function_executor.py` - Execution engine
- `app/modules/functions/custom_function_wrapper.py` - Training integration
- `app/api/custom_function_routes.py` - API endpoints
- `tests/test_custom_functions.py` - Test suite

### Modified Files
- `app/models/__init__.py` - Added CustomTrainingFunction import
- `app/__init__.py` - Registered custom_function_bp blueprint
- `app/modules/registry.py` - Added get_with_custom() method

## Advanced Features

### Custom Datasets

Provide your own dataset instead of generating:

```javascript
{
  "name": "Custom Iris Data",
  "code": "def f(x):\n    return [1.0]",
  "num_inputs": 4,
  "num_outputs": 1,
  "sample_strategy": "custom",
  "custom_dataset": [
    {"x": [5.1, 3.5, 1.4, 0.2], "y": [0.0]},
    {"x": [7.0, 3.2, 4.7, 1.4], "y": [1.0]},
    {"x": [6.3, 3.3, 6.0, 2.5], "y": [2.0]}
  ]
}
```

### Estimated Samples

- Linspace 1D: `samples_per_input` samples
- Linspace 2D: `samples_per_input^2` samples
- Linspace 3D+: Falls back to random sampling
- Random: Fixed `num_samples`

## Performance Tips

1. **Keep functions simple** — Complex math is slower
2. **Use random sampling for high dimensions** — Linspace explodes combinatorially
3. **Reduce samples_per_input** — Affects training time
4. **Test before training** — Catch errors early
5. **Monitor execution** — Check exec_time in test results

