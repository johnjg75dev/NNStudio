# Custom Functions Quick Start Guide

## 5-Minute Setup

### Step 1: Create a Simple Function

**Python Example:**
```python
def f(x):
    """Returns sum of inputs."""
    return [sum(x)]
```

**JavaScript Example:**
```javascript
function f(x) {
    return x.reduce((a, b) => a + b, 0);
}
```

### Step 2: Define Input/Output

```
Inputs:  2  (A, B)
Outputs: 1  (Sum)
Classification: No
Strategy: linspace
```

### Step 3: Create via API

```bash
curl -X POST http://localhost:5000/api/functions/custom \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Sum Function",
    "language": "python",
    "code": "def f(x):\n    return [sum(x)]",
    "num_inputs": 2,
    "num_outputs": 1,
    "input_labels": ["A", "B"],
    "output_labels": ["Result"]
  }'
```

### Step 4: Test It

```bash
curl -X POST http://localhost:5000/api/functions/custom/1/test \
  -H "Content-Type: application/json" \
  -d '{"input": [3, 4]}'

# Response: {"success": true, "output": [7.0]}
```

### Step 5: Preview Dataset

```bash
curl -X POST http://localhost:5000/api/functions/custom/1/preview \
  -H "Content-Type: application/json" \
  -d '{"samples_per_input": 5}'

# Returns 25 samples (5x5 grid)
```

### Step 6: Use in Training

1. Create training session
2. Select function: `custom_1_Sum Function`
3. Choose network architecture based on 2 inputs → 1 output
4. Train normally

## Common Starter Functions

### Sum
```python
def f(x):
    return [sum(x)]
```

### Product
```python
def f(x):
    return [x[0] * x[1]]
```

### Maximum
```python
def f(x):
    return [max(x)]
```

### Squared Distance
```python
def f(x):
    return [x[0]**2 + x[1]**2]
```

### XOR (Binary)
```python
def f(x):
    return [1.0 if (x[0] > 0.5) != (x[1] > 0.5) else 0.0]
```

### Sine Wave
```python
def f(x):
    import math
    return [math.sin(x[0] * 2 * math.pi)]
```

## Function Templates

Get templates:
```bash
curl http://localhost:5000/api/functions/custom/templates
```

## Workflow Example

```bash
# 1. Create
curl -X POST http://localhost:5000/api/functions/custom \
  -d '{"name":"Test","language":"python",...}'

# 2. Get ID from response: 42

# 3. Test
curl -X POST http://localhost:5000/api/functions/custom/42/test \
  -d '{"input":[0.5,0.3]}'

# 4. Preview
curl -X POST http://localhost:5000/api/functions/custom/42/preview

# 5. Use in training (select "custom_42_Test")

# 6. Update (if needed)
curl -X PUT http://localhost:5000/api/functions/custom/42 \
  -d '{"code":"new code..."}'

# 7. Delete (if done)
curl -X DELETE http://localhost:5000/api/functions/custom/42
```

## Tips & Tricks

### Tip 1: Keep Functions Simple
Execution time matters for training. Simpler = faster.

### Tip 2: Test Early
Always test with `/test` before using in training.

### Tip 3: Check Output Shape
Verify function outputs expected number of values.

### Tip 4: Use Linspace for Classification
Grid sampling covers input space uniformly.

### Tip 5: Use Random for Complex Functions
Random sampling is better for high dimensions.

## Common Mistakes

❌ **Wrong function name**
```python
def compute(x):  # Wrong - must be 'f'
    return [sum(x)]
```

✅ **Correct**
```python
def f(x):
    return [sum(x)]
```

---

❌ **Forgetting to return list**
```python
def f(x):
    return sum(x)  # Returns number, ok but inconsistent
```

✅ **Better**
```python
def f(x):
    return [sum(x)]  # Explicit list
```

---

❌ **Wrong output size**
```python
def f(x):
    return [1, 2, 3]  # But num_outputs=1
```

✅ **Automatic handling**
```python
def f(x):
    return [1, 2, 3]  # Auto-truncated to [1]
```

---

❌ **Missing numpy import**
```python
def f(x):
    return [np.sum(x)]  # NameError!
```

✅ **Import inside function**
```python
def f(x):
    import numpy as np
    return [np.sum(x)]
```

## Debugging

### Function won't execute
Check test output:
```bash
curl -X POST http://localhost:5000/api/functions/custom/42/test \
  -d '{"input":[1,2]}'
```
Look for error message.

### Dataset is weird
Preview first:
```bash
curl -X POST http://localhost:5000/api/functions/custom/42/preview
```
Check sample values make sense.

### Training isn't learning
- Function might be too simple (need hidden layers)
- Function might be too complex (network too small)
- Learning rate might be wrong
- Try simpler function first to verify setup

## Advanced Example: Multi-Output

```python
def f(x):
    """Returns [sum, product, max, min]."""
    import numpy as np
    return [
        np.sum(x),
        np.prod(x),
        np.max(x),
        np.min(x)
    ]
```

Setup: `num_inputs=2, num_outputs=4`

## API Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/functions/custom` | POST | Create function |
| `/api/functions/custom` | GET | List functions |
| `/api/functions/custom/<id>` | GET | Get function code |
| `/api/functions/custom/<id>` | PUT | Update function |
| `/api/functions/custom/<id>` | DELETE | Delete function |
| `/api/functions/custom/<id>/test` | POST | Execute test |
| `/api/functions/custom/<id>/preview` | POST | Preview dataset |
| `/api/functions/custom/templates` | GET | Get templates |

## Next Steps

1. ✅ Create your first custom function
2. ✅ Test it with sample inputs
3. ✅ Preview the generated dataset
4. ✅ Select it in training session
5. ✅ Train your network
6. ✅ Experiment with different functions

Happy training! 🚀
