"""
app/modules/functions/custom_function_wrapper.py
Wrapper that converts CustomTrainingFunction database records into 
TrainingFunction modules compatible with the existing training system.
"""
from .base_function import TrainingFunction
from ...core.function_executor import FunctionExecutor, DatasetGenerator
import numpy as np


class DynamicCustomFunction(TrainingFunction):
    """
    Dynamically created TrainingFunction wrapping a CustomTrainingFunction from database.
    """
    
    def __init__(self, custom_func_record):
        """
        Args:
            custom_func_record: CustomTrainingFunction database model instance
        """
        self.custom_func_record = custom_func_record
        
        # Set BaseModule attributes
        self.key = f"custom_{custom_func_record.id}"
        self.label = custom_func_record.name
        self.description = custom_func_record.description or "Custom training function"
        self.category = "functions"
        
        # Set TrainingFunction attributes
        self.inputs = custom_func_record.num_inputs
        self.outputs = custom_func_record.num_outputs
        self.input_labels = custom_func_record.input_labels or []
        self.output_labels = custom_func_record.output_labels or []
        self.is_classification = custom_func_record.is_classification
        
        # Default recommended config
        self.recommended = {
            "layers": [{"neurons": 8, "activation": "tanh", "type": "dense"}],
            "optimizer": "adam",
            "loss": "mse" if not self.is_classification else "bce",
            "dropout": 0.0,
            "lr": 0.01,
        }
        
        # Cached dataset
        self._dataset_cache = None
    
    def f(self, x: np.ndarray) -> list[float]:
        """Directly call the custom function with an input array."""
        func = self._get_executor_func()
        return func(x)

    def _get_executor_func(self):
        """Get executable function from code."""
        code = self.custom_func_record.code
        language = self.custom_func_record.language
        num_outputs = self.custom_func_record.num_outputs
        
        if language == 'python':
            # Compile Python function
            safe_globals = {
                'np': np,
                'numpy': np,
                'math': __import__('math'),
                'abs': abs,
                'min': min,
                'max': max,
                'round': round,
                'int': int,
                'float': float,
                'list': list,
                'sum': sum,
                'len': len,
            }
            safe_locals = {}
            
            try:
                exec(code, safe_globals, safe_locals)
                func = safe_locals.get('f')
                
                if func is None:
                    raise RuntimeError("Function 'f' not found in code")
                
                # Wrap to ensure consistent output format
                def wrapper(x):
                    result = func(x)
                    if isinstance(result, (int, float, np.number)):
                        output = [float(result)]
                    else:
                        output = [float(v) for v in result]
                    
                    # Normalize to expected size
                    if len(output) < num_outputs:
                        output.extend([0.0] * (num_outputs - len(output)))
                    else:
                        output = output[:num_outputs]
                    
                    return output
                
                return wrapper
            except Exception as e:
                raise RuntimeError(f"Failed to compile function: {str(e)}")
        
        elif language == 'javascript':
            # For JavaScript, we need to evaluate it somehow
            # For now, raise error - this should be handled on frontend
            raise RuntimeError(
                "JavaScript functions must be executed on the frontend. "
                "Use Python for backend execution."
            )
        
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    def generate_dataset(self):
        """
        Generate training dataset from custom function.
        
        Strategies:
        - 'custom': Use provided custom_dataset from database
        - 'linspace': Generate by sampling across [0, 1] grid
        - 'random': Generate random samples
        """
        # Return cached if available
        if self._dataset_cache is not None:
            return self._dataset_cache
        
        strategy = self.custom_func_record.sample_strategy or 'linspace'
        
        try:
            func = self._get_executor_func()
        except Exception:
            # Fallback to empty dataset if code is invalid
            return [{"x": [0.0] * self.inputs, "y": [0.0] * self.outputs}]
        
        # Generate based on strategy
        if strategy == 'custom':
            # Use provided dataset
            if self.custom_func_record.custom_dataset:
                dataset = self.custom_func_record.custom_dataset
            else:
                # Fallback to linspace
                dataset = DatasetGenerator.generate_linspace(
                    func, self.inputs, self.outputs, samples_per_input=10
                )
        
        elif strategy == 'linspace':
            # Generate grid samples
            dataset = DatasetGenerator.generate_linspace(
                func, self.inputs, self.outputs, samples_per_input=10
            )
        
        elif strategy == 'random':
            # Generate random samples
            dataset = DatasetGenerator.generate_random(
                func, self.inputs, self.outputs, num_samples=100
            )
        
        else:
            # Unknown strategy - use linspace
            dataset = DatasetGenerator.generate_linspace(
                func, self.inputs, self.outputs, samples_per_input=10
            )
        
        self._dataset_cache = dataset
        return dataset
    
    def to_dict(self):
        """Serialize to dict."""
        result = super().to_dict()
        result['custom_id'] = self.custom_func_record.id
        result['language'] = self.custom_func_record.language
        result['is_custom'] = True
        return result
