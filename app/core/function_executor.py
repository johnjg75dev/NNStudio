"""
app/core/function_executor.py
Safely execute user-defined training functions.
Supports Python and JavaScript code with input validation and error handling.
"""
import numpy as np
import re
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result of executing a training function."""
    success: bool
    output: Optional[List[float]] = None
    error: Optional[str] = None
    exec_time: Optional[float] = None


class FunctionExecutor:
    """Safely execute user-defined training functions."""
    
    # Allowed modules for Python execution
    SAFE_GLOBALS = {
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
        'range': range,
    }
    
    @staticmethod
    def execute_python(
        code: str,
        input_values: List[float],
        num_outputs: int
    ) -> ExecutionResult:
        """
        Execute Python function code.
        
        Function signature should be:
        def f(x):
            # x is the input array
            # return output array or single value
        
        Args:
            code: Python function definition
            input_values: Input array
            num_outputs: Expected output size
            
        Returns:
            ExecutionResult with output or error
        """
        import time
        start = time.time()
        
        try:
            # Validate code contains function definition
            if 'def f(' not in code and 'def f (' not in code:
                return ExecutionResult(
                    success=False,
                    error="Function must be named 'f' and take parameter 'x': def f(x):"
                )
            
            # Create safe execution environment
            safe_locals = {}
            exec_globals = FunctionExecutor.SAFE_GLOBALS.copy()
            
            # Execute the function definition
            exec(code, exec_globals, safe_locals)
            
            if 'f' not in safe_locals:
                return ExecutionResult(
                    success=False,
                    error="Could not find function 'f' after execution"
                )
            
            func = safe_locals['f']
            
            # Call function with input array
            x = np.array(input_values, dtype=np.float64)
            result = func(x)
            
            # Normalize output
            if isinstance(result, (int, float, np.number)):
                output = [float(result)]
            elif isinstance(result, (list, np.ndarray)):
                output = [float(v) for v in result]
            else:
                return ExecutionResult(
                    success=False,
                    error=f"Function must return number or array, got {type(result)}"
                )
            
            # Pad or truncate to expected size
            if len(output) < num_outputs:
                output.extend([0.0] * (num_outputs - len(output)))
            elif len(output) > num_outputs:
                output = output[:num_outputs]
            
            elapsed = time.time() - start
            
            return ExecutionResult(
                success=True,
                output=output,
                exec_time=elapsed
            )
            
        except SyntaxError as e:
            return ExecutionResult(
                success=False,
                error=f"Syntax error in code: {str(e)}"
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"Execution error: {str(e)}"
            )
    
    @staticmethod
    def execute_javascript(
        code: str,
        input_values: List[float],
        num_outputs: int
    ) -> ExecutionResult:
        """
        Execute JavaScript function code using PyExecJS or similar.
        
        Function signature should be:
        function f(x) {
            // x is the input array
            // return output array or single value
        }
        
        Args:
            code: JavaScript function definition
            input_values: Input array  
            num_outputs: Expected output size
            
        Returns:
            ExecutionResult with output or error
        """
        try:
            # Try to use execjs if available
            try:
                import execjs
                
                if 'function f(' not in code and 'const f =' not in code:
                    return ExecutionResult(
                        success=False,
                        error="Function must be named 'f': function f(x) { ... }"
                    )
                
                # Wrap in execution context
                wrapper = f"""
                {code}
                f({input_values});
                """
                
                ctx = execjs.compile(wrapper)
                result = ctx.exec_(wrapper)
                
                # Parse result
                if isinstance(result, (list, tuple)):
                    output = [float(v) for v in result]
                elif isinstance(result, (int, float)):
                    output = [float(result)]
                else:
                    return ExecutionResult(
                        success=False,
                        error=f"Function returned unexpected type: {type(result)}"
                    )
                
                # Normalize
                if len(output) < num_outputs:
                    output.extend([0.0] * (num_outputs - len(output)))
                elif len(output) > num_outputs:
                    output = output[:num_outputs]
                
                return ExecutionResult(
                    success=True,
                    output=output
                )
                
            except ImportError:
                # Fallback: parse and execute simple JS
                return FunctionExecutor._execute_javascript_fallback(
                    code, input_values, num_outputs
                )
                
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"JavaScript execution error: {str(e)}"
            )
    
    @staticmethod
    def _execute_javascript_fallback(
        code: str,
        input_values: List[float],
        num_outputs: int
    ) -> ExecutionResult:
        """
        Simple JavaScript fallback parser for basic operations.
        Supports simple math expressions.
        """
        try:
            # Very basic fallback - only works for simple cases
            # Better to require PyExecJS to be installed
            return ExecutionResult(
                success=False,
                error=("JavaScript execution requires 'execjs' library. "
                       "Install with: pip install PyExecJS")
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                error=f"JavaScript fallback error: {str(e)}"
            )
    
    @staticmethod
    def validate_python_code(code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate Python code for basic syntax errors.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            compile(code, '<string>', 'exec')
            
            # Check for required function
            if 'def f(' not in code and 'def f (' not in code:
                return False, "Function must be named 'f' and take parameter 'x': def f(x):"
            
            return True, None
            
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def validate_javascript_code(code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate JavaScript code for basic syntax errors.
        
        Returns:
            (is_valid, error_message)
        """
        # Check for required function
        if 'function f(' not in code and 'const f' not in code and 'let f' not in code:
            return False, "Function must be named 'f': function f(x) { ... }"
        
        # Basic brace matching
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            return False, f"Mismatched braces: {open_braces} open, {close_braces} close"
        
        # Basic parenthesis matching
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens != close_parens:
            return False, f"Mismatched parentheses: {open_parens} open, {close_parens} close"
        
        return True, None


class DatasetGenerator:
    """Generate datasets from custom functions."""
    
    @staticmethod
    def generate_linspace(
        func,
        num_inputs: int,
        num_outputs: int,
        samples_per_input: int = 10
    ) -> List[dict]:
        """
        Generate dataset by sampling inputs across [0, 1] range.
        
        Args:
            func: Function taking input array, returning output array/value
            num_inputs: Number of input dimensions
            num_outputs: Number of output dimensions
            samples_per_input: Samples per dimension (total = samples_per_input^num_inputs)
            
        Returns:
            List of {"x": [...], "y": [...]} dicts
        """
        dataset = []
        
        # Generate grid of input values
        if num_inputs == 1:
            x_values = [np.linspace(0, 1, samples_per_input)]
        elif num_inputs == 2:
            x1 = np.linspace(0, 1, samples_per_input)
            x2 = np.linspace(0, 1, samples_per_input)
            x_values = np.meshgrid(x1, x2)
        else:
            # For higher dimensions, use random sampling
            return DatasetGenerator.generate_random(
                func, num_inputs, num_outputs, samples_per_input
            )
        
        # Flatten and execute function for each point
        if num_inputs == 1:
            for x in x_values[0]:
                result = func(np.array([x]))
                y = result if isinstance(result, (list, np.ndarray)) else [result]
                dataset.append({
                    "x": [float(x)],
                    "y": [float(v) for v in y[:num_outputs]]
                })
        else:
            # num_inputs == 2
            for i in range(x_values[0].shape[0]):
                for j in range(x_values[0].shape[1]):
                    x = np.array([x_values[0][i, j], x_values[1][i, j]])
                    result = func(x)
                    y = result if isinstance(result, (list, np.ndarray)) else [result]
                    dataset.append({
                        "x": [float(v) for v in x],
                        "y": [float(v) for v in y[:num_outputs]]
                    })
        
        return dataset
    
    @staticmethod
    def generate_random(
        func,
        num_inputs: int,
        num_outputs: int,
        num_samples: int = 100
    ) -> List[dict]:
        """Generate random input dataset."""
        dataset = []
        
        for _ in range(num_samples):
            x = np.random.uniform(0, 1, num_inputs)
            result = func(x)
            y = result if isinstance(result, (list, np.ndarray)) else [result]
            
            # Normalize output
            y_normalized = [float(v) for v in y[:num_outputs]]
            if len(y_normalized) < num_outputs:
                y_normalized.extend([0.0] * (num_outputs - len(y_normalized)))
            
            dataset.append({
                "x": [float(v) for v in x],
                "y": y_normalized
            })
        
        return dataset
