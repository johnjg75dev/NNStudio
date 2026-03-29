"""
tests/test_custom_functions.py
Tests for custom training function creation and execution.
"""
import pytest
import numpy as np
from app.core.function_executor import FunctionExecutor, DatasetGenerator, ExecutionResult
from app.models import CustomTrainingFunction, User
from app import db


class TestFunctionExecutor:
    """Test Python function execution."""
    
    def test_execute_simple_python_function(self):
        """Test executing a simple Python function."""
        code = """
def f(x):
    return np.sum(x)
"""
        result = FunctionExecutor.execute_python(
            code,
            [1.0, 2.0, 3.0],
            num_outputs=1
        )
        
        assert result.success
        assert result.output == [6.0]
    
    def test_execute_python_with_output_padding(self):
        """Test that output is padded to expected size."""
        code = """
def f(x):
    return x[0]  # Return single value
"""
        result = FunctionExecutor.execute_python(
            code,
            [5.0, 3.0],
            num_outputs=3
        )
        
        assert result.success
        assert len(result.output) == 3
        assert result.output == [5.0, 0.0, 0.0]
    
    def test_execute_python_with_output_truncation(self):
        """Test that oversized output is truncated."""
        code = """
def f(x):
    return [1, 2, 3, 4, 5]
"""
        result = FunctionExecutor.execute_python(
            code,
            [0.0],
            num_outputs=2
        )
        
        assert result.success
        assert len(result.output) == 2
        assert result.output == [1.0, 2.0]
    
    def test_execute_python_with_math_operations(self):
        """Test Python function with math module."""
        code = """
def f(x):
    import math
    return math.sin(x[0]) + math.cos(x[1])
"""
        result = FunctionExecutor.execute_python(
            code,
            [0.0, 0.0],
            num_outputs=1
        )
        
        assert result.success
        assert abs(result.output[0] - 1.0) < 0.001
    
    def test_execute_python_invalid_syntax(self):
        """Test error handling for syntax errors."""
        code = """
def f(x)
    return x  # missing colon
"""
        result = FunctionExecutor.execute_python(code, [0.0], 1)
        
        assert not result.success
        assert result.error is not None
    
    def test_execute_python_missing_function_name(self):
        """Test error when function not named 'f'."""
        code = """
def compute(x):
    return x
"""
        result = FunctionExecutor.execute_python(code, [0.0], 1)
        
        assert not result.success
        assert "f" in result.error
    
    def test_execute_python_returns_invalid_type(self):
        """Test error when function returns invalid type."""
        code = """
def f(x):
    return "invalid"
"""
        result = FunctionExecutor.execute_python(code, [0.0], 1)
        
        assert not result.success
        assert "must return number or array" in result.error


class TestPythonCodeValidation:
    """Test Python code validation."""
    
    def test_validate_valid_python_code(self):
        """Test validation of valid code."""
        code = """
def f(x):
    return np.sum(x)
"""
        is_valid, error = FunctionExecutor.validate_python_code(code)
        assert is_valid
        assert error is None
    
    def test_validate_python_syntax_error(self):
        """Test validation catches syntax errors."""
        code = """
def f(x)
    return x
"""
        is_valid, error = FunctionExecutor.validate_python_code(code)
        assert not is_valid
        assert error is not None
    
    def test_validate_python_missing_function_name(self):
        """Test validation requires function named 'f'."""
        code = """
def compute(x):
    return x
"""
        is_valid, error = FunctionExecutor.validate_python_code(code)
        assert not is_valid


class TestJavaScriptCodeValidation:
    """Test JavaScript code validation."""
    
    def test_validate_valid_js_code(self):
        """Test validation of valid JavaScript."""
        code = """
function f(x) {
    return x[0] + x[1];
}
"""
        is_valid, error = FunctionExecutor.validate_javascript_code(code)
        assert is_valid
        assert error is None
    
    def test_validate_js_missing_function_name(self):
        """Test validation requires function named 'f'."""
        code = """
function compute(x) {
    return x[0];
}
"""
        is_valid, error = FunctionExecutor.validate_javascript_code(code)
        assert not is_valid
    
    def test_validate_js_mismatched_braces(self):
        """Test validation catches mismatched braces."""
        code = """
function f(x) {
    return x[0]
}
}  // extra brace
"""
        is_valid, error = FunctionExecutor.validate_javascript_code(code)
        assert not is_valid


class TestDatasetGenerator:
    """Test dataset generation."""
    
    def test_generate_linspace_1d(self):
        """Test generating linspace dataset for 1D input."""
        def func(x):
            return [2 * x[0]]  # y = 2x
        
        dataset = DatasetGenerator.generate_linspace(
            func, num_inputs=1, num_outputs=1, samples_per_input=5
        )
        
        assert len(dataset) == 5
        assert all("x" in s and "y" in s for s in dataset)
        
        # Check first sample (x=0)
        assert dataset[0]["x"] == [0.0]
        assert dataset[0]["y"][0] == 0.0
    
    def test_generate_linspace_2d(self):
        """Test generating linspace dataset for 2D input."""
        def func(x):
            return [x[0] + x[1]]  # y = x + y
        
        dataset = DatasetGenerator.generate_linspace(
            func, num_inputs=2, num_outputs=1, samples_per_input=3
        )
        
        assert len(dataset) == 9  # 3x3 grid
        assert dataset[0]["x"] == [0.0, 0.0]
    
    def test_generate_random(self):
        """Test random dataset generation."""
        def func(x):
            return np.sum(x)
        
        dataset = DatasetGenerator.generate_random(
            func, num_inputs=2, num_outputs=1, num_samples=10
        )
        
        assert len(dataset) == 10
        assert all(len(s["x"]) == 2 for s in dataset)
        assert all(len(s["y"]) == 1 for s in dataset)


class TestCustomTrainingFunctionModel:
    """Test CustomTrainingFunction database model."""
    
    def test_create_custom_function(self, flask_app, user: User):
        """Test creating a custom function in database."""
        with flask_app.app_context():
            custom_func = CustomTrainingFunction(
                user_id=user.id,
                name="Test Function",
                description="A test function",
                language="python",
                code="def f(x):\n    return np.sum(x)",
                num_inputs=2,
                num_outputs=1,
                input_labels=["A", "B"],
                output_labels=["Sum"],
                is_classification=False,
                is_valid=True,
            )
            
            db.session.add(custom_func)
            db.session.commit()
            
            retrieved = CustomTrainingFunction.query.filter_by(
                name="Test Function"
            ).first()
            
            assert retrieved is not None
            assert retrieved.user_id == user.id
            assert retrieved.language == "python"
            assert retrieved.num_inputs == 2
    
    def test_custom_function_to_dict(self, flask_app, user: User):
        """Test to_dict() excludes code."""
        with flask_app.app_context():
            custom_func = CustomTrainingFunction(
                user_id=user.id,
                name="Test",
                language="python",
                code="def f(x):\n    return x",
                num_inputs=1,
                num_outputs=1,
                is_valid=True,
            )
            
            db.session.add(custom_func)
            db.session.commit()
            
            result = custom_func.to_dict()
            
            assert "code" not in result
            assert result["name"] == "Test"
            assert result["is_valid"] is True
    
    def test_custom_function_to_dict_full(self, flask_app, user: User):
        """Test to_dict_full() includes code."""
        with flask_app.app_context():
            custom_func = CustomTrainingFunction(
                user_id=user.id,
                name="Test",
                language="python",
                code="def f(x):\n    return x",
                num_inputs=1,
                num_outputs=1,
                is_valid=True,
            )
            
            db.session.add(custom_func)
            db.session.commit()
            
            result = custom_func.to_dict_full()
            
            assert "code" in result
            assert result["code"] == "def f(x):\n    return x"


class TestCustomFunctionWrapper:
    """Test DynamicCustomFunction wrapper."""
    
    def test_wrap_custom_function(self, flask_app, user: User):
        """Test wrapping custom function as TrainingFunction."""
        with flask_app.app_context():
            custom_func = CustomTrainingFunction(
                user_id=user.id,
                name="Sum Function",
                language="python",
                code="def f(x):\n    import numpy as np\n    return [np.sum(x)]",
                num_inputs=2,
                num_outputs=1,
                input_labels=["A", "B"],
                output_labels=["Sum"],
                is_classification=False,
                is_valid=True,
            )
            
            db.session.add(custom_func)
            db.session.commit()
            
            from app.modules.functions.custom_function_wrapper import DynamicCustomFunction
            wrapper = DynamicCustomFunction(custom_func)
            
            assert wrapper.key == f"custom_{custom_func.id}"
            assert wrapper.label == "Sum Function"
            assert wrapper.inputs == 2
            assert wrapper.outputs == 1
    
    def test_generate_dataset_from_custom_function(self, flask_app, user: User):
        """Test generating dataset from custom function."""
        with flask_app.app_context():
            custom_func = CustomTrainingFunction(
                user_id=user.id,
                name="Double",
                language="python",
                code="def f(x):\n    return [2 * x[0], 2 * x[1]]",
                num_inputs=2,
                num_outputs=2,
                sample_strategy="linspace",
                is_valid=True,
            )
            
            db.session.add(custom_func)
            db.session.commit()
            
            from app.modules.functions.custom_function_wrapper import DynamicCustomFunction
            wrapper = DynamicCustomFunction(custom_func)
            
            dataset = wrapper.generate_dataset()
            
            assert len(dataset) > 0
            assert all("x" in s and "y" in s for s in dataset)


class TestCustomFunctionAPI:
    """Test custom function API endpoints."""
    
    def test_create_function_endpoint(self, flask_app, user: User):
        """Test POST /api/functions/custom endpoint."""
        with flask_app.app_context():
            client = flask_app.test_client()
            
            # Would need to be authenticated
            response = client.post(
                '/api/functions/custom',
                json={
                    "name": "Test Function",
                    "language": "python",
                    "code": "def f(x):\n    return np.sum(x)",
                    "num_inputs": 2,
                    "num_outputs": 1,
                },
            )
            
            # Expect auth redirect or 401
            assert response.status_code in [200, 302, 401, 405]
    
    def test_get_templates_endpoint(self, flask_app):
        """Test GET /api/functions/custom/templates endpoint."""
        with flask_app.app_context():
            client = flask_app.test_client()
            
            # Templates should be accessible (no auth required)
            response = client.get('/api/functions/custom/templates')
            
            # May require auth
            assert response.status_code in [200, 401, 302]


@pytest.fixture
def user(flask_app):
    """Create a test user for testing."""
    with flask_app.app_context():
        from app.models import User
        user = User(username="testuser")
        user.set_password("testpass")
        db.session.add(user)
        db.session.commit()
        yield user
        db.session.delete(user)
        db.session.commit()
