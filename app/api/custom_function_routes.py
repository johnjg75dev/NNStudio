"""
app/api/custom_function_routes.py
API endpoints for creating and managing custom training functions.

Routes:
  POST   /api/functions/custom              - Create custom function
  GET    /api/functions/custom              - List user's custom functions
  GET    /api/functions/custom/<id>         - Get custom function code
  PUT    /api/functions/custom/<id>         - Update custom function  
  DELETE /api/functions/custom/<id>         - Delete custom function
  POST   /api/functions/custom/<id>/test    - Test function with sample input
  POST   /api/functions/custom/<id>/preview - Preview generated dataset
"""
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user

from .. import db
from ..models import CustomTrainingFunction
from ..core.function_executor import FunctionExecutor, DatasetGenerator

custom_function_bp = Blueprint('custom_functions', __name__)


# ════════════════════════════════════════════════════════════════════════
# Create Custom Function
# ════════════════════════════════════════════════════════════════════════
@custom_function_bp.route('', methods=['POST'])
@login_required
def create_custom_function():
    """
    Create a new custom training function.
    
    Request JSON:
    {
        "name": "str (required)",
        "description": "str (optional)",
        "language": "str - 'python' or 'javascript' (required)",
        "code": "str - function code (required)",
        "num_inputs": int (required),
        "num_outputs": int (required),
        "input_labels": list[str] (optional),
        "output_labels": list[str] (optional),
        "is_classification": bool (optional, default false),
        "sample_strategy": "str - 'linspace', 'random', 'custom' (default 'linspace')",
        "custom_dataset": list (optional - for 'custom' strategy)
    }
    """
    try:
        data = request.get_json()
        
        # Validate required fields
        required = ['name', 'language', 'code', 'num_inputs', 'num_outputs']
        for field in required:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        language = data['language'].lower()
        if language not in ['python', 'javascript']:
            return jsonify({"success": False, "error": "Language must be 'python' or 'javascript'"}), 400
        
        code = data['code'].strip()
        
        # Validate code
        if language == 'python':
            is_valid, error = FunctionExecutor.validate_python_code(code)
        else:  # javascript
            is_valid, error = FunctionExecutor.validate_javascript_code(code)
        
        if not is_valid:
            return jsonify({"success": False, "error": f"Code validation failed: {error}"}), 400
        
        # Create model
        custom_func = CustomTrainingFunction(
            user_id=current_user.id,
            name=data['name'],
            description=data.get('description'),
            language=language,
            code=code,
            num_inputs=data['num_inputs'],
            num_outputs=data['num_outputs'],
            input_labels=data.get('input_labels', []),
            output_labels=data.get('output_labels', []),
            is_classification=data.get('is_classification', False),
            sample_strategy=data.get('sample_strategy', 'linspace'),
            custom_dataset=data.get('custom_dataset'),
            is_valid=True,
        )
        
        db.session.add(custom_func)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "function": custom_func.to_dict(),
            "message": f"Custom function '{custom_func.name}' created"
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════
# List Custom Functions
# ════════════════════════════════════════════════════════════════════════
@custom_function_bp.route('', methods=['GET'])
@login_required
def list_custom_functions():
    """List all custom functions for current user."""
    try:
        limit = min(request.args.get('limit', 50, type=int), 100)
        offset = request.args.get('offset', 0, type=int)
        
        query = CustomTrainingFunction.query.filter_by(user_id=current_user.id)
        total = query.count()
        
        functions = query.order_by(CustomTrainingFunction.created_at.desc())\
                        .limit(limit)\
                        .offset(offset)\
                        .all()
        
        return jsonify({
            "success": True,
            "functions": [f.to_dict() for f in functions],
            "total": total,
            "limit": limit,
            "offset": offset
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════
# Get Custom Function (with code)
# ════════════════════════════════════════════════════════════════════════
@custom_function_bp.route('/<int:func_id>', methods=['GET'])
@login_required
def get_custom_function(func_id):
    """Get custom function details including code."""
    try:
        custom_func = CustomTrainingFunction.query.filter_by(
            id=func_id,
            user_id=current_user.id
        ).first()
        
        if not custom_func:
            return jsonify({"success": False, "error": "Function not found"}), 404
        
        return jsonify({
            "success": True,
            "function": custom_func.to_dict_full()
        }), 200
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════
# Update Custom Function
# ════════════════════════════════════════════════════════════════════════
@custom_function_bp.route('/<int:func_id>', methods=['PUT'])
@login_required
def update_custom_function(func_id):
    """Update custom function code and metadata."""
    try:
        custom_func = CustomTrainingFunction.query.filter_by(
            id=func_id,
            user_id=current_user.id
        ).first()
        
        if not custom_func:
            return jsonify({"success": False, "error": "Function not found"}), 404
        
        data = request.get_json()
        
        # Update fields if provided
        if 'name' in data:
            custom_func.name = data['name']
        
        if 'description' in data:
            custom_func.description = data['description']
        
        if 'code' in data:
            code = data['code'].strip()
            language = custom_func.language
            
            # Validate new code
            if language == 'python':
                is_valid, error = FunctionExecutor.validate_python_code(code)
            else:
                is_valid, error = FunctionExecutor.validate_javascript_code(code)
            
            if not is_valid:
                return jsonify({
                    "success": False,
                    "error": f"Code validation failed: {error}"
                }), 400
            
            custom_func.code = code
            custom_func.is_valid = True
        
        if 'input_labels' in data:
            custom_func.input_labels = data['input_labels']
        
        if 'output_labels' in data:
            custom_func.output_labels = data['output_labels']
        
        if 'is_classification' in data:
            custom_func.is_classification = data['is_classification']
        
        if 'sample_strategy' in data:
            custom_func.sample_strategy = data['sample_strategy']
        
        if 'custom_dataset' in data:
            custom_func.custom_dataset = data['custom_dataset']
        
        db.session.commit()
        
        return jsonify({
            "success": True,
            "function": custom_func.to_dict(),
            "message": "Function updated"
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════
# Delete Custom Function
# ════════════════════════════════════════════════════════════════════════
@custom_function_bp.route('/<int:func_id>', methods=['DELETE'])
@login_required
def delete_custom_function(func_id):
    """Delete a custom function."""
    try:
        custom_func = CustomTrainingFunction.query.filter_by(
            id=func_id,
            user_id=current_user.id
        ).first()
        
        if not custom_func:
            return jsonify({"success": False, "error": "Function not found"}), 404
        
        func_name = custom_func.name
        db.session.delete(custom_func)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "message": f"Function '{func_name}' deleted"
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════
# Test Custom Function
# ════════════════════════════════════════════════════════════════════════
@custom_function_bp.route('/<int:func_id>/test', methods=['POST'])
@login_required
def test_custom_function(func_id):
    """
    Test custom function with sample input.
    
    Request JSON:
    {
        "input": list[float] - Input array (must match num_inputs)
    }
    
    Returns execution result with output and timing.
    """
    try:
        custom_func = CustomTrainingFunction.query.filter_by(
            id=func_id,
            user_id=current_user.id
        ).first()
        
        if not custom_func:
            return jsonify({"success": False, "error": "Function not found"}), 404
        
        data = request.get_json()
        
        if 'input' not in data:
            return jsonify({"success": False, "error": "Missing 'input' array"}), 400
        
        input_values = data['input']
        if len(input_values) != custom_func.num_inputs:
            return jsonify({
                "success": False,
                "error": f"Input size mismatch: expected {custom_func.num_inputs}, got {len(input_values)}"
            }), 400
        
        # Execute function
        if custom_func.language == 'python':
            result = FunctionExecutor.execute_python(
                custom_func.code,
                input_values,
                custom_func.num_outputs
            )
        elif custom_func.language == 'javascript':
            result = FunctionExecutor.execute_javascript(
                custom_func.code,
                input_values,
                custom_func.num_outputs
            )
        else:
            return jsonify({"success": False, "error": "Unknown language"}), 400
        
        # Update test result
        test_result_data = {
            "success": result.success,
            "input": input_values,
            "output": result.output,
            "error": result.error,
            "exec_time": result.exec_time,
        }
        custom_func.last_test_result = test_result_data
        db.session.commit()
        
        if result.success:
            return jsonify({
                "success": True,
                "output": result.output,
                "exec_time": result.exec_time,
                "message": "Function executed successfully"
            }), 200
        else:
            return jsonify({
                "success": False,
                "error": result.error
            }), 400
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════
# Preview Dataset
# ════════════════════════════════════════════════════════════════════════
@custom_function_bp.route('/<int:func_id>/preview', methods=['POST'])
@login_required
def preview_dataset(func_id):
    """
    Generate and preview dataset for custom function.
    
    Request JSON (optional):
    {
        "samples_per_input": int (default 5),
        "strategy": str - 'linspace' or 'random' (default 'linspace')
    }
    
    Returns sample of generated dataset.
    """
    try:
        custom_func = CustomTrainingFunction.query.filter_by(
            id=func_id,
            user_id=current_user.id
        ).first()
        
        if not custom_func:
            return jsonify({"success": False, "error": "Function not found"}), 404
        
        if not custom_func.is_valid:
            return jsonify({
                "success": False,
                "error": "Function is not valid. Fix code errors before generating dataset."
            }), 400
        
        data = request.get_json() or {}
        
        # Get executable function
        try:
            from ..modules.functions.custom_function_wrapper import DynamicCustomFunction
            wrapper = DynamicCustomFunction(custom_func)
            func = wrapper._get_executor_func()
        except Exception as e:
            return jsonify({"success": False, "error": f"Failed to compile function: {str(e)}"}), 400
        
        strategy = data.get('strategy', custom_func.sample_strategy or 'linspace')
        samples_per_input = data.get('samples_per_input', 5)
        
        try:
            if strategy == 'custom':
                if custom_func.custom_dataset:
                    dataset = custom_func.custom_dataset
                else:
                    dataset = DatasetGenerator.generate_linspace(
                        func, custom_func.num_inputs, custom_func.num_outputs,
                        samples_per_input=samples_per_input
                    )
            elif strategy == 'linspace':
                dataset = DatasetGenerator.generate_linspace(
                    func, custom_func.num_inputs, custom_func.num_outputs,
                    samples_per_input=samples_per_input
                )
            else:  # random
                dataset = DatasetGenerator.generate_random(
                    func, custom_func.num_inputs, custom_func.num_outputs,
                    num_samples=samples_per_input
                )
            
            # Return preview (first 10 samples)
            preview = dataset[:10]
            
            return jsonify({
                "success": True,
                "preview": preview,
                "total_samples": len(dataset),
                "first_sample": dataset[0] if dataset else None,
                "last_sample": dataset[-1] if dataset else None
            }), 200
            
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Failed to generate dataset: {str(e)}"
            }), 400
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════════════════
# Get Templates
# ════════════════════════════════════════════════════════════════════════
@custom_function_bp.route('/templates', methods=['GET'])
@login_required
def get_templates():
    """Get code templates for Python and JavaScript."""
    templates = {
        "python": {
            "description": "Python function template",
            "code": '''def f(x):
    """
    Custom training function.
    
    Args:
        x: numpy array of inputs with length {num_inputs}
        
    Returns:
        output value or list of {num_outputs} values
    """
    # Example: sum of inputs
    import numpy as np
    return np.sum(x)
'''
        },
        "javascript": {
            "description": "JavaScript function template",
            "code": '''function f(x) {
    // Custom training function
    // 
    // Args:
    //   x: array of inputs with length {num_inputs}
    //
    // Returns:
    //   output value or array of {num_outputs} values
    
    // Example: sum of inputs
    return x.reduce((a, b) => a + b, 0);
}
'''
        }
    }
    
    return jsonify({
        "success": True,
        "templates": templates,
    }), 200
