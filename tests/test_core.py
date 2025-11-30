"""
Comprehensive tests for QAero core functionality.
"""
import pytest
import numpy as np
from src.qaero.core.base import Problem, OptimizationProblem, QaeroError
from src.qaero.core.results import OptimizationResult, ResultValidator
from src.qaero.problems.aerodynamics import AirfoilOptimizationProblem

def test_problem_validation():
    """Test problem parameter validation."""
    with pytest.raises(QaeroError):
        Problem(problem_id="", parameters={})
    
    with pytest.raises(QaeroError):
        Problem(problem_id="test", parameters="not_a_dict")

def test_optimization_problem():
    """Test optimization problem creation."""
    def simple_objective(x):
        return sum(x**2)
    
    problem = OptimizationProblem(
        problem_id="test_opt",
        objective=simple_objective,
        variables=['x1', 'x2'],
        bounds={'x1': (0, 1), 'x2': (0, 1)}
    )
    
    assert problem.problem_id == "test_opt"
    assert len(problem.variables) == 2
    assert problem.bounds['x1'] == (0, 1)

def test_airfoil_optimization():
    """Test airfoil optimization problem template."""
    problem = AirfoilOptimizationProblem(parameterization="naca")
    
    assert problem.parameters['parameterization'] == "naca"
    assert len(problem.variables) == 3
    assert 'm' in problem.variables
    
    # Test objective function
    test_point = np.array([0.05, 0.4, 0.12])
    result = problem.objective(test_point)
    assert isinstance(result, float)

def test_result_validation():
    """Test result validation logic."""
    valid_result = OptimizationResult(
        problem_id="test",
        backend_name="test_backend",
        success=True,
        execution_time=1.0,
        optimal_value=0.5,
        optimal_variables={'x': 1.0}
    )
    
    assert ResultValidator.validate_optimization(valid_result)
    
    invalid_result = OptimizationResult(
        problem_id="test",
        backend_name="test_backend", 
        success=True,
        execution_time=1.0,
        optimal_value=np.inf,  # Invalid value
        optimal_variables={'x': 1.0}
    )
    
    assert not ResultValidator.validate_optimization(invalid_result)