"""
Comprehensive tests for QAero core functionality.
"""
import pytest
import numpy as np
import tempfile
import json
from datetime import datetime

from qaero.core.base import (
    Problem, OptimizationProblem, PDEProblem, QaeroError, 
    BackendState, ProblemType
)
from qaero.core.results import (
    OptimizationResult, PDEResult, ResultValidator, BaseResult
)
from qaero.core.registry import BackendRegistry, AlgorithmRegistry


class TestCoreBase:
    """Test core base classes and functionality."""
    
    def test_problem_validation(self):
        """Test problem parameter validation."""
        # Valid problem
        problem = Problem(
            problem_id="test_problem",
            problem_type=ProblemType.OPTIMIZATION,
            parameters={"param1": 1.0}
        )
        assert problem.problem_id == "test_problem"
        
        # Invalid problem ID
        with pytest.raises(QaeroError):
            Problem(problem_id="", problem_type=ProblemType.OPTIMIZATION, parameters={})
        
        # Invalid parameters type
        with pytest.raises(QaeroError):
            Problem(problem_id="test", problem_type=ProblemType.OPTIMIZATION, parameters="invalid")
    
    def test_optimization_problem_creation(self):
        """Test optimization problem creation and validation."""
        def simple_objective(x):
            return np.sum(x**2)
        
        # Valid optimization problem
        problem = OptimizationProblem(
            problem_id="test_opt",
            objective=simple_objective,
            variables=['x1', 'x2'],
            bounds={'x1': (0, 1), 'x2': (0, 1)}
        )
        
        assert problem.problem_type == ProblemType.OPTIMIZATION
        assert len(problem.variables) == 2
        assert problem.bounds['x1'] == (0, 1)
        
        # Test objective evaluation
        test_point = np.array([0.5, 0.5])
        result = problem.objective(test_point)
        assert result == 0.5
        
        # Invalid: no variables
        with pytest.raises(QaeroError):
            OptimizationProblem(
                problem_id="test",
                objective=simple_objective,
                variables=[],
                bounds={}
            )
        
        # Invalid: non-callable objective
        with pytest.raises(QaeroError):
            OptimizationProblem(
                problem_id="test",
                objective="not_callable",
                variables=['x1'],
                bounds={}
            )
    
    def test_pde_problem_creation(self):
        """Test PDE problem creation and validation."""
        # Valid PDE problem
        problem = PDEProblem(
            problem_id="test_pde",
            equation="laplace",
            domain={"bounds": [(0, 1)]},
            boundary_conditions={"left": 0, "right": 1}
        )
        
        assert problem.problem_type == ProblemType.PDE
        assert problem.equation == "laplace"
        assert problem.discretization == "finite_difference"
        
        # Invalid: missing boundary conditions
        with pytest.raises(QaeroError):
            PDEProblem(
                problem_id="test",
                equation="laplace", 
                domain={"bounds": [(0, 1)]},
                boundary_conditions={}
            )
    
    def test_backend_state_enum(self):
        """Test backend state enumeration."""
        assert BackendState.READY.value == "ready"
        assert BackendState.ERROR.value == "error"
        assert len(BackendState) == 4


class TestResults:
    """Test result classes and functionality."""
    
    def test_optimization_result(self):
        """Test optimization result creation and methods."""
        result = OptimizationResult(
            problem_id="test_opt",
            backend_name="test_backend",
            success=True,
            execution_time=1.5,
            optimal_value=0.123,
            optimal_variables={'x1': 0.5, 'x2': 0.6},
            solution_history=[1.0, 0.5, 0.123],
            n_function_evaluations=100
        )
        
        assert result.success is True
        assert result.optimal_value == 0.123
        assert result.n_iterations == 3
        assert result.get_optimal_variable('x1') == 0.5
        
        # Test serialization
        result_dict = result.to_dict()
        assert result_dict['problem_id'] == "test_opt"
        assert result_dict['success'] is True
        
        # Test JSON serialization
        json_str = result.to_json()
        parsed = json.loads(json_str)
        assert parsed['backend_name'] == "test_backend"
    
    def test_pde_result(self):
        """Test PDE result creation and methods."""
        solution_field = np.random.rand(50)
        
        result = PDEResult(
            problem_id="test_pde",
            backend_name="test_solver", 
            success=True,
            execution_time=2.5,
            solution_field=solution_field,
            residual_norm=1e-6
        )
        
        assert result.success is True
        assert result.solution_field is not None
        assert 'mean' in result.field_statistics
        assert result.field_statistics['size'] == 50
        
        # Test with None solution
        result_no_solution = PDEResult(
            problem_id="test",
            backend_name="test",
            success=False,
            execution_time=0.0
        )
        assert result_no_solution.solution_field is None
    
    def test_result_validator(self):
        """Test result validation logic."""
        # Valid optimization result
        valid_opt_result = OptimizationResult(
            problem_id="test",
            backend_name="test",
            success=True,
            execution_time=1.0,
            optimal_value=0.5,
            optimal_variables={'x': 1.0}
        )
        
        is_valid, warnings = ResultValidator.validate_optimization(valid_opt_result)
        assert is_valid is True
        assert len(warnings) == 0
        
        # Invalid optimization result (non-finite value)
        invalid_opt_result = OptimizationResult(
            problem_id="test",
            backend_name="test",
            success=True, 
            execution_time=1.0,
            optimal_value=np.inf,
            optimal_variables={'x': 1.0}
        )
        
        is_valid, warnings = ResultValidator.validate_optimization(invalid_opt_result)
        assert is_valid is False
        assert any("not finite" in warning for warning in warnings)
        
        # Valid PDE result
        valid_pde_result = PDEResult(
            problem_id="test",
            backend_name="test",
            success=True,
            execution_time=1.0,
            solution_field=np.array([1.0, 2.0, 3.0]),
            residual_norm=1e-8
        )
        
        is_valid, warnings = ResultValidator.validate_pde(valid_pde_result)
        assert is_valid is True
        
        # Invalid PDE result (non-finite values)
        invalid_pde_result = PDEResult(
            problem_id="test",
            backend_name="test",
            success=True,
            execution_time=1.0,
            solution_field=np.array([1.0, np.inf, 3.0]),
            residual_norm=1e-8
        )
        
        is_valid, warnings = ResultValidator.validate_pde(invalid_pde_result)
        assert is_valid is False
        assert any("non-finite" in warning for warning in warnings)
    
    def test_result_validator_multiple(self):
        """Test validation of multiple results."""
        results = [
            OptimizationResult(
                problem_id="opt1",
                backend_name="backend1",
                success=True,
                execution_time=1.0,
                optimal_value=0.5,
                optimal_variables={'x': 1.0}
            ),
            PDEResult(
                problem_id="pde1", 
                backend_name="backend2",
                success=True,
                execution_time=2.0,
                solution_field=np.array([1.0, 2.0, 3.0]),
                residual_norm=1e-6
            )
        ]
        
        validation_results = ResultValidator.validate_all(results)
        assert len(validation_results) == 2
        assert all(is_valid for is_valid, _ in validation_results.values())


class TestRegistry:
    """Test registry system for backends and algorithms."""
    
    def test_backend_registry(self):
        """Test backend registration and retrieval."""
        # Test backend listing
        backends = BackendRegistry.list_backends()
        assert isinstance(backends, list)
        
        # Test backend creation (classical should be available)
        if 'classical_scipy' in backends:
            backend = BackendRegistry.create_backend('classical_scipy')
            assert backend.name == 'classical_scipy'
            assert backend.state.name == 'READY'
        
        # Test error for non-existent backend
        with pytest.raises(KeyError):
            BackendRegistry.create_backend('non_existent_backend')
    
    def test_algorithm_registry(self):
        """Test algorithm registration and categorization."""
        # Test algorithm listing
        algorithms = AlgorithmRegistry.list_algorithms()
        assert isinstance(algorithms, dict)
        
        # Should have some categories
        assert len(algorithms) > 0
        
        # Test getting algorithms by category
        quantum_optimization = AlgorithmRegistry.get_algorithms_by_category('quantum_optimization')
        assert isinstance(quantum_optimization, list)


class TestProblemTemplates:
    """Test aerospace problem templates."""
    
    def test_airfoil_optimization_problem(self):
        """Test airfoil optimization problem template."""
        from qaero.problems.aerodynamics import AirfoilOptimizationProblem
        
        # Test NACA parameterization
        problem = AirfoilOptimizationProblem(parameterization="naca")
        assert problem.parameters['parameterization'] == "naca"
        assert len(problem.variables) == 3
        assert 'm' in problem.variables
        
        # Test objective function evaluation
        test_point = np.array([0.05, 0.4, 0.12])
        result = problem.objective(test_point)
        assert isinstance(result, float)
        assert result < 0  # Should be negative for minimization
        
        # Test CST parameterization
        problem_cst = AirfoilOptimizationProblem(parameterization="cst")
        assert len(problem_cst.variables) == 6
        assert 'A0' in problem_cst.variables
    
    def test_wing_design_problem(self):
        """Test wing design optimization problem."""
        from qaero.problems.aerodynamics import WingDesignProblem
        
        problem = WingDesignProblem(
            disciplines=['aerodynamics', 'structures'],
            coupling='weak'
        )
        
        assert problem.parameters['disciplines'] == ['aerodynamics', 'structures']
        assert problem.parameters['coupling'] == 'weak'
        
        # Test objective function
        test_point = np.array([20.0, 8.0, 0.3, 25.0, 5.0])  # span, AR, taper, sweep, dihedral
        result = problem.objective(test_point)
        assert isinstance(result, float)
    
    def test_trajectory_problems(self):
        """Test trajectory optimization problems."""
        from qaero.problems.trajectory import AscentTrajectoryProblem, SatelliteManeuverProblem
        
        # Test ascent trajectory
        ascent_problem = AscentTrajectoryProblem(stages=2)
        assert ascent_problem.parameters['stages'] == 2
        assert len(ascent_problem.variables) == 6  # 2 stages * 3 variables each
        
        # Test satellite maneuver
        maneuver_problem = SatelliteManeuverProblem()
        assert 'burn1_magnitude' in maneuver_problem.variables
        assert maneuver_problem.bounds['burn1_magnitude'] == (0.0, 500.0)


class TestIntegration:
    """Integration tests for core functionality."""
    
    def test_end_to_end_optimization(self):
        """Test complete optimization workflow."""
        from qaero import solve_problem
        from qaero.problems.aerodynamics import AirfoilOptimizationProblem
        
        # Create problem
        problem = AirfoilOptimizationProblem(parameterization="naca")
        
        # Solve with classical backend
        result = solve_problem(problem, backend="classical_scipy")
        
        # Verify result structure
        assert isinstance(result, OptimizationResult)
        assert result.problem_id == "airfoil_optimization"
        assert result.backend_name == "classical_scipy"
        
        # Result should be valid
        is_valid, warnings = ResultValidator.validate_optimization(result)
        if result.success:
            assert is_valid is True
        else:
            # Even failed results should be valid structures
            assert len(warnings) == 0 or "Optimization failed" in warnings[0]
    
    def test_problem_serialization(self):
        """Test problem serialization for distributed computing."""
        problem = OptimizationProblem(
            problem_id="serialization_test",
            objective=lambda x: np.sum(x**2),
            variables=['x1', 'x2'],
            bounds={'x1': (0, 1), 'x2': (0, 1)},
            parameters={'test_param': 42}
        )
        
        # Test that problem can be pickled (for multiprocessing)
        import pickle
        pickled = pickle.dumps(problem)
        unpickled = pickle.loads(pickled)
        
        assert unpickled.problem_id == problem.problem_id
        assert unpickled.parameters['test_param'] == 42
        
        # Test objective still works after unpickling
        test_point = np.array([0.5, 0.5])
        assert unpickled.objective(test_point) == problem.objective(test_point)


@pytest.mark.slow
class TestPerformance:
    """Performance tests for core components."""
    
    def test_optimization_performance(self):
        """Test optimization performance with large problems."""
        import time
        
        # Create larger optimization problem
        n_vars = 20
        variables = [f'x{i}' for i in range(n_vars)]
        
        def large_objective(x):
            return np.sum(x**2) + np.sum(np.sin(x))
        
        problem = OptimizationProblem(
            problem_id="performance_test",
            objective=large_objective,
            variables=variables,
            bounds={var: (-10, 10) for var in variables}
        )
        
        # Time objective evaluation
        test_point = np.random.randn(n_vars)
        start_time = time.time()
        
        for _ in range(1000):
            _ = problem.objective(test_point)
        
        eval_time = time.time() - start_time
        assert eval_time < 1.0  # Should be fast
        
    def test_memory_usage(self):
        """Test memory usage with large arrays."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large solution field
        large_field = np.random.rand(10000, 10000)  # 100M elements
        result = PDEResult(
            problem_id="memory_test",
            backend_name="test",
            success=True,
            execution_time=1.0,
            solution_field=large_field
        )
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        # Should use reasonable memory (array is ~800MB for float64)
        assert memory_increase < 2000  # Less than 2GB increase


if __name__ == "__main__":
    pytest.main([__file__, "-v"])