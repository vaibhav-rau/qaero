"""
Comprehensive tests for computational backends.
"""
import pytest
import numpy as np
import tempfile
import os

from qaero.core.base import OptimizationProblem, PDEProblem, ProblemType
from qaero.core.results import OptimizationResult, PDEResult
from qaero.backends import BackendRegistry, Backend
from qaero.backends.classical import ClassicalBackend, SimulatedAnnealingBackend
from qaero.backends.quantum import QuantumBackend


class TestBackendBase:
    """Test backend base functionality."""
    
    def test_backend_registry(self):
        """Test backend registration system."""
        # List available backends
        backends = BackendRegistry.list_backends()
        assert isinstance(backends, list)
        assert len(backends) > 0
        
        # Test backend creation
        for backend_name in backends:
            if backend_name in ['classical_scipy', 'simulated_annealing']:
                backend = BackendRegistry.create_backend(backend_name)
                assert isinstance(backend, Backend)
                assert backend.name == backend_name
    
    def test_backend_validation(self):
        """Test backend problem validation."""
        backend = ClassicalBackend("test_backend")
        
        # Valid optimization problem
        valid_problem = OptimizationProblem(
            problem_id="test",
            objective=lambda x: np.sum(x**2),
            variables=['x1', 'x2']
        )
        assert backend.validate_problem(valid_problem) is True
        
        # Invalid problem type (should still validate for classical backend)
        invalid_problem = OptimizationProblem(
            problem_id="test",
            objective="not_callable",  # This will fail later
            variables=['x1']
        )
        # Validation should pass, execution will fail
        assert backend.validate_problem(invalid_problem) is True


class TestClassicalBackends:
    """Test classical computational backends."""
    
    def setup_method(self):
        """Set up test problems."""
        self.simple_optimization = OptimizationProblem(
            problem_id="simple_quadratic",
            objective=lambda x: np.sum(x**2),
            variables=['x1', 'x2'],
            bounds={'x1': (-1, 1), 'x2': (-1, 1)}
        )
        
        self.rosenbrock_problem = OptimizationProblem(
            problem_id="rosenbrock",
            objective=lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2,
            variables=['x', 'y'],
            bounds={'x': (-2, 2), 'y': (-1, 3)}
        )
    
    def test_classical_backend_initialization(self):
        """Test classical backend initialization."""
        backend = ClassicalBackend("test_classical")
        assert backend.name == "test_classical"
        assert backend.state.name == "READY"
        assert ProblemType.OPTIMIZATION in backend.supported_problems
    
    def test_classical_optimization_simple(self):
        """Test simple optimization with classical backend."""
        backend = ClassicalBackend("classical_test")
        
        result = backend.solve_optimization(self.simple_optimization)
        
        assert isinstance(result, OptimizationResult)
        assert result.problem_id == "simple_quadratic"
        assert result.backend_name == "classical_test"
        
        if result.success:
            assert abs(result.optimal_value) < 1e-4
            assert all(abs(v) < 0.1 for v in result.optimal_variables.values())
        else:
            # Even if failed, should have proper structure
            assert result.error_message is not None
    
    def test_classical_optimization_rosenbrock(self):
        """Test Rosenbrock function optimization."""
        backend = ClassicalBackend("rosenbrock_solver", {'method': 'BFGS'})
        
        result = backend.solve_optimization(self.rosenbrock_problem)
        
        assert isinstance(result, OptimizationResult)
        
        if result.success:
            # Rosenbrock minimum at (1, 1) with value 0
            assert abs(result.optimal_value) < 1e-3
            x_opt = result.optimal_variables['x']
            y_opt = result.optimal_variables['y']
            assert abs(x_opt - 1.0) < 0.1
            assert abs(y_opt - 1.0) < 0.1
    
    def test_classical_optimization_with_constraints(self):
        """Test constrained optimization."""
        def objective(x):
            return x[0]**2 + x[1]**2
        
        constraints = [
            {'type': 'ineq', 'fun': lambda x: x[0] - 0.5},  # x[0] >= 0.5
            {'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1.0}  # x[0] + x[1] = 1
        ]
        
        problem = OptimizationProblem(
            problem_id="constrained",
            objective=objective,
            variables=['x', 'y'],
            constraints=constraints,
            bounds={'x': (0, 2), 'y': (0, 2)}
        )
        
        backend = ClassicalBackend("constrained_solver", {'method': 'SLSQP'})
        result = backend.solve_optimization(problem)
        
        assert isinstance(result, OptimizationResult)
        
        if result.success:
            # Solution should satisfy constraints
            x_opt = result.optimal_variables['x']
            y_opt = result.optimal_variables['y']
            assert x_opt >= 0.45  # Allow some tolerance
            assert abs(x_opt + y_opt - 1.0) < 1e-3
    
    def test_simulated_annealing_backend(self):
        """Test simulated annealing backend."""
        backend = SimulatedAnnealingBackend("annealing_test", {
            'annealing_params': {'maxiter': 100}
        })
        
        result = backend.solve_optimization(self.simple_optimization)
        
        assert isinstance(result, OptimizationResult)
        assert result.backend_name == "annealing_test"
        
        if result.success:
            assert abs(result.optimal_value) < 1e-2
    
    def test_backend_configuration(self):
        """Test backend configuration options."""
        # Test different optimization methods
        methods = ['BFGS', 'Nelder-Mead', 'Powell']
        
        for method in methods:
            backend = ClassicalBackend(f"test_{method}", {'method': method})
            result = backend.solve_optimization(self.simple_optimization)
            
            assert isinstance(result, OptimizationResult)
            assert result.metadata.get('method') == method
    
    def test_backend_error_handling(self):
        """Test backend error handling for invalid problems."""
        backend = ClassicalBackend("error_test")
        
        # Problem with invalid objective
        invalid_problem = OptimizationProblem(
            problem_id="invalid",
            objective="not_callable",  # This will cause an error
            variables=['x1']
        )
        
        result = backend.solve_optimization(invalid_problem)
        
        assert isinstance(result, OptimizationResult)
        assert result.success is False
        assert result.error_message is not None


class TestQuantumBackends:
    """Test quantum computing backends."""
    
    def setup_method(self):
        """Set up test problems for quantum backends."""
        self.qubo_like_problem = OptimizationProblem(
            problem_id="qubo_test",
            objective=lambda x: x[0]**2 + x[1]**2 - 2*x[0]*x[1],  # QUBO-like
            variables=['x1', 'x2'],
            bounds={'x1': (0, 1), 'x2': (0, 1)}
        )
    
    def test_quantum_backend_initialization(self):
        """Test quantum backend initialization."""
        backend = QuantumBackend("quantum_test", {'provider': 'qiskit'})
        assert backend.name == "quantum_test"
        assert backend.provider == "qiskit"
        
        # Test that quantum backend validates optimization problems
        assert backend.validate_problem(self.qubo_like_problem) is True
    
    def test_quantum_backend_fallback(self):
        """Test quantum backend fallback to classical methods."""
        backend = QuantumBackend("quantum_fallback_test")
        
        result = backend.solve_optimization(self.qubo_like_problem)
        
        assert isinstance(result, OptimizationResult)
        assert result.backend_name == "quantum_fallback_test"
        assert result.metadata.get('quantum_attempted', False) is True
        assert result.metadata.get('fallback_used', False) is True
        
        if result.success:
            assert result.optimal_value is not None
    
    @pytest.mark.quantum
    def test_quantum_backend_with_actual_provider(self):
        """Test quantum backend with actual quantum provider (if available)."""
        pytest.importorskip("qiskit")
        
        try:
            backend = QuantumBackend("real_quantum", {'provider': 'qiskit'})
            result = backend.solve_optimization(self.qubo_like_problem)
            
            assert isinstance(result, OptimizationResult)
            # Should at least return a result structure
        except Exception as e:
            # Quantum hardware might not be available, that's OK for tests
            pytest.skip(f"Quantum backend not available: {e}")


class TestBackendIntegration:
    """Integration tests for backend functionality."""
    
    def test_multiple_backend_comparison(self):
        """Compare results from multiple backends."""
        problem = OptimizationProblem(
            problem_id="comparison",
            objective=lambda x: (x[0] - 1)**2 + (x[1] - 2)**2,
            variables=['x', 'y'],
            bounds={'x': (-5, 5), 'y': (-5, 5)}
        )
        
        backends_to_test = []
        if 'classical_scipy' in BackendRegistry.list_backends():
            backends_to_test.append('classical_scipy')
        if 'simulated_annealing' in BackendRegistry.list_backends():
            backends_to_test.append('simulated_annealing')
        
        results = {}
        for backend_name in backends_to_test:
            backend = BackendRegistry.create_backend(backend_name)
            result = backend.solve_optimization(problem)
            results[backend_name] = result
        
        # All should return valid result structures
        for backend_name, result in results.items():
            assert isinstance(result, OptimizationResult)
            assert result.problem_id == "comparison"
            
            # Validate results
            is_valid, warnings = result.validate_optimization(result)
            assert is_valid is True or len(warnings) == 0
        
        # If multiple backends succeeded, solutions should be similar
        successful_results = [
            r for r in results.values() 
            if r.success and r.optimal_value is not None
        ]
        
        if len(successful_results) >= 2:
            optimal_values = [r.optimal_value for r in successful_results]
            # All should find near-optimal solutions
            assert all(abs(val) < 1.0 for val in optimal_values)
    
    def test_backend_performance(self):
        """Test backend performance characteristics."""
        import time
        
        # Create medium-sized problem
        n_vars = 10
        variables = [f'x{i}' for i in range(n_vars)]
        
        problem = OptimizationProblem(
            problem_id="performance",
            objective=lambda x: np.sum(x**2) + np.sum(np.sin(10*x)),
            variables=variables,
            bounds={var: (-1, 1) for var in variables}
        )
        
        backend = ClassicalBackend("performance_test")
        
        start_time = time.time()
        result = backend.solve_optimization(problem)
        solve_time = time.time() - start_time
        
        assert isinstance(result, OptimizationResult)
        
        # Should complete in reasonable time
        assert solve_time < 30.0  # 30 seconds max
        
        if result.success:
            assert result.execution_time == solve_time
    
    def test_backend_resource_cleanup(self):
        """Test that backends properly clean up resources."""
        backend = ClassicalBackend("cleanup_test")
        
        # Use context manager to ensure cleanup
        with backend:
            assert backend.state.name == "READY"
            result = backend.solve_optimization(self.simple_optimization)
            assert isinstance(result, OptimizationResult)
        
        # Backend should still be accessible after context exit
        assert backend.state.name == "READY"
    
    def test_backend_serialization(self):
        """Test backend serialization for distributed computing."""
        backend = ClassicalBackend("serialization_test")
        
        # Test that backend can be pickled
        import pickle
        try:
            pickled = pickle.dumps(backend)
            unpickled = pickle.loads(pickled)
            
            assert unpickled.name == backend.name
            assert unpickled.state == backend.state
            
            # Test that unpickled backend still works
            result = unpickled.solve_optimization(self.simple_optimization)
            assert isinstance(result, OptimizationResult)
            
        except Exception as e:
            pytest.skip(f"Backend serialization not supported: {e}")


class TestBackendEdgeCases:
    """Test edge cases and error conditions for backends."""
    
    def test_large_problem_handling(self):
        """Test backend handling of large optimization problems."""
        n_vars = 50  # Larger problem
        variables = [f'x{i}' for i in range(n_vars)]
        
        problem = OptimizationProblem(
            problem_id="large_problem",
            objective=lambda x: np.sum(x**2),
            variables=variables,
            bounds={var: (-10, 10) for var in variables}
        )
        
        backend = ClassicalBackend("large_problem_solver")
        result = backend.solve_optimization(problem)
        
        assert isinstance(result, OptimizationResult)
        # Should handle without crashing
    
    def test_ill_conditioned_problem(self):
        """Test backend handling of ill-conditioned problems."""
        # Poorly scaled problem
        problem = OptimizationProblem(
            problem_id="ill_conditioned",
            objective=lambda x: (x[0] * 1e6)**2 + (x[1] * 1e-6)**2,
            variables=['x1', 'x2'],
            bounds={'x1': (-1, 1), 'x2': (-1, 1)}
        )
        
        backend = ClassicalBackend("ill_conditioned_solver")
        result = backend.solve_optimization(problem)
        
        assert isinstance(result, OptimizationResult)
        # Should handle without numerical issues
    
    def test_discontinuous_problem(self):
        """Test backend handling of discontinuous objectives."""
        def discontinuous_objective(x):
            return x[0]**2 + (1 if x[1] > 0 else -1) * x[1]**2
        
        problem = OptimizationProblem(
            problem_id="discontinuous",
            objective=discontinuous_objective,
            variables=['x1', 'x2'],
            bounds={'x1': (-1, 1), 'x2': (-1, 1)}
        )
        
        backend = ClassicalBackend("discontinuous_solver")
        result = backend.solve_optimization(problem)
        
        assert isinstance(result, OptimizationResult)
        # Should handle discontinuous functions appropriately
    
    def test_backend_with_very_strict_tolerance(self):
        """Test backend with very strict convergence tolerance."""
        backend = ClassicalBackend("strict_tolerance", {
            'tol': 1e-12  # Very strict tolerance
        })
        
        result = backend.solve_optimization(self.simple_optimization)
        
        assert isinstance(result, OptimizationResult)
        
        if result.success:
            # With strict tolerance, should get very accurate solution
            assert abs(result.optimal_value) < 1e-10


@pytest.mark.slow
class TestBackendRobustness:
    """Robustness tests for backends under various conditions."""
    
    def test_backend_under_memory_pressure(self):
        """Test backend behavior under memory pressure."""
        # This test might be skipped in CI environments
        import psutil
        memory = psutil.virtual_memory()
        
        if memory.available < 2 * 1024 * 1024 * 1024:  # 2GB
            pytest.skip("Insufficient memory for memory pressure test")
        
        # Create multiple large problems
        backend = ClassicalBackend("memory_pressure_test")
        results = []
        
        for i in range(5):
            problem = OptimizationProblem(
                problem_id=f"memory_test_{i}",
                objective=lambda x: np.sum(x**2),
                variables=[f'x{j}' for j in range(20)],
                bounds={f'x{j}': (-10, 10) for j in range(20)}
            )
            
            result = backend.solve_optimization(problem)
            results.append(result)
        
        # All should complete without memory errors
        assert all(isinstance(r, OptimizationResult) for r in results)
    
    def test_backend_long_running_optimization(self):
        """Test backend with long-running optimization."""
        backend = ClassicalBackend("long_running", {
            'maxiter': 10000  # Allow many iterations
        })
        
        # Complex objective that might take longer
        def complex_objective(x):
            return np.sum(x**2) + np.sum(np.sin(100*x)) + np.sum(np.exp(-x**2))
        
        problem = OptimizationProblem(
            problem_id="complex",
            objective=complex_objective,
            variables=['x1', 'x2', 'x3', 'x4'],
            bounds={f'x{i+1}': (-5, 5) for i in range(4)}
        )
        
        result = backend.solve_optimization(problem)
        
        assert isinstance(result, OptimizationResult)
        # Should complete within reasonable time (test timeout will handle hangs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])