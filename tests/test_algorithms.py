"""
Comprehensive tests for quantum algorithms.
"""
import pytest
import numpy as np

from qaero.core.base import OptimizationProblem, PDEProblem
from qaero.algorithms.optimization import (
    QAOAOptimizer, VQEOptimizer, AnnealingOptimizer, HybridOptimizer
)
from qaero.algorithms.pde import (
    HHLLinearSolver, VQLSSolver, FiniteDifferenceSolver
)
from qaero.core.registry import AlgorithmRegistry


class TestOptimizationAlgorithms:
    """Test quantum optimization algorithms."""
    
    def setup_method(self):
        """Set up test optimization problems."""
        self.quadratic_problem = OptimizationProblem(
            problem_id="quadratic",
            objective=lambda x: np.sum(x**2),
            variables=['x1', 'x2'],
            bounds={'x1': (-1, 1), 'x2': (-1, 1)}
        )
        
        self.qubo_problem = OptimizationProblem(
            problem_id="qubo",
            objective=lambda x: x[0]**2 + x[1]**2 - 2*x[0]*x[1],
            variables=['x1', 'x2'],
            bounds={'x1': (0, 1), 'x2': (0, 1)}
        )
    
    def test_qaoa_optimizer(self):
        """Test QAOA optimization algorithm."""
        qaoa = QAOAOptimizer({'p': 2})  # 2 QAOA layers
        assert qaoa.name == "qaoa"
        assert qaoa.p == 2
        
        result = qaoa.optimize(self.qubo_problem)
        
        assert result.problem_id == "qubo"
        assert result.backend_name == "qaoa_p2"
        assert result.metadata['algorithm'] == 'QAOA'
        assert result.metadata['p'] == 2
        
        if result.success:
            assert result.optimal_value is not None
    
    def test_vqe_optimizer(self):
        """Test VQE optimization algorithm."""
        vqe = VQEOptimizer({'ansatz': 'EfficientSU2'})
        assert vqe.name == "vqe"
        
        result = vqe.optimize(self.qubo_problem)
        
        assert result.problem_id == "qubo"
        assert result.backend_name == "vqe"
        assert result.metadata['algorithm'] == 'VQE'
        assert result.metadata['ansatz'] == 'EfficientSU2'
    
    def test_annealing_optimizer(self):
        """Test quantum annealing optimizer."""
        annealer = AnnealingOptimizer({
            'annealing_time': 50,
            'num_reads': 500
        })
        
        result = annealer.optimize(self.quadratic_problem)
        
        assert result.problem_id == "quadratic"
        assert result.backend_name == "quantum_annealing"
        assert result.metadata['algorithm'] == 'QuantumAnnealing'
        assert result.metadata['annealing_time'] == 50
    
    def test_hybrid_optimizer(self):
        """Test hybrid quantum-classical optimizer."""
        hybrid = HybridOptimizer({
            'quantum_steps': 5,
            'classical_optimizer': 'BFGS'
        })
        
        result = hybrid.optimize(self.quadratic_problem)
        
        assert result.problem_id == "quadratic"
        assert result.backend_name == "hybrid_quantum_classical"
        assert result.metadata['algorithm'] == 'HybridQuantumClassical'
        assert 'hybrid_components' in result.metadata


class TestPDEAlgorithms:
    """Test quantum PDE solving algorithms."""
    
    def setup_method(self):
        """Set up test PDE problems."""
        self.poisson_problem = PDEProblem(
            problem_id="poisson",
            equation="laplace",
            domain={"bounds": [(0, 1)]},
            boundary_conditions={"left": 0, "right": 1},
            discretization="finite_difference"
        )
    
    def test_hhl_solver(self):
        """Test HHL linear solver algorithm."""
        hhl = HHLLinearSolver({'epsilon': 1e-4})
        assert hhl.name == "hhl"
        assert hhl.epsilon == 1e-4
        
        result = hhl.solve(self.poisson_problem)
        
        assert result.problem_id == "poisson"
        assert result.backend_name == "hhl_linear_solver"
        assert result.metadata['algorithm'] == 'HHL'
        assert result.metadata['epsilon'] == 1e-4
        
        if result.success:
            assert result.residual_norm < 1e-3
    
    def test_vqls_solver(self):
        """Test VQLS linear solver algorithm."""
        vqls = VQLSSolver({'max_iter': 100})
        
        result = vqls.solve(self.poisson_problem)
        
        assert result.problem_id == "poisson"
        assert result.backend_name == "vqls"
        assert result.metadata['algorithm'] == 'VQLS'
    
    def test_finite_difference_solver(self):
        """Test classical finite difference solver."""
        fd_solver = FiniteDifferenceSolver({
            'tolerance': 1e-8,
            'max_iter': 1000
        })
        
        result = fd_solver.solve(self.poisson_problem)
        
        assert result.problem_id == "poisson"
        assert result.backend_name == "finite_difference"
        assert result.metadata['algorithm'] == 'FiniteDifference'


class TestAlgorithmRegistry:
    """Test algorithm registration system."""
    
    def test_algorithm_categories(self):
        """Test algorithm categorization."""
        categories = AlgorithmRegistry.list_algorithms()
        
        assert 'quantum_optimization' in categories
        assert 'quantum_pde' in categories
        assert 'classical_pde' in categories
        
        # Should have some algorithms registered
        assert len(categories) > 0
    
    def test_algorithm_retrieval(self):
        """Test algorithm retrieval by category."""
        quantum_opt_algorithms = AlgorithmRegistry.get_algorithms_by_category('quantum_optimization')
        
        assert isinstance(quantum_opt_algorithms, list)
        assert 'qaoa' in quantum_opt_algorithms
        assert 'vqe' in quantum_opt_algorithms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])