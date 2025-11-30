"""
QAero: Quantum Aerospace Optimization & Simulation Toolkit
"""
__version__ = "0.1.0"
__author__ = "QAero Development Team"

from .core.base import Problem, OptimizationProblem, PDEProblem, QaeroError
from .core.results import OptimizationResult, PDEResult
from .backends import registry, Backend
from .problems import AirfoilOptimizationProblem, WingDesignProblem, AscentTrajectoryProblem

# High-level API
def solve_problem(problem: Problem, backend: str = "classical_scipy", **backend_config):
    """
    High-level function to solve any QAero problem.
    
    Args:
        problem: The problem to solve (OptimizationProblem or PDEProblem)
        backend: Backend name (classical_scipy, simulated_annealing, quantum_generic, etc.)
        **backend_config: Backend-specific configuration
    
    Returns:
        OptimizationResult or PDEResult
    """
    backend_instance = registry.create_backend(backend, **backend_config)
    
    if isinstance(problem, OptimizationProblem):
        return backend_instance.solve_optimization(problem)
    elif isinstance(problem, PDEProblem):
        return backend_instance.solve_pde(problem)
    else:
        raise QaeroError(f"Unsupported problem type: {type(problem)}")

def list_available_backends():
    """List all registered computational backends."""
    return registry.list_backends()

__all__ = [
    # Core
    'solve_problem', 'list_available_backends',
    
    # Problems
    'Problem', 'OptimizationProblem', 'PDEProblem',
    
    # Results
    'OptimizationResult', 'PDEResult',
    
    # Backends
    'Backend', 'registry',
    
    # Problem templates
    'AirfoilOptimizationProblem', 'WingDesignProblem', 'AscentTrajectoryProblem',
    
    # Exceptions
    'QaeroError'
]