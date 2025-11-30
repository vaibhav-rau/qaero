"""
QAero: Quantum Aerospace Optimization & Simulation Toolkit
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("qaero")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

__author__ = "QAero Development Team"
__email__ = "dev@qaero.tech"

# Core imports
from .core.base import Problem, OptimizationProblem, PDEProblem, QaeroError
from .core.results import OptimizationResult, PDEResult, ResultValidator
from .core.registry import AlgorithmRegistry, BackendRegistry
from .core.solver import Solver, HybridOptimizerSolver, create_solver, SolverConfig
from .core.backend_interface import BackendInterface, QuantumBackendInterface, ClassicalBackendInterface
from .core.metrics import PerformanceMetrics, QualityMetrics, BenchmarkResult, BenchmarkSuite, global_benchmark_suite

# Backend imports
from .backends.base import Backend, BackendRegistry
from .backends.classical import ClassicalBackend, SimulatedAnnealingBackend
from .backends.quantum import QuantumBackend, DWaveBackend, QiskitBackend

# Problem template imports
from .problems.aerodynamics import (
    AirfoilOptimizationProblem,
    WingDesignProblem,
    CompressibleFlowProblem
)
from .problems.trajectory import (
    AscentTrajectoryProblem,
    SatelliteManeuverProblem,
    OrbitalTransferProblem
)

# Algorithm imports
from .algorithms.optimization import (
    QuantumOptimizer,
    QAOAOptimizer,
    VQEOptimizer,
    AnnealingOptimizer
)
from .algorithms.pde import (
    QuantumPDESolver,
    HHLLinearSolver,
    VQLSSolver
)

# Persona imports
from .personas import AerospaceEngineer, QuantumResearcher, create_persona, UserPreferences

# CLI imports
from .cli import main as cli_main

# Jupyter imports
from .jupyter import (
    QAeroDashboard,
    CircuitVisualizer,
    OptimizationTraceViewer,
    ParetoFrontViewer,
    show_dashboard,
    quick_solve,
    compare_backends
)

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
        
    Raises:
        QaeroError: If problem solution fails
        
    Example:
        >>> problem = AirfoilOptimizationProblem()
        >>> result = solve_problem(problem, backend="simulated_annealing")
        >>> print(f"Optimal value: {result.optimal_value}")
    """
    backend_instance = BackendRegistry.create_backend(backend, **backend_config)
    
    if not backend_instance.validate_problem(problem):
        raise QaeroError(f"Backend {backend} cannot solve problem {problem.problem_id}")
    
    if isinstance(problem, OptimizationProblem):
        return backend_instance.solve_optimization(problem)
    elif isinstance(problem, PDEProblem):
        return backend_instance.solve_pde(problem)
    else:
        raise QaeroError(f"Unsupported problem type: {type(problem)}")

def list_available_backends() -> list[str]:
    """List all registered computational backends."""
    return BackendRegistry.list_backends()

def list_available_algorithms() -> dict:
    """List all registered algorithms by category."""
    return AlgorithmRegistry.list_algorithms()

# Package-level backend registry
_backend_registry = BackendRegistry()
_algorithm_registry = AlgorithmRegistry()

__all__ = [
    # Core
    'solve_problem', 'list_available_backends', 'list_available_algorithms',
    
    # Problems
    'Problem', 'OptimizationProblem', 'PDEProblem',
    
    # Results
    'OptimizationResult', 'PDEResult', 'ResultValidator',
    
    # Backends
    'Backend', 'BackendRegistry',
    'ClassicalBackend', 'SimulatedAnnealingBackend',
    'QuantumBackend', 'DWaveBackend', 'QiskitBackend',
    
    # Algorithms
    'QuantumOptimizer', 'QAOAOptimizer', 'VQEOptimizer', 'AnnealingOptimizer',
    'QuantumPDESolver', 'HHLLinearSolver', 'VQLSSolver',
    
    # Problem templates
    'AirfoilOptimizationProblem', 'WingDesignProblem', 'CompressibleFlowProblem',
    'AscentTrajectoryProblem', 'SatelliteManeuverProblem', 'OrbitalTransferProblem',
    
    # Solver
    'Solver', 'HybridOptimizerSolver', 'create_solver', 'SolverConfig',
    
    # Backend Interface
    'BackendInterface', 'QuantumBackendInterface', 'ClassicalBackendInterface',
    
    # Metrics and Benchmarking
    'PerformanceMetrics', 'QualityMetrics', 'BenchmarkResult', 'BenchmarkSuite', 'global_benchmark_suite',
    
    # Personas
    'AerospaceEngineer', 'QuantumResearcher', 'create_persona', 'UserPreferences',
    
    # CLI
    'cli_main',
    
    # Jupyter
    'QAeroDashboard', 'CircuitVisualizer', 'OptimizationTraceViewer', 'ParetoFrontViewer',
    'show_dashboard', 'quick_solve', 'compare_backends',
    
    # Exceptions
    'QaeroError'
]