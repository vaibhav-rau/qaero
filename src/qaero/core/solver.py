"""
Solver abstraction with hybrid quantum-classical capabilities.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass
import time
import logging

from .base import Problem, OptimizationProblem, PDEProblem, QaeroError
from .results import OptimizationResult, PDEResult, ResultValidator
from ..backends import BackendRegistry
from ..algorithms.optimization import (
    QAOAOptimizer, VQEOptimizer, AnnealingOptimizer, HybridOptimizer
)
from ..algorithms.pde import HHLLinearSolver, VQLSSolver

logger = logging.getLogger("qaero.solver")


@dataclass
class SolverConfig:
    """Solver configuration with hybrid optimization settings."""
    backend: str = "auto"
    algorithm: str = "auto"
    max_iterations: int = 1000
    tolerance: float = 1e-6
    auto_fallback: bool = True
    verbose: bool = False
    save_intermediate_results: bool = False
    timeout: float = 3600.0  # seconds
    
    # Hybrid optimization parameters
    classical_optimizer: str = "BFGS"
    quantum_steps: int = 10
    parameter_server: bool = True
    
    # Quantum-specific parameters
    shots: int = 1024
    optimization_level: int = 1


class Solver(ABC):
    """
    Abstract solver class with pluggable backends and algorithms.
    Supports hybrid quantum-classical optimization.
    """
    
    def __init__(self, config: Optional[Union[SolverConfig, Dict]] = None):
        if config is None:
            self.config = SolverConfig()
        elif isinstance(config, dict):
            self.config = SolverConfig(**config)
        else:
            self.config = config
        
        self.backend = None
        self.algorithm = None
        self._callbacks: List[Callable] = []
        self._intermediate_results: List[Any] = []
        self._setup_solver()
    
    def _setup_solver(self):
        """Setup solver with configured backend and algorithm."""
        # Setup backend
        if self.config.backend == "auto":
            self.backend = self._auto_select_backend()
        else:
            self.backend = BackendRegistry.create_backend(self.config.backend)
        
        # Setup algorithm
        if self.config.algorithm == "auto":
            self.algorithm = self._auto_select_algorithm()
        else:
            self.algorithm = self._create_algorithm(self.config.algorithm)
        
        logger.info(f"Solver initialized with backend: {self.backend.name}, "
                   f"algorithm: {self.algorithm.__class__.__name__}")
    
    def _auto_select_backend(self) -> Any:
        """Automatically select the best backend based on problem type and available resources."""
        available_backends = BackendRegistry.list_backends()
        
        # Prefer quantum backends if available and appropriate
        quantum_backends = [b for b in available_backends if 'quantum' in b or 'qiskit' in b or 'dwave' in b]
        if quantum_backends and self.config.auto_fallback:
            return BackendRegistry.create_backend(quantum_backends[0])
        
        # Fall back to classical
        classical_backends = [b for b in available_backends if 'classical' in b or 'scipy' in b]
        if classical_backends:
            return BackendRegistry.create_backend(classical_backends[0])
        
        raise QaeroError("No suitable backends available")
    
    def _auto_select_algorithm(self) -> Any:
        """Automatically select the best algorithm based on problem type."""
        # Default to hybrid optimizer for flexibility
        return HybridOptimizer({
            'classical_optimizer': self.config.classical_optimizer,
            'quantum_steps': self.config.quantum_steps
        })
    
    def _create_algorithm(self, algorithm_name: str) -> Any:
        """Create algorithm instance by name."""
        algorithm_map = {
            'qaoa': QAOAOptimizer,
            'vqe': VQEOptimizer,
            'annealing': AnnealingOptimizer,
            'hybrid': HybridOptimizer,
            'hhl': HHLLinearSolver,
            'vqls': VQLSSolver
        }
        
        if algorithm_name not in algorithm_map:
            raise QaeroError(f"Unknown algorithm: {algorithm_name}")
        
        return algorithm_map[algorithm_name]()
    
    def solve(self, problem: Problem, **kwargs) -> Union[OptimizationResult, PDEResult]:
        """
        Solve a problem with the configured solver.
        
        Args:
            problem: The problem to solve
            **kwargs: Additional solver parameters
            
        Returns:
            OptimizationResult or PDEResult depending on problem type
        """
        start_time = time.time()
        
        try:
            # Validate problem
            if not self.backend.validate_problem(problem):
                raise QaeroError(f"Backend {self.backend.name} cannot solve problem {problem.problem_id}")
            
            # Execute callbacks before solving
            for callback in self._callbacks:
                callback('pre_solve', problem)
            
            # Solve based on problem type
            if isinstance(problem, OptimizationProblem):
                result = self._solve_optimization(problem, **kwargs)
            elif isinstance(problem, PDEProblem):
                result = self._solve_pde(problem, **kwargs)
            else:
                raise QaeroError(f"Unsupported problem type: {type(problem)}")
            
            # Validate result
            is_valid, warnings = ResultValidator.validate_optimization(result) if isinstance(result, OptimizationResult) else ResultValidator.validate_pde(result)
            if not is_valid and self.config.verbose:
                logger.warning(f"Result validation warnings: {warnings}")
            
            # Execute callbacks after solving
            for callback in self._callbacks:
                callback('post_solve', result)
            
            return result
            
        except Exception as e:
            logger.error(f"Solver failed: {e}")
            
            # Create error result
            if isinstance(problem, OptimizationProblem):
                return OptimizationResult(
                    problem_id=problem.problem_id,
                    backend_name=self.backend.name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                )
            else:
                return PDEResult(
                    problem_id=problem.problem_id,
                    backend_name=self.backend.name,
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message=str(e)
                )
    
    def _solve_optimization(self, problem: OptimizationProblem, **kwargs) -> OptimizationResult:
        """Solve optimization problem."""
        # Use algorithm if available, otherwise use backend directly
        if self.algorithm and hasattr(self.algorithm, 'optimize'):
            result = self.algorithm.optimize(problem, **kwargs)
        else:
            result = self.backend.solve_optimization(problem)
        
        # Store intermediate result if configured
        if self.config.save_intermediate_results:
            self._intermediate_results.append(result)
        
        return result
    
    def _solve_pde(self, problem: PDEProblem, **kwargs) -> PDEResult:
        """Solve PDE problem."""
        if self.algorithm and hasattr(self.algorithm, 'solve'):
            result = self.algorithm.solve(problem, **kwargs)
        else:
            result = self.backend.solve_pde(problem)
        
        # Store intermediate result if configured
        if self.config.save_intermediate_results:
            self._intermediate_results.append(result)
        
        return result
    
    def add_callback(self, callback: Callable):
        """Add a callback function to be executed during solving."""
        self._callbacks.append(callback)
    
    def get_intermediate_results(self) -> List[Any]:
        """Get all intermediate results if save_intermediate_results is enabled."""
        return self._intermediate_results.copy()
    
    def clear_intermediate_results(self):
        """Clear stored intermediate results."""
        self._intermediate_results.clear()
    
    def set_backend(self, backend_name: str, **backend_config):
        """Dynamically change the solver backend."""
        self.backend = BackendRegistry.create_backend(backend_name, **backend_config)
        logger.info(f"Solver backend changed to: {backend_name}")
    
    def set_algorithm(self, algorithm_name: str, **algorithm_config):
        """Dynamically change the solver algorithm."""
        self.algorithm = self._create_algorithm(algorithm_name)
        if algorithm_config:
            self.algorithm.config.update(algorithm_config)
        logger.info(f"Solver algorithm changed to: {algorithm_name}")


class HybridOptimizerSolver(Solver):
    """
    Specialized solver for hybrid quantum-classical optimization.
    Manages parameter servers and coordinates between classical and quantum components.
    """
    
    def __init__(self, config: Optional[Union[SolverConfig, Dict]] = None):
        super().__init__(config)
        self.parameter_server = None
        self.classical_optimizer = None
        self.quantum_processor = None
        
        if self.config.parameter_server:
            self._setup_parameter_server()
    
    def _setup_parameter_server(self):
        """Setup parameter server for hybrid optimization."""
        # This would interface with actual parameter server infrastructure
        # For now, implement a simple in-memory version
        self.parameter_server = {
            'current_parameters': None,
            'parameter_history': [],
            'gradient_history': [],
            'metadata': {}
        }
    
    def _solve_optimization(self, problem: OptimizationProblem, **kwargs) -> OptimizationResult:
        """Execute hybrid quantum-classical optimization."""
        import time
        start_time = time.time()
        
        try:
            from scipy.optimize import minimize
            import numpy as np
            
            solution_history = []
            parameter_history = []
            
            def hybrid_objective(x):
                # Store parameters
                if self.config.parameter_server:
                    self.parameter_server['current_parameters'] = x.copy()
                    self.parameter_server['parameter_history'].append(x.copy())
                
                parameter_history.append(x.copy())
                
                # Classical objective evaluation
                classical_val = problem.objective(x)
                solution_history.append(classical_val)
                
                # Quantum enhancement (simplified)
                quantum_enhancement = self._quantum_enhancement(x, classical_val)
                
                total_cost = classical_val + quantum_enhancement
                
                # Execute callbacks
                for callback in self._callbacks:
                    callback('hybrid_evaluation', {
                        'parameters': x,
                        'classical_value': classical_val,
                        'quantum_enhancement': quantum_enhancement,
                        'total_cost': total_cost
                    })
                
                return total