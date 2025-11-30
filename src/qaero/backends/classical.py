"""
High-performance classical backends as fallbacks and baselines.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.sparse.linalg import spsolve
import time

from .base import Backend, registry
from ..core.base import OptimizationProblem, PDEProblem, BackendState
from ..core.results import OptimizationResult, PDEResult

@registry.register("classical_scipy")
class ClassicalBackend(Backend):
    """Classical optimization backend using SciPy."""
    
    def _initialize(self):
        self._state = BackendState.READY
        self.logger.info(f"Initialized ClassicalBackend with config: {self.config}")
    
    def solve_optimization(self, problem: OptimizationProblem) -> OptimizationResult:
        start_time = time.time()
        
        try:
            # Extract bounds in SciPy format
            bounds = []
            for var in problem.variables:
                if problem.bounds and var in problem.bounds:
                    bounds.append(problem.bounds[var])
                else:
                    bounds.append((None, None))  # Unbounded
            
            # Initial guess
            x0 = np.zeros(len(problem.variables))
            
            # Solve using selected method
            method = self.config.get('method', 'SLSQP')
            result = minimize(
                problem.objective,
                x0,
                method=method,
                bounds=bounds,
                constraints=problem.constraints or []
            )
            
            # Build optimal variables dict
            optimal_vars = {
                var: result.x[i] for i, var in enumerate(problem.variables)
            }
            
            return OptimizationResult(
                problem_id=problem.problem_id,
                backend_name=self.name,
                success=result.success,
                execution_time=time.time() - start_time,
                optimal_value=result.fun,
                optimal_variables=optimal_vars,
                solution_history=getattr(result, 'x_history', None),
                metadata={
                    'n_iterations': result.nit,
                    'message': result.message,
                    'method': method
                }
            )
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return OptimizationResult(
                problem_id=problem.problem_id,
                backend_name=self.name,
                success=False,
                execution_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def solve_pde(self, problem: PDEProblem) -> PDEResult:
        # Placeholder for classical PDE solver
        # In practice, this would integrate with OpenFOAM, SU2, etc.
        raise NotImplementedError("Classical PDE solver coming in next release")

@registry.register("simulated_annealing")
class SimulatedAnnealingBackend(Backend):
    """Simulated annealing backend for combinatorial problems."""
    
    def _initialize(self):
        self._state = BackendState.READY
    
    def solve_optimization(self, problem: OptimizationProblem) -> OptimizationResult:
        start_time = time.time()
        
        try:
            from scipy.optimize import dual_annealing
            
            bounds = list(problem.bounds.values()) if problem.bounds else None
            
            result = dual_annealing(
                problem.objective,
                bounds=bounds,
                **self.config.get('annealing_params', {})
            )
            
            optimal_vars = {
                var: result.x[i] for i, var in enumerate(problem.variables)
            }
            
            return OptimizationResult(
                problem_id=problem.problem_id,
                backend_name=self.name,
                success=result.success,
                execution_time=time.time() - start_time,
                optimal_value=result.fun,
                optimal_variables=optimal_vars,
                metadata={
                    'n_iterations': result.nit,
                    'message': result.message
                }
            )
            
        except Exception as e:
            self.logger.error(f"Simulated annealing failed: {e}")
            return OptimizationResult(
                problem_id=problem.problem_id,
                backend_name=self.name,
                success=False,
                execution_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def solve_pde(self, problem: PDEProblem) -> PDEResult:
        raise NotImplementedError("PDE solving not available for annealing backend")