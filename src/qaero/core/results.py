"""
Results handling with comprehensive analytics and validation.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import time
from datetime import datetime
import json

from .base import Problem

@dataclass
class BaseResult:
    """Base result class with common analytics."""
    problem_id: str
    backend_name: str
    success: bool
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'problem_id': self.problem_id,
            'backend_name': self.backend_name,
            'success': self.success,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'error_message': self.error_message
        }
    
    def to_json(self) -> str:
        """Serialize result to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

@dataclass
class OptimizationResult(BaseResult):
    """Optimization result with comprehensive output."""
    optimal_value: Optional[float] = None
    optimal_variables: Optional[Dict[str, float]] = None
    solution_history: Optional[List[float]] = None
    convergence_data: Optional[Dict[str, Any]] = None
    constraints_violation: Optional[float] = None
    n_function_evaluations: Optional[int] = None
    
    @property
    def n_iterations(self) -> Optional[int]:
        """Number of iterations until convergence."""
        if self.solution_history:
            return len(self.solution_history)
        return None
    
    def get_optimal_variable(self, variable: str) -> float:
        """Get optimal value for a specific variable."""
        if not self.optimal_variables or variable not in self.optimal_variables:
            raise KeyError(f"Variable '{variable}' not found in optimal solution")
        return self.optimal_variables[variable]
    
    def plot_convergence(self, show: bool = True) -> Any:
        """Plot convergence history if available."""
        if not self.solution_history:
            raise ValueError("No solution history available for plotting")
        
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.solution_history, 'b-', linewidth=2)
            plt.xlabel('Iteration')
            plt.ylabel('Objective Value')
            plt.title(f'Convergence History - {self.problem_id}')
            plt.grid(True, alpha=0.3)
            plt.yscale('log' if max(self.solution_history) > 10 * min(self.solution_history) else 'linear')
            
            if show:
                plt.show()
            
            return plt.gcf()
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")

@dataclass
class PDEResult(BaseResult):
    """PDE solution result."""
    solution_field: Optional[np.ndarray] = None
    residual_norm: Optional[float] = None
    convergence_rate: Optional[float] = None
    mesh_info: Optional[Dict[str, Any]] = None
    field_statistics: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Compute field statistics if solution is available."""
        if self.solution_field is not None and self.field_statistics is None:
            self.field_statistics = self._compute_field_statistics()
    
    def _compute_field_statistics(self) -> Dict[str, Any]:
        """Compute statistics of the solution field."""
        if self.solution_field is None:
            return {}
        
        return {
            'min': float(np.min(self.solution_field)),
            'max': float(np.max(self.solution_field)),
            'mean': float(np.mean(self.solution_field)),
            'std': float(np.std(self.solution_field)),
            'l2_norm': float(np.linalg.norm(self.solution_field)),
            'size': self.solution_field.size,
            'shape': self.solution_field.shape
        }
    
    def visualize(self, show: bool = True) -> Any:
        """Visualize the solution field."""
        if self.solution_field is None:
            raise ValueError("No solution field available for visualization")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            if self.solution_field.ndim == 1:
                ax.plot(self.solution_field)
                ax.set_xlabel('Grid Point')
                ax.set_ylabel('Solution Value')
            elif self.solution_field.ndim == 2:
                im = ax.imshow(self.solution_field, cmap='viridis', origin='lower')
                plt.colorbar(im, ax=ax, label='Solution Value')
            else:
                raise ValueError(f"Cannot visualize {self.solution_field.ndim}D field")
            
            ax.set_title(f'PDE Solution - {self.problem_id}')
            ax.grid(True, alpha=0.3)
            
            if show:
                plt.show()
            
            return fig
        except ImportError:
            raise ImportError("Matplotlib is required for visualization")

class ResultValidator:
    """Validate results for physical plausibility and numerical stability."""
    
    @staticmethod
    def validate_optimization(result: OptimizationResult) -> Tuple[bool, List[str]]:
        """Validate optimization result."""
        warnings = []
        
        if not result.success:
            return True, ["Optimization failed - this is a valid result state"]
        
        # Basic checks
        checks = [
            (result.optimal_value is not None, "Missing optimal value"),
            (result.optimal_variables is not None, "Missing optimal variables"),
            (result.execution_time >= 0, "Negative execution time"),
        ]
        
        for condition, warning in checks:
            if not condition:
                warnings.append(warning)
        
        # Numerical stability checks
        if result.optimal_value is not None:
            if not np.isfinite(result.optimal_value):
                warnings.append("Optimal value is not finite")
            if abs(result.optimal_value) > 1e10:
                warnings.append("Optimal value suspiciously large")
        
        # Constraint violation check
        if result.constraints_violation is not None and result.constraints_violation > 1e-3:
            warnings.append(f"Significant constraint violation: {result.constraints_violation}")
        
        return len(warnings) == 0, warnings
    
    @staticmethod
    def validate_pde(result: PDEResult) -> Tuple[bool, List[str]]:
        """Validate PDE solution."""
        warnings = []
        
        if not result.success:
            return True, ["PDE solution failed - this is a valid result state"]
        
        if result.solution_field is None:
            warnings.append("Missing solution field")
            return False, warnings
        
        # Numerical checks
        if not np.all(np.isfinite(result.solution_field)):
            warnings.append("Solution field contains non-finite values")
        
        if result.residual_norm is not None:
            if result.residual_norm > 1e-3:
                warnings.append(f"Large residual norm: {result.residual_norm}")
            if result.residual_norm < 0:
                warnings.append("Negative residual norm")
        
        # Physical plausibility (basic checks)
        field_stats = result.field_statistics or {}
        if 'max' in field_stats and 'min' in field_stats:
            if abs(field_stats['max'] - field_stats['min']) > 1e6:
                warnings.append("Extreme variation in solution field")
        
        return len(warnings) == 0, warnings
    
    @staticmethod
    def validate_all(results: List[BaseResult]) -> Dict[str, Tuple[bool, List[str]]]:
        """Validate multiple results."""
        validation_results = {}
        
        for i, result in enumerate(results):
            if isinstance(result, OptimizationResult):
                is_valid, warnings = ResultValidator.validate_optimization(result)
            elif isinstance(result, PDEResult):
                is_valid, warnings = ResultValidator.validate_pde(result)
            else:
                is_valid, warnings = False, ["Unknown result type"]
            
            key = f"result_{i}_{result.problem_id}"
            validation_results[key] = (is_valid, warnings)
        
        return validation_results