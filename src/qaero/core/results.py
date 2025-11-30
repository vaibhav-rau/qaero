"""
Results handling with comprehensive analytics and validation.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import time
from datetime import datetime

@dataclass
class BaseResult:
    """Base result class with common analytics."""
    problem_id: str
    backend_name: str
    success: bool
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'problem_id': self.problem_id,
            'backend_name': self.backend_name,
            'success': self.success,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class OptimizationResult(BaseResult):
    """Optimization result with comprehensive output."""
    optimal_value: Optional[float] = None
    optimal_variables: Optional[Dict[str, float]] = None
    solution_history: Optional[List[float]] = None
    convergence_data: Optional[Dict[str, Any]] = None
    constraints_violation: Optional[float] = None
    
    @property
    def n_iterations(self) -> Optional[int]:
        """Number of iterations until convergence."""
        if self.solution_history:
            return len(self.solution_history)
        return None

@dataclass
class PDEResult(BaseResult):
    """PDE solution result."""
    solution_field: Optional[np.ndarray] = None
    residual_norm: Optional[float] = None
    convergence_rate: Optional[float] = None
    mesh_info: Optional[Dict[str, Any]] = None
    
    def visualize(self):
        """Placeholder for solution visualization."""
        pass

class ResultValidator:
    """Validate results for physical plausibility and numerical stability."""
    
    @staticmethod
    def validate_optimization(result: OptimizationResult) -> bool:
        """Validate optimization result."""
        if not result.success:
            return True  # Failed runs are still valid results
            
        checks = [
            result.optimal_value is not None,
            result.optimal_variables is not None,
            result.execution_time >= 0
        ]
        
        if result.optimal_value is not None:
            checks.append(np.isfinite(result.optimal_value))
            
        return all(checks)
    
    @staticmethod
    def validate_pde(result: PDEResult) -> bool:
        """Validate PDE solution."""
        if not result.success:
            return True
            
        checks = [
            result.solution_field is not None,
            np.all(np.isfinite(result.solution_field)),
            result.residual_norm is not None,
            result.residual_norm >= 0
        ]
        
        return all(checks)