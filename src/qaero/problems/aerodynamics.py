"""
Aerodynamics-specific problem templates.
"""
import numpy as np
from typing import Dict, List, Callable, Optional
from ..core.base import OptimizationProblem

class AirfoilOptimizationProblem(OptimizationProblem):
    """Airfoil shape optimization using parameterized geometry."""
    
    def __init__(
        self,
        parameterization: str = "naca",
        design_conditions: Optional[Dict] = None,
        objective: Optional[Callable] = None,
        **kwargs
    ):
        # Default design conditions
        design_conditions = design_conditions or {
            'mach': 0.3,
            'alpha': 2.0,  # degrees
            'reynolds': 1e6
        }
        
        # Default objective: maximize L/D ratio
        if objective is None:
            objective = self.default_aerodynamic_objective
        
        parameters = {
            'parameterization': parameterization,
            'design_conditions': design_conditions,
            **kwargs
        }
        
        # Define optimization variables based on parameterization
        if parameterization == "naca":
            variables = ['m', 'p', 't']  # NACA 4-digit parameters
            bounds = {'m': (0.0, 0.09), 'p': (0.2, 0.8), 't': (0.08, 0.2)}
        elif parameterization == "cst":
            variables = [f'A{i}' for i in range(6)]  # CST parameters
            bounds = {var: (-0.1, 0.1) for var in variables}
        else:
            raise ValueError(f"Unknown parameterization: {parameterization}")
        
        super().__init__(
            problem_id="airfoil_optimization",
            parameters=parameters,
            objective=objective,
            variables=variables,
            bounds=bounds
        )
    
    @staticmethod
    def default_aerodynamic_objective(x: np.ndarray) -> float:
        """
        Default objective: negative lift-to-drag ratio (for minimization).
        In practice, this would call a CFD solver.
        """
        # Placeholder aerodynamic model - would integrate with XFOIL, SU2, etc.
        m, p, t = x
        
        # Simple analytical approximation for demonstration
        cl = 0.1 + 5.0 * m  # Lift coefficient roughly proportional to camber
        cd = 0.02 + 0.1 * t + 0.05 * abs(p - 0.4)  # Drag model
        
        l_d_ratio = cl / (cd + 1e-8)  # Avoid division by zero
        
        return -l_d_ratio  # Negative for minimization

class WingDesignProblem(OptimizationProblem):
    """3D wing design optimization with multiple disciplines."""
    
    def __init__(self, planform_variables: List[str] = None, **kwargs):
        planform_variables = planform_variables or [
            'span', 'chord_root', 'chord_tip', 'sweep', 'taper'
        ]
        
        parameters = {
            'disciplines': ['aerodynamics', 'structures'],
            'coupling': 'weak',  # weak/strong multidisciplinary coupling
            **kwargs
        }
        
        bounds = {
            'span': (5.0, 50.0),
            'chord_root': (1.0, 5.0), 
            'chord_tip': (0.5, 3.0),
            'sweep': (0.0, 35.0),
            'taper': (0.2, 1.0)
        }
        
        super().__init__(
            problem_id="wing_design",
            parameters=parameters,
            objective=self.multidisciplinary_objective,
            variables=planform_variables,
            bounds=bounds
        )
    
    def multidisciplinary_objective(self, x: np.ndarray) -> float:
        """Multi-disciplinary objective combining aerodynamics and structures."""
        span, chord_root, chord_tip, sweep, taper = x
        
        # Aerodynamic performance (simplified)
        aspect_ratio = span / ((chord_root + chord_tip) / 2)
        aero_perf = aspect_ratio * np.cos(np.radians(sweep))
        
        # Structural weight (simplified)
        struct_weight = span**2 * (chord_root + chord_tip)
        
        # Combined objective (maximize performance, minimize weight)
        combined = aero_perf / (struct_weight + 1e-8)
        
        return -combined  # Negative for minimization