"""
Aerospace trajectory optimization problems.
"""
import numpy as np
from typing import Dict, List
from ..core.base import OptimizationProblem

class AscentTrajectoryProblem(OptimizationProblem):
    """Rocket ascent trajectory optimization."""
    
    def __init__(self, stages: int = 2, **kwargs):
        # Variables: throttle settings, pitch angles per stage
        variables = []
        for stage in range(stages):
            variables.extend([
                f'stage_{stage}_throttle',
                f'stage_{stage}_pitch_initial', 
                f'stage_{stage}_pitch_final'
            ])
        
        bounds = {}
        for var in variables:
            if 'throttle' in var:
                bounds[var] = (0.3, 1.0)  # Throttle limits
            elif 'pitch' in var:
                bounds[var] = (0.0, 90.0)  # Pitch angle limits
        
        parameters = {
            'stages': stages,
            'target_orbit': kwargs.get('target_orbit', 'LEO'),
            'constraints': ['max_q', 'max_g', 'orbit_injection'],
            **kwargs
        }
        
        super().__init__(
            problem_id="ascent_trajectory",
            parameters=parameters,
            objective=self.trajectory_objective,
            variables=variables,
            bounds=bounds
        )
    
    def trajectory_objective(self, x: np.ndarray) -> float:
        """Objective: maximize payload mass to orbit."""
        # Simplified rocket equation-based model
        total_delta_v = 0.0
        stages = self.parameters['stages']
        
        for stage in range(stages):
            throttle = x[stage * 3]
            # Simplified delta-v calculation
            stage_dv = 3000 + 1000 * throttle  # m/s
            total_delta_v += stage_dv
        
        # Target LEO requires ~9400 m/s
        required_dv = 9400
        payload_frac = max(0, 1 - abs(total_delta_v - required_dv) / required_dv)
        
        return -payload_frac  # Negative for minimization

class SatelliteManeuverProblem(OptimizationProblem):
    """Satellite orbital maneuver optimization."""
    
    def __init__(self, **kwargs):
        variables = ['burn1_magnitude', 'burn1_angle', 'burn2_magnitude', 'burn2_angle']
        
        bounds = {
            'burn1_magnitude': (0.0, 500.0),  # m/s
            'burn1_angle': (0.0, 360.0),      # degrees
            'burn2_magnitude': (0.0, 500.0),
            'burn2_angle': (0.0, 360.0)
        }
        
        parameters = {
            'initial_orbit': kwargs.get('initial_orbit', {'altitude': 400, 'inclination': 28.5}),
            'target_orbit': kwargs.get('target_orbit', {'altitude': 1200, 'inclination': 0.0}),
            **kwargs
        }
        
        super().__init__(
            problem_id="satellite_maneuver",
            parameters=parameters,
            objective=self.maneuver_objective,
            variables=variables,
            bounds=bounds
        )
    
    def maneuver_objective(self, x: np.ndarray) -> float:
        """Objective: minimize propellant used for maneuver."""
        burn1_dv, burn1_angle, burn2_dv, burn2_angle = x
        
        # Simplified orbital mechanics
        # In practice, this would use patched conics or high-fidelity propagator
        total_dv = burn1_dv + burn2_dv
        
        # Add penalty for inefficient burn angles
        angle_penalty = abs(np.sin(np.radians(burn1_angle))) + abs(np.sin(np.radians(burn2_angle)))
        
        return total_dv + 0.1 * angle_penalty