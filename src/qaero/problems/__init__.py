"""
Aerospace-specific problem templates for quantum optimization and simulation.
"""
from .aerodynamics import (
    AirfoilOptimizationProblem,
    WingDesignProblem,
    CompressibleFlowProblem,
    TransonicFlowProblem
)
from .trajectory import (
    AscentTrajectoryProblem,
    SatelliteManeuverProblem,
    OrbitalTransferProblem,
    ReentryTrajectoryProblem
)
from .structures import (
    StructuralOptimizationProblem,
    CompositeMaterialProblem,
    AeroelasticProblem
)

__all__ = [
    'AirfoilOptimizationProblem',
    'WingDesignProblem', 
    'CompressibleFlowProblem',
    'TransonicFlowProblem',
    'AscentTrajectoryProblem',
    'SatelliteManeuverProblem',
    'OrbitalTransferProblem',
    'ReentryTrajectoryProblem',
    'StructuralOptimizationProblem',
    'CompositeMaterialProblem',
    'AeroelasticProblem'
]