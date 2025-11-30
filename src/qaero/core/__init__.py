"""
QAero core module containing base classes and fundamental abstractions.
"""
from .base import Problem, OptimizationProblem, PDEProblem, QaeroError, BackendState
from .registry import AlgorithmRegistry, BackendRegistry
from .results import OptimizationResult, PDEResult, ResultValidator

__all__ = [
    'Problem', 'OptimizationProblem', 'PDEProblem', 'QaeroError', 'BackendState',
    'AlgorithmRegistry', 'BackendRegistry',
    'OptimizationResult', 'PDEResult', 'ResultValidator'
]