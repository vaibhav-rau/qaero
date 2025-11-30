"""
QAero Core Base Classes - Robust foundation for quantum aerospace computing.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger("qaero")

class QaeroError(Exception):
    """Base exception for all QAero-specific errors."""
    pass

class BackendState(Enum):
    """Backend state management."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"

@runtime_checkable
class BackendProtocol(Protocol):
    """Protocol defining backend interface for plugin compatibility."""
    
    def solve_optimization(self, problem: 'OptimizationProblem') -> 'OptimizationResult':
        ...
    
    def solve_pde(self, problem: 'PDEProblem') -> 'PDEResult':
        ...
    
    @property
    def state(self) -> BackendState:
        ...
    
    def validate_problem(self, problem: 'Problem') -> bool:
        ...

@dataclass
class Problem:
    """Base class for all computational problems."""
    problem_id: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        self.metadata = self.metadata or {}
        self.validate()
    
    def validate(self):
        """Validate problem parameters."""
        if not self.problem_id:
            raise QaeroError("Problem must have an ID")
        if not isinstance(self.parameters, dict):
            raise QaeroError("Parameters must be a dictionary")

@dataclass
class OptimizationProblem(Problem):
    """Optimization problem definition."""
    objective: Any  # Callable or symbolic objective
    variables: List[str]
    constraints: Optional[List[Any]] = None
    bounds: Optional[Dict[str, tuple]] = None
    
    def validate(self):
        super().validate()
        if not self.variables:
            raise QaeroError("Optimization problem must have variables")
        if not callable(self.objective) and not hasattr(self.objective, 'evalf'):
            raise QaeroError("Objective must be callable or symbolic")

@dataclass 
class PDEProblem(Problem):
    """PDE problem definition."""
    equation: Any
    domain: Any
    boundary_conditions: Dict[str, Any]
    discretization: str = "finite_difference"

class Backend(ABC):
    """Abstract base class for all computational backends."""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self._state = BackendState.INITIALIZING
        self.logger = logging.getLogger(f"qaero.backend.{name}")
        self._initialize()
    
    @abstractmethod
    def _initialize(self):
        """Initialize backend-specific resources."""
        pass
    
    @abstractmethod
    def solve_optimization(self, problem: OptimizationProblem) -> 'OptimizationResult':
        """Solve optimization problem."""
        pass
    
    @abstractmethod
    def solve_pde(self, problem: PDEProblem) -> 'PDEResult':
        """Solve PDE problem."""
        pass
    
    @property
    def state(self) -> BackendState:
        return self._state
    
    def validate_problem(self, problem: Problem) -> bool:
        """Validate if backend can handle this problem."""
        return True
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def cleanup(self):
        """Clean up backend resources."""
        self._state = BackendState.READY