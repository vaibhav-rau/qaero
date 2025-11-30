"""
Tests for computational backends.
"""
import numpy as np
from src.qaero.backends import registry
from src.qaero.core.base import OptimizationProblem

def test_backend_registry():
    """Test backend registration and creation."""
    backends = registry.list_backends()
    assert "classical_scipy" in backends
    assert "simulated_annealing" in backends
    
    # Test backend creation