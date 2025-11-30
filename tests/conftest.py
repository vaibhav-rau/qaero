"""
Pytest configuration and fixtures for QAero tests.
"""
import pytest
import numpy as np
import tempfile
import os
from typing import Generator

from qaero.core.base import OptimizationProblem, PDEProblem
from qaero.problems.aerodynamics import AirfoilOptimizationProblem
from qaero.backends.classical import ClassicalBackend


@pytest.fixture
def simple_optimization_problem() -> OptimizationProblem:
    """Fixture for a simple quadratic optimization problem."""
    return OptimizationProblem(
        problem_id="simple_quadratic",
        objective=lambda x: np.sum(x**2),
        variables=['x1', 'x2'],
        bounds={'x1': (-1, 1), 'x2': (-1, 1)}
    )


@pytest.fixture
def rosenbrock_problem() -> OptimizationProblem:
    """Fixture for Rosenbrock function optimization problem."""
    return OptimizationProblem(
        problem_id="rosenbrock",
        objective=lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2,
        variables=['x', 'y'],
        bounds={'x': (-2, 2), 'y': (-1, 3)}
    )


@pytest.fixture
def airfoil_problem() -> AirfoilOptimizationProblem:
    """Fixture for airfoil optimization problem."""
    return AirfoilOptimizationProblem(parameterization="naca")


@pytest.fixture
def poisson_problem() -> PDEProblem:
    """Fixture for Poisson equation PDE problem."""
    return PDEProblem(
        problem_id="poisson",
        equation="laplace",
        domain={"bounds": [(0, 1)]},
        boundary_conditions={"left": 0, "right": 1},
        discretization="finite_difference"
    )


@pytest.fixture
def classical_backend() -> ClassicalBackend:
    """Fixture for classical optimization backend."""
    return ClassicalBackend("test_backend")


@pytest.fixture
def temp_results_dir() -> Generator[str, None, None]:
    """Fixture for temporary results directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def random_seed() -> int:
    """Fixture to set random seed for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    return seed


@pytest.fixture(scope="session")
def quantum_available() -> bool:
    """Check if quantum computing dependencies are available."""
    try:
        import qiskit
        return True
    except ImportError:
        return False


@pytest.fixture
def skip_if_no_quantum(quantum_available: bool) -> None:
    """Skip test if quantum dependencies are not available."""
    if not quantum_available:
        pytest.skip("Quantum computing dependencies not available")


def pytest_configure(config):
    """Pytest configuration hook."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "quantum: marks tests that require quantum dependencies"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--run-quantum", 
        action="store_true",
        default=False,
        help="run quantum tests (requires quantum dependencies)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command-line options."""
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_quantum = pytest.mark.skip(reason="need --run-quantum option to run")
    
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)
        if "quantum" in item.keywords and not config.getoption("--run-quantum"):
            item.add_marker(skip_quantum)