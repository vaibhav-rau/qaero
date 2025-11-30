"""
Backend plugins for quantum and classical computation.
"""
from .base import Backend, BackendRegistry, register_backend
from .classical import ClassicalBackend, SimulatedAnnealingBackend
from .quantum import QuantumBackend, DWaveBackend, QiskitBackend, PennyLaneBackend

# Register all back