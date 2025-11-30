"""
Backend plugins for quantum and classical computation.
"""
from .base import Backend
from .classical import ClassicalBackend, SimulatedAnnealingBackend
from .quantum import QuantumBackend, DWaveBackend, QiskitBackend

__all__ = [
    'Backend',
    'ClassicalBackend', 
    'SimulatedAnnealingBackend',
    'QuantumBackend',
    'DWaveBackend',
    'QiskitBackend'
]