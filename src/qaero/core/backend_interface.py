"""
Backend interface for quantum and classical computation.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import numpy as np
from dataclasses import dataclass

from .base import BackendState, QaeroError


@dataclass
class QuantumJob:
    """Quantum job representation."""
    job_id: str
    backend: str
    circuit_or_qubo: Any
    shots: int = 1024
    parameters: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        self.parameters = self.parameters or {}
        self.metadata = self.metadata or {}


@dataclass
class JobStatus:
    """Job status information."""
    job_id: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    progress: float = 0.0  # 0.0 to 1.0
    queue_position: Optional[int] = None
    estimated_completion: Optional[float] = None
    error_message: Optional[str] = None


class BackendInterface(ABC):
    """
    Abstract backend interface for quantum and classical computation.
    Provides unified API for circuit execution, sampling, and expectation estimation.
    """
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        self.name = name
        self.config = config or {}
        self._state = BackendState.INITIALIZING
        self._job_queue: List[QuantumJob] = []
        self._completed_jobs: Dict[str, Any] = {}
    
    @abstractmethod
    def submit(self, circuit_or_qubo: Any, **kwargs) -> QuantumJob:
        """Submit a quantum circuit or QUBO for execution."""
        pass
    
    @abstractmethod
    def estimate_expectation(self, operator: Any, params: Any, **kwargs) -> float:
        """Estimate expectation value of an operator."""
        pass
    
    @abstractmethod
    def run_sampler(self, circuit_or_qubo: Any, shots: int = 1024, **kwargs) -> Dict[str, int]:
        """Run sampler and return measurement results."""
        pass
    
    def get_job_status(self, job_id: str) -> JobStatus:
        """Get status of a submitted job."""
        # Check if job is completed
        if job_id in self._completed_jobs:
            return JobStatus(job_id=job_id, status='completed', progress=1.0)
        
        # Check if job is in queue
        for i, job in enumerate(self._job_queue):
            if job.job_id == job_id:
                return JobStatus(
                    job_id=job_id,
                    status='pending',
                    progress=0.0,
                    queue_position=i
                )
        
        raise QaeroError(f"Job {job_id} not found")
    
    def get_job_result(self, job_id: str) -> Any:
        """Get result of a completed job."""
        if job_id not in self._completed_jobs:
            raise QaeroError(f"Job {job_id} not found or not completed")
        return self._completed_jobs[job_id]
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        for i, job in enumerate(self._job_queue):
            if job.job_id == job_id:
                self._job_queue.pop(i)
                return True
        return False
    
    def list_jobs(self, status: Optional[str] = None) -> List[QuantumJob]:
        """List jobs with optional status filter."""
        if status == 'completed':
            return list(self._completed_jobs.values())
        elif status == 'pending':
            return self._job_queue.copy()
        else:
            return self._job_queue.copy() + list(self._completed_jobs.values())
    
    def clear_jobs(self):
        """Clear all jobs from the backend."""
        self._job_queue.clear()
        self._completed_jobs.clear()
    
    @property
    def queue_length(self) -> int:
        """Get current queue length."""
        return len(self._job_queue)
    
    @property
    def state(self) -> BackendState:
        return self._state


class QuantumBackendInterface(BackendInterface):
    """Quantum backend interface with circuit compilation and execution."""
    
    def __init__(self, name: str, config: Optional[Dict] = None):
        super().__init__(name, config)
        self._quantum_instance = None
        self._transpiler = None
        self._noise_model = None
        
        self._initialize_quantum_resources()
    
    def _initialize_quantum_resources(self):
        """Initialize quantum-specific resources."""
        # This would setup quantum hardware/simulator connection
        # For now, implement a simulated version
        self._state = BackendState.READY
    
    def submit(self, circuit_or_qubo: Any, **kwargs) -> QuantumJob:
        """Submit quantum circuit for execution."""
        import uuid
        
        job_id = str(uuid.uuid4())[:8]
        shots = kwargs.get('shots', self.config.get('shots', 1024))
        
        job = QuantumJob(
            job_id=job_id,
            backend=self.name,
            circuit_or_qubo=circuit_or_qubo,
            shots=shots,
            parameters=kwargs
        )
        
        # Simulate quantum execution
        self._job_queue.append(job)
        self._simulate_quantum_execution(job)
        
        return job
    
    def estimate_expectation(self, operator: Any, params: Any, **kwargs) -> float:
        """Estimate expectation value using quantum computation."""
        # Simplified expectation estimation
        # In practice, this would involve actual quantum circuit execution
        if isinstance(params, (int, float)):
            return float(params ** 2)  # Simple quadratic
        elif isinstance(params, np.ndarray):
            return float(np.sum(params ** 2))
        else:
            return 0.0
    
    def run_sampler(self, circuit_or_qubo: Any, shots: int = 1024, **kwargs) -> Dict[str, int]:
        """Run quantum sampler and return measurement counts."""
        # Simulate quantum sampling
        if hasattr(circuit_or_qubo, 'num_qubits'):
            n_qubits = circuit_or_qubo.num_qubits
        else:
            n_qubits = 2  # Default
        
        # Generate random measurement results
        results = {}
        for i in range(2 ** n_qubits):
            bitstring = format(i, f'0{n_qubits}b')
            results[bitstring] = np.random.randint(0, shots // (2 ** n_qubits))
        
        # Normalize to total shots
        total = sum(results.values())
        if total < shots:
            # Add remaining shots to most probable outcome
            most_probable = max(results, key=results.get)
            results[most_probable] += shots - total
        
        return results
    
    def _simulate_quantum_execution(self, job: QuantumJob):
        """Simulate quantum job execution (would be real hardware in production)."""
        # In production, this would submit to actual quantum hardware
        # For simulation, we immediately "complete" the job
        self._job_queue.remove(job)
        
        # Generate simulated results
        if hasattr(job.circuit_or_qubo, 'num_qubits'):
            n_qubits = job.circuit_or_qubo.num_qubits
            results = self.run_sampler(job.circuit_or_qubo, job.shots)
        else:
            # QUBO problem - simulate annealing results
            results = {'solution': np.random.randn(10).tolist()}
        
        self._completed_jobs[job.job_id] = {
            'job_id': job.job_id,
            'results': results,
            'metadata': job.metadata
        }


class ClassicalBackendInterface(BackendInterface):
    """Classical backend interface for simulation and fallback."""
    
    def submit(self, circuit_or_qubo: Any, **kwargs) -> QuantumJob:
        """Submit for classical simulation."""
        import uuid
        
        job_id = str(uuid.uuid4())[:8]
        
        job = QuantumJob(
            job_id=job_id,
            backend=self.name,
            circuit_or_qubo=circuit_or_qubo,
            parameters=kwargs
        )
        
        # Classical simulation is immediate
        results = self._classical_simulation(circuit_or_qubo, **kwargs)
        self._completed_jobs[job.job_id] = {
            'job_id': job.job_id,
            'results': results,
            'metadata': {'simulation_type': 'classical'}
        }
        
        return job
    
    def estimate_expectation(self, operator: Any, params: Any, **kwargs) -> float:
        """Classical expectation estimation."""
        if isinstance(params, (int, float)):
            return float(params)
        elif isinstance(params, np.ndarray):
            return float(np.mean(params))
        else:
            return 0.0
    
    def run_sampler(self, circuit_or_qubo: Any, shots: int = 1024, **kwargs) -> Dict[str, int]:
        """Classical sampling simulation."""
        # Classical simulation of quantum sampling
        if hasattr(circuit_or_qubo, 'num_qubits'):
            n_qubits = circuit_or_qubo.num_qubits
        else:
            n_qubits = 2
        
        # Generate deterministic or probabilistic results
        results = {}
        for i in range(2 ** n_qubits):
            bitstring = format(i, f'0{n_qubits}b')
            # Prefer all-zeros state in classical simulation
            if bitstring == '0' * n_qubits:
                results[bitstring] = shots // 2
            else:
                results[bitstring] = shots // (2 * (2 ** n_qubits - 1))
        
        return results
    
    def _classical_simulation(self, circuit_or_qubo: Any, **kwargs) -> Dict[str, Any]:
        """Perform classical simulation."""
        if hasattr(circuit_or_qubo, 'num_qubits'):
            # Quantum circuit simulation
            return {
                'statevector': np.random.randn(2 ** circuit_or_qubo.num_qubits),
                'density_matrix': None,
                'measurement_counts': self.run_sampler(circuit_or_qubo, kwargs.get('shots', 1024))
            }
        else:
            # QUBO problem - classical optimization
            return {
                'solution': np.zeros(10),  # Placeholder
                'energy': 0.0,
                'optimization_result': 'success'
            }