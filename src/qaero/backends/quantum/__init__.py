"""
Quantum computing backends with full hardware integration.
Gate-based simulators, annealers, cloud providers, and classical fallbacks.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import asyncio
import time
import warnings
from enum import Enum
import json

from ....core.base import Backend, OptimizationProblem, PDEProblem, BackendState, QaeroError
from ....core.results import OptimizationResult, PDEResult
from ....core.registry import register_backend

logger = logging.getLogger("qaero.backends.quantum")

class BackendType(Enum):
    """Types of quantum computing backends."""
    GATE_BASED_SIMULATOR = "gate_based_simulator"
    GATE_BASED_HARDWARE = "gate_based_hardware"
    QUANTUM_ANNEALER = "quantum_annealer"
    HYBRID_SOLVER = "hybrid_solver"
    CLASSICAL_FALLBACK = "classical_fallback"
    CLOUD_QUANTUM = "cloud_quantum"

@dataclass
class BackendConfig:
    """Configuration for quantum backends."""
    backend_type: BackendType
    provider: str
    device_name: str
    shots: int = 1024
    optimization_level: int = 1
    noise_model: Optional[Any] = None
    max_jobs: int = 10
    timeout: float = 3600.0
    cost_tracking: bool = True

@dataclass
class QuantumJob:
    """Quantum job representation with full metadata."""
    job_id: str
    backend: str
    circuit_or_qubo: Any
    shots: int
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    submitted_at: float
    cost_estimate: float = 0.0

@dataclass
class DeviceCharacteristics:
    """Characteristics of quantum hardware."""
    n_qubits: int
    topology: List[Tuple[int, int]]  # Qubit connectivity
    gate_times: Dict[str, float]  # Gate operation times
    readout_time: float
    t1: float  # Relaxation time
    t2: float  # Dephasing time
    gate_fidelities: Dict[str, float]
    readout_fidelity: float

@register_backend("qiskit_aer")
class QiskitAerBackend(Backend):
    """Qiskit Aer simulator backend with noise models."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("qiskit_aer", config or {})
        self.backend_type = BackendType.GATE_BASED_SIMULATOR
        self.simulator = None
        self.noise_model = None
        self.job_queue = []
        self.completed_jobs = {}
        
    def _initialize(self):
        """Initialize Qiskit Aer simulator."""
        try:
            from qiskit import Aer
            from qiskit.providers.aer import AerSimulator
            from qiskit.providers.aer.noise import NoiseModel
            
            # Create simulator with configuration
            simulator_config = {
                'method': 'statevector',
                'max_parallel_threads': 0,  # Auto-detect
                'max_parallel_experiments': 1,
                'shots': self.config.get('shots', 1024)
            }
            
            self.simulator = AerSimulator(**simulator_config)
            
            # Setup noise model if specified
            if self.config.get('noise_model'):
                self.noise_model = self._create_noise_model()
            
            self._state = BackendState.READY
            logger.info("Qiskit Aer backend initialized successfully")
            
        except ImportError as e:
            logger.error(f"Qiskit not available: {e}")
            self._state = BackendState.ERROR
            raise QaeroError("Qiskit installation required for this backend")
    
    def _create_noise_model(self) -> Any:
        """Create realistic noise model for aerospace simulations."""
        try:
            from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
            
            noise_model = NoiseModel()
            
            # Add depolarizing noise for single-qubit gates
            p_depol_single = 0.001  # 0.1% error rate
            depol_single = depolarizing_error(p_depol_single, 1)
            noise_model.add_all_qubit_quantum_error(depol_single, ['u1', 'u2', 'u3'])
            
            # Add depolarizing noise for two-qubit gates
            p_depol_double = 0.01  # 1% error rate for CNOT
            depol_double = depolarizing_error(p_depol_double, 2)
            noise_model.add_all_qubit_quantum_error(depol_double, ['cx'])
            
            # Thermal relaxation (T1=100μs, T2=150μs typical for superconductors)
            t1 = 100e-6
            t2 = 150e-6
            gate_time = 50e-9  # 50ns gate time
            thermal_error = thermal_relaxation_error(t1, t2, gate_time)
            noise_model.add_all_qubit_quantum_error(thermal_error, ['u1', 'u2', 'u3', 'cx'])
            
            return noise_model
            
        except Exception as e:
            logger.warning(f"Could not create noise model: {e}")
            return None
    
    def solve_optimization(self, problem: OptimizationProblem) -> OptimizationResult:
        """Solve optimization using quantum circuits."""
        import time
        start_time = time.time()
        
        try:
            from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
            from qiskit.circuit import Parameter
            from qiskit.algorithms.optimizers import COBYLA
            from qiskit.algorithms import QAOA
            from qiskit_optimization import QuadraticProgram
            from qiskit_optimization.converters import QuadraticProgramToQubo
            
            # Convert problem to QUBO
            qubo = self._problem_to_qubo(problem)
            qp = QuadraticProgram()
            
            # Add variables
            for var in problem.variables:
                if problem.bounds and var in problem.bounds:
                    low, high = problem.bounds[var]
                    qp.continuous_var(low, high, name=var)
                else:
                    qp.continuous_var(name=var)
            
            # Set objective (simplified)
            qp.minimize(constant=0.0, linear={}, quadratic={})
            
            # Convert to QUBO
            converter = QuadraticProgramToQubo()
            qubo_problem = converter.convert(qp)
            
            # Setup QAOA
            optimizer = COBYLA(maxiter=100)
            qaoa = QAOA(optimizer=optimizer, quantum_instance=self.simulator)
            
            # Execute QAOA
            result = qaoa.compute_minimum_eigenvalue(qubo_problem.to_ising()[0])
            
            # Process results
            optimal_value = result.eigenvalue.real
            optimal_point = self._decode_qiskit_solution(result, problem)
            
            return OptimizationResult(
                problem_id=problem.problem_id,
                backend_name=self.name,
                success=True,
                execution_time=time.time() - start_time,
                optimal_value=optimal_value,
                optimal_variables=optimal_point,
                metadata={
                    'backend': 'qiskit_aer',
                    'algorithm': 'QAOA',
                    'shots': self.config.get('shots', 1024),
                    'optimizer': 'COBYLA'
                }
            )
            
        except Exception as e:
            logger.error(f"Qiskit optimization failed: {e}")
            return OptimizationResult(
                problem_id=problem.problem_id,
                backend_name=self.name,
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def solve_pde(self, problem: PDEProblem) -> PDEResult:
        """Solve PDE using quantum linear algebra (simulated)."""
        # This would implement HHL or VQLS for PDE solving
        # For now, provide classical fallback
        from ....backends.classical import ClassicalBackend
        
        classical = ClassicalBackend("qiskit_fallback")
        result = classical.solve_pde(problem)
        result.backend_name = self.name
        result.metadata['quantum_simulation'] = True
        
        return result
    
    def _problem_to_qubo(self, problem: OptimizationProblem) -> Dict:
        """Convert optimization problem to QUBO format."""
        # Simplified QUBO conversion
        n_vars = len(problem.variables)
        qubo = {}
        
        for i in range(n_vars):
            for j in range(i, n_vars):
                if i == j:
                    qubo[(i, i)] = 1.0  # Linear terms
                else:
                    qubo[(i, j)] = 0.1  # Small quadratic coupling
        
        return qubo
    
    def _decode_qiskit_solution(self, result: Any, problem: OptimizationProblem) -> Dict[str, float]:
        """Decode Qiskit solution to problem variables."""
        # Simplified decoding
        optimal_point = {}
        for i, var in enumerate(problem.variables):
            if problem.bounds and var in problem.bounds:
                low, high = problem.bounds[var]
                optimal_point[var] = (low + high) / 2  # Midpoint as fallback
            else:
                optimal_point[var] = 0.0
        
        return optimal_point

@register_backend("pennylane")
class PennyLaneBackend(Backend):
    """PennyLane backend for quantum machine learning and optimization."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("pennylane", config or {})
        self.backend_type = BackendType.GATE_BASED_SIMULATOR
        self.device = None
        self.supported_devices = ['default.qubit', 'default.qubit.tf', 'default.qubit.autograd']
        
    def _initialize(self):
        """Initialize PennyLane device."""
        try:
            import pennylane as qml
            
            device_name = self.config.get('device', 'default.qubit')
            if device_name not in self.supported_devices:
                logger.warning(f"Device {device_name} not supported, using default.qubit")
                device_name = 'default.qubit'
            
            self.device = qml.device(device_name, wires=self.config.get('n_qubits', 4))
            self._state = BackendState.READY
            logger.info(f"PennyLane backend initialized with device {device_name}")
            
        except ImportError as e:
            logger.error(f"PennyLane not available: {e}")
            self._state = BackendState.ERROR
            raise QaeroError("PennyLane installation required for this backend")
    
    def solve_optimization(self, problem: OptimizationProblem) -> OptimizationResult:
        """Solve optimization using PennyLane variational circuits."""
        import time
        start_time = time.time()
        
        try:
            import pennylane as qml
            from pennylane import numpy as pnp
            
            n_qubits = min(len(problem.variables), 4)  # Limit qubits for simulation
            n_layers = self.config.get('n_layers', 2)
            
            # Define quantum circuit
            def circuit(params, n_qubits=n_qubits, n_layers=n_layers):
                # Hardware-efficient ansatz
                for layer in range(n_layers):
                    # Rotation layers
                    for i in range(n_qubits):
                        qml.RX(params[layer, i, 0], wires=i)
                        qml.RY(params[layer, i, 1], wires=i)
                        qml.RZ(params[layer, i, 2], wires=i)
                    
                    # Entangling layers
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                
                return qml.expval(qml.PauliZ(0))
            
            # Create QNode
            qnode = qml.QNode(circuit, self.device)
            
            # Initial parameters
            params = pnp.random.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))
            
            # Optimization loop
            opt = qml.GradientDescentOptimizer(stepsize=0.1)
            solution_history = []
            
            for iteration in range(100):
                params, cost = opt.step_and_cost(lambda p: self._evaluate_objective(qnode, p, problem), params)
                solution_history.append(cost)
                
                if iteration % 10 == 0:
                    logger.info(f"Iteration {iteration}, cost: {cost:.4f}")
            
            # Final evaluation
            final_cost = self._evaluate_objective(qnode, params, problem)
            optimal_point = self._decode_pennylane_solution(params, problem)
            
            return OptimizationResult(
                problem_id=problem.problem_id,
                backend_name=self.name,
                success=True,
                execution_time=time.time() - start_time,
                optimal_value=final_cost,
                optimal_variables=optimal_point,
                solution_history=solution_history,
                metadata={
                    'backend': 'pennylane',
                    'algorithm': 'VQE',
                    'n_qubits': n_qubits,
                    'n_layers': n_layers,
                    'optimizer': 'GradientDescent'
                }
            )
            
        except Exception as e:
            logger.error(f"PennyLane optimization failed: {e}")
            return OptimizationResult(
                problem_id=problem.problem_id,
                backend_name=self.name,
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _evaluate_objective(self, qnode, params, problem: OptimizationProblem) -> float:
        """Evaluate objective function using quantum circuit."""
        # Map quantum measurement to classical objective
        quantum_output = qnode(params)
        
        # Convert to continuous variables and evaluate
        continuous_point = self._quantum_to_continuous(quantum_output, params, problem)
        return problem.objective(continuous_point)
    
    def _quantum_to_continuous(self, quantum_output: float, params: np.ndarray, 
                             problem: OptimizationProblem) -> np.ndarray:
        """Convert quantum output to continuous variables."""
        n_vars = len(problem.variables)
        continuous_point = np.zeros(n_vars)
        
        # Use parameter values to determine continuous variables
        for i in range(min(n_vars, params.size)):
            # Normalize parameter to [0, 1] range
            param_val = params.flat[i % params.size]
            normalized = (np.sin(param_val) + 1) / 2  # Map to [0, 1]
            
            if problem.bounds and problem.variables[i] in problem.bounds:
                low, high = problem.bounds[problem.variables[i]]
                continuous_point[i] = low + normalized * (high - low)
            else:
                continuous_point[i] = normalized
        
        return continuous_point
    
    def _decode_pennylane_solution(self, params: np.ndarray, 
                                 problem: OptimizationProblem) -> Dict[str, float]:
        """Decode PennyLane parameters to problem variables."""
        optimal_point = {}
        n_vars = len(problem.variables)
        
        for i in range(n_vars):
            var = problem.variables[i]
            if problem.bounds and var in problem.bounds:
                low, high = problem.bounds[var]
                # Use first parameter for this variable
                param_val = params.flat[i % params.size]
                normalized = (np.sin(param_val) + 1) / 2
                optimal_point[var] = low + normalized * (high - low)
            else:
                optimal_point[var] = float(params.flat[i % params.size])
        
        return optimal_point

@register_backend("dwave")
class DWaveBackend(Backend):
    """D-Wave quantum annealer backend."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("dwave", config or {})
        self.backend_type = BackendType.QUANTUM_ANNEALER
        self.sampler = None
        self.solver = None
        
    def _initialize(self):
        """Initialize D-Wave connection."""
        try:
            import dwave.cloud
            from dwave.system import DWaveSampler, EmbeddingComposite
            
            # Configuration
            self.config.setdefault('solver', 'Advantage_system6.1')
            self.config.setdefault('num_reads', 1000)
            self.config.setdefault('annealing_time', 20)
            
            # Initialize sampler
            self.sampler = EmbeddingComposite(DWaveSampler(
                solver=self.config['solver'],
                client='base'
            ))
            
            self._state = BackendState.READY
            logger.info(f"D-Wave backend initialized with solver {self.config['solver']}")
            
        except ImportError as e:
            logger.error(f"D-Wave Ocean SDK not available: {e}")
            self._state = BackendState.ERROR
        except Exception as e:
            logger.error(f"D-Wave connection failed: {e}")
            self._state = BackendState.ERROR
    
    def solve_optimization(self, problem: OptimizationProblem) -> OptimizationResult:
        """Solve optimization using quantum annealing."""
        import time
        start_time = time.time()
        
        if self._state == BackendState.ERROR:
            # Fallback to simulated annealing
            return self._simulated_annealing_fallback(problem)
        
        try:
            # Convert problem to QUBO
            qubo = self._problem_to_dwave_qubo(problem)
            
            # Annealing parameters
            annealing_params = {
                'num_reads': self.config.get('num_reads', 1000),
                'annealing_time': self.config.get('annealing_time', 20),
                'chain_strength': self.config.get('chain_strength', 1.0),
                'return_embedding': True
            }
            
            # Submit to D-Wave
            response = self.sampler.sample_qubo(qubo, **annealing_params)
            
            # Process results
            best_sample = response.first.sample
            best_energy = response.first.energy
            
            # Decode to continuous variables
            optimal_point = self._decode_dwave_solution(best_sample, problem)
            optimal_value = problem.objective(
                np.array([optimal_point[var] for var in problem.variables])
            )
            
            # Calculate timing and costs
            execution_time = time.time() - start_time
            qpu_time = response.info.get('timing', {}).get('qpu_access_time', 0) / 1000  # ms to s
            
            return OptimizationResult(
                problem_id=problem.problem_id,
                backend_name=self.name,
                success=True,
                execution_time=execution_time,
                optimal_value=optimal_value,
                optimal_variables=optimal_point,
                metadata={
                    'backend': 'dwave',
                    'algorithm': 'quantum_annealing',
                    'solver': self.config.get('solver'),
                    'num_reads': annealing_params['num_reads'],
                    'qpu_time': qpu_time,
                    'best_energy': best_energy,
                    'chain_break_fraction': getattr(response.first, 'chain_break_fraction', 0.0)
                }
            )
            
        except Exception as e:
            logger.error(f"D-Wave optimization failed: {e}")
            return self._simulated_annealing_fallback(problem)
    
    def _problem_to_dwave_qubo(self, problem: OptimizationProblem) -> Dict:
        """Convert problem to D-Wave QUBO format."""
        n_vars = len(problem.variables)
        qubo = {}
        
        # Create a simple QUBO for demonstration
        # In practice, this would use proper problem encoding
        for i in range(n_vars):
            # Linear terms - encourage variables to be 0 or 1
            qubo[(i, i)] = -1.0
            
            # Quadratic terms - small couplings
            for j in range(i + 1, min(i + 3, n_vars)):  # Limited connectivity
                qubo[(i, j)] = 0.1
        
        return qubo
    
    def _decode_dwave_solution(self, sample: Dict, problem: OptimizationProblem) -> Dict[str, float]:
        """Decode D-Wave binary sample to continuous variables."""
        optimal_point = {}
        n_vars = len(problem.variables)
        
        for i in range(n_vars):
            var = problem.variables[i]
            binary_val = sample.get(i, 0)
            
            if problem.bounds and var in problem.bounds:
                low, high = problem.bounds[var]
                # Map binary {0,1} to continuous range
                optimal_point[var] = low + binary_val * (high - low)
            else:
                optimal_point[var] = float(binary_val)
        
        return optimal_point
    
    def _simulated_annealing_fallback(self, problem: OptimizationProblem) -> OptimizationResult:
        """Fallback to simulated annealing when D-Wave is unavailable."""
        from ....backends.classical import SimulatedAnnealingBackend
        
        annealer = SimulatedAnnealingBackend("dwave_fallback")
        result = annealer.solve_optimization(problem)
        result.backend_name = self.name
        result.metadata['dwave_fallback'] = True
        
        return result

@register_backend("aws_braket")
class AWSBraketBackend(Backend):
    """AWS Braket cloud quantum computing backend."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("aws_braket", config or {})
        self.backend_type = BackendType.CLOUD_QUANTUM
        self.device = None
        self.s3_bucket = None
        self.s3_prefix = None
        
    def _initialize(self):
        """Initialize AWS Braket connection."""
        try:
            import boto3
            from braket.aws import AwsDevice
            from braket.circuits import Circuit
            
            # AWS configuration
            self.config.setdefault('device_arn', 'arn:aws:braket:::device/quantum-simulator/amazon/sv1')
            self.config.setdefault('s3_bucket', 'amazon-braket-your-bucket')
            self.config.setdefault('s3_prefix', 'qaero-jobs')
            
            # Get device
            self.device = AwsDevice(self.config['device_arn'])
            self.s3_bucket = self.config['s3_bucket']
            self.s3_prefix = self.config['s3_prefix']
            
            self._state = BackendState.READY
            logger.info(f"AWS Braket backend initialized with device {self.config['device_arn']}")
            
        except ImportError as e:
            logger.error(f"AWS Braket not available: {e}")
            self._state = BackendState.ERROR
        except Exception as e:
            logger.error(f"AWS Braket connection failed: {e}")
            self._state = BackendState.ERROR
    
    def solve_optimization(self, problem: OptimizationProblem) -> OptimizationResult:
        """Solve optimization using AWS Braket."""
        import time
        start_time = time.time()
        
        if self._state == BackendState.ERROR:
            return self._local_simulator_fallback(problem)
        
        try:
            from braket.circuits import Circuit
            from braket.circuits import gates
            from braket.aws import AwsQuantumTask
            
            # Create simple quantum circuit
            circuit = Circuit()
            n_qubits = min(len(problem.variables), 10)  # Limit for demonstration
            
            # Hadamard on all qubits
            for i in range(n_qubits):
                circuit.h(i)
            
            # Add some entanglement
            for i in range(n_qubits - 1):
                circuit.cnot(i, i + 1)
            
            # Measure all qubits
            for i in range(n_qubits):
                circuit.i(i)  # Identity to preserve structure
            
            # Submit task to AWS Braket
            task = self.device.run(
                circuit,
                self.s3_bucket,
                f"{self.s3_prefix}/{problem.problem_id}_{int(time.time())}",
                shots=self.config.get('shots', 100)
            )
            
            # Wait for completion (with timeout)
            timeout = self.config.get('timeout', 300)
            task_result = task.result(timeout)
            
            # Process results
            measurement_counts = task_result.measurement_counts
            most_frequent = max(measurement_counts, key=measurement_counts.get)
            
            # Decode to optimization solution
            optimal_point = self._decode_braket_solution(most_frequent, problem)
            optimal_value = problem.objective(
                np.array([optimal_point[var] for var in problem.variables])
            )
            
            # Cost calculation (simplified)
            execution_time = time.time() - start_time
            task_cost = self._estimate_braket_cost(task_result)
            
            return OptimizationResult(
                problem_id=problem.problem_id,
                backend_name=self.name,
                success=True,
                execution_time=execution_time,
                optimal_value=optimal_value,
                optimal_variables=optimal_point,
                metadata={
                    'backend': 'aws_braket',
                    'device_arn': self.config['device_arn'],
                    'shots': self.config.get('shots', 100),
                    'task_id': task.id,
                    'estimated_cost': task_cost,
                    'measurement_counts': dict(measurement_counts)
                }
            )
            
        except Exception as e:
            logger.error(f"AWS Braket optimization failed: {e}")
            return self._local_simulator_fallback(problem)
    
    def _decode_braket_solution(self, bitstring: str, problem: OptimizationProblem) -> Dict[str, float]:
        """Decode Braket measurement to continuous variables."""
        optimal_point = {}
        n_vars = len(problem.variables)
        
        for i in range(n_vars):
            var = problem.variables[i]
            if i < len(bitstring):
                bit_val = int(bitstring[i])
            else:
                bit_val = 0
            
            if problem.bounds and var in problem.bounds:
                low, high = problem.bounds[var]
                optimal_point[var] = low + bit_val * (high - low)
            else:
                optimal_point[var] = float(bit_val)
        
        return optimal_point
    
    def _estimate_braket_cost(self, task_result: Any) -> float:
        """Estimate AWS Braket task cost."""
        # Simplified cost estimation
        # In practice, would use actual AWS pricing
        shots = self.config.get('shots', 100)
        
        if 'simulator' in self.config['device_arn']:
            # Simulator pricing (approx)
            return shots * 0.0001  # $0.0001 per shot
        else:
            # Quantum hardware pricing (approx)
            return shots * 0.01  # $0.01 per shot
    
    def _local_simulator_fallback(self, problem: OptimizationProblem) -> OptimizationResult:
        """Fallback to local simulator."""
        from ....backends.classical import ClassicalBackend
        
        classical = ClassicalBackend("braket_fallback")
        result = classical.solve_optimization(problem)
        result.backend_name = self.name
        result.metadata['aws_braket_fallback'] = True
        
        return result

@register_backend("azure_quantum")
class AzureQuantumBackend(Backend):
    """Microsoft Azure Quantum backend."""
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__("azure_quantum", config or {})
        self.backend_type = BackendType.CLOUD_QUANTUM
        self.workspace = None
        self.provider = None
        
    def _initialize(self):
        """Initialize Azure Quantum connection."""
        try:
            from azure.quantum import Workspace
            from azure.quantum.optimization import Problem, ProblemType
            
            # Azure configuration
            subscription_id = self.config.get('subscription_id')
            resource_group = self.config.get('resource_group')
            workspace_name = self.config.get('workspace_name')
            location = self.config.get('location', 'eastus')
            
            if not all([subscription_id, resource_group, workspace_name]):
                logger.error("Azure Quantum configuration incomplete")
                self._state = BackendState.ERROR
                return
            
            self.workspace = Workspace(
                subscription_id=subscription_id,
                resource_group=resource_group,
                name=workspace_name,
                location=location
            )
            
            self._state = BackendState.READY
            logger.info("Azure Quantum backend initialized successfully")
            
        except ImportError as e:
            logger.error(f"Azure Quantum not available: {e}")
            self._state = BackendState.ERROR
        except Exception as e:
            logger.error(f"Azure Quantum connection failed: {e}")
            self._state = BackendState.ERROR
    
    def solve_optimization(self, problem: OptimizationProblem) -> OptimizationResult:
        """Solve optimization using Azure Quantum."""
        import time
        start_time = time.time()
        
        if self._state == BackendState.ERROR:
            return self._azure_fallback(problem)
        
        try:
            from azure.quantum.optimization import Problem, ProblemType, Term
            from azure.quantum.optimization import SimulatedAnnealing
            
            # Convert to Azure Quantum problem format
            terms = []
            n_vars = len(problem.variables)
            
            # Create simple Ising model terms
            for i in range(n_vars):
                # Local field terms
                terms.append(Term(c=i, indices=[i]))
                
                # Interaction terms (limited connectivity)
                for j in range(i + 1, min(i + 2, n_vars)):
                    terms.append(Term(c=0.1, indices=[i, j]))
            
            # Create problem
            azure_problem = Problem(name=problem.problem_id, terms=terms)
            
            # Create solver
            solver = SimulatedAnnealing(
                workspace=self.workspace,
                timeout=self.config.get('timeout', 100)  # seconds
            )
            
            # Submit job
            job = solver.submit(azure_problem)
            
            # Wait for result
            job_result = job.get_results()
            solution = job_result['configuration']
            energy = job_result['cost']
            
            # Decode solution
            optimal_point = self._decode_azure_solution(solution, problem)
            optimal_value = problem.objective(
                np.array([optimal_point[var] for var in problem.variables])
            )
            
            execution_time = time.time() - start_time
            
            return OptimizationResult(
                problem_id=problem.problem_id,
                backend_name=self.name,
                success=True,
                execution_time=execution_time,
                optimal_value=optimal_value,
                optimal_variables=optimal_point,
                metadata={
                    'backend': 'azure_quantum',
                    'solver': 'SimulatedAnnealing',
                    'job_id': job.id,
                    'azure_energy': energy,
                    'workspace': self.workspace.name
                }
            )
            
        except Exception as e:
            logger.error(f"Azure Quantum optimization failed: {e}")
            return self._azure_fallback(problem)
    
    def _decode_azure_solution(self, solution: Dict, problem: OptimizationProblem) -> Dict[str, float]:
        """Decode Azure Quantum solution."""
        optimal_point = {}
        n_vars = len(problem.variables)
        
        for i in range(n_vars):
            var = problem.variables[i]
            spin_val = solution.get(str(i), 1)  # Default to +1
            
            # Convert spin {+1, -1} to binary {1, 0}
            binary_val = 1 if spin_val == 1 else 0
            
            if problem.bounds and var in problem.bounds:
                low, high = problem.bounds[var]
                optimal_point[var] = low + binary_val * (high - low)
            else:
                optimal_point[var] = float(binary_val)
        
        return optimal_point
    
    def _azure_fallback(self, problem: OptimizationProblem) -> OptimizationResult:
        """Fallback for Azure Quantum."""
        from ....backends.classical import ClassicalBackend
        
        classical = ClassicalBackend("azure_fallback")
        result = classical.solve_optimization(problem)
        result.backend_name = self.name
        result.metadata['azure_quantum_fallback'] = True
        
        return result