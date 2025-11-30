"""
Metrics and benchmarking system for quantum aerospace computing.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import numpy as np
import time
from datetime import datetime
import json

from .base import Problem
from .results import OptimizationResult, PDEResult


@dataclass
class PerformanceMetrics:
    """Performance metrics for quantum-classical algorithms."""
    execution_time: float
    memory_usage: float  # MB
    cpu_utilization: float  # percentage
    quantum_time: Optional[float] = None
    classical_time: Optional[float] = None
    circuit_depth: Optional[int] = None
    n_qubits: Optional[int] = None
    shots_used: Optional[int] = None
    convergence_iterations: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'cpu_utilization': self.cpu_utilization,
            'quantum_time': self.quantum_time,
            'classical_time': self.classical_time,
            'circuit_depth': self.circuit_depth,
            'n_qubits': self.n_qubits,
            'shots_used': self.shots_used,
            'convergence_iterations': self.convergence_iterations
        }


@dataclass
class QualityMetrics:
    """Solution quality metrics."""
    optimality_gap: Optional[float] = None
    feasibility_violation: Optional[float] = None
    constraint_satisfaction: Optional[float] = None
    solution_accuracy: Optional[float] = None
    residual_norm: Optional[float] = None
    physical_plausibility: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'optimality_gap': self.optimality_gap,
            'feasibility_violation': self.feasibility_violation,
            'constraint_satisfaction': self.constraint_satisfaction,
            'solution_accuracy': self.solution_accuracy,
            'residual_norm': self.residual_norm,
            'physical_plausibility': self.physical_plausibility
        }


@dataclass
class BenchmarkResult:
    """Benchmark result containing comprehensive metrics."""
    benchmark_id: str
    problem_id: str
    backend_name: str
    algorithm_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = False
    performance: Optional[PerformanceMetrics] = None
    quality: Optional[QualityMetrics] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'benchmark_id': self.benchmark_id,
            'problem_id': self.problem_id,
            'backend_name': self.backend_name,
            'algorithm_name': self.algorithm_name,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'performance': self.performance.to_dict() if self.performance else None,
            'quality': self.quality.to_dict() if self.quality else None,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class MetricsCollector:
    """Collect performance and quality metrics during computation."""
    
    def __init__(self):
        self._start_time = None
        self._performance_data = []
        self._quality_data = []
        self._resource_usage = []
    
    def start_timing(self):
        """Start timing execution."""
        self._start_time = time.time()
        self._performance_data = []
        self._quality_data = []
        self._resource_usage = []
    
    def record_performance(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self._performance_data.append(metrics)
    
    def record_quality(self, metrics: QualityMetrics):
        """Record quality metrics."""
        self._quality_data.append(metrics)
    
    def record_resource_usage(self, memory_mb: float, cpu_percent: float):
        """Record resource usage."""
        self._resource_usage.append({
            'timestamp': time.time(),
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent
        })
    
    def get_average_performance(self) -> Optional[PerformanceMetrics]:
        """Get average performance metrics."""
        if not self._performance_data:
            return None
        
        # Calculate averages
        perf = self._performance_data[0]
        return PerformanceMetrics(
            execution_time=np.mean([p.execution_time for p in self._performance_data]),
            memory_usage=np.mean([p.memory_usage for p in self._performance_data]),
            cpu_utilization=np.mean([p.cpu_utilization for p in self._performance_data]),
            quantum_time=np.mean([p.quantum_time for p in self._performance_data if p.quantum_time]),
            classical_time=np.mean([p.classical_time for p in self._performance_data if p.classical_time]),
            circuit_depth=perf.circuit_depth,
            n_qubits=perf.n_qubits,
            shots_used=perf.shots_used,
            convergence_iterations=perf.convergence_iterations
        )
    
    def get_final_quality(self) -> Optional[QualityMetrics]:
        """Get final quality metrics."""
        if not self._quality_data:
            return None
        return self._quality_data[-1]


class Benchmark(ABC):
    """Abstract benchmark class for quantum aerospace computing."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.metrics_collector = MetricsCollector()
    
    @abstractmethod
    def setup(self) -> Problem:
        """Setup benchmark problem."""
        pass
    
    @abstractmethod
    def run(self, solver: Any, **kwargs) -> BenchmarkResult:
        """Run benchmark with given solver."""
        pass
    
    @abstractmethod
    def evaluate(self, result: Any) -> QualityMetrics:
        """Evaluate solution quality."""
        pass


class AirfoilOptimizationBenchmark(Benchmark):
    """Benchmark for airfoil optimization problems."""
    
    def __init__(self):
        super().__init__(
            name="airfoil_optimization",
            description="Benchmark for airfoil shape optimization using quantum-classical methods"
        )
    
    def setup(self) -> Problem:
        """Setup airfoil optimization problem."""
        from ..problems.aerodynamics import AirfoilOptimizationProblem
        
        return AirfoilOptimizationProblem(
            parameterization="naca",
            design_conditions={
                'mach': 0.3,
                'alpha': 2.0,
                'reynolds': 1e6
            }
        )
    
    def run(self, solver: Any, **kwargs) -> BenchmarkResult:
        """Run airfoil optimization benchmark."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        self.metrics_collector.start_timing()
        
        # Record initial resource usage
        initial_memory = process.memory_info().rss / 1024 / 1024
        initial_cpu = process.cpu_percent()
        
        problem = self.setup()
        
        # Solve problem
        start_time = time.time()
        result = solver.solve(problem, **kwargs)
        execution_time = time.time() - start_time
        
        # Record final resource usage
        final_memory = process.memory_info().rss / 1024 / 1024
        final_cpu = process.cpu_percent()
        
        # Collect performance metrics
        performance = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage=(initial_memory + final_memory) / 2,
            cpu_utilization=(initial_cpu + final_cpu) / 2,
            quantum_time=result.metadata.get('quantum_time'),
            classical_time=result.metadata.get('classical_time'),
            circuit_depth=result.metadata.get('circuit_depth'),
            n_qubits=result.metadata.get('n_qubits'),
            shots_used=result.metadata.get('shots_used'),
            convergence_iterations=result.n_iterations
        )
        
        # Evaluate quality
        quality = self.evaluate(result)
        
        benchmark_result = BenchmarkResult(
            benchmark_id=f"airfoil_opt_{int(time.time())}",
            problem_id=problem.problem_id,
            backend_name=result.backend_name,
            algorithm_name=result.metadata.get('algorithm', 'unknown'),
            success=result.success,
            performance=performance,
            quality=quality,
            metadata={
                'optimal_value': result.optimal_value,
                'n_variables': len(problem.variables),
                'solver_config': getattr(solver, 'config', {}),
                'result_metadata': result.metadata
            }
        )
        
        return benchmark_result
    
    def evaluate(self, result: OptimizationResult) -> QualityMetrics:
        """Evaluate airfoil optimization quality."""
        if not result.success or result.optimal_value is None:
            return QualityMetrics()
        
        # Calculate optimality gap (simplified)
        # In practice, this would compare against known optimal solution
        optimality_gap = abs(result.optimal_value)  # Assuming minimization
        
        # Calculate feasibility (simplified)
        feasibility_violation = 0.0
        if result.constraints_violation is not None:
            feasibility_violation = result.constraints_violation
        
        # Physical plausibility check
        physical_plausibility = 1.0  # Would implement actual checks
        
        return QualityMetrics(
            optimality_gap=optimality_gap,
            feasibility_violation=feasibility_violation,
            physical_plausibility=physical_plausibility
        )


class WingDesignBenchmark(Benchmark):
    """Benchmark for wing design optimization."""
    
    def setup(self) -> Problem:
        """Setup wing design problem."""
        from ..problems.aerodynamics import WingDesignProblem
        
        return WingDesignProblem(
            disciplines=['aerodynamics', 'structures'],
            coupling='weak'
        )
    
    def run(self, solver: Any, **kwargs) -> BenchmarkResult:
        """Run wing design benchmark."""
        # Implementation similar to AirfoilOptimizationBenchmark
        problem = self.setup()
        result = solver.solve(problem, **kwargs)
        quality = self.evaluate(result)
        
        # Simplified performance metrics
        performance = PerformanceMetrics(
            execution_time=result.execution_time,
            memory_usage=100.0,  # Would measure actual usage
            cpu_utilization=50.0  # Would measure actual usage
        )
        
        return BenchmarkResult(
            benchmark_id=f"wing_design_{int(time.time())}",
            problem_id=problem.problem_id,
            backend_name=result.backend_name,
            algorithm_name=result.metadata.get('algorithm', 'unknown'),
            success=result.success,
            performance=performance,
            quality=quality
        )
    
    def evaluate(self, result: OptimizationResult) -> QualityMetrics:
        """Evaluate wing design quality."""
        # Similar to airfoil evaluation but with wing-specific metrics
        return QualityMetrics(
            optimality_gap=abs(result.optimal_value) if result.optimal_value else None,
            physical_plausibility=0.9
        )


class TrajectoryOptimizationBenchmark(Benchmark):
    """Benchmark for trajectory optimization problems."""
    
    def setup(self) -> Problem:
        """Setup trajectory optimization problem."""
        from ..problems.trajectory import AscentTrajectoryProblem
        
        return AscentTrajectoryProblem(stages=2, target_orbit='LEO')
    
    def run(self, solver: Any, **kwargs) -> BenchmarkResult:
        """Run trajectory optimization benchmark."""
        problem = self.setup()
        result = solver.solve(problem, **kwargs)
        quality = self.evaluate(result)
        
        performance = PerformanceMetrics(
            execution_time=result.execution_time,
            memory_usage=150.0,
            cpu_utilization=60.0
        )
        
        return BenchmarkResult(
            benchmark_id=f"trajectory_opt_{int(time.time())}",
            problem_id=problem.problem_id,
            backend_name=result.backend_name,
            algorithm_name=result.metadata.get('algorithm', 'unknown'),
            success=result.success,
            performance=performance,
            quality=quality
        )
    
    def evaluate(self, result: OptimizationResult) -> QualityMetrics:
        """Evaluate trajectory optimization quality."""
        return QualityMetrics(
            optimality_gap=abs(result.optimal_value) if result.optimal_value else None,
            feasibility_violation=0.1  # Simplified
        )


class BenchmarkSuite:
    """Suite of benchmarks for comprehensive evaluation."""
    
    def __init__(self):
        self.benchmarks = {
            'airfoil_optimization': AirfoilOptimizationBenchmark(),
            'wing_design': WingDesignBenchmark(),
            'trajectory_optimization': TrajectoryOptimizationBenchmark()
        }
        self.results = []
    
    def run_all(self, solver: Any, **kwargs) -> List[BenchmarkResult]:
        """Run all benchmarks in the suite."""
        self.results = []
        
        for name, benchmark in self.benchmarks.items():
            print(f"Running benchmark: {name}")
            result = benchmark.run(solver, **kwargs)
            self.results.append(result)
            print(f"Completed {name}: success={result.success}, "
                  f"time={result.performance.execution_time:.2f}s")
        
        return self.results.copy()
    
    def run_single(self, benchmark_name: str, solver: Any, **kwargs) -> BenchmarkResult:
        """Run a single benchmark."""
        if benchmark_name not in self.benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        benchmark = self.benchmarks[benchmark_name]
        result = benchmark.run(solver, **kwargs)
        self.results.append(result)
        return result
    
    def get_comparison_report(self) -> Dict[str, Any]:
        """Generate comparison report across all benchmarks."""
        if not self.results:
            return {}
        
        report = {
            'total_benchmarks': len(self.results),
            'successful_benchmarks': sum(1 for r in self.results if r.success),
            'average_execution_time': np.mean([r.performance.execution_time for r in self.results if r.performance]),
            'benchmark_details': {}
        }
        
        for result in self.results:
            report['benchmark_details'][result.benchmark_id] = {
                'problem': result.problem_id,
                'backend': result.backend_name,
                'algorithm': result.algorithm_name,
                'success': result.success,
                'execution_time': result.performance.execution_time if result.performance else None,
                'optimality_gap': result.quality.optimality_gap if result.quality else None
            }
        
        return report
    
    def save_results(self, filename: str):
        """Save benchmark results to file."""
        with open(filename, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2, default=str)
    
    def load_results(self, filename: str):
        """Load benchmark results from file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.results = []
        for item in data:
            # Reconstruct BenchmarkResult objects
            # This is simplified - would need proper reconstruction
            pass


# Global benchmark suite instance
global_benchmark_suite = BenchmarkSuite()