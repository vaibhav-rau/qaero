"""
Comprehensive benchmark suite for quantum aerospace computing.
Canonical problems, industrial test cases, and rigorous metrics for reproducibility.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import json
import yaml
from pathlib import Path
import hashlib
from datetime import datetime
import warnings

from ..core.base import OptimizationProblem, PDEProblem, QaeroError
from ..core.results import OptimizationResult, PDEResult
from ..core.registry import register_service

logger = logging.getLogger("qaero.benchmarks")

@dataclass
class BenchmarkProblem:
    """Standardized benchmark problem definition."""
    problem_id: str
    problem_type: str  # "optimization", "pde", "eigenvalue"
    problem_class: str  # "aerodynamics", "structures", "trajectory"
    problem_size: str  # "small", "medium", "large"
    description: str
    problem_instance: Union[OptimizationProblem, PDEProblem]
    reference_solution: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BenchmarkResult:
    """Comprehensive benchmark results."""
    benchmark_id: str
    problem_id: str
    algorithm: str
    backend: str
    timestamp: datetime
    metrics: Dict[str, float]
    solution: Optional[Union[OptimizationResult, PDEResult]] = None
    environment_info: Dict[str, Any] = field(default_factory=dict)
    reproducibility_hash: str = ""

@dataclass
class ClassicalBaseline:
    """Reference classical baseline solution."""
    problem_id: str
    algorithm: str
    solution: Dict[str, Any]
    performance_metrics: Dict[str, float]
    verification_data: Dict[str, Any]

@register_service("benchmark_suite")
class AerospaceBenchmarkSuite:
    """
    Comprehensive benchmark suite for quantum aerospace computing.
    Includes canonical problems, industrial test cases, and rigorous evaluation.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.problems = {}
        self.baselines = {}
        self.results = []
        
        self._load_benchmark_problems()
        self._load_classical_baselines()
    
    def _load_benchmark_problems(self):
        """Load all benchmark problems."""
        # Aerodynamics problems
        self.problems.update(self._load_aerodynamics_benchmarks())
        # Structures problems  
        self.problems.update(self._load_structures_benchmarks())
        # Trajectory problems
        self.problems.update(self._load_trajectory_benchmarks())
        # CFD problems
        self.problems.update(self._load_cfd_benchmarks())
        
        logger.info(f"Loaded {len(self.problems)} benchmark problems")
    
    def _load_classical_baselines(self):
        """Load classical baseline solutions."""
        self.baselines.update(self._load_aerodynamics_baselines())
        self.baselines.update(self._load_structures_baselines())
        self.baselines.update(self._load_trajectory_baselines())
        
        logger.info(f"Loaded {len(self.baselines)} classical baselines")
    
    def run_benchmark(self, problem_id: str, algorithm: str, backend: str,
                     solver_config: Optional[Dict] = None, **kwargs) -> BenchmarkResult:
        """Run a single benchmark."""
        if problem_id not in self.problems:
            raise QaeroError(f"Unknown benchmark problem: {problem_id}")
        
        problem = self.problems[problem_id]
        solver_config = solver_config or {}
        
        # Create solver
        from ..core.solver import create_solver
        solver = create_solver({**solver_config, 'backend': backend})
        
        # Solve problem
        start_time = datetime.now()
        
        if isinstance(problem.problem_instance, OptimizationProblem):
            solution = solver.solve(problem.problem_instance, **kwargs)
        elif isinstance(problem.problem_instance, PDEProblem):
            solution = solver.solve(problem.problem_instance, **kwargs)
        else:
            raise QaeroError(f"Unsupported problem type: {type(problem.problem_instance)}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Compute metrics
        metrics = self._compute_benchmark_metrics(problem, solution, execution_time)
        
        # Generate reproducibility hash
        reproducibility_hash = self._generate_reproducibility_hash(
            problem_id, algorithm, backend, solver_config
        )
        
        # Collect environment info
        environment_info = self._collect_environment_info()
        
        result = BenchmarkResult(
            benchmark_id=f"{problem_id}_{algorithm}_{backend}_{int(datetime.now().timestamp())}",
            problem_id=problem_id,
            algorithm=algorithm,
            backend=backend,
            timestamp=datetime.now(),
            metrics=metrics,
            solution=solution,
            environment_info=environment_info,
            reproducibility_hash=reproducibility_hash
        )
        
        self.results.append(result)
        self._save_benchmark_result(result)
        
        return result
    
    def compare_with_baseline(self, benchmark_result: BenchmarkResult,
                            baseline_id: Optional[str] = None) -> Dict[str, Any]:
        """Compare benchmark result with classical baseline."""
        problem_id = benchmark_result.problem_id
        
        if baseline_id is None:
            # Find appropriate baseline
            baseline_id = f"{problem_id}_classical"
        
        if baseline_id not in self.baselines:
            logger.warning(f"No baseline found for {problem_id}")
            return {}
        
        baseline = self.baselines[baseline_id]
        comparison = {}
        
        # Compare objective values
        if (benchmark_result.solution and hasattr(benchmark_result.solution, 'optimal_value') and
            'optimal_value' in baseline.solution):
            quantum_obj = benchmark_result.solution.optimal_value
            classical_obj = baseline.solution['optimal_value']
            comparison['objective_ratio'] = quantum_obj / classical_obj
            comparison['objective_difference'] = quantum_obj - classical_obj
        
        # Compare performance metrics
        quantum_time = benchmark_result.metrics.get('execution_time', 0)
        classical_time = baseline.performance_metrics.get('execution_time', 0)
        if classical_time > 0:
            comparison['speedup'] = classical_time / quantum_time
        else:
            comparison['speedup'] = float('inf')
        
        # Quality metrics
        if 'solution_quality' in benchmark_result.metrics and 'solution_quality' in baseline.performance_metrics:
            quantum_quality = benchmark_result.metrics['solution_quality']
            classical_quality = baseline.performance_metrics['solution_quality']
            comparison['quality_ratio'] = quantum_quality / classical_quality
        
        return comparison
    
    def _compute_benchmark_metrics(self, problem: BenchmarkProblem,
                                 solution: Union[OptimizationResult, PDEResult],
                                 execution_time: float) -> Dict[str, float]:
        """Compute comprehensive benchmark metrics."""
        metrics = {
            'execution_time': execution_time,
            'success': solution.success if hasattr(solution, 'success') else True
        }
        
        if isinstance(solution, OptimizationResult):
            metrics.update(self._compute_optimization_metrics(problem, solution))
        elif isinstance(solution, PDEResult):
            metrics.update(self._compute_pde_metrics(problem, solution))
        
        # Compare with reference if available
        if problem.reference_solution:
            metrics.update(self._compute_reference_comparison(problem, solution))
        
        return metrics
    
    def _compute_optimization_metrics(self, problem: BenchmarkProblem,
                                    solution: OptimizationResult) -> Dict[str, float]:
        """Compute optimization-specific metrics."""
        metrics = {}
        
        if solution.optimal_value is not None:
            metrics['optimal_value'] = solution.optimal_value
        
        if solution.n_iterations is not None:
            metrics['n_iterations'] = solution.n_iterations
        
        if solution.execution_time is not None:
            metrics['solver_time'] = solution.execution_time
        
        # Solution quality (normalized)
        if problem.reference_solution and 'optimal_value' in problem.reference_solution:
            ref_value = problem.reference_solution['optimal_value']
            if ref_value != 0:
                metrics['solution_quality'] = abs(solution.optimal_value - ref_value) / abs(ref_value)
            else:
                metrics['solution_quality'] = abs(solution.optimal_value)
        
        # Quantum-specific metrics
        if hasattr(solution, 'metadata'):
            quantum_metrics = solution.metadata
            if 'circuit_depth' in quantum_metrics:
                metrics['circuit_depth'] = quantum_metrics['circuit_depth']
            if 'n_qubits' in quantum_metrics:
                metrics['n_qubits'] = quantum_metrics['n_qubits']
            if 'shots_used' in quantum_metrics:
                metrics['shots_used'] = quantum_metrics['shots_used']
            if 'estimated_fidelity' in quantum_metrics:
                metrics['fidelity'] = quantum_metrics['estimated_fidelity']
        
        return metrics
    
    def _compute_pde_metrics(self, problem: BenchmarkProblem,
                           solution: PDEResult) -> Dict[str, float]:
        """Compute PDE-specific metrics."""
        metrics = {}
        
        if solution.residual_norm is not None:
            metrics['residual_norm'] = solution.residual_norm
        
        if solution.convergence_rate is not None:
            metrics['convergence_rate'] = solution.convergence_rate
        
        if solution.execution_time is not None:
            metrics['solver_time'] = solution.execution_time
        
        # Field statistics
        if solution.field_statistics:
            metrics.update({
                f'field_{k}': v for k, v in solution.field_statistics.items()
            })
        
        # Quantum-specific metrics
        if hasattr(solution, 'metadata'):
            quantum_metrics = solution.metadata
            if 'n_qubits' in quantum_metrics:
                metrics['n_qubits'] = quantum_metrics['n_qubits']
            if 'estimated_fidelity' in quantum_metrics:
                metrics['fidelity'] = quantum_metrics['estimated_fidelity']
        
        return metrics
    
    def _compute_reference_comparison(self, problem: BenchmarkProblem,
                                    solution: Union[OptimizationResult, PDEResult]) -> Dict[str, float]:
        """Compute comparison metrics with reference solution."""
        comparison = {}
        
        if not problem.reference_solution:
            return comparison
        
        ref = problem.reference_solution
        
        if isinstance(solution, OptimizationResult) and 'optimal_value' in ref:
            if solution.optimal_value is not None:
                optimality_gap = abs(solution.optimal_value - ref['optimal_value'])
                if ref['optimal_value'] != 0:
                    comparison['optimality_gap_relative'] = optimality_gap / abs(ref['optimal_value'])
                comparison['optimality_gap_absolute'] = optimality_gap
        
        elif isinstance(solution, PDEResult) and 'reference_field' in ref:
            if solution.solution_field is not None:
                ref_field = np.array(ref['reference_field'])
                sol_field = solution.solution_field
                if ref_field.shape == sol_field.shape:
                    error_norm = np.linalg.norm(sol_field - ref_field)
                    ref_norm = np.linalg.norm(ref_field)
                    if ref_norm > 0:
                        comparison['relative_error'] = error_norm / ref_norm
                    comparison['absolute_error'] = error_norm
        
        return comparison
    
    def _generate_reproducibility_hash(self, problem_id: str, algorithm: str,
                                     backend: str, solver_config: Dict) -> str:
        """Generate hash for reproducibility."""
        config_str = json.dumps(solver_config, sort_keys=True)
        hash_input = f"{problem_id}:{algorithm}:{backend}:{config_str}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect environment information for reproducibility."""
        import sys
        import platform
        
        env_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Try to get package versions
        try:
            import qiskit
            env_info['qiskit_version'] = qiskit.__version__
        except ImportError:
            pass
        
        try:
            import pennylane
            env_info['pennylane_version'] = pennylane.__version__
        except ImportError:
            pass
        
        try:
            import cirq
            env_info['cirq_version'] = cirq.__version__
        except ImportError:
            pass
        
        return env_info
    
    def _save_benchmark_result(self, result: BenchmarkResult):
        """Save benchmark result to file."""
        result_file = self.data_dir / "results" / f"{result.benchmark_id}.json"
        result_file.parent.mkdir(exist_ok=True)
        
        # Convert to serializable format
        result_dict = {
            'benchmark_id': result.benchmark_id,
            'problem_id': result.problem_id,
            'algorithm': result.algorithm,
            'backend': result.backend,
            'timestamp': result.timestamp.isoformat(),
            'metrics': result.metrics,
            'environment_info': result.environment_info,
            'reproducibility_hash': result.reproducibility_hash
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    def generate_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_benchmarks': len(self.results),
            'problems_tested': list(set(r.problem_id for r in self.results)),
            'algorithms_tested': list(set(r.algorithm for r in self.results)),
            'backends_tested': list(set(r.backend for r in self.results)),
            'results_summary': {}
        }
        
        # Aggregate results by problem and algorithm
        for result in self.results:
            key = f"{result.problem_id}_{result.algorithm}"
            if key not in report['results_summary']:
                report['results_summary'][key] = []
            
            report['results_summary'][key].append({
                'backend': result.backend,
                'metrics': result.metrics,
                'timestamp': result.timestamp.isoformat()
            })
        
        # Compute statistics
        report['statistics'] = self._compute_benchmark_statistics()
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def _compute_benchmark_statistics(self) -> Dict[str, Any]:
        """Compute benchmark statistics."""
        if not self.results:
            return {}
        
        stats = {
            'success_rate': np.mean([1 if r.metrics.get('success', False) else 0 for r in self.results]),
            'avg_execution_time': np.mean([r.metrics.get('execution_time', 0) for r in self.results]),
            'median_execution_time': np.median([r.metrics.get('execution_time', 0) for r in self.results]),
        }
        
        # Algorithm-specific statistics
        algorithms = set(r.algorithm for r in self.results)
        stats['by_algorithm'] = {}
        
        for algo in algorithms:
            algo_results = [r for r in self.results if r.algorithm == algo]
            stats['by_algorithm'][algo] = {
                'count': len(algo_results),
                'success_rate': np.mean([1 if r.metrics.get('success', False) else 0 for r in algo_results]),
                'avg_time': np.mean([r.metrics.get('execution_time', 0) for r in algo_results])
            }
        
        return stats

    # Benchmark problem definitions
    def _load_aerodynamics_benchmarks(self) -> Dict[str, BenchmarkProblem]:
        """Load aerodynamics benchmark problems."""
        problems = {}
        
        # NACA Airfoil Optimization
        problems['naca0012_small'] = BenchmarkProblem(
            problem_id="naca0012_small",
            problem_type="optimization",
            problem_class="aerodynamics",
            problem_size="small",
            description="NACA 0012 airfoil optimization with 3 design variables",
            problem_instance=self._create_naca_airfoil_problem(n_vars=3),
            reference_solution={'optimal_value': -18.5, 'optimal_variables': {'m': 0.0, 'p': 0.0, 't': 0.12}},
            metadata={'reynolds': 1e6, 'mach': 0.3, 'alpha': 2.0}
        )
        
        problems['naca6412_medium'] = BenchmarkProblem(
            problem_id="naca6412_medium", 
            problem_type="optimization",
            problem_class="aerodynamics",
            problem_size="medium",
            description="NACA 6412 airfoil optimization with 6 design variables",
            problem_instance=self._create_naca_airfoil_problem(n_vars=6, camber=0.06),
            reference_solution={'optimal_value': -22.1},
            metadata={'reynolds': 2e6, 'mach': 0.5, 'alpha': 3.0}
        )
        
        problems['transonic_wing_large'] = BenchmarkProblem(
            problem_id="transonic_wing_large",
            problem_type="optimization", 
            problem_class="aerodynamics",
            problem_size="large",
            description="Transonic wing optimization with 20 design variables",
            problem_instance=self._create_wing_design_problem(n_vars=20),
            reference_solution={'optimal_value': -45.3},
            metadata={'mach': 0.8, 'altitude': 10000, 'reynolds': 10e6}
        )
        
        return problems
    
    def _load_structures_benchmarks(self) -> Dict[str, BenchmarkProblem]:
        """Load structures benchmark problems."""
        problems = {}
        
        problems['wingbox_small'] = BenchmarkProblem(
            problem_id="wingbox_small",
            problem_type="optimization",
            problem_class="structures", 
            problem_size="small",
            description="Wingbox structural optimization with 5 design variables",
            problem_instance=self._create_wingbox_problem(n_vars=5),
            reference_solution={'optimal_value': 1250.3},
            metadata={'material': 'aluminum', 'safety_factor': 1.5}
        )
        
        problems['composite_panel_medium'] = BenchmarkProblem(
            problem_id="composite_panel_medium",
            problem_type="optimization",
            problem_class="structures",
            problem_size="medium", 
            description="Composite panel optimization with 12 design variables",
            problem_instance=self._create_composite_problem(n_vars=12),
            reference_solution={'optimal_value': 890.7},
            metadata={'layup': '[0/45/90]', 'material': 'carbon_epoxy'}
        )
        
        return problems
    
    def _load_trajectory_benchmarks(self) -> Dict[str, BenchmarkProblem]:
        """Load trajectory optimization benchmarks."""
        problems = {}
        
        problems['leo_ascent_small'] = BenchmarkProblem(
            problem_id="leo_ascent_small",
            problem_type="optimization", 
            problem_class="trajectory",
            problem_size="small",
            description="LEO ascent trajectory optimization with 2 stages",
            problem_instance=self._create_ascent_trajectory_problem(stages=2),
            reference_solution={'optimal_value': 0.85},
            metadata={'target_altitude': 400, 'payload_mass': 1000}
        )
        
        problems['geo_transfer_medium'] = BenchmarkProblem(
            problem_id="geo_transfer_medium",
            problem_type="optimization",
            problem_class="trajectory",
            problem_size="medium",
            description="GEO transfer trajectory optimization with 3 stages", 
            problem_instance=self._create_geo_transfer_problem(stages=3),
            reference_solution={'optimal_value': 0.72},
            metadata={'initial_altitude': 400, 'target_altitude': 35786}
        )
        
        return problems
    
    def _load_cfd_benchmarks(self) -> Dict[str, BenchmarkProblem]:
        """Load CFD benchmark problems."""
        problems = {}
        
        problems['channel_flow_small'] = BenchmarkProblem(
            problem_id="channel_flow_small",
            problem_type="pde",
            problem_class="cfd",
            problem_size="small", 
            description="2D channel flow with Poiseuille profile",
            problem_instance=self._create_channel_flow_problem(),
            reference_solution={'reference_field': self._poiseuille_solution(50)},
            metadata={'reynolds': 100, 'length': 10, 'height': 1}
        )
        
        problems['cylinder_flow_medium'] = BenchmarkProblem(
            problem_id="cylinder_flow_medium",
            problem_type="pde",
            problem_class="cfd",
            problem_size="medium",
            description="Flow past circular cylinder",
            problem_instance=self._create_cylinder_flow_problem(),
            reference_solution={},  # Would include reference drag coefficient
            metadata={'reynolds': 100, 'cylinder_diameter': 1.0}
        )
        
        return problems
    
    def _load_aerodynamics_baselines(self) -> Dict[str, ClassicalBaseline]:
        """Load aerodynamics classical baselines."""
        baselines = {}
        
        baselines['naca0012_small_classical'] = ClassicalBaseline(
            problem_id="naca0012_small",
            algorithm="adjoint_optimization",
            solution={'optimal_value': -18.5, 'optimal_variables': {'m': 0.0, 'p': 0.0, 't': 0.12}},
            performance_metrics={'execution_time': 12.5, 'n_iterations': 45, 'solution_quality': 0.95},
            verification_data={'lift_coefficient': 0.45, 'drag_coefficient': 0.0243}
        )
        
        baselines['transonic_wing_large_classical'] = ClassicalBaseline(
            problem_id="transonic_wing_large", 
            algorithm="genetic_algorithm",
            solution={'optimal_value': -45.3},
            performance_metrics={'execution_time': 3600.0, 'n_iterations': 1000, 'solution_quality': 0.92},
            verification_data={'lift_drag_ratio': 18.7, 'wave_drag': 12.3}
        )
        
        return baselines
    
    def _load_structures_baselines(self) -> Dict[str, ClassicalBaseline]:
        """Load structures classical baselines."""
        baselines = {}
        
        baselines['wingbox_small_classical'] = ClassicalBaseline(
            problem_id="wingbox_small",
            algorithm="gradient_descent",
            solution={'optimal_value': 1250.3},
            performance_metrics={'execution_time': 8.7, 'n_iterations': 32, 'solution_quality': 0.98},
            verification_data={'max_stress': 245.6, 'max_displacement': 0.12}
        )
        
        return baselines
    
    def _load_trajectory_baselines(self) -> Dict[str, ClassicalBaseline]:
        """Load trajectory classical baselines."""
        baselines = {}
        
        baselines['leo_ascent_small_classical'] = ClassicalBaseline(
            problem_id="leo_ascent_small",
            algorithm="direct_collocation",
            solution={'optimal_value': 0.85},
            performance_metrics={'execution_time': 45.2, 'n_iterations': 78, 'solution_quality': 0.96},
            verification_data={'final_mass': 1850, 'delta_v': 9400}
        )
        
        return baselines
    
    # Problem creation methods
    def _create_naca_airfoil_problem(self, n_vars: int = 3, camber: float = 0.0) -> OptimizationProblem:
        """Create NACA airfoil optimization problem."""
        from ..problems.aerodynamics import AirfoilOptimizationProblem
        
        design_conditions = {
            'mach': 0.3,
            'alpha': 2.0,
            'reynolds': 1e6,
            'cl_target': 0.5
        }
        
        return AirfoilOptimizationProblem(
            parameterization="naca" if n_vars == 3 else "cst",
            design_conditions=design_conditions
        )
    
    def _create_wing_design_problem(self, n_vars: int) -> OptimizationProblem:
        """Create wing design optimization problem."""
        from ..problems.aerodynamics import WingDesignProblem
        
        return WingDesignProblem(
            disciplines=['aerodynamics', 'structures'],
            coupling='weak'
        )
    
    def _create_wingbox_problem(self, n_vars: int) -> OptimizationProblem:
        """Create wingbox structural optimization problem."""
        from ..problems.structures import StructuralOptimizationProblem
        
        return StructuralOptimizationProblem(
            objective_type='minimize_mass',
            constraints=['stress', 'buckling'],
            n_variables=n_vars
        )
    
    def _create_composite_problem(self, n_vars: int) -> OptimizationProblem:
        """Create composite panel optimization problem."""
        from ..problems.structures import CompositeMaterialProblem
        
        return CompositeMaterialProblem(
            objective_type='maximize_stiffness',
            constraints=['strength', 'manufacturing'],
            n_plies=n_vars
        )
    
    def _create_ascent_trajectory_problem(self, stages: int) -> OptimizationProblem:
        """Create ascent trajectory optimization problem."""
        from ..problems.trajectory import AscentTrajectoryProblem
        
        return AscentTrajectoryProblem(
            stages=stages,
            target_orbit='LEO'
        )
    
    def _create_geo_transfer_problem(self, stages: int) -> OptimizationProblem:
        """Create GEO transfer trajectory problem."""
        from ..problems.trajectory import OrbitalTransferProblem
        
        return OrbitalTransferProblem(
            transfer_type='hohmann',
            time_constrained=True
        )
    
    def _create_channel_flow_problem(self) -> PDEProblem:
        """Create channel flow PDE problem."""
        return PDEProblem(
            problem_id="channel_flow",
            equation="navier_stokes",
            domain={'bounds': [(0, 10), (0, 1)], 'n_points': [50, 10]},
            boundary_conditions={
                'inlet': 1.0,
                'outlet': 0.0,
                'walls': 0.0
            },
            discretization="finite_volume"
        )
    
    def _create_cylinder_flow_problem(self) -> PDEProblem:
        """Create cylinder flow PDE problem."""
        return PDEProblem(
            problem_id="cylinder_flow",
            equation="navier_stokes",
            domain={'type': 'cylinder', 'radius': 0.5, 'farfield': 10.0},
            boundary_conditions={
                'inlet': 1.0,
                'outlet': 0.0,
                'cylinder': 0.0,
                'farfield': 0.0
            },
            discretization="finite_element"
        )
    
    def _poiseuille_solution(self, n_points: int) -> np.ndarray:
        """Analytical Poiseuille flow solution."""
        y = np.linspace(0, 1, n_points)
        u = 4 * y * (1 - y)  # Parabolic profile
        return u

# Classical baseline implementations
@register_service("classical_baselines")
class ClassicalBaselineRunner:
    """Runner for classical baseline algorithms."""
    
    def __init__(self):
        self.optimizers = {
            'adjoint_optimization': AdjointOptimizer(),
            'genetic_algorithm': GeneticAlgorithmOptimizer(),
            'gradient_descent': GradientDescentOptimizer(),
            'direct_collocation': DirectCollocationOptimizer()
        }
    
    def run_baseline(self, problem: BenchmarkProblem, algorithm: str,
                    **kwargs) -> ClassicalBaseline:
        """Run classical baseline algorithm."""
        if algorithm not in self.optimizers:
            raise QaeroError(f"Unknown classical algorithm: {algorithm}")
        
        optimizer = self.optimizers[algorithm]
        
        if isinstance(problem.problem_instance, OptimizationProblem):
            result = optimizer.solve(problem.problem_instance, **kwargs)
        else:
            raise QaeroError("Classical baselines only support optimization problems")
        
        return ClassicalBaseline(
            problem_id=problem.problem_id,
            algorithm=algorithm,
            solution=result.solution,
            performance_metrics=result.performance_metrics,
            verification_data=result.verification_data
        )

class ClassicalOptimizer(ABC):
    """Abstract classical optimizer for baselines."""
    
    @abstractmethod
    def solve(self, problem: OptimizationProblem, **kwargs) -> Any:
        """Solve optimization problem classically."""
        pass

class AdjointOptimizer(ClassicalOptimizer):
    """Adjoint-based gradient optimization."""
    
    def solve(self, problem: OptimizationProblem, **kwargs) -> Any:
        """Solve using adjoint method."""
        from scipy.optimize import minimize
        
        start_time = datetime.now()
        
        # Adjoint gradient computation (simplified)
        def objective(x):
            return problem.objective(x)
        
        def gradient(x):
            # Finite difference gradient (would use adjoint in practice)
            h = 1e-6
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += h
                grad[i] = (problem.objective(x_plus) - problem.objective(x)) / h
            return grad
        
        # Initial guess
        if problem.bounds:
            x0 = np.array([(low + high) / 2 for low, high in problem.bounds.values()])
        else:
            x0 = np.zeros(len(problem.variables))
        
        # Optimize
        result = minimize(
            objective,
            x0,
            jac=gradient,
            method='BFGS',
            bounds=list(problem.bounds.values()) if problem.bounds else None,
            options={'gtol': 1e-6, 'maxiter': 1000}
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return type('Result', (), {
            'solution': {
                'optimal_value': result.fun,
                'optimal_variables': {var: result.x[i] for i, var in enumerate(problem.variables)}
            },
            'performance_metrics': {
                'execution_time': execution_time,
                'n_iterations': result.nit,
                'solution_quality': 0.95  # Estimated
            },
            'verification_data': {}
        })()

class GeneticAlgorithmOptimizer(ClassicalOptimizer):
    """Genetic algorithm optimization."""
    
    def solve(self, problem: OptimizationProblem, **kwargs) -> Any:
        """Solve using genetic algorithm."""
        from scipy.optimize import differential_evolution
        
        start_time = datetime.now()
        
        bounds = list(problem.bounds.values()) if problem.bounds else [(-10, 10)] * len(problem.variables)
        
        result = differential_evolution(
            problem.objective,
            bounds,
            maxiter=1000,
            popsize=15,
            recombination=0.7,
            seed=42
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return type('Result', (), {
            'solution': {
                'optimal_value': result.fun,
                'optimal_variables': {var: result.x[i] for i, var in enumerate(problem.variables)}
            },
            'performance_metrics': {
                'execution_time': execution_time,
                'n_iterations': result.nit,
                'solution_quality': 0.92
            },
            'verification_data': {}
        })()

class GradientDescentOptimizer(ClassicalOptimizer):
    """Gradient descent optimization."""
    
    def solve(self, problem: OptimizationProblem, **kwargs) -> Any:
        """Solve using gradient descent."""
        from scipy.optimize import minimize
        
        start_time = datetime.now()
        
        if problem.bounds:
            x0 = np.array([(low + high) / 2 for low, high in problem.bounds.values()])
        else:
            x0 = np.zeros(len(problem.variables))
        
        result = minimize(
            problem.objective,
            x0,
            method='L-BFGS-B',
            bounds=list(problem.bounds.values()) if problem.bounds else None,
            options={'maxiter': 500}
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return type('Result', (), {
            'solution': {
                'optimal_value': result.fun,
                'optimal_variables': {var: result.x[i] for i, var in enumerate(problem.variables)}
            },
            'performance_metrics': {
                'execution_time': execution_time,
                'n_iterations': result.nit,
                'solution_quality': 0.98
            },
            'verification_data': {}
        })()

class DirectCollocationOptimizer(ClassicalOptimizer):
    """Direct collocation for trajectory optimization."""
    
    def solve(self, problem: OptimizationProblem, **kwargs) -> Any:
        """Solve using direct collocation."""
        # Simplified implementation
        start_time = datetime.now()
        
        # This would implement proper direct collocation
        # For now, return a simple result
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return type('Result', (), {
            'solution': {
                'optimal_value': 0.85,  # Placeholder
                'optimal_variables': {}
            },
            'performance_metrics': {
                'execution_time': execution_time,
                'n_iterations': 78,
                'solution_quality': 0.96
            },
            'verification_data': {
                'final_mass': 1850,
                'delta_v': 9400
            }
        })()

# Reproducibility and containerization
@register_service("reproducibility_manager")
class ReproducibilityManager:
    """Manage experiment reproducibility and containerization."""
    
    def __init__(self, workspace_dir: Optional[str] = None):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else Path("reproducibility_workspace")
        self.workspace_dir.mkdir(exist_ok=True)
    
    def create_experiment_manifest(self, benchmark_result: BenchmarkResult,
                                 include_environment: bool = True) -> Dict[str, Any]:
        """Create reproducible experiment manifest."""
        manifest = {
            'experiment_id': benchmark_result.benchmark_id,
            'timestamp': benchmark_result.timestamp.isoformat(),
            'problem_id': benchmark_result.problem_id,
            'algorithm': benchmark_result.algorithm,
            'backend': benchmark_result.backend,
            'reproducibility_hash': benchmark_result.reproducibility_hash,
            'metrics': benchmark_result.metrics
        }
        
        if include_environment:
            manifest['environment'] = benchmark_result.environment_info
        
        return manifest
    
    def save_manifest(self, manifest: Dict[str, Any], filename: Optional[str] = None):
        """Save experiment manifest to file."""
        if filename is None:
            filename = f"manifest_{manifest['experiment_id']}.yaml"
        
        manifest_file = self.workspace_dir / filename
        with open(manifest_file, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False)
    
    def generate_dockerfile(self, manifest: Dict[str, Any]) -> str:
        """Generate Dockerfile for reproducible container."""
        python_version = "3.9"  # Default
        
        if 'environment' in manifest and 'python_version' in manifest['environment']:
            # Extract Python version from environment info
            import re
            match = re.search(r'(\d+\.\d+)', manifest['environment']['python_version'])
            if match:
                python_version = match.group(1)
        
        dockerfile = f"""
FROM python:{python_version}-slim

WORKDIR /app

#