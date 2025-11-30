"""
User personas for QAero - Aerospace Engineer and Quantum Researcher interfaces.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np

from ..core.base import Problem, OptimizationProblem, PDEProblem
from ..core.results import OptimizationResult, PDEResult


@dataclass
class UserPreferences:
    """User preferences and expertise level."""
    expertise_level: str  # 'beginner', 'intermediate', 'expert'
    preferred_backend: str
    visualization_enabled: bool = True
    auto_fallback: bool = True
    max_execution_time: float = 3600.0  # seconds


class BaseUserPersona(ABC):
    """Base class for user personas."""
    
    def __init__(self, name: str, preferences: UserPreferences):
        self.name = name
        self.preferences = preferences
        self._problem_builder = None
        self._solver = None
    
    @abstractmethod
    def create_problem(self, problem_type: str, **kwargs) -> Problem:
        """Create a problem appropriate for this user persona."""
        pass
    
    @abstractmethod
    def solve_problem(self, problem: Problem, **solver_kwargs) -> Any:
        """Solve a problem using persona-appropriate methods."""
        pass
    
    @abstractmethod
    def visualize_results(self, results: Any, **kwargs):
        """Visualize results in a persona-appropriate way."""
        pass


class AerospaceEngineer(BaseUserPersona):
    """
    Aerospace Engineer persona - uses high-level templates and domain-specific interfaces.
    Focus on results, not implementation details.
    """
    
    def __init__(self, preferences: Optional[UserPreferences] = None):
        if preferences is None:
            preferences = UserPreferences(
                expertise_level='intermediate',
                preferred_backend='auto',
                visualization_enabled=True,
                auto_fallback=True
            )
        super().__init__("Aerospace Engineer", preferences)
        
        # High-level problem templates
        self._problem_templates = {
            'airfoil_optimization': self._create_airfoil_problem,
            'wing_design': self._create_wing_problem,
            'trajectory_optimization': self._create_trajectory_problem,
            'cfd_simulation': self._create_cfd_problem,
            'structural_optimization': self._create_structural_problem
        }
    
    def create_problem(self, problem_type: str, **kwargs) -> Problem:
        """Create a problem using high-level templates."""
        if problem_type not in self._problem_templates:
            raise ValueError(f"Unknown problem type: {problem_type}. "
                           f"Available: {list(self._problem_templates.keys())}")
        
        return self._problem_templates[problem_type](**kwargs)
    
    def solve_problem(self, problem: Problem, **solver_kwargs) -> Any:
        """Solve with automatic backend selection and fallbacks."""
        from ..core.solver import Solver
        
        # Auto-configure solver based on preferences
        solver_config = {
            'backend': self.preferences.preferred_backend,
            'auto_fallback': self.preferences.auto_fallback,
            'timeout': self.preferences.max_execution_time,
            **solver_kwargs
        }
        
        solver = Solver(**solver_config)
        return solver.solve(problem)
    
    def visualize_results(self, results: Any, **kwargs):
        """Create domain-specific visualizations."""
        if isinstance(results, OptimizationResult):
            self._visualize_optimization_results(results, **kwargs)
        elif isinstance(results, PDEResult):
            self._visualize_pde_results(results, **kwargs)
        else:
            print(f"Visualization not available for {type(results)}")
    
    def _create_airfoil_problem(self, **kwargs) -> OptimizationProblem:
        """Create airfoil optimization problem."""
        from ..problems.aerodynamics import AirfoilOptimizationProblem
        
        design_conditions = kwargs.get('design_conditions', {
            'mach': kwargs.get('mach', 0.3),
            'alpha': kwargs.get('alpha', 2.0),
            'reynolds': kwargs.get('reynolds', 1e6)
        })
        
        return AirfoilOptimizationProblem(
            parameterization=kwargs.get('parameterization', 'naca'),
            design_conditions=design_conditions,
            **kwargs
        )
    
    def _create_wing_problem(self, **kwargs) -> OptimizationProblem:
        """Create wing design problem."""
        from ..problems.aerodynamics import WingDesignProblem
        
        return WingDesignProblem(
            disciplines=kwargs.get('disciplines', ['aerodynamics', 'structures']),
            coupling=kwargs.get('coupling', 'weak'),
            **kwargs
        )
    
    def _create_trajectory_problem(self, **kwargs) -> OptimizationProblem:
        """Create trajectory optimization problem."""
        from ..problems.trajectory import AscentTrajectoryProblem
        
        return AscentTrajectoryProblem(
            stages=kwargs.get('stages', 2),
            target_orbit=kwargs.get('target_orbit', 'LEO'),
            **kwargs
        )
    
    def _create_cfd_problem(self, **kwargs) -> PDEProblem:
        """Create CFD simulation problem."""
        from ..problems.aerodynamics import CompressibleFlowProblem
        
        return CompressibleFlowProblem(
            mach_number=kwargs.get('mach_number', 0.8),
            flow_type=kwargs.get('flow_type', 'subsonic'),
            **kwargs
        )
    
    def _create_structural_problem(self, **kwargs) -> OptimizationProblem:
        """Create structural optimization problem."""
        from ..problems.structures import StructuralOptimizationProblem
        
        return StructuralOptimizationProblem(
            objective_type=kwargs.get('objective_type', 'minimize_mass'),
            constraints=kwargs.get('constraints', ['stress', 'displacement']),
            **kwargs
        )
    
    def _visualize_optimization_results(self, results: OptimizationResult, **kwargs):
        """Create engineering-focused optimization visualizations."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(2, 3, figure=fig)
            
            # Convergence plot
            ax1 = fig.add_subplot(gs[0, 0])
            if results.solution_history:
                ax1.plot(results.solution_history, 'b-', linewidth=2, label='Objective')
                ax1.set_ylabel('Objective Value')
                ax1.set_xlabel('Iteration')
                ax1.set_title('Convergence History')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
            
            # Design variables
            ax2 = fig.add_subplot(gs[0, 1])
            if results.optimal_variables:
                variables = list(results.optimal_variables.keys())
                values = list(results.optimal_variables.values())
                ax2.bar(variables, values, alpha=0.7)
                ax2.set_ylabel('Optimal Value')
                ax2.set_title('Optimal Design Variables')
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # Performance metrics
            ax3 = fig.add_subplot(gs[0, 2])
            metrics = {
                'Execution Time': f"{results.execution_time:.2f}s",
                'Success': str(results.success),
                'Optimal Value': f"{results.optimal_value:.6f}" if results.optimal_value else 'N/A'
            }
            ax3.axis('off')
            ax3.text(0.1, 0.9, 'Performance Metrics', fontsize=12, fontweight='bold')
            for i, (key, value) in enumerate(metrics.items()):
                ax3.text(0.1, 0.7 - i*0.15, f"{key}: {value}", fontsize=10)
            
            # Domain-specific visualization
            ax4 = fig.add_subplot(gs[1, :])
            self._create_domain_visualization(ax4, results, **kwargs)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")
    
    def _create_domain_visualization(self, ax, results: OptimizationResult, **kwargs):
        """Create domain-specific visualization (airfoil, wing, etc.)."""
        problem_type = getattr(results, 'problem_type', 'unknown')
        
        if 'airfoil' in results.problem_id:
            self._plot_airfoil_design(ax, results, **kwargs)
        elif 'wing' in results.problem_id:
            self._plot_wing_design(ax, results, **kwargs)
        elif 'trajectory' in results.problem_id:
            self._plot_trajectory(ax, results, **kwargs)
        else:
            ax.text(0.5, 0.5, 'Domain visualization\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Design Visualization')
    
    def _plot_airfoil_design(self, ax, results: OptimizationResult, **kwargs):
        """Plot optimized airfoil shape."""
        try:
            # Simplified airfoil generation from parameters
            if results.optimal_variables and 'm' in results.optimal_variables:
                # NACA airfoil visualization
                m = results.optimal_variables['m']
                p = results.optimal_variables.get('p', 0.4)
                t = results.optimal_variables.get('t', 0.12)
                
                x = np.linspace(0, 1, 100)
                yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
                
                if m > 0 and p > 0:
                    yc = np.where(x < p, 
                                 m/p**2 * (2*p*x - x**2),
                                 m/(1-p)**2 * ((1-2*p) + 2*p*x - x**2))
                    theta = np.where(x < p,
                                    np.arctan(2*m/p**2 * (p - x)),
                                    np.arctan(2*m/(1-p)**2 * (p - x)))
                    
                    xu = x - yt * np.sin(theta)
                    yu = yc + yt * np.cos(theta)
                    xl = x + yt * np.sin(theta)
                    yl = yc - yt * np.cos(theta)
                    
                    ax.plot(xu, yu, 'b-', linewidth=2, label='Upper Surface')
                    ax.plot(xl, yl, 'r-', linewidth=2, label='Lower Surface')
                else:
                    ax.plot(x, yt, 'b-', linewidth=2, label='Upper Surface')
                    ax.plot(x, -yt, 'r-', linewidth=2, label='Lower Surface')
                
                ax.set_xlabel('x/c')
                ax.set_ylabel('y/c')
                ax.set_title('Optimized Airfoil Shape')
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.axis('equal')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Visualization error: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_wing_design(self, ax, results: OptimizationResult, **kwargs):
        """Plot wing planform design."""
        if results.optimal_variables:
            span = results.optimal_variables.get('span', 10.0)
            ar = results.optimal_variables.get('aspect_ratio', 8.0)
            taper = results.optimal_variables.get('taper_ratio', 0.3)
            sweep = results.optimal_variables.get('sweep', 25.0)
            
            # Calculate chord lengths
            root_chord = 2 * span / (ar * (1 + taper))
            tip_chord = taper * root_chord
            
            # Wing planform coordinates
            x_root = [0, root_chord]
            y_root = [0, 0]
            
            x_tip = [span * np.tan(np.radians(sweep)), 
                    span * np.tan(np.radians(sweep)) + tip_chord]
            y_tip = [span, span]
            
            ax.plot(x_root, y_root, 'k-', linewidth=3, label='Root Chord')
            ax.plot(x_tip, y_tip, 'k-', linewidth=3, label='Tip Chord')
            ax.plot([x_root[0], x_tip[0]], [y_root[0], y_tip[0]], 'b--', alpha=0.7)
            ax.plot([x_root[1], x_tip[1]], [y_root[1], y_tip[1]], 'b--', alpha=0.7)
            
            ax.set_xlabel('x (m)')
            ax.set_ylabel('y (m)')
            ax.set_title('Wing Planform Design')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.axis('equal')
    
    def _plot_trajectory(self, ax, results: OptimizationResult, **kwargs):
        """Plot trajectory optimization results."""
        # Simplified trajectory visualization
        time = np.linspace(0, 100, 50)
        altitude = 100 * (1 - np.cos(np.pi * time / 100))
        
        ax.plot(time, altitude, 'g-', linewidth=2, label='Altitude')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Altitude (km)')
        ax.set_title('Ascent Trajectory')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _visualize_pde_results(self, results: PDEResult, **kwargs):
        """Create engineering-focused PDE visualizations."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Solution field
            if results.solution_field is not None:
                if results.solution_field.ndim == 1:
                    ax1.plot(results.solution_field, 'b-', linewidth=2)
                    ax1.set_xlabel('Grid Point')
                    ax1.set_ylabel('Solution')
                elif results.solution_field.ndim == 2:
                    im = ax1.imshow(results.solution_field, cmap='viridis', origin='lower')
                    plt.colorbar(im, ax=ax1, label='Solution Value')
                ax1.set_title('Solution Field')
            
            # Residual and convergence
            ax2.axis('off')
            ax2.text(0.1, 0.9, 'Simulation Metrics', fontsize=12, fontweight='bold')
            metrics = [
                f"Success: {results.success}",
                f"Execution Time: {results.execution_time:.2f}s",
                f"Residual Norm: {results.residual_norm:.2e}" if results.residual_norm else "Residual: N/A",
                f"Convergence Rate: {results.convergence_rate:.3f}" if results.convergence_rate else "Conv Rate: N/A"
            ]
            
            for i, metric in enumerate(metrics):
                ax2.text(0.1, 0.7 - i*0.1, metric, fontsize=10)
            
            if results.field_statistics:
                ax2.text(0.1, 0.3, 'Field Statistics', fontsize=10, fontweight='bold')
                stats = [
                    f"Min: {results.field_statistics['min']:.3f}",
                    f"Max: {results.field_statistics['max']:.3f}",
                    f"Mean: {results.field_statistics['mean']:.3f}"
                ]
                for i, stat in enumerate(stats):
                    ax2.text(0.1, 0.2 - i*0.08, stat, fontsize=9)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for visualization")


class QuantumResearcher(BaseUserPersona):
    """
    Quantum Researcher persona - focuses on algorithm development, ansatz design,
    and low-level quantum computing details.
    """
    
    def __init__(self, preferences: Optional[UserPreferences] = None):
        if preferences is None:
            preferences = UserPreferences(
                expertise_level='expert',
                preferred_backend='quantum_generic',
                visualization_enabled=True,
                auto_fallback=False  # Researchers want to see quantum failures
            )
        super().__init__("Quantum Researcher", preferences)
        
        self._ansatz_library = {}
        self._hardware_characteristics = {}
    
    def create_problem(self, problem_type: str, **kwargs) -> Problem:
        """Create a problem with full control over quantum formulation."""
        if problem_type == "custom_optimization":
            return self._create_custom_optimization(**kwargs)
        elif problem_type == "quantum_pde":
            return self._create_quantum_pde(**kwargs)
        elif problem_type == "vqe_problem":
            return self._create_vqe_problem(**kwargs)
        elif problem_type == "qaoa_problem":
            return self._create_qaoa_problem(**kwargs)
        else:
            # Fall back to aerospace engineer interface
            engineer = AerospaceEngineer()
            return engineer.create_problem(problem_type, **kwargs)
    
    def solve_problem(self, problem: Problem, **solver_kwargs) -> Any:
        """Solve with full control over quantum parameters and algorithms."""
        from ..core.solver import Solver
        
        # Researcher-specific configuration
        solver_config = {
            'backend': self.preferences.preferred_backend,
            'auto_fallback': self.preferences.auto_fallback,
            'verbose': True,  # Researchers want detailed output
            'save_intermediate_results': True,
            **solver_kwargs
        }
        
        solver = Solver(**solver_config)
        
        # Add quantum-specific callbacks
        if 'quantum_callbacks' in solver_kwargs:
            for callback in solver_kwargs['quantum_callbacks']:
                solver.add_callback(callback)
        
        return solver.solve(problem)
    
    def visualize_results(self, results: Any, **kwargs):
        """Create quantum computing focused visualizations."""
        if isinstance(results, OptimizationResult):
            self._visualize_quantum_optimization(results, **kwargs)
        else:
            # Fall back to basic visualization
            super().visualize_results(results, **kwargs)
    
    def register_ansatz(self, name: str, ansatz_func, description: str = ""):
        """Register a custom quantum ansatz."""
        self._ansatz_library[name] = {
            'function': ansatz_func,
            'description': description
        }
    
    def get_ansatz(self, name: str):
        """Retrieve a registered ansatz."""
        return self._ansatz_library.get(name)
    
    def list_ansatzes(self) -> List[str]:
        """List all registered ansatzes."""
        return list(self._ansatz_library.keys())
    
    def set_hardware_characteristics(self, backend_name: str, characteristics: Dict):
        """Set hardware characteristics for a specific backend."""
        self._hardware_characteristics[backend_name] = characteristics
    
    def _create_custom_optimization(self, **kwargs) -> OptimizationProblem:
        """Create a custom optimization problem with quantum formulation."""
        from ..core.base import OptimizationProblem
        
        return OptimizationProblem(
            problem_id=kwargs.get('problem_id', 'custom_quantum_optimization'),
            objective=kwargs['objective'],
            variables=kwargs.get('variables', ['x']),
            bounds=kwargs.get('bounds', {}),
            parameters={
                'formulation': kwargs.get('formulation', 'qubo'),
                'encoding': kwargs.get('encoding', 'binary'),
                'hardware_aware': kwargs.get('hardware_aware', False),
                **kwargs
            }
        )
    
    def _create_quantum_pde(self, **kwargs) -> PDEProblem:
        """Create a PDE problem with quantum solution method."""
        from ..core.base import PDEProblem
        
        return PDEProblem(
            problem_id=kwargs.get('problem_id', 'quantum_pde'),
            equation=kwargs['equation'],
            domain=kwargs['domain'],
            boundary_conditions=kwargs['boundary_conditions'],
            parameters={
                'quantum_method': kwargs.get('quantum_method', 'hhl'),
                'discretization': kwargs.get('discretization', 'quantum_friendly'),
                **kwargs
            }
        )
    
    def _create_vqe_problem(self, **kwargs) -> OptimizationProblem:
        """Create a VQE-specific problem."""
        from ..core.base import OptimizationProblem
        
        def vqe_objective(params):
            # This would be replaced with actual VQE computation
            return np.sum(params**2)
        
        return OptimizationProblem(
            problem_id='vqe_custom',
            objective=vqe_objective,
            variables=[f'theta_{i}' for i in range(kwargs.get('n_parameters', 4))],
            parameters={
                'algorithm': 'vqe',
                'ansatz': kwargs.get('ansatz', 'EfficientSU2'),
                'initial_parameters': kwargs.get('initial_parameters'),
                **kwargs
            }
        )
    
    def _create_qaoa_problem(self, **kwargs) -> OptimizationProblem:
        """Create a QAOA-specific problem."""
        from ..core.base import OptimizationProblem
        
        def qaoa_objective(params):
            # QAOA cost function would go here
            return np.sum(params**2)
        
        return OptimizationProblem(
            problem_id='qaoa_custom',
            objective=qaoa_objective,
            variables=[f'gamma_{i}' for i in range(kwargs.get('p', 1))] + 
                     [f'beta_{i}' for i in range(kwargs.get('p', 1))],
            parameters={
                'algorithm': 'qaoa',
                'p': kwargs.get('p', 1),
                'mixer': kwargs.get('mixer', 'x'),
                **kwargs
            }
        )
    
    def _visualize_quantum_optimization(self, results: OptimizationResult, **kwargs):
        """Create quantum computing focused visualizations."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            # Convergence history with quantum-specific metrics
            if results.solution_history:
                axes[0].plot(results.solution_history, 'b-', linewidth=2, label='Cost')
                axes[0].set_ylabel('Cost Function')
                axes[0].set_xlabel('Iteration')
                axes[0].set_title('Quantum Optimization Convergence')
                axes[0].grid(True, alpha=0.3)
                axes[0].legend()
            
            # Parameter evolution (for VQE/QAOA)
            if 'parameter_history' in results.metadata:
                param_history = results.metadata['parameter_history']
                for i in range(min(4, len(param_history[0]))):
                    params = [p[i] for p in param_history]
                    axes[1].plot(params, label=f'$\\theta_{i}$')
                axes[1].set_ylabel('Parameter Value')
                axes[1].set_xlabel('Iteration')
                axes[1].set_title('Parameter Evolution')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
            
            # Quantum circuit metrics
            axes[2].axis('off')
            quantum_metrics = [
                f"Algorithm: {results.metadata.get('algorithm', 'Unknown')}",
                f"Backend: {results.backend_name}",
                f"Quantum Time: {results.metadata.get('quantum_time', 'N/A')}",
                f"Classical Time: {results.metadata.get('classical_time', 'N/A')}",
                f"Circuit Depth: {results.metadata.get('circuit_depth', 'N/A')}",
                f"Number of Qubits: {results.metadata.get('n_qubits', 'N/A')}"
            ]
            
            axes[2].text(0.1, 0.9, 'Quantum Metrics', fontsize=12, fontweight='bold')
            for i, metric in enumerate(quantum_metrics):
                axes[2].text(0.1, 0.7 - i*0.1, metric, fontsize=10)
            
            # Hardware information
            axes[3].axis('off')
            if 'hardware_info' in results.metadata:
                hw_info = results.metadata['hardware_info']
                axes[3].text(0.1, 0.9, 'Hardware Info', fontsize=12, fontweight='bold')
                for i, (key, value) in enumerate(hw_info.items()):
                    axes[3].text(0.1, 0.7 - i*0.08, f"{key}: {value}", fontsize=9)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for quantum visualization")


# Factory function for creating personas
def create_persona(persona_type: str, **preferences) -> BaseUserPersona:
    """Factory function to create user personas."""
    if persona_type.lower() == "aerospace_engineer":
        return AerospaceEngineer(UserPreferences(**preferences))
    elif persona_type.lower() == "quantum_researcher":
        return QuantumResearcher(UserPreferences(**preferences))
    else:
        raise ValueError(f"Unknown persona type: {persona_type}")