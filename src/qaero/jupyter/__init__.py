"""
Jupyter integration for QAero with rich widgets and visualizations.
"""
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from ..core.base import Problem, OptimizationProblem, PDEProblem
from ..core.solver import create_solver
from ..core.results import OptimizationResult, PDEResult
from ..personas import create_persona


class QAeroDashboard:
    """Interactive dashboard for QAero in Jupyter notebooks."""
    
    def __init__(self):
        self.problem = None
        self.solver = None
        self.results = None
        self.persona = None
        
        self._create_widgets()
        self._setup_event_handlers()
    
    def _create_widgets(self):
        """Create interactive widgets."""
        # Problem selection
        self.problem_type = widgets.Dropdown(
            options=['airfoil_optimization', 'wing_design', 'trajectory_optimization', 'cfd_simulation'],
            value='airfoil_optimization',
            description='Problem:',
            style={'description_width': 'initial'}
        )
        
        # Backend selection
        self.backend_selection = widgets.Dropdown(
            options=['auto', 'classical_scipy', 'simulated_annealing', 'quantum_generic'],
            value='auto',
            description='Backend:',
            style={'description_width': 'initial'}
        )
        
        # Algorithm selection
        self.algorithm_selection = widgets.Dropdown(
            options=['auto', 'qaoa', 'vqe', 'annealing', 'hybrid'],
            value='auto',
            description='Algorithm:',
            style={'description_width': 'initial'}
        )
        
        # Persona selection
        self.persona_selection = widgets.Dropdown(
            options=['aerospace_engineer', 'quantum_researcher'],
            value='aerospace_engineer',
            description='User Persona:',
            style={'description_width': 'initial'}
        )
        
        # Control buttons
        self.create_button = widgets.Button(
            description='Create Problem',
            button_style='primary',
            tooltip='Create the selected problem'
        )
        
        self.solve_button = widgets.Button(
            description='Solve Problem',
            button_style='success',
            tooltip='Solve the current problem'
        )
        
        self.visualize_button = widgets.Button(
            description='Visualize Results',
            button_style='info',
            tooltip='Visualize current results'
        )
        
        # Progress and status
        self.progress = widgets.FloatProgress(
            value=0.0,
            min=0.0,
            max=1.0,
            description='Progress:',
            bar_style='info',
            style={'bar_color': '#1f77b4'}
        )
        
        self.status_text = widgets.HTML(
            value='<i>Ready to create problem...</i>'
        )
        
        # Output area
        self.output_area = widgets.Output()
        
        # Layout
        self.control_panel = widgets.VBox([
            widgets.HTML('<h3>QAero Control Panel</h3>'),
            self.problem_type,
            self.backend_selection,
            self.algorithm_selection,
            self.persona_selection,
            widgets.HBox([self.create_button, self.solve_button, self.visualize_button]),
            self.progress,
            self.status_text
        ])
        
        self.dashboard = widgets.HBox([
            self.control_panel,
            self.output_area
        ])
    
    def _setup_event_handlers(self):
        """Setup widget event handlers."""
        self.create_button.on_click(self._on_create_click)
        self.solve_button.on_click(self._on_solve_click)
        self.visualize_button.on_click(self._on_visualize_click)
    
    def _on_create_click(self, button):
        """Handle create problem button click."""
        with self.output_area:
            clear_output()
            
            try:
                self._update_status('Creating problem...')
                self.progress.value = 0.2
                
                # Create persona and problem
                self.persona = create_persona(self.persona_selection.value)
                self.problem = self.persona.create_problem(self.problem_type.value)
                
                self.progress.value = 1.0
                self._update_status(f'Created {self.problem_type.value} problem successfully!')
                
                # Display problem info
                print(f"Problem ID: {self.problem.problem_id}")
                print(f"Variables: {self.problem.variables}")
                if hasattr(self.problem, 'bounds') and self.problem.bounds:
                    print("Bounds:")
                    for var, bounds in self.problem.bounds.items():
                        print(f"  {var}: {bounds}")
                
            except Exception as e:
                self._update_status(f'Error creating problem: {e}', is_error=True)
                self.progress.value = 0.0
    
    def _on_solve_click(self, button):
        """Handle solve problem button click."""
        with self.output_area:
            clear_output()
            
            if self.problem is None:
                self._update_status('Please create a problem first!', is_error=True)
                return
            
            try:
                self._update_status('Solving problem...')
                self.progress.value = 0.3
                
                # Create solver
                solver_config = {
                    'backend': self.backend_selection.value,
                    'algorithm': self.algorithm_selection.value
                }
                self.solver = create_solver(solver_config)
                
                self.progress.value = 0.6
                
                # Solve problem
                self.results = self.solver.solve(self.problem)
                
                self.progress.value = 1.0
                self._update_status('Problem solved successfully!')
                
                # Display results
                print(f"Backend: {self.results.backend_name}")
                print(f"Success: {self.results.success}")
                print(f"Execution Time: {self.results.execution_time:.2f}s")
                
                if self.results.success:
                    if isinstance(self.results, OptimizationResult):
                        print(f"Optimal Value: {self.results.optimal_value}")
                        if self.results.optimal_variables:
                            print("Optimal Variables:")
                            for var, value in self.results.optimal_variables.items():
                                print(f"  {var}: {value:.6f}")
                    
                    # Show algorithm-specific info
                    if 'algorithm' in self.results.metadata:
                        print(f"Algorithm: {self.results.metadata['algorithm']}")
                
            except Exception as e:
                self._update_status(f'Error solving problem: {e}', is_error=True)
                self.progress.value = 0.0
    
    def _on_visualize_click(self, button):
        """Handle visualize results button click."""
        with self.output_area:
            clear_output()
            
            if self.results is None:
                self._update_status('Please solve a problem first!', is_error=True)
                return
            
            try:
                self._update_status('Creating visualizations...')
                
                if self.persona:
                    self.persona.visualize_results(self.results)
                else:
                    # Fallback visualization
                    self._create_basic_visualization(self.results)
                
                self._update_status('Visualization completed!')
                
            except Exception as e:
                self._update_status(f'Error creating visualization: {e}', is_error=True)
    
    def _update_status(self, message: str, is_error: bool = False):
        """Update status text."""
        color = 'red' if is_error else 'green'
        self.status_text.value = f'<span style="color: {color}">{message}</span>'
    
    def _create_basic_visualization(self, results):
        """Create basic visualization fallback."""
        if isinstance(results, OptimizationResult):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Convergence plot
            if results.solution_history:
                ax1.plot(results.solution_history, 'b-', linewidth=2)
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Objective Value')
                ax1.set_title('Convergence History')
                ax1.grid(True, alpha=0.3)
            
            # Variables plot
            if results.optimal_variables:
                variables = list(results.optimal_variables.keys())
                values = list(results.optimal_variables.values())
                ax2.bar(variables, values, alpha=0.7)
                ax2.set_ylabel('Optimal Value')
                ax2.set_title('Optimal Variables')
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.show()
        
        elif isinstance(results, PDEResult):
            if results.solution_field is not None:
                plt.figure(figsize=(10, 6))
                
                if results.solution_field.ndim == 1:
                    plt.plot(results.solution_field, 'b-', linewidth=2)
                    plt.xlabel('Grid Point')
                    plt.ylabel('Solution')
                elif results.solution_field.ndim == 2:
                    plt.imshow(results.solution_field, cmap='viridis', origin='lower')
                    plt.colorbar(label='Solution Value')
                
                plt.title(f'PDE Solution - {results.problem_id}')
                plt.grid(True, alpha=0.3)
                plt.show()
    
    def show(self):
        """Display the dashboard."""
        display(self.dashboard)


class CircuitVisualizer:
    """Quantum circuit visualizer for Jupyter notebooks."""
    
    def __init__(self):
        self.circuit = None
        
    def display_circuit(self, circuit):
        """Display quantum circuit."""
        try:
            # Try to use Qiskit's circuit drawer if available
            from qiskit import QuantumCircuit
            if isinstance(circuit, QuantumCircuit):
                return circuit.draw(output='mpl')
        except ImportError:
            pass
        
        # Fallback text representation
        print("Quantum Circuit (text representation):")
        print(f"Number of qubits: {getattr(circuit, 'num_qubits', 'Unknown')}")
        print(f"Circuit depth: {getattr(circuit, 'depth', 'Unknown')}")
        
        # Create a simple ASCII art representation
        self._create_ascii_circuit(circuit)
    
    def _create_ascii_circuit(self, circuit):
        """Create ASCII art representation of circuit."""
        n_qubits = getattr(circuit, 'num_qubits', 3)
        
        print("\nASCII Circuit Diagram:")
        for i in range(n_qubits):
            line = f"q[{i}]: --"
            # Add some gates for demonstration
            if i == 0:
                line += "H--"
            if i < n_qubits - 1:
                line += "CX-"
            else:
                line += "---"
            line += "M--"
            print(line)


class OptimizationTraceViewer:
    """Interactive optimization trace viewer."""
    
    def __init__(self, results: OptimizationResult):
        self.results = results
        self.fig = None
        self.ax = None
        
    def show_trace(self, interactive: bool = True):
        """Show optimization trace."""
        if not self.results.solution_history:
            print("No solution history available")
            return
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(len(self.results.solution_history))
        self.ax.plot(iterations, self.results.solution_history, 'b-', linewidth=2, label='Objective')
        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('Objective Value')
        self.ax.set_title('Optimization Convergence Trace')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        if interactive and len(self.results.solution_history) > 10:
            self._add_interactive_features()
        
        plt.tight_layout()
        plt.show()
    
    def _add_interactive_features(self):
        """Add interactive features to the plot."""
        # Add annotation at minimum point
        min_idx = np.argmin(self.results.solution_history)
        min_val = self.results.solution_history[min_idx]
        
        self.ax.annotate(f'Min: {min_val:.6f}',
                        xy=(min_idx, min_val),
                        xytext=(min_idx + 5, min_val + 0.1 * min_val),
                        arrowprops=dict(arrowstyle='->', color='red'),
                        fontsize=10,
                        color='red')
        
        # Add convergence rate indicator
        if len(self.results.solution_history) > 20:
            last_10 = self.results.solution_history[-10:]
            convergence_rate = (last_10[-1] - last_10[0]) / last_10[0] if last_10[0] != 0 else 0
            self.ax.text(0.05, 0.95, f'Final Convergence Rate: {convergence_rate:.2e}',
                        transform=self.ax.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))


class ParetoFrontViewer:
    """Pareto front viewer for multi-objective optimization."""
    
    def __init__(self, pareto_points: np.ndarray, objectives: list):
        self.pareto_points = pareto_points
        self.objectives = objectives
        
    def show_pareto_front(self, show_dominated: bool = True):
        """Show Pareto front visualization."""
        if self.pareto_points.shape[1] != 2:
            print("Pareto front visualization currently supports 2 objectives only")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot Pareto front
        ax.scatter(self.pareto_points[:, 0], self.pareto_points[:, 1],
                  c='red', s=50, label='Pareto Front', zorder=3)
        
        # Connect Pareto points
        sorted_indices = np.argsort(self.pareto_points[:, 0])
        sorted_points = self.pareto_points[sorted_indices]
        ax.plot(sorted_points[:, 0], sorted_points[:, 1], 'r--', alpha=0.7, zorder=2)
        
        ax.set_xlabel(self.objectives[0])
        ax.set_ylabel(self.objectives[1])
        ax.set_title('Pareto Front')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add annotations for interesting points
        if len(self.pareto_points) > 0:
            # Best for objective 1
            best_obj1_idx = np.argmin(self.pareto_points[:, 0])
            ax.annotate('Best Obj1',
                       xy=(self.pareto_points[best_obj1_idx, 0], self.pareto_points[best_obj1_idx, 1]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            # Best for objective 2
            best_obj2_idx = np.argmin(self.pareto_points[:, 1])
            ax.annotate('Best Obj2',
                       xy=(self.pareto_points[best_obj2_idx, 0], self.pareto_points[best_obj2_idx, 1]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
        
        plt.tight_layout()
        plt.show()


# Convenience functions for Jupyter usage
def show_dashboard():
    """Display the QAero dashboard."""
    dashboard = QAeroDashboard()
    dashboard.show()
    return dashboard

def quick_solve(problem_type: str, backend: str = 'auto'):
    """Quick solve a problem (convenience function for notebooks)."""
    persona = create_persona('aerospace_engineer')
    problem = persona.create_problem(problem_type)
    solver = create_solver({'backend': backend})
    results = solver.solve(problem)
    
    print(f"Solved {problem_type} with {backend} backend")
    print(f"Success: {results.success}")
    if results.success:
        print(f"Optimal Value: {results.optimal_value}")
    
    persona.visualize_results(results)
    return results

def compare_backends(problem_type: str, backends: list):
    """Compare multiple backends on the same problem."""
    persona = create_persona('aerospace_engineer')
    problem = persona.create_problem(problem_type)
    
    results = {}
    for backend in backends:
        print(f"Solving with {backend}...")
        solver = create_solver({'backend': backend})
        results[backend] = solver.solve(problem)
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    backends_list = []
    execution_times = []
    optimal_values = []
    
    for backend, result in results.items():
        backends_list.append(backend)
        execution_times.append(result.execution_time)
        if result.success and result.optimal_value is not None:
            optimal_values.append(abs(result.optimal_value))
        else:
            optimal_values.append(np.nan)
    
    x = range(len(backends_list))
    width = 0.35
    
    # Execution times
    ax.bar([i - width/2 for i in x], execution_times, width, label='Execution Time (s)', alpha=0.7)
    
    # Optimal values (normalized)
    if not all(np.isnan(optimal_values)):
        optimal_norm = [v / max(optimal_values) if not np.isnan(v) else 0 for v in optimal_values]
        ax.bar([i + width/2 for i in x], optimal_norm, width, label='Optimal Value (norm)', alpha=0.7)
    
    ax.set_xlabel('Backend')
    ax.set_ylabel('Metrics')
    ax.set_title(f'Backend Comparison for {problem_type}')
    ax.set_xticks(x)
    ax.set_xticklabels(backends_list, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results