"""
Multi-objective optimization and constraint handling for aerospace design.
Implements Pareto front tracking, scalarization methods, and gradient-enhanced optimization.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.optimize import differential_evolution
import warnings

from ...core.base import OptimizationProblem, QaeroError
from ...core.results import OptimizationResult
from ...core.registry import register_algorithm

logger = logging.getLogger("qaero.algorithms.multi_objective")

@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    method: str = "weighted_sum"  # "weighted_sum", "epsilon_constraint", "nsga2"
    n_pareto_points: int = 20
    constraint_handling: str = "penalty"  # "penalty", "augmented_lagrangian", "feasibility"
    penalty_weight: float = 1000.0
    use_gradients: bool = True
    parameter_shift: bool = True

@dataclass
class ParetoPoint:
    """Represents a point on the Pareto front."""
    objectives: np.ndarray
    variables: np.ndarray
    constraints: np.ndarray
    is_feasible: bool

class ParetoFront:
    """Pareto front representation and management."""
    
    def __init__(self):
        self.points: List[ParetoPoint] = []
        self.n_objectives = 0
    
    def add_point(self, point: ParetoPoint):
        """Add a point to the Pareto front."""
        if self.n_objectives == 0:
            self.n_objectives = len(point.objectives)
        elif len(point.objectives) != self.n_objectives:
            raise ValueError("Inconsistent number of objectives")
        
        # Check if point is dominated by any existing point
        is_dominated = False
        to_remove = []
        
        for i, existing_point in enumerate(self.points):
            if self._dominates(existing_point.objectives, point.objectives):
                is_dominated = True
                break
            elif self._dominates(point.objectives, existing_point.objectives):
                to_remove.append(i)
        
        # Remove dominated points
        for i in sorted(to_remove, reverse=True):
            self.points.pop(i)
        
        # Add point if not dominated
        if not is_dominated:
            self.points.append(point)
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (minimization)."""
        # obj1 dominates obj2 if it's better in all objectives and strictly better in at least one
        better_or_equal = np.all(obj1 <= obj2)
        strictly_better = np.any(obj1 < obj2)
        return better_or_equal and strictly_better
    
    def get_pareto_front(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get Pareto front as arrays of objectives and variables."""
        objectives = np.array([point.objectives for point in self.points])
        variables = np.array([point.variables for point in self.points])
        return objectives, variables
    
    def get_hypervolume(self, reference_point: np.ndarray) -> float:
        """Calculate hypervolume indicator."""
        if len(self.points) == 0:
            return 0.0
        
        objectives = np.array([point.objectives for point in self.points])
        return self._calculate_hypervolume(objectives, reference_point)
    
    def _calculate_hypervolume(self, points: np.ndarray, reference: np.ndarray) -> float:
        """Calculate hypervolume using inclusion-exclusion principle."""
        # Simplified hypervolume calculation for 2-3 objectives
        if len(points) == 0:
            return 0.0
        
        if points.shape[1] == 2:
            # 2D hypervolume (area)
            sorted_points = points[np.argsort(points[:, 0])]
            volume = 0.0
            prev_x = reference[0]
            
            for point in sorted_points:
                if point[0] > reference[0] and point[1] > reference[1]:
                    volume += (point[0] - prev_x) * (point[1] - reference[1])
                    prev_x = point[0]
            
            return volume
        else:
            # Approximate for higher dimensions
            return np.prod(reference - np.min(points, axis=0))
    
    def get_diversity(self) -> float:
        """Calculate diversity metric of Pareto front."""
        if len(self.points) < 2:
            return 0.0
        
        objectives = np.array([point.objectives for point in self.points])
        distances = []
        
        for i in range(len(objectives)):
            for j in range(i + 1, len(objectives)):
                distances.append(np.linalg.norm(objectives[i] - objectives[j]))
        
        return np.mean(distances) if distances else 0.0

@register_algorithm("multi_objective", "multi_objective_optimization")
class MultiObjectiveOptimizer:
    """
    Multi-objective optimization with Pareto front tracking and constraint handling.
    Supports weighted sum, epsilon-constraint, and hybrid methods.
    """
    
    def __init__(self, config: Optional[Union[MultiObjectiveConfig, Dict]] = None):
        if config is None:
            self.config = MultiObjectiveConfig()
        elif isinstance(config, dict):
            self.config = MultiObjectiveConfig(**config)
        else:
            self.config = config
        
        self.pareto_front = ParetoFront()
        self.objective_functions = []
        self.constraint_functions = []
    
    def add_objective(self, objective_func: Callable):
        """Add an objective function."""
        self.objective_functions.append(objective_func)
    
    def add_constraint(self, constraint_func: Callable, constraint_type: str = "ineq"):
        """Add a constraint function."""
        self.constraint_functions.append((constraint_func, constraint_type))
    
    def optimize(self, problem: OptimizationProblem, **kwargs) -> OptimizationResult:
        """Execute multi-objective optimization."""
        import time
        start_time = time.time()
        
        try:
            # For multi-objective, we modify the single objective problem
            if len(self.objective_functions) == 0:
                # Use the problem's objective as single objective
                self.objective_functions = [problem.objective]
            
            # Generate Pareto front
            pareto_objectives, pareto_variables = self._generate_pareto_front(problem)
            
            # Store Pareto front in metadata
            metadata = {
                'algorithm': 'MultiObjective',
                'method': self.config.method,
                'n_objectives': len(self.objective_functions),
                'n_pareto_points': len(pareto_objectives),
                'pareto_front': {
                    'objectives': pareto_objectives.tolist(),
                    'variables': pareto_variables.tolist()
                },
                'hypervolume': float(self.pareto_front.get_hypervolume(np.max(pareto_objectives, axis=0))),
                'diversity': float(self.pareto_front.get_diversity())
            }
            
            # Return a representative solution (knee point)
            knee_point = self._find_knee_point(pareto_objectives, pareto_variables)
            optimal_vars = {
                var: knee_point[i] for i, var in enumerate(problem.variables)
            }
            
            # Evaluate main objective at knee point
            main_objective_value = problem.objective(knee_point)
            
            return OptimizationResult(
                problem_id=problem.problem_id,
                backend_name="multi_objective",
                success=True,
                execution_time=time.time() - start_time,
                optimal_value=main_objective_value,
                optimal_variables=optimal_vars,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Multi-objective optimization failed: {e}")
            return OptimizationResult(
                problem_id=problem.problem_id,
                backend_name="multi_objective",
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _generate_pareto_front(self, problem: OptimizationProblem) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Pareto front using specified method."""
        if self.config.method == "weighted_sum":
            return self._weighted_sum_method(problem)
        elif self.config.method == "epsilon_constraint":
            return self._epsilon_constraint_method(problem)
        elif self.config.method == "nsga2":
            return self._nsga2_method(problem)
        else:
            raise QaeroError(f"Unknown multi-objective method: {self.config.method}")
    
    def _weighted_sum_method(self, problem: OptimizationProblem) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted sum method for generating Pareto front."""
        n_objectives = len(self.objective_functions)
        pareto_objectives = []
        pareto_variables = []
        
        # Generate different weight combinations
        weights_list = self._generate_weights(n_objectives, self.config.n_pareto_points)
        
        for weights in weights_list:
            def weighted_objective(x):
                total = 0.0
                for i, obj_func in enumerate(self.objective_functions):
                    total += weights[i] * obj_func(x)
                
                # Add constraint penalties
                total += self._constraint_penalty(x)
                return total
            
            # Solve weighted problem
            result = self._solve_single_objective(problem, weighted_objective)
            
            if result.success:
                # Evaluate all objectives at solution
                all_objectives = np.array([obj_func(result.x) for obj_func in self.objective_functions])
                
                # Create Pareto point
                point = ParetoPoint(
                    objectives=all_objectives,
                    variables=result.x,
                    constraints=self._evaluate_constraints(result.x),
                    is_feasible=self._is_feasible(result.x)
                )
                
                self.pareto_front.add_point(point)
        
        return self.pareto_front.get_pareto_front()
    
    def _epsilon_constraint_method(self, problem: OptimizationProblem) -> Tuple[np.ndarray, np.ndarray]:
        """Epsilon-constraint method for generating Pareto front."""
        if len(self.objective_functions) < 2:
            raise QaeroError("Epsilon-constraint method requires at least 2 objectives")
        
        # Use first objective as primary, constrain others
        primary_objective = self.objective_functions[0]
        secondary_objectives = self.objective_functions[1:]
        
        # Find ranges for secondary objectives
        epsilon_ranges = self._find_epsilon_ranges(problem, secondary_objectives)
        
        for epsilon_values in epsilon_ranges:
            def constrained_objective(x):
                value = primary_objective(x)
                value += self._constraint_penalty(x)
                return value
            
            # Add epsilon constraints
            constraints = []
            for i, (obj_func, epsilon) in enumerate(zip(secondary_objectives, epsilon_values)):
                def epsilon_constraint(x, obj_func=obj_func, epsilon=epsilon):
                    return epsilon - obj_func(x)
                
                constraints.append({'type': 'ineq', 'fun': epsilon_constraint})
            
            # Solve constrained problem
            result = self._solve_single_objective(problem, constrained_objective, constraints)
            
            if result.success:
                all_objectives = np.array([obj_func(result.x) for obj_func in self.objective_functions])
                
                point = ParetoPoint(
                    objectives=all_objectives,
                    variables=result.x,
                    constraints=self._evaluate_constraints(result.x),
                    is_feasible=self._is_feasible(result.x)
                )
                
                self.pareto_front.add_point(point)
        
        return self.pareto_front.get_pareto_front()
    
    def _nsga2_method(self, problem: OptimizationProblem) -> Tuple[np.ndarray, np.ndarray]:
        """NSGA-II inspired method for generating Pareto front."""
        n_vars = len(problem.variables)
        population_size = self.config.n_pareto_points * 2
        
        # Initialize population
        population = self._initialize_population(problem, population_size)
        
        # Evolutionary loop
        for generation in range(100):
            # Evaluate population
            evaluated_pop = []
            for individual in population:
                objectives = np.array([obj_func(individual) for obj_func in self.objective_functions])
                constraints = self._evaluate_constraints(individual)
                is_feasible = self._is_feasible(individual)
                
                point = ParetoPoint(
                    objectives=objectives,
                    variables=individual,
                    constraints=constraints,
                    is_feasible=is_feasible
                )
                evaluated_pop.append(point)
            
            # Non-dominated sorting
            fronts = self._non_dominated_sorting(evaluated_pop)
            
            # Select next generation
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= self.config.n_pareto_points:
                    new_population.extend(front)
                else:
                    # Crowding distance sorting for the last front
                    remaining = self.config.n_pareto_points - len(new_population)
                    sorted_front = self._crowding_distance_sorting(front)
                    new_population.extend(sorted_front[:remaining])
                    break
            
            population = [point.variables for point in new_population]
            
            # Crossover and mutation
            if generation < 99:  # Don't mutate last generation
                population = self._evolve_population(problem, population)
        
        # Extract Pareto front from final population
        for individual in population:
            objectives = np.array([obj_func(individual) for obj_func in self.objective_functions])
            point = ParetoPoint(
                objectives=objectives,
                variables=individual,
                constraints=self._evaluate_constraints(individual),
                is_feasible=self._is_feasible(individual)
            )
            self.pareto_front.add_point(point)
        
        return self.pareto_front.get_pareto_front()
    
    def _generate_weights(self, n_objectives: int, n_points: int) -> List[np.ndarray]:
        """Generate weight combinations for weighted sum method."""
        if n_objectives == 1:
            return [np.array([1.0])]
        
        weights = []
        if n_objectives == 2:
            for i in range(n_points):
                w1 = i / (n_points - 1) if n_points > 1 else 0.5
                w2 = 1 - w1
                weights.append(np.array([w1, w2]))
        else:
            # Simplex lattice design for higher dimensions
            for i in range(n_points):
                weight = np.random.dirichlet(np.ones(n_objectives))
                weights.append(weight)
        
        return weights
    
    def _find_epsilon_ranges(self, problem: OptimizationProblem, objectives: List[Callable]) -> List[np.ndarray]:
        """Find epsilon value ranges for constraint method."""
        # Sample the objective space to find ranges
        n_samples = 50
        n_objectives = len(objectives)
        objective_values = []
        
        for _ in range(n_samples):
            if problem.bounds:
                sample = np.array([np.random.uniform(low, high) 
                                 for var in problem.variables 
                                 for low, high in [problem.bounds[var]]])
            else:
                sample = np.random.randn(len(problem.variables))
            
            values = np.array([obj_func(sample) for obj_func in objectives])
            objective_values.append(values)
        
        objective_values = np.array(objective_values)
        min_vals = np.min(objective_values, axis=0)
        max_vals = np.max(objective_values, axis=0)
        
        # Generate epsilon values
        epsilon_sets = []
        n_epsilon_points = self.config.n_pareto_points
        
        for i in range(n_epsilon_points):
            epsilon = min_vals + (max_vals - min_vals) * (i / (n_epsilon_points - 1))
            epsilon_sets.append(epsilon)
        
        return epsilon_sets
    
    def _solve_single_objective(self, problem: OptimizationProblem, objective: Callable, 
                              additional_constraints: List = None) -> Any:
        """Solve a single objective optimization problem."""
        from scipy.optimize import minimize
        
        constraints = []
        if problem.constraints:
            constraints.extend(problem.constraints)
        if additional_constraints:
            constraints.extend(additional_constraints)
        
        # Add problem constraints
        constraints.extend(self._get_problem_constraints())
        
        # Initial guess
        if problem.bounds:
            x0 = np.array([(low + high) / 2 for low, high in problem.bounds.values()])
        else:
            x0 = np.zeros(len(problem.variables))
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=list(problem.bounds.values()) if problem.bounds else None,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        return result
    
    def _constraint_penalty(self, x: np.ndarray) -> float:
        """Calculate constraint penalty."""
        penalty = 0.0
        
        if self.config.constraint_handling == "penalty":
            for constraint_func, constraint_type in self.constraint_functions:
                violation = self._calculate_violation(constraint_func(x), constraint_type)
                penalty += self.config.penalty_weight * violation**2
        
        return penalty
    
    def _calculate_violation(self, constraint_value: float, constraint_type: str) -> float:
        """Calculate constraint violation."""
        if constraint_type == "ineq":
            return max(0, -constraint_value)  # g(x) >= 0
        elif constraint_type == "eq":
            return abs(constraint_value)  # h(x) = 0
        else:
            return 0.0
    
    def _evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all constraints at point x."""
        constraints = []
        for constraint_func, constraint_type in self.constraint_functions:
            constraints.append(constraint_func(x))
        return np.array(constraints)
    
    def _is_feasible(self, x: np.ndarray) -> bool:
        """Check if point x satisfies all constraints."""
        for constraint_func, constraint_type in self.constraint_functions:
            value = constraint_func(x)
            if constraint_type == "ineq" and value < 0:
                return False
            elif constraint_type == "eq" and abs(value) > 1e-6:
                return False
        return True
    
    def _get_problem_constraints(self) -> List[Dict]:
        """Convert constraint functions to scipy constraints."""
        constraints = []
        for constraint_func, constraint_type in self.constraint_functions:
            if constraint_type == "ineq":
                constraints.append({'type': 'ineq', 'fun': constraint_func})
            elif constraint_type == "eq":
                constraints.append({'type': 'eq', 'fun': constraint_func})
        return constraints
    
    def _initialize_population(self, problem: OptimizationProblem, size: int) -> List[np.ndarray]:
        """Initialize population for evolutionary algorithms."""
        population = []
        n_vars = len(problem.variables)
        
        for _ in range(size):
            if problem.bounds:
                individual = np.array([np.random.uniform(low, high) 
                                     for var in problem.variables 
                                     for low, high in [problem.bounds[var]]])
            else:
                individual = np.random.randn(n_vars)
            population.append(individual)
        
        return population
    
    def _non_dominated_sorting(self, population: List[ParetoPoint]) -> List[List[ParetoPoint]]:
        """Perform non-dominated sorting."""
        fronts = [[]]
        
        for i, point_i in enumerate(population):
            point_i.domination_count = 0
            point_i.dominated_set = []
            
            for j, point_j in enumerate(population):
                if i == j:
                    continue
                
                if self._dominates(point_i.objectives, point_j.objectives):
                    point_i.dominated_set.append(j)
                elif self._dominates(point_j.objectives, point_i.objectives):
                    point_i.domination_count += 1
            
            if point_i.domination_count == 0:
                fronts[0].append(point_i)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for point in fronts[i]:
                for dominated_idx in point.dominated_set:
                    dominated_point = population[dominated_idx]
                    dominated_point.domination_count -= 1
                    if dominated_point.domination_count == 0:
                        next_front.append(dominated_point)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def _crowding_distance_sorting(self, front: List[ParetoPoint]) -> List[ParetoPoint]:
        """Sort front by crowding distance."""
        if len(front) == 0:
            return front
        
        n_objectives = len(front[0].objectives)
        crowding_distances = np.zeros(len(front))
        
        for obj_idx in range(n_objectives):
            # Sort by current objective
            sorted_front = sorted(front, key=lambda x: x.objectives[obj_idx])
            obj_min = sorted_front[0].objectives[obj_idx]
            obj_max = sorted_front[-1].objectives[obj_idx]
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # Assign crowding distance
            crowding_distances[0] = np.inf
            crowding_distances[-1] = np.inf
            
            for i in range(1, len(front) - 1):
                prev_obj = sorted_front[i-1].objectives[obj_idx]
                next_obj = sorted_front[i+1].objectives[obj_idx]
                crowding_distances[i] += (next_obj - prev_obj) / obj_range
        
        # Sort by crowding distance (descending)
        sorted_indices = np.argsort(crowding_distances)[::-1]
        return [front[i] for i in sorted_indices]
    
    def _evolve_population(self, problem: OptimizationProblem, population: List[np.ndarray]) -> List[np.ndarray]:
        """Evolve population using crossover and mutation."""
        new_population = []
        n_vars = len(problem.variables)
        
        while len(new_population) < len(population):
            # Tournament selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Crossover
            child1, child2 = self._crossover(parent1, parent2)
            
            # Mutation
            child1 = self._mutate(problem, child1)
            child2 = self._mutate(problem, child2)
            
            new_population.extend([child1, child2])
        
        return new_population[:len(population)]
    
    def _tournament_selection(self, population: List[np.ndarray]) -> np.ndarray:
        """Tournament selection."""
        tournament_size = 3
        tournament = np.random.choice(len(population), tournament_size, replace=False)
        
        # For simplicity, return random individual from tournament
        # In practice, would use fitness-based selection
        return population[np.random.choice(tournament)]
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated binary crossover."""
        child1 = 0.5 * (parent1 + parent2)
        child2 = 0.5 * (parent1 - parent2)  # Diversity
        return child1, child2
    
    def _mutate(self, problem: OptimizationProblem, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation."""
        mutation_rate = 0.1
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if np.random.random() < mutation_rate:
                if problem.bounds and problem.variables[i] in problem.bounds:
                    low, high = problem.bounds[problem.variables[i]]
                    mutated[i] = np.random.uniform(low, high)
                else:
                    mutated[i] += 0.1 * np.random.randn()
        
        return mutated
    
    def _find_knee_point(self, objectives: np.ndarray, variables: np.ndarray) -> np.ndarray:
        """Find knee point on Pareto front."""
        if len(objectives) == 0:
            return np.zeros(variables.shape[1]) if variables.shape[0] > 0 else np.array([])
        
        if objectives.shape[1] == 1:
            return variables[np.argmin(objectives[:, 0])]
        
        # Use Utopia point method for knee point identification
        ideal_point = np.min(objectives, axis=0)
        nadir_point = np.max(objectives, axis=0)
        
        # Normalize objectives
        ranges = nadir_point - ideal_point
        ranges[ranges == 0] = 1.0  # Avoid division by zero
        normalized_obj = (objectives - ideal_point) / ranges
        
        # Find point closest to ideal point
        distances = np.linalg.norm(normalized_obj, axis=1)
        knee_idx = np.argmin(distances)
        
        return variables[knee_idx]

@register_algorithm("gradient_enhanced", "quantum_optimization")
class GradientEnhancedOptimizer:
    """
    Gradient-enhanced optimization with differentiable surrogates.
    Uses parameter shift rules for quantum circuits and automatic differentiation.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'differentiation_method': 'parameter_shift',  # 'parameter_shift', 'finite_difference', 'autodiff'
            'step_size': 0.01,
            'use_surrogate': True,
            'surrogate_type': 'neural_network'  # 'neural_network', 'gaussian_process', 'polynomial'
        }
        
        self.surrogate_model = None
        self.gradient_function = None
    
    def optimize(self, problem: OptimizationProblem, **kwargs) -> OptimizationResult:
        """Execute gradient-enhanced optimization."""
        import time
        start_time = time.time()
        
        try:
            # Build surrogate model if enabled
            if self.config['use_surrogate']:
                self._build_surrogate_model(problem)
            
            # Setup gradient function
            self._setup_gradient_function(problem)
            
            # Use gradient-based optimization
            from scipy.optimize import minimize
            
            solution_history = []
            
            def objective_with_grad(x):
                if self.config['use_surrogate'] and self.surrogate_model is not None:
                    value = self.surrogate_model.predict(x.reshape(1, -1))[0]
                else:
                    value = problem.objective(x)
                
                solution_history.append(value)
                return value
            
            # Initial guess
            if problem.bounds:
                x0 = np.array([(low + high) / 2 for low, high in problem.bounds.values()])
            else:
                x0 = np.zeros(len(problem.variables))
            
            # Optimize with gradient
            result = minimize(
                objective_with_grad,
                x0,
                method='BFGS',
                jac=self.gradient_function,
                bounds=list(problem.bounds.values()) if problem.bounds else None,
                options={'gtol': 1e-6, 'maxiter': 1000}
            )
            
            optimal_vars = {
                var: result.x[i] for i, var in enumerate(problem.variables)
            }
            
            return OptimizationResult(
                problem_id=problem.problem_id,
                backend_name="gradient_enhanced",
                success=result.success,
                execution_time=time.time() - start_time,
                optimal_value=result.fun,
                optimal_variables=optimal_vars,
                solution_history=solution_history,
                metadata={
                    'algorithm': 'GradientEnhanced',
                    'differentiation_method': self.config['differentiation_method'],
                    'use_surrogate': self.config['use_surrogate'],
                    'n_iterations': result.nit,
                    'gradient_evaluations': result.njev
                }
            )
            
        except Exception as e:
            logger.error(f"Gradient-enhanced optimization failed: {e}")
            return OptimizationResult(
                problem_id=problem.problem_id,
                backend_name="gradient_enhanced",
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _build_surrogate_model(self, problem: OptimizationProblem):
        """Build differentiable surrogate model."""
        surrogate_type = self.config.get('surrogate_type', 'neural_network')
        
        if surrogate_type == 'neural_network':
            self._build_neural_surrogate(problem)
        elif surrogate_type == 'gaussian_process':
            self._build_gaussian_surrogate(problem)
        elif surrogate_type == 'polynomial':
            self._build_polynomial_surrogate(problem)
        else:
            logger.warning(f"Unknown surrogate type: {surrogate_type}, using polynomial")
            self._build_polynomial_surrogate(problem)
    
    def _build_neural_surrogate(self, problem: OptimizationProblem):
        """Build neural network surrogate model."""
        try:
            from sklearn.neural_network import MLPRegressor
            
            # Generate training data
            n_samples = 100
            n_vars = len(problem.variables)
            
            X_train = np.random.uniform(-1, 1, (n_samples, n_vars))
            y_train = np.array([problem.objective(x) for x in X_train])
            
            # Train neural network
            self.surrogate_model = MLPRegressor(
                hidden_layer_sizes=(50, 30),
                activation='tanh',
                max_iter=1000,
                random_state=42
            )
            self.surrogate_model.fit(X_train, y_train)
            
            logger.info("Neural network surrogate model built")
            
        except ImportError:
            logger.warning("scikit-learn not available, using polynomial surrogate")
            self._build_polynomial_surrogate(problem)
    
    def _build_gaussian_surrogate(self, problem: OptimizationProblem):
        """Build Gaussian process surrogate model."""
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel
            
            # Generate training data
            n_samples = 50
            n_vars = len(problem.variables)
            
            X_train = np.random.uniform(-1, 1, (n_samples, n_vars))
            y_train = np.array([problem.objective(x) for x in X_train])
            
            # Train Gaussian process
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            self.surrogate_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
            self.surrogate_model.fit(X_train, y_train)
            
            logger.info("Gaussian process surrogate model built")
            
        except ImportError:
            logger.warning("scikit-learn not available, using polynomial surrogate")
            self._build_polynomial_surrogate(problem)
    
    def _build_polynomial_surrogate(self, problem: OptimizationProblem):
        """Build polynomial surrogate model."""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        
        # Generate training data
        n_samples = 100
        n_vars = len(problem.variables)
        
        X_train = np.random.uniform(-1, 1, (n_samples, n_vars))
        y_train = np.array([problem.objective(x) for x in X_train])
        
        # Train polynomial model
        self.surrogate_model = Pipeline([
            ('poly', PolynomialFeatures(degree=3)),
            ('linear', LinearRegression())
        ])
        self.surrogate_model.fit(X_train, y_train)
        
        logger.info("Polynomial surrogate model built")
    
    def _setup_gradient_function(self, problem: OptimizationProblem):
        """Setup gradient computation function."""
        differentiation_method = self.config.get('differentiation_method', 'parameter_shift')
        
        if differentiation_method == 'parameter_shift':
            self.gradient_function = self._parameter_shift_gradient
        elif differentiation_method == 'finite_difference':
            self.gradient_function = self._finite_difference_gradient
        elif differentiation_method == 'autodiff':
            self.gradient_function = self._autodiff_gradient
        else:
            logger.warning(f"Unknown differentiation method: {differentiation_method}, using finite difference")
            self.gradient_function = self._finite_difference_gradient
    
    def _parameter_shift_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient using parameter shift rule (quantum-inspired)."""
        gradient = np.zeros_like(x)
        shift = self.config.get('step_size', 0.01)
        
        if self.config['use_surrogate'] and self.surrogate_model is not None:
            # Use surrogate for gradient computation
            for i in range(len(x)):
                x_plus = x.copy()
                x_minus = x.copy()
                
                x_plus[i] += np.pi / 2  # Quantum parameter shift
                x_minus[i] -= np.pi / 2
                
                f_plus = self.surrogate_model.predict(x_plus.reshape(1, -1))[0]
                f_minus = self.surrogate_model.predict(x_minus.reshape(1, -1))[0]
                
                gradient[i] = 0.5 * (f_plus - f_minus)
        else:
            # Use actual objective function
            for i in range(len(x)):
                x_plus = x.copy()
                x_minus = x.copy()
                
                x_plus[i] += shift
                x_minus[i] -= shift
                
                f_plus = self._evaluate_objective(x_plus)
                f_minus = self._evaluate_objective(x_minus)
                
                gradient[i] = (f_plus - f_minus) / (2 * shift)
        
        return gradient
    
    def _finite_difference_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient using finite differences."""
        gradient = np.zeros_like(x)
        h = self.config.get('step_size', 1e-6)
        
        if self.config['use_surrogate'] and self.surrogate_model is not None:
            # Use surrogate for gradient computation
            for i in range(len(x)):
                x_perturbed = x.copy()
                x_perturbed[i] += h
                
                f_x = self.surrogate_model.predict(x.reshape(1, -1))[0]
                f_perturbed = self.surrogate_model.predict(x_perturbed.reshape(1, -1))[0]
                
                gradient[i] = (f_perturbed - f_x) / h
        else:
            # Use actual objective function
            f_x = self._evaluate_objective(x)
            for i in range(len(x)):
                x_perturbed = x.copy()
                x_perturbed[i] += h
                f_perturbed = self._evaluate_objective(x_perturbed)
                gradient[i] = (f_perturbed - f_x) / h
        
        return gradient
    
    def _autodiff_gradient(self, x: np.ndarray) -> np.ndarray:
        """Compute gradient using automatic differentiation."""
        try:
            import autograd.numpy as anp
            from autograd import grad
            
            # Define objective function for autograd
            def objective_autograd(x_anp):
                if self.config['use_surrogate'] and self.surrogate_model is not None:
                    # Convert back to numpy for surrogate prediction
                    x_np = anp.array(x_anp)
                    return anp.array(self.surrogate_model.predict(x_np.reshape(1, -1))[0])
                else:
                    return anp.array(self._evaluate_objective(x_anp))
            
            # Compute gradient
            gradient_func = grad(objective_autograd)
            return gradient_func(x)
            
        except ImportError:
            logger.warning("autograd not available, using finite difference")
            return self._finite_difference_gradient(x)
    
    def _evaluate_objective(self, x: np.ndarray) -> float:
        """Evaluate objective function with error handling."""
        try:
            return self.problem.objective(x)
        except:
            # Return large value for invalid points
            return 1e10