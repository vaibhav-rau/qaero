"""
Comprehensive tests for aerospace problem templates.
"""
import pytest
import numpy as np

from qaero.problems.aerodynamics import (
    AirfoilOptimizationProblem, WingDesignProblem, CompressibleFlowProblem
)
from qaero.problems.trajectory import (
    AscentTrajectoryProblem, SatelliteManeuverProblem, OrbitalTransferProblem
)
from qaero.core.base import OptimizationProblem, PDEProblem


class TestAerodynamicsProblems:
    """Test aerodynamics problem templates."""
    
    def test_airfoil_optimization_naca(self):
        """Test NACA airfoil optimization problem."""
        problem = AirfoilOptimizationProblem(parameterization="naca")
        
        assert problem.problem_id == "airfoil_optimization"
        assert problem.parameters['parameterization'] == "naca"
        assert len(problem.variables) == 3
        assert 'm' in problem.variables
        assert 'p' in problem.variables
        assert 't' in problem.variables
        
        # Test objective function
        test_point = np.array([0.05, 0.4, 0.12])
        result = problem.objective(test_point)
        assert isinstance(result, float)
        assert result < 0  # Minimizing negative L/D
    
    def test_airfoil_optimization_cst(self):
        """Test CST airfoil optimization problem."""
        problem = AirfoilOptimizationProblem(parameterization="cst")
        
        assert problem.parameters['parameterization'] == "cst"
        assert len(problem.variables) == 6
        assert all(f'A{i}' in problem.variables for i in range(6))
    
    def test_wing_design_problem(self):
        """Test wing design optimization problem."""
        problem = WingDesignProblem(
            disciplines=['aerodynamics', 'structures', 'controls'],
            coupling='strong'
        )
        
        assert problem.parameters['disciplines'] == ['aerodynamics', 'structures', 'controls']
        assert problem.parameters['coupling'] == 'strong'
        assert len(problem.variables) == 5
        
        # Test objective function
        test_point = np.array([20.0, 8.0, 0.3, 25.0, 5.0])
        result = problem.objective(test_point)
        assert isinstance(result, float)
    
    def test_compressible_flow_problem(self):
        """Test compressible flow PDE problem."""
        problem = CompressibleFlowProblem(
            mach_number=0.8,
            flow_type='transonic'
        )
        
        assert problem.problem_id == "compressible_flow"
        assert problem.parameters['mach_number'] == 0.8
        assert problem.parameters['flow_type'] == 'transonic'
        assert problem.equation == "euler"


class TestTrajectoryProblems:
    """Test trajectory optimization problems."""
    
    def test_ascent_trajectory_problem(self):
        """Test rocket ascent trajectory optimization."""
        problem = AscentTrajectoryProblem(stages=3, target_orbit='GTO')
        
        assert problem.parameters['stages'] == 3
        assert problem.parameters['target_orbit'] == 'GTO'
        assert len(problem.variables) == 9  # 3 stages * 3 variables
        
        # Test objective function
        test_point = np.array([0.8, 80.0, 60.0, 0.9, 70.0, 50.0, 1.0, 60.0, 40.0])
        result = problem.objective(test_point)
        assert isinstance(result, float)
    
    def test_satellite_maneuver_problem(self):
        """Test satellite orbital maneuver optimization."""
        problem = SatelliteManeuverProblem(
            initial_orbit={'altitude': 400, 'inclination': 28.5},
            target_orbit={'altitude': 1200, 'inclination': 0.0}
        )
        
        assert problem.parameters['initial_orbit']['altitude'] == 400
        assert problem.parameters['target_orbit']['inclination'] == 0.0
        assert 'burn1_magnitude' in problem.variables
        assert problem.bounds['burn1_magnitude'] == (0.0, 500.0)
    
    def test_orbital_transfer_problem(self):
        """Test orbital transfer optimization."""
        problem = OrbitalTransferProblem(
            transfer_type='hohmann',
            time_constrained=True
        )
        
        assert problem.parameters['transfer_type'] == 'hohmann'
        assert problem.parameters['time_constrained'] is True
        assert 'departure_time' in problem.variables


class TestProblemValidation:
    """Test problem validation and edge cases."""
    
    def test_problem_parameter_validation(self):
        """Test problem parameter validation."""
        # Should raise error for invalid parameterization
        with pytest.raises(ValueError):
            AirfoilOptimizationProblem(parameterization="invalid_method")
    
    def test_objective_function_robustness(self):
        """Test objective function robustness with edge cases."""
        problem = AirfoilOptimizationProblem(parameterization="naca")
        
        # Test with extreme values
        extreme_point = np.array([0.0, 0.0, 0.0])
        result = problem.objective(extreme_point)
        assert np.isfinite(result)
        
        # Test with boundary values
        boundary_point = np.array([0.09, 0.8, 0.2])
        result = problem.objective(boundary_point)
        assert np.isfinite(result)
    
    def test_problem_serialization(self):
        """Test that problems can be serialized."""
        import pickle
        
        problem = AirfoilOptimizationProblem(parameterization="naca")
        
        # Serialize and deserialize
        pickled = pickle.dumps(problem)
        unpickled = pickle.loads(pickled)
        
        assert unpickled.problem_id == problem.problem_id
        assert unpickled.parameters == problem.parameters
        
        # Test that objective still works
        test_point = np.array([0.05, 0.4, 0.12])
        assert unpickled.objective(test_point) == problem.objective(test_point)


class TestProblemPerformance:
    """Test problem performance characteristics."""
    
    def test_objective_evaluation_speed(self):
        """Test objective function evaluation speed."""
        import time
        
        problem = WingDesignProblem()
        test_point = np.array([20.0, 8.0, 0.3, 25.0, 5.0])
        
        # Time multiple evaluations
        start_time = time.time()
        for _ in range(1000):
            _ = problem.objective(test_point)
        eval_time = time.time() - start_time
        
        # Should be fast (less than 1 second for 1000 evaluations)
        assert eval_time < 1.0
    
    def test_large_problem_creation(self):
        """Test creation of problems with many variables."""
        # Create problem with many design variables
        n_vars = 50
        variables = [f'x{i}' for i in range(n_vars)]
        
        problem = OptimizationProblem(
            problem_id="large_design",
            objective=lambda x: np.sum(x**2),
            variables=variables,
            bounds={var: (-1, 1) for var in variables}
        )
        
        assert len(problem.variables) == n_vars
        assert len(problem.bounds) == n_vars
        
        # Test objective evaluation
        test_point = np.random.randn(n_vars)
        result = problem.objective(test_point)
        assert np.isfinite(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])