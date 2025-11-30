"""
Surrogate and Reduced Order Models for bridging high-fidelity CFD with quantum solvers.
Implements POD, DMD, and automatic ROM construction pipelines.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from scipy.linalg import svd, eig, lstsq
from scipy.interpolate import interp1d

from ...core.base import PDEProblem, QaeroError
from ...core.results import PDEResult
from ...core.registry import register_algorithm

logger = logging.getLogger("qaero.algorithms.rom")

@dataclass
class ROMConfig:
    """Configuration for Reduced Order Models."""
    method: str = "pod"  # "pod", "dmd", "autoencoder"
    n_modes: int = 10
    tolerance: float = 1e-6
    include_nonlinear: bool = False
    adaptive: bool = True

@register_algorithm("pod_rom", "surrogate_pde")
class PODReducedOrderModel:
    """
    Proper Orthogonal Decomposition for constructing efficient ROMs from CFD data.
    Maps high-dimensional PDE solutions to low-dimensional subspaces.
    """
    
    def __init__(self, config: Optional[Union[ROMConfig, Dict]] = None):
        if config is None:
            self.config = ROMConfig()
        elif isinstance(config, dict):
            self.config = ROMConfig(**config)
        else:
            self.config = config
        
        self.snapshot_matrix = None
        self.pod_modes = None
        self.singular_values = None
        self.reduced_basis = None
        self.mean_field = None
    
    def build_from_snapshots(self, snapshots: np.ndarray):
        """Build ROM from snapshot data."""
        self.snapshot_matrix = snapshots
        self.mean_field = np.mean(snapshots, axis=1, keepdims=True)
        
        # Remove mean
        fluctuations = snapshots - self.mean_field
        
        # Perform SVD
        U, s, Vt = svd(fluctuations, full_matrices=False)
        
        # Truncate to n_modes
        n_modes = min(self.config.n_modes, len(s))
        self.pod_modes = U[:, :n_modes]
        self.singular_values = s[:n_modes]
        self.reduced_basis = self.pod_modes
        
        logger.info(f"POD ROM built with {n_modes} modes, energy: {np.sum(s[:n_modes])/np.sum(s):.3f}")
    
    def project_to_reduced_space(self, full_field: np.ndarray) -> np.ndarray:
        """Project full field to reduced space."""
        if self.reduced_basis is None:
            raise QaeroError("ROM not built. Call build_from_snapshots first.")
        
        fluctuation = full_field - self.mean_field.flatten()
        coefficients = self.reduced_basis.T @ fluctuation
        return coefficients
    
    def reconstruct_from_reduced(self, coefficients: np.ndarray) -> np.ndarray:
        """Reconstruct full field from reduced coefficients."""
        if self.reduced_basis is None:
            raise QaeroError("ROM not built. Call build_from_snapshots first.")
        
        fluctuation = self.reduced_basis @ coefficients
        full_field = fluctuation + self.mean_field.flatten()
        return full_field
    
    def solve_reduced_system(self, problem: PDEProblem, **kwargs) -> PDEResult:
        """Solve PDE using ROM approximation."""
        import time
        start_time = time.time()
        
        try:
            if self.reduced_basis is None:
                # Generate training data and build ROM
                self._generate_training_data(problem)
            
            # Solve reduced system
            reduced_solution, residual_norm = self._solve_reduced_pde(problem)
            
            # Reconstruct full solution
            full_solution = self.reconstruct_from_reduced(reduced_solution)
            
            return PDEResult(
                problem_id=problem.problem_id,
                backend_name="pod_rom",
                success=residual_norm < self.config.tolerance,
                execution_time=time.time() - start_time,
                solution_field=full_solution,
                residual_norm=residual_norm,
                metadata={
                    'algorithm': 'POD_ROM',
                    'n_modes': self.config.n_modes,
                    'energy_captured': np.sum(self.singular_values) / np.sum(self.singular_values) if self.singular_values is not None else 0.0,
                    'reduced_dimension': len(reduced_solution)
                }
            )
            
        except Exception as e:
            logger.error(f"POD ROM solver failed: {e}")
            return PDEResult(
                problem_id=problem.problem_id,
                backend_name="pod_rom",
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _generate_training_data(self, problem: PDEProblem):
        """Generate training snapshots for ROM construction."""
        # Generate snapshots by solving PDE with different parameters
        n_snapshots = 50
        n_points = 100
        
        snapshots = np.zeros((n_points, n_snapshots))
        
        for i in range(n_snapshots):
            # Vary boundary conditions or parameters
            varied_problem = self._vary_problem_parameters(problem, i)
            
            # Solve using simple method (would use high-fidelity solver in practice)
            solution = self._solve_simple_pde(varied_problem, n_points)
            snapshots[:, i] = solution
        
        self.build_from_snapshots(snapshots)
    
    def _vary_problem_parameters(self, problem: PDEProblem, variation: int) -> PDEProblem:
        """Create problem variation for training."""
        # Vary boundary conditions
        bc_variation = 0.1 * np.sin(variation * 0.1)
        
        new_bc = problem.boundary_conditions.copy()
        if 'left' in new_bc:
            new_bc['left'] += bc_variation
        if 'right' in new_bc:
            new_bc['right'] += bc_variation
        
        return PDEProblem(
            problem_id=problem.problem_id + f"_variation_{variation}",
            equation=problem.equation,
            domain=problem.domain,
            boundary_conditions=new_bc,
            discretization=problem.discretization
        )
    
    def _solve_simple_pde(self, problem: PDEProblem, n_points: int) -> np.ndarray:
        """Simple PDE solver for training data generation."""
        # Finite difference solution
        grid = np.linspace(0, 1, n_points)
        h = grid[1] - grid[0]
        
        # Set up linear system
        A = np.zeros((n_points, n_points))
        b = np.zeros(n_points)
        
        for i in range(1, n_points-1):
            A[i, i-1] = 1/h**2
            A[i, i] = -2/h**2
            A[i, i+1] = 1/h**2
        
        # Boundary conditions
        A[0, 0] = 1.0
        A[-1, -1] = 1.0
        b[0] = problem.boundary_conditions.get('left', 0.0)
        b[-1] = problem.boundary_conditions.get('right', 1.0)
        
        solution = np.linalg.solve(A, b)
        return solution
    
    def _solve_reduced_pde(self, problem: PDEProblem) -> Tuple[np.ndarray, float]:
        """Solve PDE in reduced space."""
        n_modes = self.reduced_basis.shape[1]
        
        # Project full operator to reduced space
        A_reduced, b_reduced = self._project_system_to_reduced(problem)
        
        # Solve reduced system
        reduced_solution = np.linalg.solve(A_reduced, b_reduced)
        
        # Calculate residual in full space
        full_solution = self.reconstruct_from_reduced(reduced_solution)
        full_residual = self._calculate_full_residual(problem, full_solution)
        
        return reduced_solution, full_residual
    
    def _project_system_to_reduced(self, problem: PDEProblem) -> Tuple[np.ndarray, np.ndarray]:
        """Project full PDE system to reduced space."""
        n_modes = self.reduced_basis.shape[1]
        n_points = self.reduced_basis.shape[0]
        
        # Create full operator (simplified)
        A_full = np.zeros((n_points, n_points))
        h = 1.0 / (n_points - 1)
        
        for i in range(1, n_points-1):
            A_full[i, i-1] = 1/h**2
            A_full[i, i] = -2/h**2
            A_full[i, i+1] = 1/h**2
        
        # Boundary conditions
        A_full[0, 0] = 1.0
        A_full[-1, -1] = 1.0
        
        # Create full RHS
        b_full = np.zeros(n_points)
        b_full[0] = problem.boundary_conditions.get('left', 0.0)
        b_full[-1] = problem.boundary_conditions.get('right', 1.0)
        
        # Project to reduced space
        A_reduced = self.reduced_basis.T @ A_full @ self.reduced_basis
        b_reduced = self.reduced_basis.T @ b_full
        
        return A_reduced, b_reduced
    
    def _calculate_full_residual(self, problem: PDEProblem, solution: np.ndarray) -> float:
        """Calculate residual in full space."""
        n_points = len(solution)
        h = 1.0 / (n_points - 1)
        
        residual = 0.0
        for i in range(1, n_points-1):
            # Laplace residual
            u_xx = (solution[i-1] - 2*solution[i] + solution[i+1]) / h**2
            residual += u_xx**2
        
        # Boundary residual
        residual += (solution[0] - problem.boundary_conditions.get('left', 0.0))**2
        residual += (solution[-1] - problem.boundary_conditions.get('right', 1.0))**2
        
        return np.sqrt(residual / (n_points + 2))

@register_algorithm("dmd_rom", "surrogate_pde")
class DMDReducedOrderModel:
    """
    Dynamic Mode Decomposition for constructing ROMs from time-series CFD data.
    Captures dynamic behavior and mode interactions.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'n_modes': 10,
            'exact': True,
            'tolerance': 1e-6
        }
        
        self.dmd_modes = None
        self.eigenvalues = None
        self.amplitudes = None
    
    def build_from_snapshots(self, snapshots: np.ndarray):
        """Build DMD model from time-series snapshots."""
        X = snapshots[:, :-1]  # First n-1 snapshots
        Y = snapshots[:, 1:]   # Next n-1 snapshots
        
        # SVD of X
        U, s, Vt = svd(X, full_matrices=False)
        
        # Truncate based on tolerance or n_modes
        if self.config['exact']:
            r = min(self.config['n_modes'], len(s))
        else:
            # Tolerance based truncation
            energy = np.cumsum(s) / np.sum(s)
            r = np.argmax(energy > 1 - self.config['tolerance']) + 1
            r = min(r, self.config['n_modes'])
        
        U_r = U[:, :r]
        s_r = s[:r]
        Vt_r = Vt[:r, :]
        
        # Build DMD operator
        A_tilde = U_r.T @ Y @ Vt_r.T @ np.diag(1.0 / s_r)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = eig(A_tilde)
        self.dmd_modes = Y @ Vt_r.T @ np.diag(1.0 / s_r) @ eigenvectors
        self.eigenvalues = eigenvalues
        
        # Calculate mode amplitudes
        self._calculate_amplitudes(snapshots[:, 0])
        
        logger.info(f"DMD ROM built with {r} modes")
    
    def _calculate_amplitudes(self, initial_condition: np.ndarray):
        """Calculate DMD mode amplitudes."""
        if self.dmd_modes is None:
            raise QaeroError("DMD modes not computed")
        
        # Solve b = Phi * alpha for amplitudes
        self.amplitudes = lstsq(self.dmd_modes, initial_condition)[0]
    
    def predict(self, time_steps: int) -> np.ndarray:
        """Predict future states using DMD."""
        if self.dmd_modes is None or self.eigenvalues is None:
            raise QaeroError("DMD model not built")
        
        predictions = []
        current_state = self.dmd_modes @ self.amplitudes
        
        for t in range(time_steps):
            predictions.append(current_state)
            # Evolve each mode by its eigenvalue
            self.amplitudes = self.amplitudes * self.eigenvalues
            current_state = self.dmd_modes @ self.amplitudes
        
        return np.array(predictions).T

@register_algorithm("autoencoder_rom", "surrogate_pde")
class AutoencoderReducedOrderModel:
    """
    Neural network autoencoder for nonlinear ROM construction.
    Learns efficient low-dimensional representations of high-fidelity data.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'encoding_dim': 10,
            'hidden_layers': [50, 30],
            'activation': 'tanh',
            'learning_rate': 0.001,
            'epochs': 1000
        }
        
        self.encoder = None
        self.decoder = None
        self.is_trained = False
    
    def build_from_snapshots(self, snapshots: np.ndarray):
        """Build autoencoder ROM from snapshot data."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Model
            from tensorflow.keras.layers import Dense, Input
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            logger.warning("TensorFlow not available, using simplified autoencoder")
            self._build_simplified_autoencoder(snapshots)
            return
        
        input_dim = snapshots.shape[0]
        encoding_dim = self.config['encoding_dim']
        
        # Build encoder
        input_layer = Input(shape=(input_dim,))
        encoded = input_layer
        
        for units in self.config['hidden_layers']:
            encoded = Dense(units, activation=self.config['activation'])(encoded)
        
        encoded = Dense(encoding_dim, activation='linear', name='bottleneck')(encoded)
        
        # Build decoder
        decoded = encoded
        for units in reversed(self.config['hidden_layers']):
            decoded = Dense(units, activation=self.config['activation'])(decoded)
        
        decoded = Dense(input_dim, activation='linear')(decoded)
        
        # Create models
        autoencoder = Model(input_layer, decoded)
        self.encoder = Model(input_layer, encoded)
        
        # Create decoder
        encoded_input = Input(shape=(encoding_dim,))
        decoder_output = encoded_input
        for layer in autoencoder.layers[-len(self.config['hidden_layers'])-1:]:
            decoder_output = layer(decoder_output)
        self.decoder = Model(encoded_input, decoder_output)
        
        # Compile and train
        autoencoder.compile(optimizer=Adam(learning_rate=self.config['learning_rate']), 
                          loss='mse')
        
        # Train (transpose for Keras format)
        X_train = snapshots.T
        history = autoencoder.fit(X_train, X_train,
                                epochs=self.config['epochs'],
                                batch_size=32,
                                shuffle=True,
                                verbose=0)
        
        self.is_trained = True
        logger.info(f"Autoencoder ROM built with encoding dimension {encoding_dim}")
    
    def _build_simplified_autoencoder(self, snapshots: np.ndarray):
        """Build simplified autoencoder without TensorFlow."""
        input_dim = snapshots.shape[0]
        encoding_dim = self.config['encoding_dim']
        
        # Use PCA as simplified autoencoder
        from scipy.linalg import svd
        
        # Center data
        mean = np.mean(snapshots, axis=1, keepdims=True)
        centered = snapshots - mean
        
        # SVD for PCA
        U, s, Vt = svd(centered, full_matrices=False)
        
        # Truncate to encoding dimension
        self.encoder_weights = U[:, :encoding_dim]
        self.decoder_weights = U[:, :encoding_dim].T
        self.mean = mean
        
        self.is_trained = True
        logger.info(f"Simplified autoencoder (PCA) built with encoding dimension {encoding_dim}")
    
    def encode(self, full_field: np.ndarray) -> np.ndarray:
        """Encode full field to reduced representation."""
        if not self.is_trained:
            raise QaeroError("Autoencoder not trained")
        
        if hasattr(self, 'encoder_weights'):
            # Simplified autoencoder
            centered = full_field - self.mean.flatten()
            return self.decoder_weights @ centered
        else:
            # Neural network autoencoder
            return self.encoder.predict(full_field.reshape(1, -1), verbose=0)[0]
    
    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decode reduced representation to full field."""
        if not self.is_trained:
            raise QaeroError("Autoencoder not trained")
        
        if hasattr(self, 'encoder_weights'):
            # Simplified autoencoder
            reconstructed = self.encoder_weights @ encoded
            return reconstructed + self.mean.flatten()
        else:
            # Neural network autoencoder
            return self.decoder.predict(encoded.reshape(1, -1), verbose=0)[0]