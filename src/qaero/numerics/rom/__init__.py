"""
Reduced Order Model pipeline for extracting principal modes and mapping PDEs to small systems.
Implements POD, autoencoders, and other dimension reduction techniques.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from scipy.linalg import svd, eig
from scipy.interpolate import interp1d
import warnings

from ....core.base import PDEProblem, QaeroError
from ....core.registry import register_service

logger = logging.getLogger("qaero.numerics.rom")

@dataclass
class ROMResult:
    """Results from Reduced Order Model construction."""
    reduced_basis: np.ndarray
    singular_values: np.ndarray
    projection_matrix: np.ndarray
    reconstruction_error: float
    energy_captured: float
    metadata: Dict[str, Any]

@dataclass
class OnlineROM:
    """Online ROM for fast solution of parameterized problems."""
    reduced_operator: np.ndarray
    reduced_rhs: np.ndarray
    projection: Callable
    reconstruction: Callable
    metadata: Dict[str, Any]

@register_service("rom_pipeline")
class ROMPipeline:
    """
    Complete Reduced Order Model pipeline for quantum-ready small systems.
    Extracts principal modes and creates efficient reduced models.
    """
    
    def __init__(self):
        self.reduction_methods = {
            'pod': PODReducer(),
            'dmd': DMDReducer(),
            'autoencoder': AutoencoderReducer(),
            'proper_interpolation': ProperInterpolationReducer()
        }
        self.quality_metrics = {
            'reconstruction_error': ReconstructionErrorMetric(),
            'prediction_error': PredictionErrorMetric(),
            'stability': StabilityMetric()
        }
    
    def build_rom(self, snapshots: np.ndarray, method: str = "pod", 
                  **kwargs) -> ROMResult:
        """Build Reduced Order Model from snapshot data."""
        if method not in self.reduction_methods:
            raise QaeroError(f"Unsupported ROM method: {method}. "
                           f"Available: {list(self.reduction_methods.keys())}")
        
        reducer = self.reduction_methods[method]
        return reducer.reduce(snapshots, **kwargs)
    
    def build_online_rom(self, full_operator: np.ndarray, full_rhs: np.ndarray,
                        reduced_basis: np.ndarray, **kwargs) -> OnlineROM:
        """Build online ROM for fast parameterized solutions."""
        # Project full operator to reduced space
        reduced_operator = reduced_basis.T @ full_operator @ reduced_basis
        reduced_rhs = reduced_basis.T @ full_rhs
        
        def project(full_vector: np.ndarray) -> np.ndarray:
            return reduced_basis.T @ full_vector
        
        def reconstruct(reduced_vector: np.ndarray) -> np.ndarray:
            return reduced_basis @ reduced_vector
        
        return OnlineROM(
            reduced_operator=reduced_operator,
            reduced_rhs=reduced_rhs,
            projection=project,
            reconstruction=reconstruct,
            metadata={
                'reduced_dimension': reduced_operator.shape[0],
                'full_dimension': full_operator.shape[0],
                'reduction_ratio': reduced_operator.shape[0] / full_operator.shape[0]
            }
        )
    
    def evaluate_rom_quality(self, rom_result: ROMResult, test_snapshots: np.ndarray,
                           metrics: List[str] = None) -> Dict[str, float]:
        """Evaluate ROM quality using specified metrics."""
        if metrics is None:
            metrics = ['reconstruction_error', 'prediction_error']
        
        quality_scores = {}
        for metric in metrics:
            if metric in self.quality_metrics:
                evaluator = self.quality_metrics[metric]
                quality_scores[metric] = evaluator.evaluate(rom_result, test_snapshots)
        
        return quality_scores
    
    def adapt_rom(self, rom_result: ROMResult, new_snapshots: np.ndarray,
                 method: str = "incremental", **kwargs) -> ROMResult:
        """Adapt existing ROM with new snapshot data."""
        if method == "incremental":
            return self._incremental_adaptation(rom_result, new_snapshots, **kwargs)
        elif method == "retrain":
            # Combine old and new snapshots and retrain
            combined_snapshots = np.hstack([rom_result.reduced_basis, new_snapshots])
            return self.build_rom(combined_snapshots, 'pod', **kwargs)
        else:
            raise QaeroError(f"Unsupported adaptation method: {method}")
    
    def _incremental_adaptation(self, rom_result: ROMResult, 
                              new_snapshots: np.ndarray, **kwargs) -> ROMResult:
        """Incremental ROM adaptation."""
        # Simplified incremental SVD
        U_old = rom_result.reduced_basis
        s_old = rom_result.singular_values
        
        # Project new snapshots onto existing basis
        coefficients = U_old.T @ new_snapshots
        residuals = new_snapshots - U_old @ coefficients
        
        # QR decomposition of residuals to find new modes
        Q, R = np.linalg.qr(residuals, mode='reduced')
        
        # Combine old and new basis
        U_combined = np.hstack([U_old, Q])
        
        # Update singular values (simplified)
        s_combined = np.concatenate([s_old, np.linalg.norm(residuals, axis=0)])
        
        return ROMResult(
            reduced_basis=U_combined,
            singular_values=s_combined,
            projection_matrix=U_combined.T,
            reconstruction_error=rom_result.reconstruction_error,
            energy_captured=rom_result.energy_captured,
            metadata={**rom_result.metadata, 'adapted': True, 'n_new_modes': Q.shape[1]}
        )

# Reduction methods
class ROMReducer(ABC):
    """Abstract ROM reducer."""
    
    @abstractmethod
    def reduce(self, snapshots: np.ndarray, **kwargs) -> ROMResult:
        """Reduce snapshot data to low-dimensional representation."""
        pass

class PODReducer(ROMReducer):
    """Proper Orthogonal Decomposition reducer."""
    
    def reduce(self, snapshots: np.ndarray, **kwargs) -> ROMResult:
        """Perform POD reduction."""
        # Remove mean
        mean_snapshot = np.mean(snapshots, axis=1, keepdims=True)
        fluctuations = snapshots - mean_snapshot
        
        # Singular Value Decomposition
        U, s, Vt = svd(fluctuations, full_matrices=False)
        
        # Determine truncation rank
        rank = self._determine_rank(s, kwargs)
        U_r = U[:, :rank]
        s_r = s[:rank]
        
        # Compute reconstruction error
        reconstruction_error = self._compute_reconstruction_error(fluctuations, U_r, s_r, Vt[:rank, :])
        energy_captured = np.sum(s_r**2) / np.sum(s**2)
        
        return ROMResult(
            reduced_basis=U_r,
            singular_values=s_r,
            projection_matrix=U_r.T,
            reconstruction_error=reconstruction_error,
            energy_captured=energy_captured,
            metadata={
                'method': 'POD',
                'rank': rank,
                'mean_snapshot': mean_snapshot.flatten(),
                'total_energy': np.sum(s**2),
                'captured_energy': np.sum(s_r**2)
            }
        )
    
    def _determine_rank(self, singular_values: np.ndarray, kwargs: Dict) -> int:
        """Determine optimal truncation rank."""
        energy_threshold = kwargs.get('energy_threshold', 0.99)
        fixed_rank = kwargs.get('rank', None)
        
        if fixed_rank is not None:
            return min(fixed_rank, len(singular_values))
        
        # Energy-based truncation
        cumulative_energy = np.cumsum(singular_values**2) / np.sum(singular_values**2)
        rank = np.argmax(cumulative_energy >= energy_threshold) + 1
        return min(rank, len(singular_values))
    
    def _compute_reconstruction_error(self, data: np.ndarray, U: np.ndarray, 
                                    s: np.ndarray, Vt: np.ndarray) -> float:
        """Compute reconstruction error."""
        reconstructed = U @ np.diag(s) @ Vt
        error = np.linalg.norm(data - reconstructed, 'fro') / np.linalg.norm(data, 'fro')
        return error

class DMDReducer(ROMReducer):
    """Dynamic Mode Decomposition reducer."""
    
    def reduce(self, snapshots: np.ndarray, **kwargs) -> ROMResult:
        """Perform DMD reduction."""
        if snapshots.shape[1] < 2:
            raise QaeroError("DMD requires at least 2 snapshots for time series")
        
        # Split into X and Y matrices
        X = snapshots[:, :-1]
        Y = snapshots[:, 1:]
        
        # SVD of X
        U, s, Vt = svd(X, full_matrices=False)
        
        # Determine rank
        rank = self._determine_rank(s, kwargs)
        U_r = U[:, :rank]
        s_r = s[:rank]
        Vt_r = Vt[:rank, :]
        
        # Compute DMD operator
        A_tilde = U_r.T @ Y @ Vt_r.T @ np.diag(1.0 / s_r)
        
        # Eigen decomposition of A_tilde
        eigenvalues, eigenvectors = eig(A_tilde)
        
        # DMD modes
        dmd_modes = Y @ Vt_r.T @ np.diag(1.0 / s_r) @ eigenvectors
        
        # Normalize modes
        dmd_modes = dmd_modes / np.linalg.norm(dmd_modes, axis=0)
        
        return ROMResult(
            reduced_basis=dmd_modes,
            singular_values=np.abs(eigenvalues),  # Use eigenvalues as "importance"
            projection_matrix=dmd_modes.T,
            reconstruction_error=self._compute_dmd_error(X, Y, dmd_modes, eigenvalues),
            energy_captured=np.sum(np.abs(eigenvalues)) / len(eigenvalues),
            metadata={
                'method': 'DMD',
                'rank': rank,
                'dmd_eigenvalues': eigenvalues,
                'is_stable': np.all(np.abs(eigenvalues) <= 1.0)
            }
        )
    
    def _compute_dmd_error(self, X: np.ndarray, Y: np.ndarray, 
                          modes: np.ndarray, eigenvalues: np.ndarray) -> float:
        """Compute DMD reconstruction error."""
        # Simplified error computation
        reconstructed = modes @ np.diag(eigenvalues) @ (modes.T @ X)
        error = np.linalg.norm(Y - reconstructed, 'fro') / np.linalg.norm(Y, 'fro')
        return error

class AutoencoderReducer(ROMReducer):
    """Neural network autoencoder reducer."""
    
    def reduce(self, snapshots: np.ndarray, **kwargs) -> ROMResult:
        """Perform autoencoder reduction."""
        encoding_dim = kwargs.get('encoding_dim', 10)
        
        try:
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
            
            # Normalize data
            scaler = StandardScaler()
            snapshots_normalized = scaler.fit_transform(snapshots.T).T
            
            # Build autoencoder (simplified - using MLP)
            # Input -> encoding -> output
            hidden_dim = max(encoding_dim * 2, 50)
            
            autoencoder = MLPRegressor(
                hidden_layer_sizes=(hidden_dim, encoding_dim, hidden_dim),
                activation='tanh',
                max_iter=1000,
                random_state=42
            )
            
            # Train autoencoder (input = output)
            autoencoder.fit(snapshots_normalized.T, snapshots_normalized.T)
            
            # Extract encoder (first half of network)
            # This is simplified - in practice would build separate encoder/decoder
            encoder_weights = autoencoder.coefs_[0]
            reduced_basis = encoder_weights.T  # Use weights as basis
            
            # Compute reconstruction
            reconstructed = autoencoder.predict(snapshots_normalized.T).T
            reconstruction_error = np.linalg.norm(snapshots_normalized - reconstructed) / np.linalg.norm(snapshots_normalized)
            
            return ROMResult(
                reduced_basis=reduced_basis,
                singular_values=np.ones(encoding_dim),  # Placeholder
                projection_matrix=reduced_basis.T,
                reconstruction_error=reconstruction_error,
                energy_captured=1.0 - reconstruction_error,
                metadata={
                    'method': 'Autoencoder',
                    'encoding_dim': encoding_dim,
                    'hidden_dim': hidden_dim,
                    'scaler': scaler
                }
            )
            
        except ImportError:
            logger.warning("scikit-learn not available, using POD fallback")
            return PODReducer().reduce(snapshots, **kwargs)

class ProperInterpolationReducer(ROMReducer):
    """Proper Interpolation methods like DEIM and Q-DEIM."""
    
    def reduce(self, snapshots: np.ndarray, **kwargs) -> ROMResult:
        """Perform proper interpolation reduction."""
        # First get POD basis
        pod_result = PODReducer().reduce(snapshots, **kwargs)
        U_r = pod_result.reduced_basis
        
        # Apply DEIM algorithm to select interpolation points
        n_points = kwargs.get('n_interpolation_points', U_r.shape[1])
        interpolation_indices = self._deim_algorithm(U_r, n_points)
        
        # Build interpolation matrix
        P = np.eye(U_r.shape[0])[interpolation_indices, :]
        
        # Reduced basis through interpolation
        U_deim = U_r[interpolation_indices, :]
        
        return ROMResult(
            reduced_basis=U_r,
            singular_values=pod_result.singular_values,
            projection_matrix=P.T @ np.linalg.pinv(U_deim.T),  # Interpolation projection
            reconstruction_error=pod_result.reconstruction_error,
            energy_captured=pod_result.energy_captured,
            metadata={
                'method': 'DEIM',
                'interpolation_indices': interpolation_indices,
                'n_interpolation_points': n_points,
                'pod_rank': U_r.shape[1]
            }
        )
    
    def _deim_algorithm(self, basis: np.ndarray, n_points: int) -> np.ndarray:
        """Discrete Empirical Interpolation Method (DEIM)."""
        n, r = basis.shape
        indices = []
        
        # Find first index: maximum of first basis vector
        first_basis = basis[:, 0]
        indices.append(np.argmax(np.abs(first_basis)))
        
        for j in range(1, min(n_points, r)):
            # Solve for coefficients using selected indices
            selected_basis = basis[indices, :j]
            current_basis = basis[:, j]
            
            # Solve least squares problem
            coefficients = np.linalg.lstsq(selected_basis.T, current_basis[indices], rcond=None)[0]
            
            # Compute residual
            approximation = basis[:, :j] @ coefficients
            residual = current_basis - approximation
            
            # Select next point with maximum residual
            new_index = np.argmax(np.abs(residual))
            indices.append(new_index)
        
        return np.array(indices)

# Quality metrics
class ROMMetric(ABC):
    """Abstract ROM quality metric."""
    
    @abstractmethod
    def evaluate(self, rom_result: ROMResult, test_data: np.ndarray) -> float:
        """Evaluate ROM quality."""
        pass

class ReconstructionErrorMetric(ROMMetric):
    """Reconstruction error metric."""
    
    def evaluate(self, rom_result: ROMResult, test_data: np.ndarray) -> float:
        """Evaluate reconstruction error on test data."""
        reconstructed = rom_result.reduced_basis @ (rom_result.projection_matrix @ test_data)
        error = np.linalg.norm(test_data - reconstructed, 'fro') / np.linalg.norm(test_data, 'fro')
        return error

class PredictionErrorMetric(ROMMetric):
    """Prediction error metric for dynamical systems."""
    
    def evaluate(self, rom_result: ROMResult, test_data: np.ndarray) -> float:
        """Evaluate prediction error."""
        # For dynamical systems, predict next state
        if test_data.shape[1] < 2:
            return float('inf')
        
        # Use first state to predict subsequent states
        initial_state = test_data[:, 0]
        predicted_states = [initial_state]
        
        for i in range(1, test_data.shape[1]):
            # Simplified prediction: linear evolution in reduced space
            reduced_state = rom_result.projection_matrix @ predicted_states[-1]
            next_reduced = reduced_state  # Identity evolution (simplified)
            next_state = rom_result.reduced_basis @ next_reduced
            predicted_states.append(next_state)
        
        predicted = np.column_stack(predicted_states)
        error = np.linalg.norm(test_data - predicted, 'fro') / np.linalg.norm(test_data, 'fro')
        return error

class StabilityMetric(ROMMetric):
    """Stability metric for ROM."""
    
    def evaluate(self, rom_result: ROMResult, test_data: np.ndarray) -> float:
        """Evaluate numerical stability."""
        # Check condition number of reduced operator
        if hasattr(rom_result, 'reduced_operator'):
            cond_number = np.linalg.cond(rom_result.reduced_operator)
        else:
            # Use projection matrix condition number
            cond_number = np.linalg.cond(rom_result.projection_matrix @ rom_result.reduced_basis)
        
        # Lower condition number is better (more stable)
        stability_score = 1.0 / (1.0 + np.log10(cond_number))
        return stability_score