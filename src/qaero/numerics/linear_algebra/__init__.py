"""
Linear algebra pipeline for quantum-ready matrix processing.
Sparsity analysis, block decomposition, low-rank compression, and preconditioning.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix, block_diag
from scipy.sparse.linalg import spsolve, eigsh
from scipy.linalg import svd, lu, qr
import warnings

from ....core.base import QaeroError
from ....core.registry import register_service

logger = logging.getLogger("qaero.numerics.linear_algebra")

@dataclass
class MatrixAnalysis:
    """Matrix analysis results."""
    condition_number: float
    sparsity_pattern: Dict[str, float]
    eigenvalue_range: Tuple[float, float]
    rank: int
    symmetry: str
    properties: Dict[str, Any]

@dataclass  
class Preconditioner:
    """Preconditioner representation."""
    type: str
    matrix: Any
    apply: Callable
    metadata: Dict[str, Any]

@register_service("linear_algebra_pipeline")
class LinearAlgebraPipeline:
    """
    Complete linear algebra pipeline for quantum-classical hybrid solving.
    Handles matrix analysis, decomposition, compression, and preconditioning.
    """
    
    def __init__(self):
        self.analyzers = {
            'sparsity': SparsityAnalyzer(),
            'eigenvalue': EigenvalueAnalyzer(),
            'condition': ConditionAnalyzer()
        }
        self.decomposers = {
            'block': BlockDecomposer(),
            'low_rank': LowRankDecomposer(),
            'tensor': TensorTrainDecomposer()
        }
        self.preconditioners = {
            'jacobi': JacobiPreconditioner(),
            'ilu': ILUPreconditioner(),
            'multigrid': MultigridPreconditioner(),
            'deflation': DeflationPreconditioner()
        }
    
    def analyze_matrix(self, A: Union[np.ndarray, sparse.spmatrix], 
                      analyses: List[str] = None) -> MatrixAnalysis:
        """Comprehensive matrix analysis."""
        if analyses is None:
            analyses = ['sparsity', 'condition', 'eigenvalue']
        
        results = {}
        for analysis in analyses:
            if analysis in self.analyzers:
                analyzer = self.analyzers[analysis]
                results.update(analyzer.analyze(A))
        
        return MatrixAnalysis(**results)
    
    def decompose_matrix(self, A: Union[np.ndarray, sparse.spmatrix], 
                        method: str = "block", **kwargs) -> Any:
        """Decompose matrix for efficient solving."""
        if method not in self.decomposers:
            raise QaeroError(f"Unsupported decomposition method: {method}")
        
        decomposer = self.decomposers[method]
        return decomposer.decompose(A, **kwargs)
    
    def build_preconditioner(self, A: Union[np.ndarray, sparse.spmatrix],
                           preconditioner_type: str = "jacobi", **kwargs) -> Preconditioner:
        """Build preconditioner for matrix."""
        if preconditioner_type not in self.preconditioners:
            raise QaeroError(f"Unsupported preconditioner type: {preconditioner_type}")
        
        preconditioner = self.preconditioners[preconditioner_type]
        return preconditioner.build(A, **kwargs)
    
    def prepare_for_quantum(self, A: Union[np.ndarray, sparse.spmatrix],
                          b: np.ndarray, method: str = "vqls", **kwargs) -> Tuple[Any, Any]:
        """Prepare linear system for quantum solvers."""
        if method == "vqls":
            return self._prepare_for_vqls(A, b, **kwargs)
        elif method == "hhl":
            return self._prepare_for_hhl(A, b, **kwargs)
        elif method == "quantum_linear_solver":
            return self._prepare_for_general_quantum(A, b, **kwargs)
        else:
            raise QaeroError(f"Unsupported quantum method: {method}")
    
    def _prepare_for_vqls(self, A: Union[np.ndarray, sparse.spmatrix], 
                         b: np.ndarray, **kwargs) -> Tuple[Any, Any]:
        """Prepare system for Variational Quantum Linear Solver."""
        # Normalize system
        A_norm, b_norm = self._normalize_system(A, b)
        
        # Apply preconditioning if requested
        if kwargs.get('precondition', True):
            preconditioner = self.build_preconditioner(A_norm, 'jacobi')
            A_precond = preconditioner.apply(A_norm)
            b_precond = preconditioner.apply(b_norm.reshape(-1, 1)).flatten()
        else:
            A_precond, b_precond = A_norm, b_norm
        
        # Ensure Hermitian positive definite for VQLS
        if not self._is_hermitian_positive_definite(A_precond):
            A_precond = self._make_hermitian_positive_definite(A_precond)
        
        return A_precond, b_precond
    
    def _prepare_for_hhl(self, A: Union[np.ndarray, sparse.spmatrix],
                        b: np.ndarray, **kwargs) -> Tuple[Any, Any]:
        """Prepare system for HHL algorithm."""
        # HHL requires specific matrix properties
        A_hhl, b_hhl = self._normalize_system(A, b)
        
        # Ensure matrix is Hermitian
        if not self._is_hermitian(A_hhl):
            A_hhl = self._make_hermitian(A_hhl)
        
        # Scale for eigenvalue range suitable for quantum phase estimation
        A_hhl = self._scale_for_quantum_phase_estimation(A_hhl)
        
        return A_hhl, b_hhl
    
    def _prepare_for_general_quantum(self, A: Union[np.ndarray, sparse.spmatrix],
                                   b: np.ndarray, **kwargs) -> Tuple[Any, Any]:
        """General preparation for quantum linear solvers."""
        A_prep, b_prep = self._normalize_system(A, b)
        
        # Reduce condition number
        if kwargs.get('reduce_condition', True):
            A_prep = self._reduce_condition_number(A_prep)
        
        return A_prep, b_prep
    
    def _normalize_system(self, A: Union[np.ndarray, sparse.spmatrix], 
                         b: np.ndarray) -> Tuple[Any, Any]:
        """Normalize linear system for numerical stability."""
        # Scale matrix and RHS
        if sparse.issparse(A):
            A_dense = A.toarray()
        else:
            A_dense = A
        
        # Scale by matrix norm
        norm_A = np.linalg.norm(A_dense, ord=2)
        if norm_A > 0:
            A_norm = A_dense / norm_A
            b_norm = b / norm_A
        else:
            A_norm, b_norm = A_dense, b
        
        return A_norm, b_norm
    
    def _is_hermitian(self, A: Union[np.ndarray, sparse.spmatrix]) -> bool:
        """Check if matrix is Hermitian."""
        if sparse.issparse(A):
            A_dense = A.toarray()
        else:
            A_dense = A
        
        return np.allclose(A_dense, A_dense.conj().T)
    
    def _is_hermitian_positive_definite(self, A: Union[np.ndarray, sparse.spmatrix]) -> bool:
        """Check if matrix is Hermitian positive definite."""
        if not self._is_hermitian(A):
            return False
        
        if sparse.issparse(A):
            try:
                # Check for positive eigenvalues
                eigvals = eigsh(A, k=min(5, A.shape[0]-1), which='SA', return_eigenvectors=False)
                return np.all(eigvals > 0)
            except:
                return False
        else:
            try:
                return np.all(np.linalg.eigvals(A) > 0)
            except:
                return False
    
    def _make_hermitian(self, A: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
        """Make matrix Hermitian."""
        if sparse.issparse(A):
            A_dense = A.toarray()
        else:
            A_dense = A
        
        return 0.5 * (A_dense + A_dense.conj().T)
    
    def _make_hermitian_positive_definite(self, A: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
        """Make matrix Hermitian positive definite."""
        A_herm = self._make_hermitian(A)
        
        # Add small diagonal to ensure positive definiteness
        n = A_herm.shape[0]
        A_pd = A_herm + 1e-8 * np.eye(n)
        
        return A_pd
    
    def _scale_for_quantum_phase_estimation(self, A: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
        """Scale matrix for quantum phase estimation eigenvalue range."""
        if sparse.issparse(A):
            A_dense = A.toarray()
        else:
            A_dense = A
        
        # Scale eigenvalues to [0, 1] range
        max_eig = np.max(np.abs(np.linalg.eigvals(A_dense)))
        if max_eig > 0:
            return A_dense / (2 * max_eig)
        else:
            return A_dense
    
    def _reduce_condition_number(self, A: Union[np.ndarray, sparse.spmatrix]) -> np.ndarray:
        """Reduce condition number of matrix."""
        if sparse.issparse(A):
            A_dense = A.toarray()
        else:
            A_dense = A
        
        # Simple diagonal scaling
        n = A_dense.shape[0]
        D = np.diag(1.0 / np.sqrt(np.abs(np.diag(A_dense)) + 1e-10))
        A_scaled = D @ A_dense @ D
        
        return A_scaled

# Analysis components
class MatrixAnalyzer(ABC):
    """Abstract matrix analyzer."""
    
    @abstractmethod
    def analyze(self, A: Union[np.ndarray, sparse.spmatrix]) -> Dict[str, Any]:
        """Analyze matrix properties."""
        pass

class SparsityAnalyzer(MatrixAnalyzer):
    """Analyze matrix sparsity pattern."""
    
    def analyze(self, A: Union[np.ndarray, sparse.spmatrix]) -> Dict[str, Any]:
        """Analyze sparsity properties."""
        if sparse.issparse(A):
            n_rows, n_cols = A.shape
            nnz = A.nnz
            density = nnz / (n_rows * n_cols)
            
            # Sparsity pattern analysis
            row_nnz = np.diff(A.indptr)
            avg_nnz_per_row = np.mean(row_nnz)
            max_nnz_per_row = np.max(row_nnz)
            
            return {
                'sparsity_pattern': {
                    'density': density,
                    'nnz': nnz,
                    'avg_nnz_per_row': avg_nnz_per_row,
                    'max_nnz_per_row': max_nnz_per_row,
                    'sparsity_ratio': 1 - density
                }
            }
        else:
            # Dense matrix
            n_rows, n_cols = A.shape
            density = 1.0
            nnz = n_rows * n_cols
            
            return {
                'sparsity_pattern': {
                    'density': density,
                    'nnz': nnz,
                    'avg_nnz_per_row': n_cols,
                    'max_nnz_per_row': n_cols,
                    'sparsity_ratio': 0.0
                }
            }

class EigenvalueAnalyzer(MatrixAnalyzer):
    """Analyze matrix eigenvalues."""
    
    def analyze(self, A: Union[np.ndarray, sparse.spmatrix]) -> Dict[str, Any]:
        """Analyze eigenvalue properties."""
        try:
            if sparse.issparse(A):
                # Compute a few eigenvalues for large sparse matrices
                k = min(10, A.shape[0] - 1)
                if k > 0:
                    eigvals = eigsh(A, k=k, which='LM', return_eigenvectors=False)
                    min_eig, max_eig = np.min(eigvals), np.max(eigvals)
                else:
                    min_eig, max_eig = 1.0, 1.0
            else:
                # Full eigenvalue decomposition for small dense matrices
                eigvals = np.linalg.eigvals(A)
                min_eig, max_eig = np.min(eigvals), np.max(eigvals)
            
            return {
                'eigenvalue_range': (float(min_eig), float(max_eig)),
                'properties': {
                    'has_negative_eigenvalues': min_eig < 0,
                    'spectral_radius': float(max_eig)
                }
            }
        except Exception as e:
            logger.warning(f"Eigenvalue analysis failed: {e}")
            return {
                'eigenvalue_range': (1.0, 1.0),
                'properties': {'analysis_failed': True}
            }

class ConditionAnalyzer(MatrixAnalyzer):
    """Analyze matrix condition number."""
    
    def analyze(self, A: Union[np.ndarray, sparse.spmatrix]) -> Dict[str, Any]:
        """Analyze condition number and related properties."""
        try:
            if sparse.issparse(A):
                A_dense = A.toarray()
            else:
                A_dense = A
            
            cond_number = np.linalg.cond(A_dense)
            rank = np.linalg.matrix_rank(A_dense)
            
            # Check symmetry
            is_symmetric = np.allclose(A_dense, A_dense.T)
            symmetry_type = "symmetric" if is_symmetric else "non-symmetric"
            
            return {
                'condition_number': float(cond_number),
                'rank': rank,
                'symmetry': symmetry_type,
                'properties': {
                    'is_singular': rank < A_dense.shape[0],
                    'log_condition': np.log10(cond_number) if cond_number > 0 else 0
                }
            }
        except Exception as e:
            logger.warning(f"Condition analysis failed: {e}")
            return {
                'condition_number': 1.0,
                'rank': A.shape[0],
                'symmetry': 'unknown',
                'properties': {'analysis_failed': True}
            }

# Decomposition components
class MatrixDecomposer(ABC):
    """Abstract matrix decomposer."""
    
    @abstractmethod
    def decompose(self, A: Union[np.ndarray, sparse.spmatrix], **kwargs) -> Any:
        """Decompose matrix."""
        pass

class BlockDecomposer(MatrixDecomposer):
    """Block matrix decomposition."""
    
    def decompose(self, A: Union[np.ndarray, sparse.spmatrix], **kwargs) -> Dict[str, Any]:
        """Decompose matrix into blocks."""
        n_blocks = kwargs.get('n_blocks', 2)
        
        if sparse.issparse(A):
            A_dense = A.toarray()
        else:
            A_dense = A
        
        n = A_dense.shape[0]
        block_size = n // n_blocks
        
        blocks = {}
        for i in range(n_blocks):
            for j in range(n_blocks):
                start_i = i * block_size
                end_i = (i + 1) * block_size if i < n_blocks - 1 else n
                start_j = j * block_size
                end_j = (j + 1) * block_size if j < n_blocks - 1 else n
                
                block = A_dense[start_i:end_i, start_j:end_j]
                blocks[f'block_{i}_{j}'] = block
        
        return {
            'blocks': blocks,
            'block_structure': (n_blocks, n_blocks),
            'block_sizes': [(block.shape[0], block.shape[1]) for block in blocks.values()]
        }

class LowRankDecomposer(MatrixDecomposer):
    """Low-rank matrix decomposition."""
    
    def decompose(self, A: Union[np.ndarray, sparse.spmatrix], **kwargs) -> Dict[str, Any]:
        """Perform low-rank approximation."""
        rank = kwargs.get('rank', 10)
        tolerance = kwargs.get('tolerance', 1e-6)
        
        if sparse.issparse(A):
            A_dense = A.toarray()
        else:
            A_dense = A
        
        # SVD decomposition
        U, s, Vt = svd(A_dense, full_matrices=False)
        
        # Truncate to specified rank
        effective_rank = min(rank, len(s))
        if tolerance is not None:
            # Determine rank based on tolerance
            cumulative_energy = np.cumsum(s) / np.sum(s)
            effective_rank = np.argmax(cumulative_energy > 1 - tolerance) + 1
            effective_rank = min(effective_rank, rank)
        
        U_r = U[:, :effective_rank]
        s_r = s[:effective_rank]
        Vt_r = Vt[:effective_rank, :]
        
        # Low-rank approximation
        A_approx = U_r @ np.diag(s_r) @ Vt_r
        approximation_error = np.linalg.norm(A_dense - A_approx) / np.linalg.norm(A_dense)
        
        return {
            'U': U_r,
            's': s_r,
            'Vt': Vt_r,
            'approximation': A_approx,
            'rank': effective_rank,
            'approximation_error': approximation_error,
            'energy_captured': np.sum(s_r) / np.sum(s)
        }

class TensorTrainDecomposer(MatrixDecomposer):
    """Tensor Train decomposition for high-dimensional problems."""
    
    def decompose(self, A: Union[np.ndarray, sparse.spmatrix], **kwargs) -> Dict[str, Any]:
        """Perform Tensor Train decomposition (simplified)."""
        # Simplified TT decomposition for 2D matrices
        # In practice, would use proper TT decomposition for high-dimensional tensors
        
        if sparse.issparse(A):
            A_dense = A.toarray()
        else:
            A_dense = A
        
        # Reshape to higher dimensions if possible
        original_shape = A_dense.shape
        if len(original_shape) == 2:
            # Try to factor dimensions
            n, m = original_shape
            # Find factors
            factors = self._find_factors(n, m)
            
            if factors:
                d1, d2 = factors
                # Reshape to 4D tensor
                tensor = A_dense.reshape(d1, d2, d1, d2)
                
                # Simplified TT decomposition
                tt_cores = self._simplified_tt_decomposition(tensor)
                
                return {
                    'tt_cores': tt_cores,
                    'tensor_shape': (d1, d2, d1, d2),
                    'original_shape': original_shape,
                    'method': 'tensor_train_simplified'
                }
        
        # Fallback: treat as matrix
        return {
            'tt_cores': [A_dense],
            'tensor_shape': original_shape,
            'original_shape': original_shape,
            'method': 'matrix_fallback'
        }
    
    def _find_factors(self, n: int, m: int) -> Tuple[int, int]:
        """Find factors for reshaping."""
        # Find d1, d2 such that d1 * d2 = n and d1 * d2 = m
        # For square matrices, we want n = d1 * d2 and m = d1 * d2
        if n == m:
            # Find factors of n
            for i in range(int(np.sqrt(n)), 0, -1):
                if n % i == 0:
                    return i, n // i
        return None
    
    def _simplified_tt_decomposition(self, tensor: np.ndarray) -> List[np.ndarray]:
        """Simplified Tensor Train decomposition."""
        # This is a simplified version for demonstration
        # In practice, would use proper TT-SVD algorithm
        
        shape = tensor.shape
        cores = []
        
        # First core
        core1 = tensor.reshape(shape[0], -1)
        U, s, Vt = svd(core1, full_matrices=False)
        rank = min(10, len(s))
        cores.append(U[:, :rank])
        
        # Middle core
        middle = np.diag(s[:rank]) @ Vt[:rank, :]
        middle_reshaped = middle.reshape(rank * shape[1], -1)
        U2, s2, Vt2 = svd(middle_reshaped, full_matrices=False)
        rank2 = min(10, len(s2))
        cores.append(U2.reshape(rank, shape[1], rank2))
        
        # Last core
        last = np.diag(s2[:rank2]) @ Vt2[:rank2, :]
        cores.append(last.reshape(rank2, shape[2], shape[3]))
        
        return cores

# Preconditioner components
class PreconditionerBuilder(ABC):
    """Abstract preconditioner builder."""
    
    @abstractmethod
    def build(self, A: Union[np.ndarray, sparse.spmatrix], **kwargs) -> Preconditioner:
        """Build preconditioner."""
        pass

class JacobiPreconditioner(PreconditionerBuilder):
    """Jacobi (diagonal) preconditioner."""
    
    def build(self, A: Union[np.ndarray, sparse.spmatrix], **kwargs) -> Preconditioner:
        """Build Jacobi preconditioner."""
        if sparse.issparse(A):
            # Extract diagonal
            diag = A.diagonal()
            # Avoid division by zero
            diag_inv = 1.0 / np.where(diag != 0, diag, 1.0)
            M_inv = sparse.diags(diag_inv)
            
            def apply(x):
                return M_inv @ x
        else:
            diag = np.diag(A)
            diag_inv = 1.0 / np.where(diag != 0, diag, 1.0)
            M_inv = np.diag(diag_inv)
            
            def apply(x):
                return M_inv @ x
        
        return Preconditioner(
            type='jacobi',
            matrix=M_inv,
            apply=apply,
            metadata={'diagonal_elements': len(diag)}
        )

class ILUPreconditioner(PreconditionerBuilder):
    """Incomplete LU preconditioner."""
    
    def build(self, A: Union[np.ndarray, sparse.spmatrix], **kwargs) -> Preconditioner:
        """Build ILU preconditioner."""
        try:
            from scipy.sparse.linalg import spilu
            
            # ILU decomposition
            ilu = spilu(A.tocsc() if sparse.issparse(A) else sparse.csc_matrix(A))
            
            def apply(x):
                return ilu.solve(x)
            
            return Preconditioner(
                type='ilu',
                matrix=ilu,
                apply=apply,
                metadata={'fill_factor': kwargs.get('fill_factor', 10)}
            )
        except Exception as e:
            logger.warning(f"ILU preconditioner failed: {e}, using Jacobi fallback")
            return JacobiPreconditioner().build(A, **kwargs)

class MultigridPreconditioner(PreconditionerBuilder):
    """Multigrid preconditioner."""
    
    def build(self, A: Union[np.ndarray, sparse.spmatrix], **kwargs) -> Preconditioner:
        """Build multigrid preconditioner (simplified)."""
        # Simplified multigrid - in practice would use proper multigrid cycles
        
        def apply(x):
            # Simple smoothing operation
            if sparse.issparse(A):
                return self._smooth(A, x, iterations=3)
            else:
                return self._smooth_dense(A, x, iterations=3)
        
        return Preconditioner(
            type='multigrid',
            matrix=None,
            apply=apply,
            metadata={'smoothing_iterations': 3, 'levels': 2}
        )
    
    def _smooth(self, A: sparse.spmatrix, x: np.ndarray, iterations: int) -> np.ndarray:
        """Gauss-Seidel smoothing for sparse matrices."""
        x_smooth = x.copy()
        for _ in range(iterations):
            for i in range(A.shape[0]):
                # Extract row
                start, end = A.indptr[i], A.indptr[i+1]
                row_indices = A.indices[start:end]
                row_data = A.data[start:end]
                
                # Gauss-Seidel update
                diagonal = 0.0
                off_diag_sum = 0.0
                for j, a_ij in zip(row_indices, row_data):
                    if j == i:
                        diagonal = a_ij
                    else:
                        off_diag_sum += a_ij * x_smooth[j]
                
                if diagonal != 0:
                    x_smooth[i] = (x[i] - off_diag_sum) / diagonal
        
        return x_smooth
    
    def _smooth_dense(self, A: np.ndarray, x: np.ndarray, iterations: int) -> np.ndarray:
        """Gauss-Seidel smoothing for dense matrices."""
        x_smooth = x.copy()
        n = A.shape[0]
        
        for _ in range(iterations):
            for i in range(n):
                off_diag_sum = np.dot(A[i, :i], x_smooth[:i]) + np.dot(A[i, i+1:], x_smooth[i+1:])
                if A[i, i] != 0:
                    x_smooth[i] = (x[i] - off_diag_sum) / A[i, i]
        
        return x_smooth

class DeflationPreconditioner(PreconditionerBuilder):
    """Deflation preconditioner for handling small eigenvalues."""
    
    def build(self, A: Union[np.ndarray, sparse.spmatrix], **kwargs) -> Preconditioner:
        """Build deflation preconditioner."""
        n_deflate = kwargs.get('n_deflate', 5)
        
        try:
            if sparse.issparse(A):
                # Compute smallest eigenvalues and eigenvectors
                eigvals, eigvecs = eigsh(A, k=n_deflate, which='SM')
            else:
                # Full eigendecomposition for small matrices
                eigvals, eigvecs = np.linalg.eigh(A)
                eigvecs = eigvecs[:, :n_deflate]
                eigvals = eigvals[:n_deflate]
            
            # Deflation subspace
            Z = eigvecs
            
            def apply(x):
                # Deflation operation: (I - Z(Z^T A Z)^{-1} Z^T A) x
                AZ = A @ Z
                ZTAZ = Z.T @ AZ
                ZTAZ_inv = np.linalg.inv(ZTAZ)
                deflation = Z @ ZTAZ_inv @ (Z.T @ x)
                return x - deflation
            
            return Preconditioner(
                type='deflation',
                matrix=Z,
                apply=apply,
                metadata={'n_deflated_modes': n_deflate, 'deflated_eigenvalues': eigvals.tolist()}
            )
        except Exception as e:
            logger.warning(f"Deflation preconditioner failed: {e}, using Jacobi fallback")
            return JacobiPreconditioner().build(A, **kwargs)