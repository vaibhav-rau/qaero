"""
Uncertainty quantification and data assimilation with quantum-enhanced methods.
Implements Bayesian hybrid pipelines, quantum sampling, and inverse problem solvers.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize
from scipy.linalg import solve, inv
import warnings

from ...core.base import OptimizationProblem, PDEProblem, QaeroError
from ...core.results import OptimizationResult, PDEResult
from ...core.registry import register_algorithm

logger = logging.getLogger("qaero.algorithms.uncertainty")

@dataclass
class BayesianConfig:
    """Configuration for Bayesian uncertainty quantification."""
    method: str = "mcmc"  # "mcmc", "hmc", "quantum_sampling", "variational_bayes"
    n_samples: int = 1000
    burn_in: int = 100
    proposal_scale: float = 0.1
    use_quantum_sampler: bool = False
    quantum_shots: int = 1024

@dataclass
class UncertaintyResult:
    """Results from uncertainty quantification."""
    posterior_mean: np.ndarray
    posterior_std: np.ndarray
    posterior_samples: np.ndarray
    credibility_intervals: Dict[str, Tuple[float, float]]
    evidence: float
    convergence_diagnostics: Dict[str, float]

@register_algorithm("bayesian_hybrid", "uncertainty_quantification")
class BayesianHybridOptimizer:
    """
    Bayesian hybrid pipelines with quantum sampling for combinatorial components
    and classical MCMC/HMC for continuous parts.
    """
    
    def __init__(self, config: Optional[Union[BayesianConfig, Dict]] = None):
        if config is None:
            self.config = BayesianConfig()
        elif isinstance(config, dict):
            self.config = BayesianConfig(**config)
        else:
            self.config = config
        
        self.prior_distributions = {}
        self.likelihood_function = None
        self.quantum_sampler = None
        
    def set_prior(self, parameter: str, distribution: Callable):
        """Set prior distribution for a parameter."""
        self.prior_distributions[parameter] = distribution
    
    def set_likelihood(self, likelihood_func: Callable):
        """Set likelihood function."""
        self.likelihood_function = likelihood_func
    
    def quantify_uncertainty(self, problem: OptimizationProblem, data: np.ndarray, **kwargs) -> UncertaintyResult:
        """Perform Bayesian uncertainty quantification."""
        import time
        start_time = time.time()
        
        try:
            # Setup posterior distribution
            posterior = self._setup_posterior(problem, data)
            
            # Sample from posterior
            if self.config.method == "mcmc":
                samples = self._mcmc_sampling(posterior, problem)
            elif self.config.method == "hmc":
                samples = self._hmc_sampling(posterior, problem)
            elif self.config.method == "quantum_sampling":
                samples = self._quantum_sampling(posterior, problem)
            elif self.config.method == "variational_bayes":
                samples = self._variational_bayes(posterior, problem)
            else:
                raise QaeroError(f"Unknown Bayesian method: {self.config.method}")
            
            # Compute posterior statistics
            posterior_mean = np.mean(samples, axis=0)
            posterior_std = np.std(samples, axis=0)
            
            # Credibility intervals (95%)
            credibility_intervals = {}
            for i, var in enumerate(problem.variables):
                lower = np.percentile(samples[:, i], 2.5)
                upper = np.percentile(samples[:, i], 97.5)
                credibility_intervals[var] = (float(lower), float(upper))
            
            # Compute evidence (marginal likelihood)
            evidence = self._compute_evidence(posterior, samples)
            
            # Convergence diagnostics
            convergence = self._compute_convergence(samples)
            
            result = UncertaintyResult(
                posterior_mean=posterior_mean,
                posterior_std=posterior_std,
                posterior_samples=samples,
                credibility_intervals=credibility_intervals,
                evidence=evidence,
                convergence_diagnostics=convergence
            )
            
            logger.info(f"Bayesian UQ completed in {time.time() - start_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Bayesian uncertainty quantification failed: {e}")
            raise
    
    def _setup_posterior(self, problem: OptimizationProblem, data: np.ndarray) -> Callable:
        """Setup posterior distribution function."""
        def posterior(x):
            # Prior term
            log_prior = 0.0
            for i, var in enumerate(problem.variables):
                if var in self.prior_distributions:
                    log_prior += np.log(self.prior_distributions[var](x[i]))
                else:
                    # Default uniform prior
                    log_prior += 0.0
            
            # Likelihood term
            if self.likelihood_function:
                log_likelihood = np.log(self.likelihood_function(data, x))
            else:
                # Default Gaussian likelihood
                predicted = problem.objective(x)
                log_likelihood = -0.5 * np.sum((data - predicted)**2)
            
            return log_prior + log_likelihood
        
        return posterior
    
    def _mcmc_sampling(self, posterior: Callable, problem: OptimizationProblem) -> np.ndarray:
        """Metropolis-Hastings MCMC sampling."""
        n_vars = len(problem.variables)
        n_samples = self.config.n_samples
        burn_in = self.config.burn_in
        
        # Initial state
        if problem.bounds:
            current_state = np.array([np.random.uniform(low, high) 
                                   for var in problem.variables 
                                   for low, high in [problem.bounds[var]]])
        else:
            current_state = np.random.randn(n_vars)
        
        current_posterior = posterior(current_state)
        samples = []
        accepted = 0
        
        for i in range(n_samples + burn_in):
            # Proposal
            proposal = current_state + self.config.proposal_scale * np.random.randn(n_vars)
            
            # Handle bounds
            if problem.bounds:
                for j, var in enumerate(problem.variables):
                    if var in problem.bounds:
                        low, high = problem.bounds[var]
                        if proposal[j] < low or proposal[j] > high:
                            # Reflect from boundaries
                            if proposal[j] < low:
                                proposal[j] = 2 * low - proposal[j]
                            else:
                                proposal[j] = 2 * high - proposal[j]
            
            # Acceptance probability
            proposal_posterior = posterior(proposal)
            acceptance_ratio = np.exp(proposal_posterior - current_posterior)
            
            if np.random.random() < acceptance_ratio:
                current_state = proposal
                current_posterior = proposal_posterior
                accepted += 1
            
            # Store sample after burn-in
            if i >= burn_in:
                samples.append(current_state.copy())
        
        acceptance_rate = accepted / (n_samples + burn_in)
        logger.info(f"MCMC acceptance rate: {acceptance_rate:.3f}")
        
        return np.array(samples)
    
    def _hmc_sampling(self, posterior: Callable, problem: OptimizationProblem) -> np.ndarray:
        """Hamiltonian Monte Carlo sampling."""
        n_vars = len(problem.variables)
        n_samples = self.config.n_samples
        burn_in = self.config.burn_in
        
        # Gradient of posterior (simplified finite differences)
        def posterior_gradient(x):
            grad = np.zeros_like(x)
            h = 1e-6
            current = posterior(x)
            
            for i in range(len(x)):
                x_perturbed = x.copy()
                x_perturbed[i] += h
                grad[i] = (posterior(x_perturbed) - current) / h
            
            return grad
        
        # Initial state
        if problem.bounds:
            current_q = np.array([np.random.uniform(low, high) 
                                for var in problem.variables 
                                for low, high in [problem.bounds[var]]])
        else:
            current_q = np.random.randn(n_vars)
        
        samples = []
        accepted = 0
        
        for i in range(n_samples + burn_in):
            # Sample momentum
            current_p = np.random.randn(n_vars)
            
            # Hamiltonian dynamics (leapfrog integration)
            q = current_q.copy()
            p = current_p.copy()
            
            # Gradient at current position
            grad_U = -posterior_gradient(q)  # Negative gradient for potential
            
            # Half step for momentum
            p = p - 0.5 * self.config.proposal_scale * grad_U
            
            # Full step for position
            q = q + self.config.proposal_scale * p
            
            # Gradient at new position
            grad_U_new = -posterior_gradient(q)
            
            # Half step for momentum
            p = p - 0.5 * self.config.proposal_scale * grad_U_new
            
            # Hamiltonian
            current_H = -posterior(current_q) + 0.5 * np.sum(current_p**2)
            proposed_H = -posterior(q) + 0.5 * np.sum(p**2)
            
            # Acceptance probability
            acceptance_ratio = np.exp(current_H - proposed_H)
            
            if np.random.random() < acceptance_ratio:
                current_q = q
                accepted += 1
            
            # Store sample after burn-in
            if i >= burn_in:
                samples.append(current_q.copy())
        
        acceptance_rate = accepted / (n_samples + burn_in)
        logger.info(f"HMC acceptance rate: {acceptance_rate:.3f}")
        
        return np.array(samples)
    
    def _quantum_sampling(self, posterior: Callable, problem: OptimizationProblem) -> np.ndarray:
        """Quantum-enhanced sampling using quantum annealers or circuit-based samplers."""
        n_vars = len(problem.variables)
        n_samples = self.config.n_samples
        
        if self.config.use_quantum_sampler and self.quantum_sampler is not None:
            # Use actual quantum sampler
            try:
                # Convert posterior to QUBO for quantum annealing
                qubo = self._posterior_to_qubo(posterior, problem)
                samples = self.quantum_sampler.sample_qubo(qubo, num_reads=n_samples)
                return self._decode_quantum_samples(samples, problem)
            except Exception as e:
                logger.warning(f"Quantum sampling failed: {e}, falling back to classical")
        
        # Classical fallback: use MCMC with quantum-inspired proposals
        return self._quantum_inspired_sampling(posterior, problem)
    
    def _quantum_inspired_sampling(self, posterior: Callable, problem: OptimizationProblem) -> np.ndarray:
        """Quantum-inspired sampling with tunneling effects."""
        n_vars = len(problem.variables)
        n_samples = self.config.n_samples
        burn_in = self.config.burn_in
        
        current_state = np.random.randn(n_vars)
        current_posterior = posterior(current_state)
        samples = []
        
        for i in range(n_samples + burn_in):
            # Quantum-inspired proposal with tunneling
            if np.random.random() < 0.1:  # 10% chance of quantum tunneling
                # Large jump to escape local minima
                proposal = current_state + 0.5 * np.random.randn(n_vars)
            else:
                # Normal random walk
                proposal = current_state + 0.1 * np.random.randn(n_vars)
            
            # Quantum fluctuation term
            quantum_fluctuation = 0.01 * np.sum(np.sin(10 * proposal))
            proposal_posterior = posterior(proposal) + quantum_fluctuation
            
            # Metropolis acceptance
            acceptance_ratio = np.exp(proposal_posterior - current_posterior)
            
            if np.random.random() < acceptance_ratio:
                current_state = proposal
                current_posterior = proposal_posterior
            
            if i >= burn_in:
                samples.append(current_state.copy())
        
        return np.array(samples)
    
    def _variational_bayes(self, posterior: Callable, problem: OptimizationProblem) -> np.ndarray:
        """Variational Bayesian inference."""
        n_vars = len(problem.variables)
        n_samples = self.config.n_samples
        
        # Mean-field variational approximation
        # Assume posterior is multivariate Gaussian
        def variational_objective(params):
            mean = params[:n_vars]
            log_std = params[n_vars:]
            std = np.exp(log_std)
            
            # KL divergence between variational distribution and prior
            kl_divergence = 0.5 * np.sum(std**2 + mean**2 - 1 - 2 * log_std)
            
            # Monte Carlo estimate of expected log likelihood
            n_mc_samples = 10
            expected_ll = 0.0
            
            for _ in range(n_mc_samples):
                sample = mean + std * np.random.randn(n_vars)
                expected_ll += posterior(sample)
            
            expected_ll /= n_mc_samples
            
            # Evidence lower bound (ELBO)
            elbo = expected_ll - kl_divergence
            return -elbo  # Minimize negative ELBO
        
        # Optimize variational parameters
        initial_params = np.concatenate([np.zeros(n_vars), np.zeros(n_vars)])
        result = minimize(variational_objective, initial_params, method='BFGS')
        
        # Extract optimal parameters
        optimal_mean = result.x[:n_vars]
        optimal_log_std = result.x[n_vars:]
        optimal_std = np.exp(optimal_log_std)
        
        # Generate samples from variational distribution
        samples = np.random.normal(optimal_mean, optimal_std, (n_samples, n_vars))
        
        return samples
    
    def _posterior_to_qubo(self, posterior: Callable, problem: OptimizationProblem) -> Dict:
        """Convert posterior distribution to QUBO format for quantum sampling."""
        # Simplified QUBO construction
        n_vars = len(problem.variables)
        qubo = {}
        
        # Sample posterior to estimate QUBO coefficients
        n_samples = 100
        samples = []
        for _ in range(n_samples):
            if problem.bounds:
                sample = np.array([np.random.uniform(low, high) 
                                 for var in problem.variables 
                                 for low, high in [problem.bounds[var]]])
            else:
                sample = np.random.randn(n_vars)
            samples.append(sample)
        
        samples = np.array(samples)
        
        # Estimate quadratic coefficients
        for i in range(n_vars):
            for j in range(i, n_vars):
                if i == j:
                    # Linear terms
                    qubo[(i, i)] = -np.mean(samples[:, i])  # Simplified
                else:
                    # Quadratic terms
                    qubo[(i, j)] = 0.1  # Small coupling
        
        return qubo
    
    def _decode_quantum_samples(self, samples: Any, problem: OptimizationProblem) -> np.ndarray:
        """Decode quantum samples to continuous variables."""
        # This would convert binary samples from quantum annealer to continuous variables
        # For now, return random samples
        n_vars = len(problem.variables)
        n_samples = self.config.n_samples
        
        if problem.bounds:
            decoded_samples = np.array([
                [np.random.uniform(low, high) for var in problem.variables 
                 for low, high in [problem.bounds[var]]]
                for _ in range(n_samples)
            ])
        else:
            decoded_samples = np.random.randn(n_samples, n_vars)
        
        return decoded_samples
    
    def _compute_evidence(self, posterior: Callable, samples: np.ndarray) -> float:
        """Compute marginal likelihood (evidence) using harmonic mean estimator."""
        # Harmonic mean estimator (simplified)
        log_posterior_values = np.array([posterior(sample) for sample in samples])
        max_log = np.max(log_posterior_values)
        shifted_values = log_posterior_values - max_log
        
        # Harmonic mean
        harmonic_mean = 1.0 / np.mean(np.exp(-shifted_values))
        evidence = harmonic_mean * np.exp(max_log)
        
        return float(evidence)
    
    def _compute_convergence(self, samples: np.ndarray) -> Dict[str, float]:
        """Compute MCMC convergence diagnostics."""
        n_chains = 4
        n_samples_per_chain = len(samples) // n_chains
        
        if n_samples_per_