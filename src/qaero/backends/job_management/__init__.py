"""
Advanced job management and orchestration for quantum computing.
Async submission, result retrieval, error handling, and cost accounting.
"""
import asyncio
import time
import uuid
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from enum import Enum
import queue
import json
from concurrent.futures import ThreadPoolExecutor, Future

from ....core.base import QaeroError
from ....core.registry import register_service

logger = logging.getLogger("qaero.backends.job_management")

class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class JobInfo:
    """Complete job information."""
    job_id: str
    backend: str
    problem_id: str
    submitted_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: JobStatus = JobStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    cost: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        self.metadata = self.metadata or {}
    
    @property
    def execution_time(self) -> Optional[float]:
        """Calculate execution time."""
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return None
    
    @property
    def queue_time(self) -> Optional[float]:
        """Calculate queue time."""
        if self.started_at and self.submitted_at:
            return self.started_at - self.submitted_at
        return None

@dataclass
class CostAccounting:
    """Cost accounting for quantum computing jobs."""
    backend: str
    job_id: str
    shots: int
    qpu_time: float  # seconds
    simulator_time: float  # seconds
    data_transfer: float  # MB
    total_cost: float
    currency: str = "USD"
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@register_service("job_manager")
class JobManager:
    """
    Advanced job management system for quantum computing workflows.
    Handles async execution, retries, cost tracking, and resource management.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.jobs: Dict[str, JobInfo] = {}
        self.job_queue = queue.Queue()
        self.completed_jobs: Dict[str, JobInfo] = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 5))
        self.running = False
        self.worker_thread = None
        self.cost_tracker = CostTracker()
        self.retry_policy = RetryPolicy()
        
        # Statistics
        self.stats = {
            'jobs_submitted': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'total_cost': 0.0,
            'total_execution_time': 0.0
        }
    
    def start(self):
        """Start the job manager."""
        if self.running:
            logger.warning("Job manager already running")
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._process_jobs, daemon=True)
        self.worker_thread.start()
        logger.info("Job manager started")
    
    def stop(self):
        """Stop the job manager."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=10)
        self.executor.shutdown(wait=False)
        logger.info("Job manager stopped")
    
    def submit_job(self, backend: Any, problem: Any, **kwargs) -> str:
        """Submit a job for execution."""
        job_id = str(uuid.uuid4())[:8]
        
        job_info = JobInfo(
            job_id=job_id,
            backend=backend.name,
            problem_id=getattr(problem, 'problem_id', 'unknown'),
            submitted_at=time.time(),
            metadata={
                'backend_type': getattr(backend, 'backend_type', 'unknown'),
                'problem_type': type(problem).__name__,
                'parameters': kwargs
            }
        )
        
        self.jobs[job_id] = job_info
        self.job_queue.put((job_id, backend, problem, kwargs))
        self.stats['jobs_submitted'] += 1
        
        logger.info(f"Job {job_id} submitted to {backend.name}")
        return job_id
    
    async def submit_job_async(self, backend: Any, problem: Any, **kwargs) -> str:
        """Submit a job asynchronously."""
        # For true async operation, would use asyncio queues
        return self.submit_job(backend, problem, **kwargs)
    
    def get_job_status(self, job_id: str) -> JobInfo:
        """Get current status of a job."""
        if job_id in self.jobs:
            return self.jobs[job_id]
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        else:
            raise QaeroError(f"Job {job_id} not found")
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        if job_id not in self.jobs:
            return False
        
        job_info = self.jobs[job_id]
        if job_info.status in [JobStatus.PENDING, JobStatus.RUNNING]:
            job_info.status = JobStatus.CANCELLED
            job_info.completed_at = time.time()
            self._move_to_completed(job_id)
            logger.info(f"Job {job_id} cancelled")
            return True
        
        return False
    
    def get_results(self, job_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for and get job results."""
        start_time = time.time()
        
        while True:
            job_info = self.get_job_status(job_id)
            
            if job_info.status == JobStatus.COMPLETED:
                return job_info.result
            elif job_info.status in [JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.TIMEOUT]:
                raise QaeroError(f"Job {job_id} failed: {job_info.error}")
            
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                job_info.status = JobStatus.TIMEOUT
                job_info.error = f"Job timeout after {timeout} seconds"
                self._move_to_completed(job_id)
                raise QaeroError(f"Job {job_id} timed out")
            
            # Wait before checking again
            time.sleep(0.1)
    
    async def get_results_async(self, job_id: str, timeout: Optional[float] = None) -> Any:
        """Wait for and get job results asynchronously."""
        # For true async operation, would use asyncio events
        return await asyncio.get_event_loop().run_in_executor(
            None, self.get_results, job_id, timeout
        )
    
    def _process_jobs(self):
        """Background job processing loop."""
        while self.running:
            try:
                # Get next job with timeout to allow graceful shutdown
                try:
                    job_id, backend, problem, kwargs = self.job_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Update job status
                job_info = self.jobs[job_id]
                job_info.status = JobStatus.RUNNING
                job_info.started_at = time.time()
                
                # Submit to thread pool for execution
                future = self.executor.submit(self._execute_job, job_id, backend, problem, kwargs)
                future.add_done_callback(lambda f: self._job_completed(job_id, f))
                
            except Exception as e:
                logger.error(f"Error in job processing loop: {e}")
    
    def _execute_job(self, job_id: str, backend: Any, problem: Any, kwargs: Dict) -> Any:
        """Execute a single job."""
        job_info = self.jobs[job_id]
        
        try:
            # Determine problem type and call appropriate solver
            from ....core.base import OptimizationProblem, PDEProblem
            
            if isinstance(problem, OptimizationProblem):
                result = backend.solve_optimization(problem, **kwargs)
            elif isinstance(problem, PDEProblem):
                result = backend.solve_pde(problem, **kwargs)
            else:
                raise QaeroError(f"Unsupported problem type: {type(problem)}")
            
            # Track costs
            cost = self._calculate_job_cost(backend, problem, result, kwargs)
            job_info.cost = cost
            self.stats['total_cost'] += cost
            
            return result
            
        except Exception as e:
            logger.error(f"Job {job_id} execution failed: {e}")
            raise
    
    def _job_completed(self, job_id: str, future: Future):
        """Handle job completion."""
        job_info = self.jobs[job_id]
        job_info.completed_at = time.time()
        
        try:
            result = future.result()
            job_info.result = result
            job_info.status = JobStatus.COMPLETED
            self.stats['jobs_completed'] += 1
            
            if hasattr(result, 'execution_time'):
                self.stats['total_execution_time'] += result.execution_time
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            job_info.status = JobStatus.FAILED
            job_info.error = str(e)
            self.stats['jobs_failed'] += 1
            
            # Apply retry policy
            if self.retry_policy.should_retry(job_info):
                retry_delay = self.retry_policy.get_retry_delay(job_info)
                logger.info(f"Job {job_id} scheduled for retry in {retry_delay}s")
                
                # Resubmit after delay
                threading.Timer(retry_delay, self._retry_job, [job_id]).start()
                return
            
            logger.error(f"Job {job_id} failed: {e}")
        
        # Move to completed jobs
        self._move_to_completed(job_id)
    
    def _retry_job(self, job_id: str):
        """Retry a failed job."""
        if job_id not in self.jobs:
            return
        
        job_info = self.jobs[job_id]
        job_info.status = JobStatus.PENDING
        job_info.started_at = None
        job_info.completed_at = None
        job_info.error = None
        
        # Get original job parameters from metadata
        backend = self._get_backend_by_name(job_info.backend)
        problem = self._reconstruct_problem(job_info)
        kwargs = job_info.metadata.get('parameters', {})
        
        # Resubmit
        self.job_queue.put((job_id, backend, problem, kwargs))
        logger.info(f"Job {job_id} retry submitted")
    
    def _move_to_completed(self, job_id: str):
        """Move job from active to completed storage."""
        if job_id in self.jobs:
            job_info = self.jobs.pop(job_id)
            self.completed_jobs[job_id] = job_info
            
            # Clean up old completed jobs (keep last 1000)
            if len(self.completed_jobs) > 1000:
                oldest_job_id = min(self.completed_jobs.keys(), 
                                  key=lambda k: self.completed_jobs[k].completed_at or 0)
                self.completed_jobs.pop(oldest_job_id)
    
    def _calculate_job_cost(self, backend: Any, problem: Any, result: Any, kwargs: Dict) -> float:
        """Calculate cost for a quantum computing job."""
        backend_type = getattr(backend, 'backend_type', 'unknown')
        shots = kwargs.get('shots', 1024)
        
        # Cost models for different backends
        cost_models = {
            'gate_based_simulator': lambda: shots * 0.000001,  # $0.000001 per shot
            'gate_based_hardware': lambda: shots * 0.01,       # $0.01 per shot
            'quantum_annealer': lambda: shots * 0.001,         # $0.001 per shot
            'cloud_quantum': lambda: shots * 0.005,            # $0.005 per shot
        }
        
        cost_function = cost_models.get(backend_type, lambda: 0.0)
        base_cost = cost_function()
        
        # Add problem complexity factor
        complexity_factor = self._estimate_problem_complexity(problem)
        total_cost = base_cost * complexity_factor
        
        # Store cost in result metadata
        if hasattr(result, 'metadata'):
            result.metadata['estimated_cost'] = total_cost
        
        return total_cost
    
    def _estimate_problem_complexity(self, problem: Any) -> float:
        """Estimate problem complexity for cost calculation."""
        from ....core.base import OptimizationProblem, PDEProblem
        
        if isinstance(problem, OptimizationProblem):
            # Complexity based on number of variables and constraints
            n_vars = len(getattr(problem, 'variables', []))
            n_constraints = len(getattr(problem, 'constraints', []))
            return 1.0 + (n_vars / 10) + (n_constraints / 5)
        
        elif isinstance(problem, PDEProblem):
            # Complexity based on discretization size
            domain = getattr(problem, 'domain', {})
            n_points = domain.get('n_points', 50)
            return 1.0 + (n_points / 100)
        
        return 1.0
    
    def _get_backend_by_name(self, backend_name: str) -> Any:
        """Get backend instance by name."""
        # This would interface with the backend registry
        # Simplified implementation
        from ....backends import BackendRegistry
        
        try:
            return BackendRegistry.create_backend(backend_name)
        except Exception:
            logger.error(f"Backend {backend_name} not available for retry")
            raise
    
    def _reconstruct_problem(self, job_info: JobInfo) -> Any:
        """Reconstruct problem from job info."""
        # In practice, would deserialize problem from metadata
        # Simplified implementation
        from ....problems.aerodynamics import AirfoilOptimizationProblem
        
        return AirfoilOptimizationProblem()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get job manager statistics."""
        current_time = time.time()
        active_jobs = len(self.jobs)
        completed_jobs = len(self.completed_jobs)
        
        # Calculate average times
        avg_execution_time = 0.0
        avg_queue_time = 0.0
        
        if self.stats['jobs_completed'] > 0:
            avg_execution_time = self.stats['total_execution_time'] / self.stats['jobs_completed']
            
            # Calculate average queue time from completed jobs
            queue_times = [
                job.queue_time for job in self.completed_jobs.values() 
                if job.queue_time is not None
            ]
            if queue_times:
                avg_queue_time = sum(queue_times) / len(queue_times)
        
        return {
            **self.stats,
            'active_jobs': active_jobs,
            'completed_jobs': completed_jobs,
            'avg_execution_time': avg_execution_time,
            'avg_queue_time': avg_queue_time,
            'uptime': current_time - getattr(self, '_start_time', current_time)
        }
    
    def save_job_history(self, filepath: str):
        """Save job history to file."""
        history = {
            'completed_jobs': {
                job_id: {
                    'job_id': job_info.job_id,
                    'backend': job_info.backend,
                    'problem_id': job_info.problem_id,
                    'status': job_info.status.value,
                    'submitted_at': job_info.submitted_at,
                    'started_at': job_info.started_at,
                    'completed_at': job_info.completed_at,
                    'cost': job_info.cost,
                    'error': job_info.error,
                    'metadata': job_info.metadata
                }
                for job_id, job_info in self.completed_jobs.items()
            },
            'statistics': self.get_statistics(),
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        logger.info(f"Job history saved to {filepath}")
    
    def load_job_history(self, filepath: str):
        """Load job history from file."""
        try:
            with open(filepath, 'r') as f:
                history = json.load(f)
            
            # Restore completed jobs
            for job_id, job_data in history.get('completed_jobs', {}).items():
                job_info = JobInfo(
                    job_id=job_data['job_id'],
                    backend=job_data['backend'],
                    problem_id=job_data['problem_id'],
                    submitted_at=job_data['submitted_at'],
                    started_at=job_data.get('started_at'),
                    completed_at=job_data.get('completed_at'),
                    status=JobStatus(job_data['status']),
                    cost=job_data.get('cost', 0.0),
                    error=job_data.get('error'),
                    metadata=job_data.get('metadata', {})
                )
                self.completed_jobs[job_id] = job_info
            
            logger.info(f"Job history loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load job history: {e}")

class CostTracker:
    """Advanced cost tracking for quantum computing resources."""
    
    def __init__(self):
        self.costs: List[CostAccounting] = []
        self.budget_limits: Dict[str, float] = {}
        self.currency = "USD"
    
    def record_cost(self, cost_data: CostAccounting):
        """Record a cost transaction."""
        self.costs.append(cost_data)
        
        # Check budget limits
        self._check_budget_limits(cost_data.backend, cost_data.total_cost)
    
    def _check_budget_limits(self, backend: str, cost: float):
        """Check if cost exceeds budget limits."""
        if backend in self.budget_limits and cost > self.budget_limits[backend]:
            logger.warning(f"Cost {cost} exceeds budget limit {self.budget_limits[backend]} for {backend}")
    
    def set_budget_limit(self, backend: str, limit: float):
        """Set budget limit for a backend."""
        self.budget_limits[backend] = limit
    
    def get_total_cost(self, backend: Optional[str] = None) -> float:
        """Get total cost for all jobs or specific backend."""
        if backend:
            costs = [c.total_cost for c in self.costs if c.backend == backend]
        else:
            costs = [c.total_cost for c in self.costs]
        
        return sum(costs)
    
    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by backend."""
        breakdown = {}
        for cost_data in self.costs:
            backend = cost_data.backend
            breakdown[backend] = breakdown.get(backend, 0.0) + cost_data.total_cost
        return breakdown

class RetryPolicy:
    """Configurable retry policy for failed jobs."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'max_retries': 3,
            'backoff_factor': 2.0,
            'initial_delay': 1.0,  # seconds
            'max_delay': 60.0,     # seconds
            'retryable_errors': [
                'timeout', 'connection', 'rate_limit', 'device_busy'
            ]
        }
    
    def should_retry(self, job_info: JobInfo) -> bool:
        """Determine if a job should be retried."""
        if job_info.metadata.get('retry_count', 0) >= self.config['max_retries']:
            return False
        
        error = job_info.error or ''
        error_lower = error.lower()
        
        # Check if error is retryable
        for retryable_error in self.config['retryable_errors']:
            if retryable_error in error_lower:
                return True
        
        return False
    
    def get_retry_delay(self, job_info: JobInfo) -> float:
        """Calculate retry delay using exponential backoff."""
        retry_count = job_info.metadata.get('retry_count', 0)
        delay = self.config['initial_delay'] * (self.config['backoff_factor'] ** retry_count)
        return min(delay, self.config['max_delay'])