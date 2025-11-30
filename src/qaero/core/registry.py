"""
Registry system for algorithms and backends with plugin support.
"""
from typing import Dict, Type, List, Any, Optional
import logging
from threading import RLock

from .base import Backend, Problem

logger = logging.getLogger("qaero.registry")

class Registry:
    """Base registry class with thread-safe operations."""
    
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Any] = {}
        self._lock = RLock()
        self.logger = logging.getLogger(f"qaero.registry.{name}")
    
    def register(self, name: str, item: Any) -> Any:
        """Register an item with thread safety."""
        with self._lock:
            if name in self._registry:
                self.logger.warning(f"Overwriting {self.name} '{name}'")
            self._registry[name] = item
            self.logger.info(f"Registered {self.name} '{name}'")
            return item
    
    def get(self, name: str) -> Any:
        """Get an item by name."""
        with self._lock:
            if name not in self._registry:
                available = list(self._registry.keys())
                raise KeyError(f"{self.name.capitalize()} '{name}' not found. Available: {available}")
            return self._registry[name]
    
    def list(self) -> List[str]:
        """List all registered items."""
        with self._lock:
            return list(self._registry.keys())
    
    def create(self, name: str, **kwargs) -> Any:
        """Create an instance of a registered item."""
        item_class = self.get(name)
        try:
            return item_class(name=name, **kwargs)
        except Exception as e:
            self.logger.error(f"Failed to create {self.name} '{name}': {e}")
            raise

class BackendRegistry(Registry):
    """Registry for computational backends."""
    
    def __init__(self):
        super().__init__("backend")
    
    def create_backend(self, name: str, **config) -> Backend:
        """Create a backend instance with proper typing."""
        backend = super().create(name, **config)
        if not isinstance(backend, Backend):
            raise TypeError(f"Registered backend '{name}' must be a Backend instance")
        return backend

class AlgorithmRegistry(Registry):
    """Registry for computational algorithms."""
    
    def __init__(self):
        super().__init__("algorithm")
        self.categories: Dict[str, List[str]] = {}
    
    def register_algorithm(self, name: str, algorithm: Any, category: str = "general") -> Any:
        """Register an algorithm with category information."""
        algorithm = self.register(name, algorithm)
        
        with self._lock:
            if category not in self.categories:
                self.categories[category] = []
            if name not in self.categories[category]:
                self.categories[category].append(name)
        
        return algorithm
    
    def list_algorithms(self) -> Dict[str, List[str]]:
        """List all algorithms by category."""
        with self._lock:
            return self.categories.copy()
    
    def get_algorithms_by_category(self, category: str) -> List[str]:
        """Get all algorithms in a specific category."""
        with self._lock:
            return self.categories.get(category, []).copy()

# Global registry instances
backend_registry = BackendRegistry()
algorithm_registry = AlgorithmRegistry()

# Convenience functions
def register_backend(name: str):
    """Decorator to register a backend class."""
    def decorator(cls):
        backend_registry.register(name, cls)
        return cls
    return decorator

def register_algorithm(name: str, category: str = "general"):
    """Decorator to register an algorithm."""
    def decorator(cls):
        algorithm_registry.register_algorithm(name, cls, category)
        return cls
    return decorator

# Re-export for convenience
BackendRegistry = backend_registry
AlgorithmRegistry = algorithm_registry