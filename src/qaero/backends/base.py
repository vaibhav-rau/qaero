"""
Base backend implementation with plugin registration.
"""
from typing import Dict, Type, List
from ..core.base import Backend, BackendProtocol

class BackendRegistry:
    """Global backend registry for plugin system."""
    
    _backends: Dict[str, Type[Backend]] = {}
    
    @classmethod
    def register(cls, name: str, backend_class: Type[Backend]):
        """Register a backend class."""
        if not issubclass(backend_class, Backend):
            raise TypeError(f"Backend must subclass {Backend}")
        cls._backends[name] = backend_class
        return backend_class
    
    @classmethod
    def get_backend(cls, name: str) -> Type[Backend]:
        """Get backend class by name."""
        if name not in cls._backends:
            raise KeyError(f"Backend '{name}' not registered. Available: {list(cls._backends.keys())}")
        return cls._backends[name]
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """List all registered backends."""
        return list(cls._backends.keys())
    
    @classmethod
    def create_backend(cls, name: str, **config) -> Backend:
        """Create a backend instance."""
        backend_class = cls.get_backend(name)
        return backend_class(name=name, config=config)

# Global registry instance
registry = BackendRegistry()