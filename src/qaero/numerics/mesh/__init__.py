"""
Mesh and discretization drivers for quantum-classical PDE solving.
Interoperability with OpenFOAM, SU2, FEniCS, and other established solvers.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
import warnings
from pathlib import Path

from ....core.base import PDEProblem, QaeroError
from ....core.registry import register_service

logger = logging.getLogger("qaero.numerics.mesh")

@dataclass
class Mesh:
    """Unified mesh representation for quantum-classical computations."""
    nodes: np.ndarray  # [n_nodes, dim]
    elements: np.ndarray  # [n_elements, nodes_per_element]
    boundaries: Dict[str, np.ndarray]  # {boundary_name: node_indices}
    dimension: int
    mesh_type: str  # "structured", "unstructured", "hybrid"
    
    def __post_init__(self):
        self.n_nodes = self.nodes.shape[0]
        self.n_elements = self.elements.shape[0]
        
    def get_boundary_nodes(self, boundary_name: str) -> np.ndarray:
        """Get nodes for specific boundary."""
        return self.nodes[self.boundaries.get(boundary_name, [])]

@dataclass
class DiscretizationConfig:
    """Configuration for PDE discretization."""
    method: str = "finite_element"  # "finite_element", "finite_volume", "finite_difference"
    element_type: str = "linear"  # "linear", "quadratic", "cubic"
    quadrature_order: int = 2
    stabilization: bool = True
    adaptive: bool = False

@register_service("mesh_manager")
class MeshManager:
    """Unified mesh management with adapter support for various formats."""
    
    def __init__(self):
        self.adapters = {
            'openfoam': OpenFOAMAdapter(),
            'su2': SU2Adapter(),
            'fenics': FEniCSAdapter(),
            'gmsh': GMSHAdapter(),
            'simple': SimpleMeshGenerator()
        }
        self.current_mesh = None
    
    def load_mesh(self, file_path: str, format: str, **kwargs) -> Mesh:
        """Load mesh from file in specified format."""
        if format not in self.adapters:
            raise QaeroError(f"Unsupported mesh format: {format}. Available: {list(self.adapters.keys())}")
        
        adapter = self.adapters[format]
        self.current_mesh = adapter.load_mesh(file_path, **kwargs)
        return self.current_mesh
    
    def generate_mesh(self, domain: Dict, method: str = "simple", **kwargs) -> Mesh:
        """Generate mesh for given domain."""
        if method not in self.adapters:
            raise QaeroError(f"Unsupported mesh generation method: {method}")
        
        generator = self.adapters[method]
        self.current_mesh = generator.generate_mesh(domain, **kwargs)
        return self.current_mesh
    
    def export_mesh(self, file_path: str, format: str, **kwargs):
        """Export mesh to file in specified format."""
        if self.current_mesh is None:
            raise QaeroError("No mesh loaded to export")
        
        if format not in self.adapters:
            raise QaeroError(f"Unsupported export format: {format}")
        
        adapter = self.adapters[format]
        adapter.export_mesh(self.current_mesh, file_path, **kwargs)
    
    def get_mesh_info(self) -> Dict[str, Any]:
        """Get information about current mesh."""
        if self.current_mesh is None:
            return {}
        
        return {
            'n_nodes': self.current_mesh.n_nodes,
            'n_elements': self.current_mesh.n_elements,
            'dimension': self.current_mesh.dimension,
            'mesh_type': self.current_mesh.mesh_type,
            'boundaries': list(self.current_mesh.boundaries.keys())
        }

class MeshAdapter(ABC):
    """Abstract base class for mesh format adapters."""
    
    @abstractmethod
    def load_mesh(self, file_path: str, **kwargs) -> Mesh:
        """Load mesh from file."""
        pass
    
    @abstractmethod
    def export_mesh(self, mesh: Mesh, file_path: str, **kwargs):
        """Export mesh to file."""
        pass

class OpenFOAMAdapter(MeshAdapter):
    """Adapter for OpenFOAM mesh format."""
    
    def load_mesh(self, file_path: str, **kwargs) -> Mesh:
        """Load OpenFOAM mesh."""
        try:
            # Simplified OpenFOAM mesh reading
            # In practice, would use pyfoam or similar
            mesh_dir = Path(file_path)
            
            # Read points
            points_file = mesh_dir / "points"
            if not points_file.exists():
                raise QaeroError(f"Points file not found in {file_path}")
            
            nodes = self._read_openfoam_points(points_file)
            
            # Read faces and boundaries (simplified)
            boundaries = self._read_openfoam_boundaries(mesh_dir)
            
            # Create simple element connectivity (2D triangles for now)
            n_nodes = nodes.shape[0]
            elements = self._create_simple_elements(n_nodes)
            
            return Mesh(
                nodes=nodes,
                elements=elements,
                boundaries=boundaries,
                dimension=3 if nodes.shape[1] == 3 else 2,
                mesh_type="unstructured"
            )
            
        except Exception as e:
            logger.error(f"Failed to load OpenFOAM mesh: {e}")
            raise QaeroError(f"OpenFOAM mesh loading failed: {e}")
    
    def export_mesh(self, mesh: Mesh, file_path: str, **kwargs):
        """Export to OpenFOAM format (simplified)."""
        # This would implement full OpenFOAM mesh writing
        logger.warning("OpenFOAM export not fully implemented")
    
    def _read_openfoam_points(self, points_file: Path) -> np.ndarray:
        """Read OpenFOAM points file."""
        # Simplified points reading
        with open(points_file, 'r') as f:
            lines = f.readlines()
        
        # Find points data (simplified parsing)
        points_start = -1
        points_end = -1
        
        for i, line in enumerate(lines):
            if line.strip() == "(":
                points_start = i + 1
            elif line.strip() == ")" and points_start != -1:
                points_end = i
                break
        
        if points_start == -1 or points_end == -1:
            raise QaeroError("Could not find points data in OpenFOAM file")
        
        points = []
        for line in lines[points_start:points_end]:
            # Parse (x y z) format
            coords = line.strip().strip('()').split()
            if len(coords) == 3:
                points.append([float(coords[0]), float(coords[1]), float(coords[2])])
        
        return np.array(points)
    
    def _read_openfoam_boundaries(self, mesh_dir: Path) -> Dict[str, np.ndarray]:
        """Read OpenFOAM boundary definitions."""
        boundaries = {}
        
        boundary_file = mesh_dir / "boundary"
        if boundary_file.exists():
            # Simplified boundary reading
            boundaries['inlet'] = np.array([0])  # Placeholder
            boundaries['outlet'] = np.array([1])  # Placeholder
            boundaries['walls'] = np.array([2, 3])  # Placeholder
        
        return boundaries
    
    def _create_simple_elements(self, n_nodes: int) -> np.ndarray:
        """Create simple element connectivity for demonstration."""
        # Create triangular elements (simplified)
        elements = []
        for i in range(0, n_nodes - 2, 2):
            if i + 2 < n_nodes:
                elements.append([i, i + 1, i + 2])
        
        return np.array(elements) if elements else np.array([[0, 1, 2]])

class SU2Adapter(MeshAdapter):
    """Adapter for SU2 mesh format."""
    
    def load_mesh(self, file_path: str, **kwargs) -> Mesh:
        """Load SU2 format mesh."""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            dimension = 2  # Default
            nodes = []
            elements = []
            boundaries = {}
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                if line.startswith("NDIME="):
                    dimension = int(line.split("=")[1])
                elif line.startswith("NPOIN="):
                    n_points = int(line.split("=")[1])
                    i += 1
                    for j in range(n_points):
                        coords = list(map(float, lines[i + j].split()[:dimension]))
                        nodes.append(coords)
                    i += n_points
                elif line.startswith("NELEM="):
                    n_elements = int(line.split("=")[1])
                    i += 1
                    for j in range(n_elements):
                        element_data = list(map(int, lines[i + j].split()))
                        elem_type = element_data[0]
                        if elem_type == 5:  # Triangle
                            elements.append(element_data[1:4])
                        elif elem_type == 9:  # Quad
                            elements.append(element_data[1:5])
                    i += n_elements
                elif line.startswith("NMARK="):
                    n_markers = int(line.split("=")[1])
                    i += 1
                    for _ in range(n_markers):
                        marker_line = lines[i].strip()
                        if marker_line.startswith("MARKER_TAG="):
                            marker_name = marker_line.split("=")[1]
                            i += 1
                            n_elements_marker = int(lines[i].split("=")[1])
                            i += 1
                            boundary_nodes = []
                            for j in range(n_elements_marker):
                                element_data = list(map(int, lines[i + j].split()))
                                boundary_nodes.extend(element_data[1:])
                            boundaries[marker_name] = np.unique(boundary_nodes)
                        i += n_elements_marker
                else:
                    i += 1
            
            return Mesh(
                nodes=np.array(nodes),
                elements=np.array(elements),
                boundaries=boundaries,
                dimension=dimension,
                mesh_type="unstructured"
            )
            
        except Exception as e:
            logger.error(f"Failed to load SU2 mesh: {e}")
            raise QaeroError(f"SU2 mesh loading failed: {e}")
    
    def export_mesh(self, mesh: Mesh, file_path: str, **kwargs):
        """Export to SU2 format."""
        with open(file_path, 'w') as f:
            # Write dimension
            f.write(f"NDIME= {mesh.dimension}\n")
            
            # Write points
            f.write(f"NPOIN= {mesh.n_nodes}\n")
            for node in mesh.nodes:
                f.write(" ".join(f"{coord:.10f}" for coord in node) + "\n")
            
            # Write elements (simplified - assume triangles)
            f.write(f"NELEM= {mesh.n_elements}\n")
            for element in mesh.elements:
                if len(element) == 3:  # Triangle
                    f.write(f"5 {' '.join(map(str, element))}\n")
                elif len(element) == 4:  # Quad
                    f.write(f"9 {' '.join(map(str, element))}\n")
            
            # Write boundaries
            f.write(f"NMARK= {len(mesh.boundaries)}\n")
            for boundary_name, nodes in mesh.boundaries.items():
                f.write(f"MARKER_TAG= {boundary_name}\n")
                f.write(f"MARKER_ELEMS= {len(nodes)}\n")
                for node in nodes:
                    f.write(f"3 {node}\n")  # Line element for 2D

class FEniCSAdapter(MeshAdapter):
    """Adapter for FEniCS mesh format."""
    
    def load_mesh(self, file_path: str, **kwargs) -> Mesh:
        """Load FEniCS/XDMF mesh."""
        try:
            import dolfin as df
        except ImportError:
            logger.warning("FEniCS not available, using fallback mesh")
            return self._create_fallback_mesh()
        
        try:
            mesh = df.Mesh()
            with df.XDMFFile(file_path) as f:
                f.read(mesh)
            
            # Extract nodes
            nodes = mesh.coordinates()
            
            # Extract elements
            elements = []
            for cell in df.cells(mesh):
                elements.append(cell.entities(0))
            
            # Extract boundaries (simplified)
            boundaries = self._extract_fenics_boundaries(mesh)
            
            return Mesh(
                nodes=nodes,
                elements=np.array(elements),
                boundaries=boundaries,
                dimension=mesh.geometry().dim(),
                mesh_type="unstructured"
            )
            
        except Exception as e:
            logger.error(f"Failed to load FEniCS mesh: {e}")
            return self._create_fallback_mesh()
    
    def _extract_fenics_boundaries(self, mesh) -> Dict[str, np.ndarray]:
        """Extract boundary information from FEniCS mesh."""
        boundaries = {}
        
        try:
            import dolfin as df
            
            # This would use mesh functions in practice
            boundaries['boundary'] = np.array([0])  # Placeholder
            
        except Exception:
            # Fallback boundaries
            boundaries['default'] = np.array([0, 1])
        
        return boundaries
    
    def _create_fallback_mesh(self) -> Mesh:
        """Create fallback mesh when FEniCS is unavailable."""
        nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        elements = np.array([[0, 1, 2], [0, 2, 3]])
        boundaries = {'bottom': np.array([0, 1]), 'right': np.array([1, 2]), 
                     'top': np.array([2, 3]), 'left': np.array([3, 0])}
        
        return Mesh(nodes, elements, boundaries, 2, "structured")

class GMSHAdapter(MeshAdapter):
    """Adapter for GMSH mesh format."""
    
    def load_mesh(self, file_path: str, **kwargs) -> Mesh:
        """Load GMSH .msh file."""
        try:
            import gmsh
        except ImportError:
            logger.warning("GMSH not available, using fallback")
            return self._create_fallback_mesh()
        
        try:
            gmsh.initialize()
            gmsh.open(file_path)
            
            # Get nodes
            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            nodes = node_coords.reshape(-1, 3)[:, :2]  # Use only 2D for now
            
            # Get elements
            element_types, element_tags, element_nodes = gmsh.model.mesh.getElements()
            elements = []
            
            for elem_type, tags, nodes_in_elem in zip(element_types, element_tags, element_nodes):
                if elem_type == 2:  # 3-node triangle
                    n_nodes_per_elem = 3
                    elements.append(nodes_in_elem.reshape(-1, n_nodes_per_elem) - 1)  # 0-based indexing
            
            if elements:
                elements = np.vstack(elements)
            else:
                elements = np.array([[0, 1, 2]])  # Fallback
            
            # Get boundaries (simplified)
            boundaries = self._extract_gmsh_boundaries()
            
            gmsh.finalize()
            
            return Mesh(
                nodes=nodes,
                elements=elements,
                boundaries=boundaries,
                dimension=2,
                mesh_type="unstructured"
            )
            
        except Exception as e:
            logger.error(f"Failed to load GMSH mesh: {e}")
            gmsh.finalize()
            return self._create_fallback_mesh()
    
    def _extract_gmsh_boundaries(self) -> Dict[str, np.ndarray]:
        """Extract boundary information from GMSH."""
        # Simplified boundary extraction
        boundaries = {
            'inlet': np.array([0]),
            'outlet': np.array([1]),
            'wall': np.array([2, 3])
        }
        return boundaries
    
    def _create_fallback_mesh(self) -> Mesh:
        """Create fallback mesh."""
        return FEniCSAdapter()._create_fallback_mesh()

class SimpleMeshGenerator(MeshAdapter):
    """Simple mesh generator for basic domains."""
    
    def load_mesh(self, file_path: str, **kwargs) -> Mesh:
        """Load mesh from file (not implemented for generator)."""
        raise QaeroError("SimpleMeshGenerator is for generation, not loading")
    
    def export_mesh(self, mesh: Mesh, file_path: str, **kwargs):
        """Export mesh to file."""
        # Export as simple text format
        with open(file_path, 'w') as f:
            f.write(f"# QAero Simple Mesh\n")
            f.write(f"# Nodes: {mesh.n_nodes}\n")
            f.write(f"# Elements: {mesh.n_elements}\n")
            f.write(f"# Dimension: {mesh.dimension}\n\n")
            
            f.write("NODES\n")
            for i, node in enumerate(mesh.nodes):
                f.write(f"{i} " + " ".join(f"{coord:.6f}" for coord in node) + "\n")
            
            f.write("\nELEMENTS\n")
            for i, element in enumerate(mesh.elements):
                f.write(f"{i} " + " ".join(map(str, element)) + "\n")
            
            f.write("\nBOUNDARIES\n")
            for boundary_name, nodes in mesh.boundaries.items():
                f.write(f"{boundary_name} " + " ".join(map(str, nodes)) + "\n")
    
    def generate_mesh(self, domain: Dict, **kwargs) -> Mesh:
        """Generate mesh for given domain description."""
        domain_type = domain.get('type', 'rectangle')
        
        if domain_type == 'rectangle':
            return self._generate_rectangle_mesh(domain, **kwargs)
        elif domain_type == 'circle':
            return self._generate_circle_mesh(domain, **kwargs)
        elif domain_type == 'airfoil':
            return self._generate_airfoil_mesh(domain, **kwargs)
        else:
            raise QaeroError(f"Unsupported domain type: {domain_type}")
    
    def _generate_rectangle_mesh(self, domain: Dict, **kwargs) -> Mesh:
        """Generate structured rectangular mesh."""
        nx = domain.get('nx', 10)
        ny = domain.get('ny', 10)
        xmin, xmax = domain.get('x_bounds', [0, 1])
        ymin, ymax = domain.get('y_bounds', [0, 1])
        
        # Generate nodes
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        nodes = np.array([[xi, yj] for yj in y for xi in x])
        
        # Generate elements (triangles)
        elements = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                n0 = j * nx + i
                n1 = j * nx + i + 1
                n2 = (j + 1) * nx + i + 1
                n3 = (j + 1) * nx + i
                
                # Two triangles per quad
                elements.append([n0, n1, n2])
                elements.append([n0, n2, n3])
        
        # Define boundaries
        boundaries = {}
        # Bottom
        boundaries['bottom'] = np.arange(nx)
        # Top
        boundaries['top'] = np.arange((ny - 1) * nx, ny * nx)
        # Left
        boundaries['left'] = np.arange(0, ny * nx, nx)
        # Right
        boundaries['right'] = np.arange(nx - 1, ny * nx, nx)
        
        return Mesh(
            nodes=nodes,
            elements=np.array(elements),
            boundaries=boundaries,
            dimension=2,
            mesh_type="structured"
        )
    
    def _generate_circle_mesh(self, domain: Dict, **kwargs) -> Mesh:
        """Generate circular mesh (simplified)."""
        radius = domain.get('radius', 1.0)
        n_radial = domain.get('n_radial', 5)
        n_circumferential = domain.get('n_circumferential', 12)
        
        # Generate nodes
        nodes = []
        for i in range(n_radial):
            r = (i + 1) * radius / n_radial
            for j in range(n_circumferential):
                theta = 2 * np.pi * j / n_circumferential
                nodes.append([r * np.cos(theta), r * np.sin(theta)])
        
        # Add center point
        nodes.append([0, 0])
        center_idx = len(nodes) - 1
        
        # Generate elements (triangles from center)
        elements = []
        for i in range(n_circumferential):
            n1 = center_idx
            n2 = i
            n3 = (i + 1) % n_circumferential
            elements.append([n1, n2, n3])
        
        # Additional radial elements (simplified)
        for i in range(n_radial - 1):
            for j in range(n_circumferential):
                current_ring_start = i * n_circumferential
                next_ring_start = (i + 1) * n_circumferential
                
                n0 = current_ring_start + j
                n1 = current_ring_start + (j + 1) % n_circumferential
                n2 = next_ring_start + j
                n3 = next_ring_start + (j + 1) % n_circumferential
                
                elements.append([n0, n1, n2])
                elements.append([n1, n3, n2])
        
        boundaries = {'outer': np.arange((n_radial - 1) * n_circumferential, n_radial * n_circumferential)}
        
        return Mesh(
            nodes=np.array(nodes),
            elements=np.array(elements),
            boundaries=boundaries,
            dimension=2,
            mesh_type="unstructured"
        )
    
    def _generate_airfoil_mesh(self, domain: Dict, **kwargs) -> Mesh:
        """Generate mesh around airfoil (simplified)."""
        # Simplified airfoil mesh - in practice would use proper airfoil coordinates
        chord = domain.get('chord', 1.0)
        n_around = domain.get('n_around', 20)
        n_radial = domain.get('n_radial', 10)
        farfield_radius = domain.get('farfield_radius', 5.0)
        
        # Generate nodes around airfoil (simplified NACA 0012)
        airfoil_nodes = []
        for i in range(n_around):
            x = i / (n_around - 1)
            # Simplified symmetric airfoil
            y = 0.12 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
            airfoil_nodes.append([x * chord, y])
            if i > 0 and i < n_around - 1:
                airfoil_nodes.append([x * chord, -y])
        
        # Generate farfield nodes
        farfield_nodes = []
        for i in range(n_around):
            angle = 2 * np.pi * i / n_around
            farfield_nodes.append([
                farfield_radius * np.cos(angle),
                farfield_radius * np.sin(angle)
            ])
        
        nodes = np.array(airfoil_nodes + farfield_nodes)
        
        # Simple triangular elements (placeholder)
        elements = []
        for i in range(len(nodes) - 2):
            elements.append([i, i + 1, i + 2])
        
        boundaries = {
            'airfoil': np.arange(len(airfoil_nodes)),
            'farfield': np.arange(len(airfoil_nodes), len(airfoil_nodes) + len(farfield_nodes))
        }
        
        return Mesh(
            nodes=nodes,
            elements=np.array(elements),
            boundaries=boundaries,
            dimension=2,
            mesh_type="unstructured"
        )