"""surfmesh.core — core data structures and shared utilities.

Submodules
----------
mesh
    :class:`~surfmesh.core.mesh.QuadMesh` — the primary container for quad
    surface meshes, exposing BEM-relevant panel quantities (centroids,
    normals, areas).
topology
    :func:`~surfmesh.core.topology.extract_vertices_faces` — convert the raw
    panel array to the compact ``(vertices, faces)`` representation.
validate
    Centralised input-validation helpers used throughout the package.
"""

from surfmesh.core.mesh import QuadMesh
from surfmesh.core.topology import extract_vertices_faces

__all__ = [
    "QuadMesh",
    "extract_vertices_faces",
]
