"""surfmesh — structured quadrilateral surface mesh generation for BEM and beyond.

surfmesh is a pure-NumPy library for creating structured quadrilateral
surface meshes of standard 3-D shapes.  It is designed as a **go-to tool
for Boundary Element Method (BEM) pre-processing**, where high-quality
quad panels with accurate centroids, normals, and areas are essential.

Package layout
--------------
The public API is flat — every symbol listed in ``__all__`` is importable
directly from ``surfmesh``.  Advanced users can also import from the
specific subpackages for better discoverability:

:mod:`surfmesh.core`
    Core data structures (:class:`~surfmesh.core.mesh.QuadMesh`) and
    topology utilities (:func:`~surfmesh.core.topology.extract_vertices_faces`).

:mod:`surfmesh.geometry`
    2-D building blocks: curve generators and quad-grid constructors.

:mod:`surfmesh.primitives`
    Ready-to-use 3-D primitive meshers: disk, cuboid, cylinder, sphere.

:mod:`surfmesh.operations`
    Mesh construction operations: revolution, translation, scaling,
    rotation, and normal flipping.

:mod:`surfmesh.io`
    *(Planned)* Mesh import / export to OBJ, VTK, STL, and BEM formats.

Quick start
-----------
>>> import surfmesh as sm
>>> panels = sm.sphere_mesher_from_projection(radius=1.0, resolution=4)
>>> mesh = sm.QuadMesh(panels)
>>> mesh
QuadMesh(n_panels=96, n_vertices=98, total_area=12.1137)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------
from surfmesh.core.mesh import QuadMesh
from surfmesh.core.topology import extract_vertices_faces

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
from surfmesh.geometry.curves import arc_edges, circumference_edges, rectangle_perimeter
from surfmesh.geometry.grid import mesh_between_edges, quad_faces_from_edges

# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------
from surfmesh.operations.revolve import circular_revolve, revolve_curve_along_path
from surfmesh.operations.transform import (
    convert_2d_face_to_3d,
    flip_normals,
    rotate_z,
    scale,
    translate,
)

# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
from surfmesh.primitives.cuboid import cuboid_mesher, cuboid_mesher_with_resolution
from surfmesh.primitives.cylinder import (
    cylinder_mesher_radial,
    cylinder_mesher_square_centered,
)
from surfmesh.primitives.disk import disk_mesher_radial, disk_mesher_square_centered
from surfmesh.primitives.sphere import sphere_mesher_from_projection, sphere_mesher_from_radial

__all__ = [
    # Core
    "QuadMesh",
    "extract_vertices_faces",
    # Geometry — curves
    "arc_edges",
    "circumference_edges",
    "rectangle_perimeter",
    # Geometry — grid
    "mesh_between_edges",
    "quad_faces_from_edges",
    # Operations — revolve
    "circular_revolve",
    "revolve_curve_along_path",
    # Operations — transform
    "convert_2d_face_to_3d",
    "flip_normals",
    "rotate_z",
    "scale",
    "translate",
    # Primitives — disk
    "disk_mesher_radial",
    "disk_mesher_square_centered",
    # Primitives — cuboid
    "cuboid_mesher",
    "cuboid_mesher_with_resolution",
    # Primitives — cylinder
    "cylinder_mesher_radial",
    "cylinder_mesher_square_centered",
    # Primitives — sphere
    "sphere_mesher_from_projection",
    "sphere_mesher_from_radial",
]
