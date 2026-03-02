"""surfmesh.geometry — 2-D geometric building blocks.

Submodules
----------
curves
    Boundary curve generators: :func:`~surfmesh.geometry.curves.circumference_edges`,
    :func:`~surfmesh.geometry.curves.arc_edges`,
    :func:`~surfmesh.geometry.curves.rectangle_perimeter`.
grid
    Structured quad grid generators: :func:`~surfmesh.geometry.grid.quad_faces_from_edges`,
    :func:`~surfmesh.geometry.grid.mesh_between_edges`.
"""

from surfmesh.geometry.curves import arc_edges, circumference_edges, rectangle_perimeter
from surfmesh.geometry.grid import mesh_between_edges, quad_faces_from_edges

__all__ = [
    "arc_edges",
    "circumference_edges",
    "mesh_between_edges",
    "quad_faces_from_edges",
    "rectangle_perimeter",
]
