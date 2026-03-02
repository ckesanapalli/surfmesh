"""surfmesh.operations — mesh construction and geometric transformation operations.

Submodules
----------
revolve
    Revolution / sweep operations:
    :func:`~surfmesh.operations.revolve.circular_revolve`,
    :func:`~surfmesh.operations.revolve.revolve_curve_along_path`.
transform
    Geometric transformations applied to raw panel arrays:
    :func:`~surfmesh.operations.transform.convert_2d_face_to_3d`,
    :func:`~surfmesh.operations.transform.translate`,
    :func:`~surfmesh.operations.transform.scale`,
    :func:`~surfmesh.operations.transform.rotate_z`,
    :func:`~surfmesh.operations.transform.flip_normals`.
"""

from surfmesh.operations.revolve import circular_revolve, revolve_curve_along_path
from surfmesh.operations.transform import (
    convert_2d_face_to_3d,
    flip_normals,
    rotate_z,
    scale,
    translate,
)

__all__ = [
    "circular_revolve",
    "convert_2d_face_to_3d",
    "flip_normals",
    "revolve_curve_along_path",
    "rotate_z",
    "scale",
    "translate",
]
