"""surfmesh.primitives — standard 3-D surface-mesh primitives.

Each submodule provides one or more mesher functions that return a raw
``(n_panels, 4, 3)`` quad-face array, which can be wrapped in a
:class:`~surfmesh.core.mesh.QuadMesh` for BEM analysis or passed directly
to the transformation utilities in :mod:`surfmesh.operations`.

Submodules
----------
disk
    Filled circular disks:
    :func:`~surfmesh.primitives.disk.disk_mesher_radial`,
    :func:`~surfmesh.primitives.disk.disk_mesher_square_centered`.
cuboid
    Closed rectangular boxes:
    :func:`~surfmesh.primitives.cuboid.cuboid_mesher`,
    :func:`~surfmesh.primitives.cuboid.cuboid_mesher_with_resolution`.
cylinder
    Closed cylinders with disk caps:
    :func:`~surfmesh.primitives.cylinder.cylinder_mesher_radial`,
    :func:`~surfmesh.primitives.cylinder.cylinder_mesher_square_centered`.
sphere
    Spherical surfaces:
    :func:`~surfmesh.primitives.sphere.sphere_mesher_from_projection`,
    :func:`~surfmesh.primitives.sphere.sphere_mesher_from_radial`.
"""

from surfmesh.primitives.cuboid import cuboid_mesher, cuboid_mesher_with_resolution
from surfmesh.primitives.cylinder import (
    cylinder_mesher_radial,
    cylinder_mesher_square_centered,
)
from surfmesh.primitives.disk import disk_mesher_radial, disk_mesher_square_centered
from surfmesh.primitives.sphere import sphere_mesher_from_projection, sphere_mesher_from_radial

__all__ = [
    "cuboid_mesher",
    "cuboid_mesher_with_resolution",
    "cylinder_mesher_radial",
    "cylinder_mesher_square_centered",
    "disk_mesher_radial",
    "disk_mesher_square_centered",
    "sphere_mesher_from_projection",
    "sphere_mesher_from_radial",
]
