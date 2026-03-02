"""Quadrilateral surface meshers for sphere geometries.

Two independent algorithms are provided:

* **Cube projection** (:func:`sphere_mesher_from_projection`) — sub-divides
  a cube and projects all vertices onto the sphere surface.  Produces
  nearly uniform panels, no polar singularities, and works well for full
  spheres.

* **Radial / spherical** (:func:`sphere_mesher_from_radial`) — revolves a
  latitude arc around the Z-axis.  Naturally parametrised, supports
  partial spheres (caps, bands), but panels shrink near the poles.

Both functions return the raw ``(n_panels, 4, 3)`` array.
"""

from __future__ import annotations

import numpy as np

from surfmesh.core.validate import validate_positive, validate_positive_int
from surfmesh.operations.revolve import circular_revolve
from surfmesh.primitives.cuboid import cuboid_mesher_with_resolution

__all__ = [
    "sphere_mesher_from_projection",
    "sphere_mesher_from_radial",
]


def sphere_mesher_from_projection(
    radius: float,
    resolution: int,
) -> np.ndarray:
    """Generate a sphere mesh via cube-to-sphere projection.

    Creates a subdivided unit-cube surface mesh with
    :func:`~surfmesh.primitives.cuboid.cuboid_mesher_with_resolution`, then
    normalises every vertex to lie on the sphere of the given *radius*.

    Parameters
    ----------
    radius:
        Sphere radius.  Must be positive.
    resolution:
        Number of quad cells along each edge of each cube face.
        Must be >= 1.

    Returns
    -------
    ndarray of shape ``(6 * resolution ** 2, 4, 3)``
        Sphere surface mesh panels.

    Raises
    ------
    ValueError
        If *radius* is not positive or *resolution* < 1.

    Examples
    --------
    >>> from surfmesh.primitives.sphere import sphere_mesher_from_projection
    >>> mesh = sphere_mesher_from_projection(radius=1.0, resolution=4)
    >>> mesh.shape
    (96, 4, 3)
    """
    validate_positive(radius, "radius")
    validate_positive_int(resolution, "resolution")

    cube_mesh = cuboid_mesher_with_resolution(
        length=2.0 * radius,
        width=2.0 * radius,
        height=2.0 * radius,
        resolution=resolution,
    )
    # Project each vertex radially onto the sphere.
    norms = np.linalg.norm(cube_mesh, axis=2, keepdims=True)
    return radius * cube_mesh / norms


def sphere_mesher_from_radial(
    radius: float,
    radial_resolution: int,
    segment_resolution: int,
    start_angle: float = 0.0,
    end_angle: float = 2.0 * np.pi,
) -> np.ndarray:
    """Generate a sphere mesh by revolving a latitude arc.

    Discretises a great-circle arc from the south pole (``-π/2``) to the
    north pole (``+π/2``) using *radial_resolution* + 1 latitude points,
    then revolves the arc around the Z-axis with
    :func:`~surfmesh.operations.revolve.circular_revolve`.

    Parameters
    ----------
    radius:
        Sphere radius.  Must be positive.
    radial_resolution:
        Number of latitude bands (quad rows from pole to pole).
        Must be >= 1.
    segment_resolution:
        Number of longitude segments (quad columns around the sphere).
        Must be >= 1.
    start_angle:
        Starting longitude for the revolution in radians.  Default ``0``.
    end_angle:
        Ending longitude for the revolution in radians.
        Default ``2π`` (complete sphere).

    Returns
    -------
    ndarray of shape ``(radial_resolution * segment_resolution, 4, 3)``
        Sphere surface mesh panels.

    Raises
    ------
    ValueError
        If *radius* is not positive, or either resolution is < 1.

    Examples
    --------
    >>> from surfmesh.primitives.sphere import sphere_mesher_from_radial
    >>> mesh = sphere_mesher_from_radial(radius=1.0, radial_resolution=8, segment_resolution=16)
    >>> mesh.shape
    (128, 4, 3)
    """
    validate_positive(radius, "radius")
    validate_positive_int(radial_resolution, "radial_resolution")
    validate_positive_int(segment_resolution, "segment_resolution")

    # Latitude arc from south pole to north pole.
    latitudes = np.linspace(-np.pi / 2.0, np.pi / 2.0, radial_resolution + 1)
    # Columns: (radius_in_xy_plane, z_coordinate)
    curve = radius * np.stack([np.cos(latitudes), np.sin(latitudes)], axis=1)

    return circular_revolve(
        curve, segment_resolution,
        start_angle=start_angle,
        end_angle=end_angle,
    )
