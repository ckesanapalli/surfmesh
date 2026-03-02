"""Quadrilateral surface meshers for disk (filled-circle) geometries.

Two complementary strategies are provided:

* **Radial** (:func:`disk_mesher_radial`) — purely polar mesh with one quad
  layer per radial division.  Simple and uniform but produces highly
  skewed cells near the centre.
* **Square-centred** (:func:`disk_mesher_square_centered`) — embeds a
  structured square mesh at the centre and smoothly blends out to a
  circular boundary.  Eliminates the degenerate centre cells and gives
  far better element quality for BEM and FEM analyses.
"""

from __future__ import annotations

import numpy as np

from surfmesh.core.validate import validate_positive, validate_positive_int, validate_ratio
from surfmesh.geometry.curves import circumference_edges, rectangle_perimeter
from surfmesh.geometry.grid import mesh_between_edges, quad_faces_from_edges

__all__ = [
    "disk_mesher_radial",
    "disk_mesher_square_centered",
]


def disk_mesher_radial(
    radius: float,
    radial_resolution: int,
    segment_resolution: int,
) -> np.ndarray:
    """Generate a 2-D circular mesh using concentric polar rings.

    The mesh is built in polar coordinates ``(r, θ)`` using
    :func:`~surfmesh.geometry.grid.quad_faces_from_edges` and then
    converted to Cartesian ``(x, y)``.

    Parameters
    ----------
    radius:
        Radius of the disk.  Must be positive.
    radial_resolution:
        Number of quad rings along the radial direction (centre to edge).
        Must be >= 1.
    segment_resolution:
        Number of quad cells in the angular direction (around the circle).
        Must be >= 1.

    Returns
    -------
    ndarray of shape ``(radial_resolution * segment_resolution, 4, 2)``
        2-D quadrilateral panel array in ``(x, y)`` coordinates.

    Raises
    ------
    ValueError
        If *radius* is negative, or either resolution is < 1.
        A zero *radius* is permitted and produces an all-zero mesh.

    Examples
    --------
    >>> from surfmesh.primitives.disk import disk_mesher_radial
    >>> mesh = disk_mesher_radial(radius=1.0, radial_resolution=4, segment_resolution=8)
    >>> mesh.shape
    (32, 4, 2)
    """
    if radius < 0:
        msg = f"radius must be non-negative, got {radius!r}."
        raise ValueError(msg)
    validate_positive_int(radial_resolution, "radial_resolution")
    validate_positive_int(segment_resolution, "segment_resolution")

    r_divisions = np.linspace(0.0, radius, radial_resolution + 1)
    theta_divisions = np.linspace(0.0, 2.0 * np.pi, segment_resolution + 1)

    # Build faces in polar coords, then map to Cartesian.
    polar_faces = quad_faces_from_edges(r_divisions, theta_divisions)
    r = polar_faces[..., 0]
    theta = polar_faces[..., 1]
    return np.stack((r * np.cos(theta), r * np.sin(theta)), axis=-1)


def disk_mesher_square_centered(
    radius: float,
    square_resolution: int,
    radial_resolution: int,
    square_side_radius_ratio: float = 1.0,
    square_disk_rotation: float = 0.0,
) -> np.ndarray:
    """Generate a 2-D disk mesh with a structured square core.

    A square mesh fills the central region, and a radial transition zone
    (built with :func:`~surfmesh.geometry.grid.mesh_between_edges`)
    smoothly connects the square boundary to the circular perimeter.
    This produces significantly better element quality than a purely polar
    mesh and is the recommended choice for BEM analyses.

    Parameters
    ----------
    radius:
        Radius of the disk.  Must be positive.
    square_resolution:
        Number of quad cells along each side of the square core.
        Must be >= 1.
    radial_resolution:
        Number of radial transition layers between the square and the
        circular boundary.  Must be >= 1.
    square_side_radius_ratio:
        Half-side-length of the square as a fraction of *radius*.
        Must be in ``(0, 1]``.  Default ``1.0`` makes the square
        inscribed in the circle.
    square_disk_rotation:
        Additional rotation (in radians) applied to the circumference
        sample points relative to the square boundary.  Default ``0.0``.

    Returns
    -------
    ndarray of shape ``(N, 4, 2)``
        2-D quadrilateral panel array combining the square core and the
        radial transition zone.

    Raises
    ------
    ValueError
        If *radius* is not positive, any resolution is < 1, or
        *square_side_radius_ratio* is outside ``(0, 1]``.

    Examples
    --------
    >>> from surfmesh.primitives.disk import disk_mesher_square_centered
    >>> mesh = disk_mesher_square_centered(
    ...     radius=1.0, square_resolution=4, radial_resolution=4
    ... )
    >>> mesh.shape
    (80, 4, 2)
    """
    validate_positive(radius, "radius")
    validate_positive_int(square_resolution, "square_resolution")
    validate_positive_int(radial_resolution, "radial_resolution")
    validate_ratio(square_side_radius_ratio, "square_side_radius_ratio")

    square_side = radius * square_side_radius_ratio
    side_coords = np.linspace(-square_side / 2.0, square_side / 2.0, square_resolution + 1)

    # --- Square core mesh ---
    square_mesh = quad_faces_from_edges(side_coords, side_coords)

    # --- Circular boundary ---
    segment_resolution = square_resolution * 4 + 1
    start_angle = np.pi / 4 + square_disk_rotation
    circumference = circumference_edges(
        radius, segment_resolution, start_angle=start_angle, counter_clockwise=True
    )

    # --- Radial transition zone ---
    # np.flip aligns face normals with the square core mesh.
    square_boundary = rectangle_perimeter(side_coords, side_coords)
    radial_edges = np.flip(np.stack([square_boundary, circumference]), axis=1)
    radial_mesh = mesh_between_edges(radial_edges, radial_resolution)

    return np.vstack([square_mesh, radial_mesh])
