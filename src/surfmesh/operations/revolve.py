"""Surface-of-revolution mesh generators.

Revolving a 2-D profile curve around an axis is one of the most flexible
techniques for creating structured surface meshes.  The two public
functions here cover the most common cases:

* :func:`revolve_curve_along_path` — full control: the revolve path can
  vary in both angle and radius.
* :func:`circular_revolve` — convenience wrapper for the common case of
  a constant-radius circular revolution around the Z-axis.

Both functions return a raw ``(n_panels, 4, 3)`` array compatible with
every other function in the library.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from surfmesh.core.validate import (
    validate_curve_2d,
    validate_positive_int,
    validate_revolve_path,
)

__all__ = [
    "circular_revolve",
    "revolve_curve_along_path",
]


def revolve_curve_along_path(
    curve: ArrayLike,
    revolve_path: ArrayLike,
) -> np.ndarray:
    """Revolve a 2-D profile curve along an arbitrary polar path.

    Each point on *revolve_path* defines a ``(angle, radius)`` position in
    polar coordinates.  The 2-D *curve* is placed at each path position and
    the resulting quad faces connect adjacent path stations.

    Parameters
    ----------
    curve:
        2-D profile curve, shape ``(n_curve_pts, 2)``.  Columns are
        interpreted as ``(local_radius, axial_z)``.
    revolve_path:
        Polar path, shape ``(n_path_pts, 2)``.  Columns are
        ``(angle_rad, radial_scale)``.

    Returns
    -------
    ndarray of shape ``((n_path_pts - 1) * (n_curve_pts - 1), 4, 3)``
        3-D quad-face panel array.

    Raises
    ------
    ValueError
        If *curve* is not ``(n, 2)`` or *revolve_path* is not ``(m, 2)``.

    Examples
    --------
    >>> import numpy as np
    >>> from surfmesh.operations.revolve import revolve_curve_along_path
    >>> curve = np.array([[1.0, 0.0], [1.0, 1.0]])
    >>> path  = np.array([[0.0, 1.0], [np.pi / 2, 1.0]])
    >>> mesh  = revolve_curve_along_path(curve, path)
    >>> mesh.shape
    (1, 4, 3)
    """
    curve = np.asarray(curve, dtype=float)
    revolve_path = np.asarray(revolve_path, dtype=float)

    validate_curve_2d(curve)
    validate_revolve_path(revolve_path)

    # Curve radii and axial positions for adjacent pairs of profile points.
    x0, z0 = curve[1:, 0], curve[1:, 1]
    x1, z1 = curve[:-1, 0], curve[:-1, 1]

    # Shape: (4 corners, 3 xyz-components, n_curve_segments)
    curve_matrix = np.array([
        [x0, x0, z0],
        [x0, x0, z0],
        [x1, x1, z1],
        [x1, x1, z1],
    ])

    # Convert revolve path from polar to Cartesian.
    angles, radii = revolve_path.T
    path_x = radii * np.cos(angles)
    path_y = radii * np.sin(angles)

    x_start, x_end = path_x[:-1], path_x[1:]
    y_start, y_end = path_y[:-1], path_y[1:]
    ones = np.ones_like(x_start)

    # Shape: (4 corners, 3 xyz-components, n_path_segments)
    path_matrix = np.array([
        [x_end,   y_end,   ones],
        [x_start, y_start, ones],
        [x_start, y_start, ones],
        [x_end,   y_end,   ones],
    ])

    # Einsum: combine (curve segment, path segment) → 3-D vertex positions.
    # curve_matrix: (4, 3, n_curve)  → axes: v, a, n
    # path_matrix:  (4, 3, n_path)   → axes: v, a, g
    # output:       (g, n, v, a)     → reshaped to (g*n, 4, 3)
    mesh = np.einsum("van,vag->gnva", curve_matrix, path_matrix)
    return mesh.reshape(-1, 4, 3)


def circular_revolve(
    curve: ArrayLike,
    segment_resolution: int,
    start_angle: float = 0.0,
    end_angle: float = 2.0 * np.pi,
) -> np.ndarray:
    """Revolve a 2-D profile curve around the Z-axis along a circular arc.

    This is a high-level convenience wrapper around
    :func:`revolve_curve_along_path`.  The revolve path is a uniform
    circular arc with unit radial scale, so the curve's own radial
    coordinates control the final geometry.

    Parameters
    ----------
    curve:
        2-D profile curve, shape ``(n_curve_pts, 2)``.  Columns are
        ``(radius, axial_z)``.  At least 2 points required.
    segment_resolution:
        Number of angular divisions (quad strips) around the axis.
        Must be >= 1.
    start_angle:
        Starting angle in radians.  Default ``0.0``.
    end_angle:
        Ending angle in radians.  Default ``2π`` (full circle).

    Returns
    -------
    ndarray of shape ``(segment_resolution * (n_curve_pts - 1), 4, 3)``
        3-D quad-face panel array.

    Raises
    ------
    ValueError
        If *curve* is not ``(n, 2)``, or *segment_resolution* < 1.

    Examples
    --------
    >>> import numpy as np
    >>> from surfmesh.operations.revolve import circular_revolve
    >>> curve = np.array([[1.0, -1.0], [1.0, 1.0]])   # vertical cylinder wall
    >>> mesh  = circular_revolve(curve, segment_resolution=8)
    >>> mesh.shape
    (8, 4, 3)
    """
    curve = np.asarray(curve, dtype=float)
    validate_curve_2d(curve)
    validate_positive_int(segment_resolution, "segment_resolution")

    angles = np.linspace(start_angle, end_angle, segment_resolution + 1)
    radii = np.ones_like(angles)  # unit-radius path; curve radii do the work
    revolve_path = np.stack([angles, radii], axis=1)

    return revolve_curve_along_path(curve, revolve_path)
