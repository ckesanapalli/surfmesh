"""Quadrilateral surface mesher for axis-aligned cuboid geometries.

A cuboid (box) is the fundamental test case for BEM solvers.  Two
interfaces are provided:

* :func:`cuboid_mesher` — explicit coordinate arrays for full control over
  non-uniform grid spacing (grading).
* :func:`cuboid_mesher_with_resolution` — specify dimensions and an integer
  resolution; coordinates are generated automatically via ``linspace``.

Both functions return all six faces of the closed box surface with
consistent counter-clockwise vertex ordering (outward normals).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from surfmesh.core.validate import (
    validate_1d_array,
    validate_positive_int,
    validate_strictly_increasing,
)
from surfmesh.geometry.grid import quad_faces_from_edges
from surfmesh.operations.transform import convert_2d_face_to_3d

__all__ = [
    "cuboid_mesher",
    "cuboid_mesher_with_resolution",
]


def cuboid_mesher(
    x_coords: ArrayLike,
    y_coords: ArrayLike,
    z_coords: ArrayLike,
) -> np.ndarray:
    """Generate a closed cuboid surface mesh from explicit coordinate arrays.

    Creates all six axis-aligned faces of the cuboid spanned by
    *x_coords* × *y_coords* × *z_coords*.  Quad vertices are ordered
    counter-clockwise when viewed from the outside so that normals point
    outward.

    Parameters
    ----------
    x_coords:
        1-D strictly increasing array of x-axis node positions.
        Must have at least 2 elements.
    y_coords:
        1-D strictly increasing array of y-axis node positions.
        Must have at least 2 elements.
    z_coords:
        1-D strictly increasing array of z-axis node positions.
        Must have at least 2 elements.

    Returns
    -------
    ndarray of shape ``(N, 4, 3)``
        All six cuboid face meshes concatenated.  The total panel count is
        ``2 * (nx*ny + ny*nz + nz*nx)`` where ``nx = len(x_coords) - 1``
        etc.

    Raises
    ------
    ValueError
        If any coordinate array is not 1-D, has fewer than 2 elements,
        or is not strictly increasing.

    Examples
    --------
    >>> from surfmesh.primitives.cuboid import cuboid_mesher
    >>> import numpy as np
    >>> mesh = cuboid_mesher([0., 1.], [0., 1.], [0., 1.])
    >>> mesh.shape
    (6, 4, 3)
    """
    x = validate_1d_array(x_coords, "x_coords")
    y = validate_1d_array(y_coords, "y_coords")
    z = validate_1d_array(z_coords, "z_coords")
    validate_strictly_increasing(x, "x_coords")
    validate_strictly_increasing(y, "y_coords")
    validate_strictly_increasing(z, "z_coords")

    # 2-D face meshes in (u, v) coordinates for each axis pairing.
    xy = quad_faces_from_edges(x, y)
    yz = quad_faces_from_edges(y, z)
    zx = quad_faces_from_edges(z, x)

    # Flip vertex order to get the opposite-facing (inward→outward) panels.
    yx = np.flip(xy, axis=1)
    zy = np.flip(yz, axis=1)
    xz = np.flip(zx, axis=1)

    # Coordinate extremes used for placing each face.
    xf0, xf1 = x[0], x[-1]
    yf0, yf1 = y[0], y[-1]
    zf0, zf1 = z[0], z[-1]

    return np.concatenate([
        convert_2d_face_to_3d(yx, axis=2, offset=zf0),  # bottom (−z)
        convert_2d_face_to_3d(xy, axis=2, offset=zf1),  # top    (+z)
        convert_2d_face_to_3d(zy, axis=0, offset=xf0),  # left   (−x)
        convert_2d_face_to_3d(yz, axis=0, offset=xf1),  # right  (+x)
        convert_2d_face_to_3d(xz, axis=1, offset=yf0),  # front  (−y)
        convert_2d_face_to_3d(zx, axis=1, offset=yf1),  # back   (+y)
    ], axis=0)


def cuboid_mesher_with_resolution(
    length: float,
    width: float,
    height: float,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    resolution: int | tuple[int, int, int] = (1, 1, 1),
) -> np.ndarray:
    """Generate a closed cuboid surface mesh from dimensions and resolution.

    A convenience wrapper around :func:`cuboid_mesher` that auto-generates
    uniform coordinate arrays using ``np.linspace``.

    Parameters
    ----------
    length:
        Dimension along the x-axis (total span).
    width:
        Dimension along the y-axis (total span).
    height:
        Dimension along the z-axis (total span).
    origin:
        Centre point of the cuboid in 3-D space.  Default ``(0, 0, 0)``.
    resolution:
        Number of quad cells along each axis.  Either a single integer
        (applied to all three axes) or a 3-element tuple ``(nx, ny, nz)``.
        All values must be positive.

    Returns
    -------
    ndarray of shape ``(N, 4, 3)``
        Closed cuboid surface mesh.

    Raises
    ------
    ValueError
        If *resolution* is not a scalar or 3-element array, or if any
        resolution value is not positive.

    Examples
    --------
    >>> from surfmesh.primitives.cuboid import cuboid_mesher_with_resolution
    >>> mesh = cuboid_mesher_with_resolution(2.0, 1.0, 1.0, resolution=2)
    >>> mesh.shape
    (24, 4, 3)
    """
    res = np.asarray(resolution, dtype=int)
    if res.ndim == 0:
        res = np.full(3, int(res))
    elif res.shape != (3,):
        msg = "resolution must be a single int or a 3-element tuple of ints."
        raise ValueError(msg)
    if np.any(res <= 0):
        msg = "All resolution values must be positive integers."
        raise ValueError(msg)

    validate_positive_int(int(res[0]), "resolution[0]")
    validate_positive_int(int(res[1]), "resolution[1]")
    validate_positive_int(int(res[2]), "resolution[2]")

    res_x, res_y, res_z = res
    ox, oy, oz = origin

    x_coords = np.linspace(ox - length / 2.0, ox + length / 2.0, res_x + 1)
    y_coords = np.linspace(oy - width  / 2.0, oy + width  / 2.0, res_y + 1)
    z_coords = np.linspace(oz - height / 2.0, oz + height / 2.0, res_z + 1)

    return cuboid_mesher(x_coords, y_coords, z_coords)
