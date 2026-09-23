from typing import TYPE_CHECKING

import numpy as np

from .edge import convert_2d_face_to_3d, quad_faces_from_edges

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def cuboid_mesher(x_coords: ArrayLike, y_coords: ArrayLike, z_coords: ArrayLike) -> np.ndarray:
    """
    Generate a full cuboid surface mesh using explicit coordinate arrays along each axis.

    This function creates quadrilateral surface faces based on 1D coordinate arrays for the
    x, y, and z axes. The surface mesh includes all six sides of the cuboid spanned by
    the given coordinates. Vertex order for each quad is counter-clockwise.

    Parameters
    ----------
    x_coords : ArrayLike of float
        1D strictly increasing array of x-axis positions for vertical planes (YZ-facing).
    y_coords : ArrayLike of float
        1D strictly increasing array of y-axis positions for horizontal planes (XZ-facing).
    z_coords : ArrayLike of float
        1D strictly increasing array of z-axis positions for depth planes (XY-facing).

    Returns
    -------
    np.ndarray
        Array of shape (N, 4, 3), where N is the number of quadrilateral faces.
        Each face is defined by four 3D points in counter-clockwise order.

    Raises
    ------
    ValueError
        If any coordinate array is not 1D, has fewer than 2 elements,
        or is not strictly increasing.

    Examples
    --------
    >>> x = [0.0, 1.0, 2.0]
    >>> y = [0.0, 1.0]
    >>> z = [0.0, 0.5, 1.0]
    >>> faces = cuboid_mesher(x, y, z)
    >>> print(faces.shape)  # 6 faces total from the cuboid
    (16, 4, 3)
    """
    coords = [np.asarray(c, dtype=float) for c in (x_coords, y_coords, z_coords)]

    for name, arr in zip(("x", "y", "z"), coords, strict=False):
        if arr.ndim != 1:
            msg = f"{name}_coords must be 1D, got shape {arr.shape}."
            raise ValueError(msg)
        if arr.size < 2:
            msg = f"{name}_coords must have at least 2 points, got {arr.size}."
            raise ValueError(msg)
        if not np.all(np.diff(arr) > 0):
            msg = f"{name}_coords must be strictly increasing."
            raise ValueError(msg)

    x, y, z = coords

    xy = quad_faces_from_edges(x, y)
    yz = quad_faces_from_edges(y, z)
    zx = quad_faces_from_edges(z, x)
    yx = np.flip(xy, axis=1)
    zy = np.flip(yz, axis=1)
    xz = np.flip(zx, axis=1)

    xf0, xf1 = x[0], x[-1]
    yf0, yf1 = y[0], y[-1]
    zf0, zf1 = z[0], z[-1]

    return np.concatenate(
        [
            convert_2d_face_to_3d(yx, axis=2, offset=zf0),
            convert_2d_face_to_3d(xy, axis=2, offset=zf1),
            convert_2d_face_to_3d(zy, axis=0, offset=xf0),
            convert_2d_face_to_3d(yz, axis=0, offset=xf1),
            convert_2d_face_to_3d(xz, axis=1, offset=yf0),
            convert_2d_face_to_3d(zx, axis=1, offset=yf1),
        ],
        axis=0,
    )


def cuboid_mesher_with_resolution(
    length: float, width: float, height: float, origin: tuple[float, float, float] = (0.0, 0.0, 0.0), resolution: int | tuple[int, int, int] = (1, 1, 1)
) -> np.ndarray:
    """
    Generate a 3D surface mesh of a cuboid with quadrilateral faces based on resolution.

    Parameters
    ----------
    length : float
        Length of the cuboid along the x-axis.
    width : float
        Width of the cuboid along the y-axis.
    height : float
        Height of the cuboid along the z-axis.
    origin : tuple of 3 floats, optional
        Center point of the cuboid in 3D space. Default is (0.0, 0.0, 0.0).
    resolution : int or tuple of 3 ints
        Number of subdivisions along each axis. If a single int is provided,
        it's used for all axes.

    Returns
    -------
    np.ndarray
        Surface mesh of shape (N, 4, 3), where N is the number of quad faces.

    Examples
    --------
    >>> from surfmesh import cuboid_mesher_with_resolution
    >>> mesh = cuboid_mesher_with_resolution(2.0, 1.0, 1.0, resolution=2)
    >>> mesh.shape
    (24, 4, 3)

    >>> mesh = cuboid_mesher_with_resolution(2.0, 1.0, 1.0, resolution=[2, 1, 2])
    >>> mesh.shape[1:]
    (4, 3)
    """
    resolution_array = np.array(resolution, dtype=int)

    if resolution_array.ndim == 0:
        resolution_array = np.full(3, resolution_array)
    elif resolution_array.shape != (3,):
        msg = "resolution must be a single int or an array-like of three ints."
        raise ValueError(msg)
    if np.any(resolution_array <= 0):
        msg = "resolution must contain only positive values."
        raise ValueError(msg)

    res_x, res_y, res_z = resolution_array
    ox, oy, oz = origin

    x_edge_point = ox + length / 2.0
    y_edge_point = oy + width / 2.0
    z_edge_point = oz + height / 2.0

    x_coords = np.linspace(-x_edge_point, x_edge_point, res_x + 1)
    y_coords = np.linspace(-y_edge_point, y_edge_point, res_y + 1)
    z_coords = np.linspace(-z_edge_point, z_edge_point, res_z + 1)

    return cuboid_mesher(x_coords, y_coords, z_coords)
