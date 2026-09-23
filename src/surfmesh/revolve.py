from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def revolve_curve_along_path(curve: ArrayLike, revolve_path: ArrayLike) -> np.ndarray:
    """
    Revolve a 2D curve along a given path to create a 3D surface mesh.

    Parameters
    ----------
    curve : ArrayLike
        An (n, 2) array representing (x, z) coordinates of the 2D curve.
    revolve_path : ArrayLike
        An (m, 2) array representing (angle in radians, radius) polar coordinates of the revolve path.

    Returns
    -------
    mesh : np.ndarray
        An array of shape ((m-1)*(n-1), 4, 3) representing quad faces of the 3D surface mesh.

    Raises
    ------
    ValueError
        If `curve` or `revolve_path` do not have the correct shape or dimensions.

    Examples
    --------
    >>> curve = np.array([[1, 2], [3, 4]])
    >>> revolve_path = np.array([[0, 1], [np.pi/2, 2]])
    >>> mesh = revolve_curve_along_path(curve, revolve_path)
    >>> mesh.shape
    (1, 4, 3)
    """
    curve = np.asarray(curve)
    revolve_path = np.asarray(revolve_path)

    if curve.ndim != 2 or curve.shape[1] != 2:
        msg = f"Curve must be a (n, 2) array. Got {curve.shape}"
        raise ValueError(msg)
    if revolve_path.ndim != 2 or revolve_path.shape[1] != 2:
        msg = f"Revolve path must be a (m, 2) array. Got {revolve_path.shape}"
        raise ValueError(msg)

    x0, z0 = curve[1:, 0], curve[1:, 1]
    x1, z1 = curve[:-1, 0], curve[:-1, 1]

    curve_matrix = np.array([
        [x0, x0, z0],
        [x0, x0, z0],
        [x1, x1, z1],
        [x1, x1, z1],
    ])

    angles, radii = revolve_path.T
    path_x = radii * np.cos(angles)
    path_y = radii * np.sin(angles)

    x_start, x_end = path_x[:-1], path_x[1:]
    y_start, y_end = path_y[:-1], path_y[1:]
    ones = np.ones_like(x_start)

    path_matrix = np.array([
        [x_end, y_end, ones],
        [x_start, y_start, ones],
        [x_start, y_start, ones],
        [x_end, y_end, ones],
    ])

    mesh = np.einsum("van,vag->gnva", curve_matrix, path_matrix)
    return mesh.reshape(-1, 4, 3)


def circular_revolve(curve: ArrayLike, segment_resolution: int, start_angle: float = 0.0, end_angle: float = 2 * np.pi) -> np.ndarray:
    """
    Revolve a 2D curve around the Z-axis counter-clockwise along a circular path.

    Parameters
    ----------
    curve : ArrayLike
        An (n, 2) array representing (radius, axial) coordinates of the curve to revolve.
    segment_resolution : int
        Number of angular divisions for the revolution.
    start_angle : float, optional
        Starting angle for the revolution, in radians. Default is 0.
    end_angle : float, optional
        Ending angle for the revolution, in radians. Default is 2*pi (full circle).

    Returns
    -------
    vertices : ndarray of shape (segment_resolution * (len(curve) - 1), 4, 3)
        Array of 3D vertices defining quad faces formed by revolving the curve.

    Examples
    --------
    >>> import numpy as np
    >>> from surfmesh.revolve import circular_revolve
    >>>
    >>> curve = np.array([
    ...     [1.0, 0.0],  # radius, axial position
    ...     [2.0, 0.0],
    ...     [2.0, 1.0]
    ... ])
    >>> segment_resolution = 8
    >>> mesh = circular_revolve(curve, segment_resolution)
    >>> print(mesh.shape)
    (16, 4, 3)

    This revolves a simple "L"-shaped curve 360 degrees around the Z-axis,
    creating a closed cylindrical surface.
    """

    curve = np.asarray(curve, dtype=float)

    if curve.ndim != 2 or curve.shape[1] != 2:
        msg = f"curve must be a (n, 2) array. Got {curve.shape}."
        raise ValueError(msg)

    if segment_resolution < 1:
        msg = f"segment_resolution must be at least 1. Got {segment_resolution}."
        raise ValueError(msg)

    # Create uniform angular revolve path
    angles = np.linspace(start_angle, end_angle, segment_resolution + 1)
    radii = np.ones_like(angles)  # unit radius for angular path
    revolve_path = np.stack([angles, radii], axis=1)

    return revolve_curve_along_path(curve, revolve_path)
