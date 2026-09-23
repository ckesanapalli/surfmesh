import numpy as np
from numpy.typing import ArrayLike


def convert_2d_face_to_3d(quad_2d_mesh: np.ndarray, axis: int, offset: float) -> np.ndarray:
    """
    Convert a 2D quadrilateral mesh to a 3D mesh by adding a fixed coordinate.
    """
    face_count = quad_2d_mesh.shape[0]
    quads_3d_mesh = np.empty((face_count, 4, 3), dtype=float)

    match axis:
        case 0:
            quads_3d_mesh[:, :, 0] = offset
            quads_3d_mesh[:, :, 1] = quad_2d_mesh[:, :, 0]
            quads_3d_mesh[:, :, 2] = quad_2d_mesh[:, :, 1]
        case 1:
            quads_3d_mesh[:, :, 0] = quad_2d_mesh[:, :, 1]
            quads_3d_mesh[:, :, 1] = offset
            quads_3d_mesh[:, :, 2] = quad_2d_mesh[:, :, 0]
        case 2:
            quads_3d_mesh[:, :, 0] = quad_2d_mesh[:, :, 0]
            quads_3d_mesh[:, :, 1] = quad_2d_mesh[:, :, 1]
            quads_3d_mesh[:, :, 2] = offset

        case _:
            axis_error_msg = f"fixed_axis must be 0 (x), 1 (y), or 2 (z). Got {axis}."
            raise ValueError(axis_error_msg)

    return quads_3d_mesh


def quad_faces_from_edges(u_coords: ArrayLike, v_coords: ArrayLike) -> np.ndarray:
    """
    Generate quadrilateral faces on a grid where one axis is fixed,
    with counter-clockwise vertex ordering.

    Parameters
    ----------
    u_coords : ArrayLike
        Coordinates along the first varying axis (horizontal direction).
    v_coords : ArrayLike
        Coordinates along the second varying axis (vertical direction).
    fixed_axis : int
        The index of the fixed axis (0 = x, 1 = y, 2 = z).
    fixed_value : float
        The fixed coordinate value for the constant axis.

    Returns
    -------
    np.ndarray
        Shape (N, 4, 2) array of 2D quad vertices, with each face ordered counter-clockwise.

    Examples
    --------
    >>> import numpy as np
    >>> from surfmesh.cuboid import quad_faces_from_edges
    >>> u = np.array([0.0, 1.0])
    >>> v = np.array([0.0, 1.0])
    >>> quads = quad_faces_from_edges(u, v)
    >>> print(quads.shape)
    (1, 4, 2)
    >>> print(quads[0])
    [[0. 0.]
     [1. 0.]
     [1. 1.]
     [0. 1.]]
    """

    # Meshgrid + quad generation
    uu, vv = np.meshgrid(u_coords, v_coords, indexing="ij")

    corners = [
        (uu[:-1, :-1], vv[:-1, :-1]),  # bottom-left
        (uu[1:, :-1], vv[1:, :-1]),  # bottom-right
        (uu[1:, 1:], vv[1:, 1:]),  # top-right
        (uu[:-1, 1:], vv[:-1, 1:]),  # top-left
    ]

    return np.stack([np.stack([x, y], axis=-1).reshape(-1, 2) for x, y in corners], axis=1)


def mesh_between_edges(edges: ArrayLike, radial_resolution: int) -> np.ndarray:
    """
    Generate a quadrilateral mesh between two open polygonal edges by interpolation. The edges can be 2D or 3D.

    Parameters
    ----------
    edges : ArrayLike, shape (2, axis_idx, vertex_idx)
        The starting and ending boundaries to interpolate between (edge_idx, axis_idx, vertex_idx).
    radial_resolution : int
        Number of layers between the two edges.

    Returns
    -------
    quads : ndarray of shape (N_layers * N_segments + 1, 4, 2)
        Quadrilateral faces connecting the two edges, with each face ordered counter-clockwise.
    """
    edges = np.asarray(edges, dtype=float)

    if edges.ndim != 3:
        msg = f"edges must have shape (2, axis_count, vertex_count), got shape {edges.shape}."
        raise ValueError(msg)
    if radial_resolution < 1:
        msg = f"radial_resolution must be at least 1, got {radial_resolution}."
        raise ValueError(msg)

    # correct interpolation
    edge1, edge2 = edges[0], edges[1]
    axis_count = edge1.shape[0]

    weights = np.linspace(0, 1, radial_resolution + 1)
    interpolated = edge1[..., np.newaxis] * (1 - weights) + edge2[..., np.newaxis] * weights
    # shape: (2, N_vertices, radial_resolution)

    return (
        np
        .array([
            interpolated[:, 1:, :-1],
            interpolated[:, 1:, 1:],
            interpolated[:, :-1, 1:],
            interpolated[:, :-1, :-1],
        ])
        .transpose(2, 3, 0, 1)
        .reshape(-1, 4, axis_count)
    )


def rectangle_perimeter(length_edge: ArrayLike, width_edge: ArrayLike) -> np.ndarray:
    """
    Generate the perimeter of a rectangle.

    Parameters
    ----------
    length_edge : ArrayLike
        Length edge vertices of the rectangle along the x-axis (m, ...).
    width_edge : ArrayLike
        Width edge vertices of the rectangle along the y-axis (n, ...).

    Returns
    -------
    perimeter : ndarray of shape (..., m * n * 2 + 1)
        Coordinates of the rectangle's corners in counter-clockwise order.

    Examples
    --------
    >>> length_edge = np.array([0, 1])
    >>> width_edge = np.array([0, 1])
    >>> perimeter = rectangle_perimeter(length_edge, width_edge)
    >>> print(perimeter.shape)
    (2, 5)
    >>> print(perimeter)
    [[1. 0. 0. 1. 1.]
     [1. 1. 0. 0. 1.]]
    """

    length_edge = np.asarray(length_edge, dtype=float)
    width_edge = np.asarray(width_edge, dtype=float)

    if length_edge.size < 2 or width_edge.size < 2:
        msg = f"length_edge and width_edge must have at least 2 points. Got Length edge: {length_edge.size}, Width edge: {width_edge.size}."
        raise ValueError(msg)

    if length_edge.shape[1:] != width_edge.shape[1:]:
        msg = f"length_edge and width_edge must have the same shape. Got Length edge: {length_edge.shape}, Width edge: {width_edge.shape}."
        raise ValueError(msg)

    length_edge = np.asarray(length_edge, dtype=float)
    width_edge = np.asarray(width_edge, dtype=float)

    length_min = length_edge[0]
    width_min = width_edge[0]
    length_max = length_edge[-1]
    width_max = width_edge[-1]

    length_edge_strip = length_edge[:-1]
    width_edge_strip = width_edge[:-1]
    length_edge_rev_strip = np.flip(length_edge)[:-1]
    width_edge_rev_strip = np.flip(width_edge)[:-1]

    length_ones = np.ones_like(length_edge_strip)
    width_ones = np.ones_like(width_edge_strip)

    return np.hstack([
        [length_edge_rev_strip, width_max * length_ones],  # Top edge
        [length_min * width_ones, width_edge_rev_strip],  # Left edge
        [length_edge_strip, width_min * length_ones],  # Bottom edge
        [length_max * width_ones, width_edge_strip],  # Right edge
        [[length_max], [width_max]],  # Top right corner
    ])
