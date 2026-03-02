"""2-D quadrilateral grid generators.

These are the foundational building blocks for all surface meshers in the
library.  They operate purely in 2-D coordinate space; higher-level
functions embed the result in 3-D via
:func:`~surfmesh.operations.transform.convert_2d_face_to_3d`.

Functions
---------
quad_faces_from_edges
    Create a structured quad mesh from two 1-D coordinate arrays.
mesh_between_edges
    Linearly interpolate a quad mesh between two open boundary curves.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from surfmesh.core.validate import validate_edges_array, validate_positive_int

__all__ = [
    "quad_faces_from_edges",
    "mesh_between_edges",
]


def quad_faces_from_edges(
    u_coords: ArrayLike,
    v_coords: ArrayLike,
) -> np.ndarray:
    """Create a structured quad mesh from two 1-D coordinate arrays.

    The grid is built using a meshgrid with ``indexing="ij"`` so that
    *u_coords* varies along the first spatial axis and *v_coords* along the
    second.  Vertices within each face are ordered **counter-clockwise**
    (bottom-left → bottom-right → top-right → top-left).

    Parameters
    ----------
    u_coords:
        1-D coordinate array for the first (horizontal) axis.
        Requires at least 2 elements.
    v_coords:
        1-D coordinate array for the second (vertical) axis.
        Requires at least 2 elements.

    Returns
    -------
    ndarray of shape ``((len(u) - 1) * (len(v) - 1), 4, 2)``
        Quadrilateral face array.  Each face is defined by four 2-D
        vertices in counter-clockwise order.

    Examples
    --------
    >>> import numpy as np
    >>> from surfmesh.geometry.grid import quad_faces_from_edges
    >>> u = np.array([0.0, 1.0])
    >>> v = np.array([0.0, 1.0])
    >>> quads = quad_faces_from_edges(u, v)
    >>> quads.shape
    (1, 4, 2)
    >>> quads[0]
    array([[0., 0.],
           [1., 0.],
           [1., 1.],
           [0., 1.]])
    """
    u_coords = np.asarray(u_coords, dtype=float)
    v_coords = np.asarray(v_coords, dtype=float)

    uu, vv = np.meshgrid(u_coords, v_coords, indexing="ij")

    # Four corners of each cell in counter-clockwise order.
    corners = [
        (uu[:-1, :-1], vv[:-1, :-1]),  # bottom-left
        (uu[1:, :-1],  vv[1:, :-1]),   # bottom-right
        (uu[1:, 1:],   vv[1:, 1:]),    # top-right
        (uu[:-1, 1:],  vv[:-1, 1:]),   # top-left
    ]

    return np.stack(
        [np.stack([x, y], axis=-1).reshape(-1, 2) for x, y in corners],
        axis=1,
    )


def mesh_between_edges(
    edges: ArrayLike,
    radial_resolution: int,
) -> np.ndarray:
    """Linearly interpolate a quad mesh between two open boundary curves.

    Given a *start* edge and an *end* edge (both described by the same set
    of coordinate axes and the same number of vertices), this function
    inserts ``radial_resolution`` evenly-spaced intermediate layers,
    producing a structured quad mesh that fills the region between the two
    boundaries.

    Parameters
    ----------
    edges:
        Array of shape ``(2, n_axes, n_vertices)`` where ``edges[0]`` is
        the start boundary and ``edges[1]`` is the end boundary.
        *n_axes* is 2 for 2-D edges and 3 for 3-D edges.
    radial_resolution:
        Number of quad layers to insert between the two edges.
        Must be >= 1.

    Returns
    -------
    ndarray of shape ``(radial_resolution * (n_vertices - 1), 4, n_axes)``
        Quadrilateral panel array ordered counter-clockwise.

    Raises
    ------
    ValueError
        If *edges* does not have shape ``(2, n_axes, n_vertices)`` or
        *radial_resolution* < 1.

    Examples
    --------
    >>> import numpy as np
    >>> from surfmesh.geometry.grid import mesh_between_edges
    >>> inner = np.array([[0.0, 1.0], [0.0, 0.0]])   # horizontal segment
    >>> outer = np.array([[0.0, 1.0], [1.0, 1.0]])   # shifted up by 1
    >>> quads = mesh_between_edges(np.stack([inner, outer]), radial_resolution=2)
    >>> quads.shape
    (2, 4, 2)
    """
    edges = np.asarray(edges, dtype=float)
    validate_edges_array(edges)
    validate_positive_int(radial_resolution, "radial_resolution")

    edge_start, edge_end = edges[0], edges[1]
    n_axes = edge_start.shape[0]

    # Interpolation weights: shape (radial_resolution + 1,)
    weights = np.linspace(0.0, 1.0, radial_resolution + 1)

    # Interpolated layers: shape (n_axes, n_vertices, radial_resolution + 1)
    interpolated = (
        edge_start[..., np.newaxis] * (1.0 - weights)
        + edge_end[..., np.newaxis] * weights
    )

    # Build quad faces from adjacent layers and adjacent vertices.
    return (
        np.array([
            interpolated[:, 1:, :-1],   # v0: next-vertex, current-layer
            interpolated[:, 1:, 1:],    # v1: next-vertex, next-layer
            interpolated[:, :-1, 1:],   # v2: current-vertex, next-layer
            interpolated[:, :-1, :-1],  # v3: current-vertex, current-layer
        ])
        .transpose(2, 3, 0, 1)
        .reshape(-1, 4, n_axes)
    )
