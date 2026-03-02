"""Mesh topology utilities.

Functions for extracting and working with the topological structure of a
surface mesh — unique vertex lists, face connectivity, and related helpers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from surfmesh.core.validate import validate_mesh_3d

__all__ = ["extract_vertices_faces"]


def extract_vertices_faces(mesh: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Extract unique vertices and a reindexed face array from a panel mesh.

    Takes the raw ``(n_faces, n_verts_per_face, n_dims)`` panel array that
    every mesher in this library returns and converts it to the compact
    ``(vertices, faces)`` representation that most downstream solvers and
    renderers expect.

    Parameters
    ----------
    mesh:
        Raw panel array of shape ``(n_faces, n_verts_per_face, n_dims)``.
        The most common case for BEM is ``(n_faces, 4, 3)`` — quad panels
        in three-dimensional space.

    Returns
    -------
    vertices : ndarray of shape ``(n_unique_vertices, n_dims)``
        De-duplicated vertex coordinate array.
    faces : ndarray of shape ``(n_faces, n_verts_per_face)``
        Integer index array mapping each face corner to the corresponding
        row in *vertices*.

    Raises
    ------
    ValueError
        If *mesh* does not have exactly 3 dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> from surfmesh.core.topology import extract_vertices_faces
    >>> mesh = np.array([
    ...     [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    ...     [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    ... ], dtype=float)
    >>> verts, faces = extract_vertices_faces(mesh)
    >>> verts.shape
    (8, 3)
    >>> faces.shape
    (2, 4)
    """
    mesh = np.asarray(mesh, dtype=float)
    validate_mesh_3d(mesh)

    n_faces, n_verts_per_face, n_dims = mesh.shape

    # Flatten all panel corners into a single vertex list, then de-duplicate.
    flat = mesh.reshape(-1, n_dims)
    vertices, inverse = np.unique(flat, axis=0, return_inverse=True)

    # Reshape the inverse-index map back to (n_faces, n_verts_per_face).
    faces = inverse.reshape(n_faces, n_verts_per_face)

    return vertices, faces
