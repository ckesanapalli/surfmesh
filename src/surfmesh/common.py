from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def extract_vertices_faces(mesh: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract unique vertices and reindex faces from a mesh.

    This function flattens the mesh, identifies unique vertices,
    and maps the original mesh faces to indices of the unique vertices.

    Parameters
    ----------
    mesh : ArrayLike
        Input mesh array of shape (n_faces, n_vertices_per_face, n_dimensions),
        typically (n, 4, 3) for quadrilateral faces in 3D.

    Returns
    -------
    vertices : np.ndarray
        Array of unique vertex coordinates, shape (n_unique_vertices, n_dimensions).

    faces : np.ndarray
        Array of face indices into the `vertices` array, shape (n_faces, n_vertices_per_face).

    Examples
    --------
    >>> import numpy as np
    >>> from surfmesh import extract_vertices_faces
    >>>
    >>> mesh = np.array([
    ...     [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    ...     [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    ... ])
    >>> vertices, faces = extract_vertices_faces(mesh)
    >>> print(vertices.shape)
    (8, 3)
    >>> print(faces.shape)
    (2, 4)
    """
    mesh = np.asarray(mesh)
    if mesh.ndim != 3:
        msg = f"Expected mesh to have 3 dimensions (n_faces, n_vertices_per_face, n_dimensions), got {mesh.ndim}"
        raise ValueError(msg)

    n_faces, n_vertices_per_face, n_dimensions = mesh.shape

    # Flatten all vertices
    flat_mesh = mesh.reshape(-1, n_dimensions)

    # Find unique vertices and their inverse indices
    vertices, inverse_indices = np.unique(flat_mesh, axis=0, return_inverse=True)

    # Reshape inverse indices back to (n_faces, n_vertices_per_face)
    faces = inverse_indices.reshape(n_faces, n_vertices_per_face)

    return vertices, faces
