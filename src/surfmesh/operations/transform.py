"""Geometric transformation operations for surface meshes.

All functions accept and return raw panel arrays of shape
``(n_panels, n_verts_per_face, n_dims)`` so they compose naturally with
every mesher in the library.

Functions
---------
convert_2d_face_to_3d
    Embed a 2-D quad mesh in 3-D by inserting a fixed coordinate.
translate
    Shift all panel vertices by a constant 3-D vector.
scale
    Uniformly or per-axis scale all panel vertices.
rotate_z
    Rotate all panel vertices around the Z-axis.
flip_normals
    Reverse vertex ordering to flip outward panel normals.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from surfmesh.core.validate import validate_axis

__all__ = [
    "convert_2d_face_to_3d",
    "flip_normals",
    "rotate_z",
    "scale",
    "translate",
]


def convert_2d_face_to_3d(
    quad_2d_mesh: ArrayLike,
    axis: int,
    offset: float,
) -> np.ndarray:
    """Embed a 2-D quadrilateral mesh in 3-D space.

    Inserts a fixed coordinate value (*offset*) along the given *axis* and
    preserves the two existing coordinates in the remaining positions.

    Axis mapping
    ~~~~~~~~~~~~
    * ``axis=0`` (x fixed): output columns are ``[offset, u, v]``
    * ``axis=1`` (y fixed): output columns are ``[v,      offset, u]``
    * ``axis=2`` (z fixed): output columns are ``[u,      v,     offset]``

    Parameters
    ----------
    quad_2d_mesh:
        2-D quad mesh array of shape ``(n_faces, 4, 2)``.
    axis:
        Spatial axis to set to *offset*.  Must be 0 (x), 1 (y), or 2 (z).
    offset:
        Fixed coordinate value along *axis*.

    Returns
    -------
    ndarray of shape ``(n_faces, 4, 3)``
        3-D quad mesh.

    Raises
    ------
    ValueError
        If *axis* is not 0, 1, or 2.

    Examples
    --------
    >>> import numpy as np
    >>> from surfmesh.operations.transform import convert_2d_face_to_3d
    >>> mesh_2d = np.array([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]])
    >>> mesh_3d = convert_2d_face_to_3d(mesh_2d, axis=2, offset=5.0)
    >>> mesh_3d.shape
    (1, 4, 3)
    >>> mesh_3d[0, 0]
    array([0., 0., 5.])
    """
    validate_axis(axis)
    quad_2d_mesh = np.asarray(quad_2d_mesh, dtype=float)
    n_faces = quad_2d_mesh.shape[0]
    out = np.empty((n_faces, 4, 3), dtype=float)

    match axis:
        case 0:
            out[:, :, 0] = offset
            out[:, :, 1] = quad_2d_mesh[:, :, 0]
            out[:, :, 2] = quad_2d_mesh[:, :, 1]
        case 1:
            out[:, :, 0] = quad_2d_mesh[:, :, 1]
            out[:, :, 1] = offset
            out[:, :, 2] = quad_2d_mesh[:, :, 0]
        case 2:
            out[:, :, 0] = quad_2d_mesh[:, :, 0]
            out[:, :, 1] = quad_2d_mesh[:, :, 1]
            out[:, :, 2] = offset

    return out


def translate(mesh: ArrayLike, translation: ArrayLike) -> np.ndarray:
    """Translate all panel vertices by a constant 3-D vector.

    Parameters
    ----------
    mesh:
        Panel array of shape ``(n_faces, n_verts, 3)``.
    translation:
        1-D array-like of length 3: ``[dx, dy, dz]``.

    Returns
    -------
    ndarray
        Translated panel array with the same shape as *mesh*.

    Raises
    ------
    ValueError
        If *translation* does not have exactly 3 elements.

    Examples
    --------
    >>> import numpy as np
    >>> from surfmesh.operations.transform import translate
    >>> mesh = np.zeros((2, 4, 3))
    >>> shifted = translate(mesh, [1.0, 2.0, 3.0])
    >>> shifted[0, 0]
    array([1., 2., 3.])
    """
    mesh = np.asarray(mesh, dtype=float)
    translation = np.asarray(translation, dtype=float)
    if translation.shape != (3,):
        msg = f"translation must have shape (3,), got {translation.shape}."
        raise ValueError(msg)
    return mesh + translation


def scale(mesh: ArrayLike, factor: float | ArrayLike) -> np.ndarray:
    """Scale all panel vertices, either uniformly or independently per axis.

    Parameters
    ----------
    mesh:
        Panel array of shape ``(n_faces, n_verts, 3)``.
    factor:
        A scalar for uniform scaling, or a length-3 array-like
        ``[sx, sy, sz]`` for per-axis scaling.

    Returns
    -------
    ndarray
        Scaled panel array with the same shape as *mesh*.

    Examples
    --------
    >>> import numpy as np
    >>> from surfmesh.operations.transform import scale
    >>> mesh = np.ones((1, 4, 3))
    >>> scale(mesh, 2.0)[0, 0]
    array([2., 2., 2.])
    >>> scale(mesh, [1.0, 2.0, 3.0])[0, 0]
    array([1., 2., 3.])
    """
    mesh = np.asarray(mesh, dtype=float)
    factor = np.asarray(factor, dtype=float)
    return mesh * factor


def rotate_z(mesh: ArrayLike, angle: float) -> np.ndarray:
    """Rotate all panel vertices around the Z-axis.

    Parameters
    ----------
    mesh:
        Panel array of shape ``(n_faces, n_verts, 3)``.
    angle:
        Rotation angle in radians (positive = counter-clockwise when viewed
        from +z).

    Returns
    -------
    ndarray
        Rotated panel array with the same shape as *mesh*.

    Examples
    --------
    >>> import numpy as np
    >>> from surfmesh.operations.transform import rotate_z
    >>> mesh = np.array([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]]])
    >>> rotated = rotate_z(mesh, np.pi / 2)
    >>> np.allclose(rotated[0, 0], [0., 1., 0.], atol=1e-10)
    True
    """
    mesh = np.asarray(mesh, dtype=float)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rot = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a,  cos_a, 0.0],
        [0.0,    0.0,   1.0],
    ])
    return mesh @ rot.T


def flip_normals(mesh: ArrayLike) -> np.ndarray:
    """Reverse vertex ordering to flip outward panel normals.

    Reverses the per-face vertex sequence from
    ``[v0, v1, v2, v3]`` to ``[v3, v2, v1, v0]``, which mirrors the
    surface normal direction.  Useful when a closed mesh has inward-facing
    normals that need to be corrected.

    Parameters
    ----------
    mesh:
        Panel array of shape ``(n_faces, n_verts, n_dims)``.

    Returns
    -------
    ndarray
        Panel array with the same shape but reversed vertex order per face.

    Examples
    --------
    >>> import numpy as np
    >>> from surfmesh.operations.transform import flip_normals
    >>> mesh = np.arange(12).reshape(1, 4, 3).astype(float)
    >>> flipped = flip_normals(mesh)
    >>> bool((flipped[0] == mesh[0, ::-1]).all())
    True
    """
    mesh = np.asarray(mesh, dtype=float)
    return mesh[:, ::-1, :]
