"""Centralized input validation utilities for surfmesh.

All public functions raise ``ValueError`` with descriptive, user-friendly
messages.  Keeping validation in one place ensures consistent error wording
across the entire package and makes it trivial to tighten or relax rules in
the future.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "as_float_array",
    "validate_positive",
    "validate_positive_int",
    "validate_1d_array",
    "validate_strictly_increasing",
    "validate_axis",
    "validate_mesh_3d",
    "validate_curve_2d",
    "validate_revolve_path",
    "validate_ratio",
    "validate_edges_array",
]


# ---------------------------------------------------------------------------
# Array coercion
# ---------------------------------------------------------------------------


def as_float_array(arr: ArrayLike, name: str) -> np.ndarray:
    """Convert *arr* to a float64 NumPy array.

    Parameters
    ----------
    arr:
        Any array-like object.
    name:
        Human-readable parameter name used in error messages.

    Returns
    -------
    np.ndarray
        Float64 array.

    Raises
    ------
    ValueError
        If *arr* cannot be converted to a numeric array.
    """
    try:
        return np.asarray(arr, dtype=float)
    except (TypeError, ValueError) as exc:
        msg = f"Could not convert {name!r} to a numeric array: {exc}"
        raise ValueError(msg) from exc


# ---------------------------------------------------------------------------
# Scalar validators
# ---------------------------------------------------------------------------


def validate_positive(value: float, name: str) -> None:
    """Raise ``ValueError`` if *value* is not strictly positive.

    Parameters
    ----------
    value:
        Scalar to check.
    name:
        Human-readable parameter name used in error messages.
    """
    if value <= 0:
        msg = f"{name} must be strictly positive, got {value!r}."
        raise ValueError(msg)


def validate_positive_int(value: int, name: str) -> None:
    """Raise ``ValueError`` if *value* is not a positive integer (>= 1).

    Parameters
    ----------
    value:
        Integer to check.
    name:
        Human-readable parameter name used in error messages.
    """
    if not isinstance(value, (int, np.integer)) or int(value) < 1:
        msg = f"{name} must be a positive integer (>= 1), got {value!r}."
        raise ValueError(msg)


def validate_ratio(value: float, name: str, lo: float = 0.0, hi: float = 1.0) -> None:
    """Raise ``ValueError`` if *value* is not in the half-open interval (*lo*, *hi*].

    Parameters
    ----------
    value:
        Scalar to check.
    name:
        Human-readable parameter name used in error messages.
    lo:
        Exclusive lower bound. Default ``0.0``.
    hi:
        Inclusive upper bound. Default ``1.0``.
    """
    if not (lo < value <= hi):
        msg = f"{name} must be in ({lo}, {hi}], got {value!r}."
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Axis validator
# ---------------------------------------------------------------------------


def validate_axis(axis: int, name: str = "axis") -> None:
    """Raise ``ValueError`` if *axis* is not 0, 1, or 2.

    Parameters
    ----------
    axis:
        Integer axis index to check.
    name:
        Human-readable parameter name used in error messages.
    """
    if axis not in (0, 1, 2):
        msg = f"{name} must be 0 (x), 1 (y), or 2 (z), got {axis!r}."
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Array shape validators
# ---------------------------------------------------------------------------


def validate_1d_array(arr: ArrayLike, name: str, min_length: int = 2) -> np.ndarray:
    """Coerce *arr* to a 1-D float array and validate its minimum length.

    Parameters
    ----------
    arr:
        Input to coerce and validate.
    name:
        Human-readable parameter name used in error messages.
    min_length:
        Minimum required number of elements. Default ``2``.

    Returns
    -------
    np.ndarray
        1-D float64 array with at least *min_length* elements.
    """
    arr = as_float_array(arr, name)
    if arr.ndim != 1:
        msg = f"{name} must be 1-dimensional, got shape {arr.shape}."
        raise ValueError(msg)
    if arr.size < min_length:
        msg = f"{name} must have at least {min_length} element(s), got {arr.size}."
        raise ValueError(msg)
    return arr


def validate_strictly_increasing(arr: np.ndarray, name: str) -> None:
    """Raise ``ValueError`` if *arr* is not strictly increasing.

    Parameters
    ----------
    arr:
        1-D array to check.
    name:
        Human-readable parameter name used in error messages.
    """
    if not np.all(np.diff(arr) > 0):
        msg = f"{name} must be strictly increasing."
        raise ValueError(msg)


def validate_mesh_3d(mesh: np.ndarray, name: str = "mesh") -> None:
    """Raise ``ValueError`` if *mesh* does not have exactly 3 dimensions.

    The expected layout is ``(n_faces, n_vertices_per_face, n_spatial_dims)``.

    Parameters
    ----------
    mesh:
        Array to validate.
    name:
        Human-readable parameter name used in error messages.
    """
    if mesh.ndim != 3:
        msg = (
            f"{name} must be 3-dimensional with shape "
            f"(n_faces, n_vertices_per_face, n_spatial_dims), "
            f"got shape {mesh.shape}."
        )
        raise ValueError(msg)


def validate_curve_2d(curve: np.ndarray, name: str = "curve") -> None:
    """Raise ``ValueError`` if *curve* is not a valid 2-column 2-D array.

    A valid curve has shape ``(n_points, 2)`` with at least 2 points.

    Parameters
    ----------
    curve:
        Array to validate.
    name:
        Human-readable parameter name used in error messages.
    """
    if curve.ndim != 2 or curve.shape[1] != 2:
        msg = f"{name} must have shape (n_points, 2), got {curve.shape}."
        raise ValueError(msg)
    if curve.shape[0] < 2:
        msg = f"{name} must have at least 2 points, got {curve.shape[0]}."
        raise ValueError(msg)


def validate_revolve_path(path: np.ndarray, name: str = "revolve_path") -> None:
    """Raise ``ValueError`` if *path* is not a valid revolve path.

    A valid revolve path has shape ``(n_steps, 2)`` with at least 2 steps,
    where each row is ``(angle_rad, radius)``.

    Parameters
    ----------
    path:
        Array to validate.
    name:
        Human-readable parameter name used in error messages.
    """
    if path.ndim != 2 or path.shape[1] != 2:
        msg = f"{name} must have shape (n_steps, 2), got {path.shape}."
        raise ValueError(msg)
    if path.shape[0] < 2:
        msg = f"{name} must have at least 2 steps, got {path.shape[0]}."
        raise ValueError(msg)


def validate_edges_array(edges: np.ndarray, name: str = "edges") -> None:
    """Raise ``ValueError`` if *edges* is not a valid pair-of-edges array.

    A valid edges array has shape ``(2, n_axes, n_vertices)`` — exactly two
    edges (start and end), each described by the same number of coordinate
    axes and vertices.

    Parameters
    ----------
    edges:
        Array to validate.
    name:
        Human-readable parameter name used in error messages.
    """
    if edges.ndim != 3:
        msg = (
            f"{name} must have shape (2, n_axes, n_vertices), "
            f"got shape {edges.shape}."
        )
        raise ValueError(msg)
    if edges.shape[0] != 2:
        msg = (
            f"{name} must have exactly 2 edges (start and end) along axis 0, "
            f"got {edges.shape[0]}."
        )
        raise ValueError(msg)
