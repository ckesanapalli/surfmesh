"""2-D curve generators for use as meshing boundaries.

These functions produce the coordinate arrays (edges/boundaries) that are
passed into grid generators such as
:func:`~surfmesh.geometry.grid.mesh_between_edges`.

Functions
---------
circumference_edges
    Uniformly-spaced points around a circular arc or full circle.
arc_edges
    Alias for :func:`circumference_edges` with explicit start/end angles
    for improved clarity when generating partial arcs.
rectangle_perimeter
    Counter-clockwise perimeter of a rectangle defined by two 1-D edge
    arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from surfmesh.core.validate import validate_positive, validate_positive_int

__all__ = [
    "circumference_edges",
    "arc_edges",
    "rectangle_perimeter",
]


def circumference_edges(
    radius: float,
    segment_resolution: int,
    start_angle: float = 0.0,
    counter_clockwise: bool = True,
) -> np.ndarray:
    """Generate uniformly-spaced points around a full circle in 2-D.

    Parameters
    ----------
    radius:
        Radius of the circle.  Must be non-negative.
    segment_resolution:
        Number of sample points on the circumference.
    start_angle:
        Starting angle in radians.  Default ``0.0``.
    counter_clockwise:
        If ``True`` (default) the points are ordered counter-clockwise;
        if ``False`` the direction is clockwise.

    Returns
    -------
    ndarray of shape ``(2, segment_resolution)``
        Row 0 contains x-coordinates, row 1 contains y-coordinates.

    Raises
    ------
    ValueError
        If *radius* is negative or *segment_resolution* < 1.

    Examples
    --------
    >>> from surfmesh.geometry.curves import circumference_edges
    >>> pts = circumference_edges(1.0, 4)
    >>> pts.shape
    (2, 4)
    """
    if radius < 0:
        msg = f"radius must be non-negative, got {radius!r}."
        raise ValueError(msg)
    validate_positive_int(segment_resolution, "segment_resolution")

    direction = 1.0 if counter_clockwise else -1.0
    angles = start_angle + direction * np.linspace(0.0, 2.0 * np.pi, segment_resolution)
    return np.array([np.cos(angles), np.sin(angles)]) * radius


def arc_edges(
    radius: float,
    start_angle: float,
    end_angle: float,
    n_points: int,
) -> np.ndarray:
    """Generate uniformly-spaced points along a circular arc in 2-D.

    A more readable alternative to :func:`circumference_edges` when only a
    partial arc is needed.

    Parameters
    ----------
    radius:
        Radius of the arc.  Must be positive.
    start_angle:
        Starting angle in radians (inclusive).
    end_angle:
        Ending angle in radians (inclusive).
    n_points:
        Total number of sample points (including both endpoints).
        Must be >= 2.

    Returns
    -------
    ndarray of shape ``(2, n_points)``
        Row 0 contains x-coordinates, row 1 contains y-coordinates.

    Raises
    ------
    ValueError
        If *radius* is not positive or *n_points* < 2.

    Examples
    --------
    >>> import numpy as np
    >>> from surfmesh.geometry.curves import arc_edges
    >>> pts = arc_edges(1.0, 0.0, np.pi, 5)
    >>> pts.shape
    (2, 5)
    """
    validate_positive(radius, "radius")
    if n_points < 2:
        msg = f"n_points must be at least 2, got {n_points!r}."
        raise ValueError(msg)

    angles = np.linspace(start_angle, end_angle, n_points)
    return np.array([np.cos(angles), np.sin(angles)]) * radius


def rectangle_perimeter(
    length_edge: ArrayLike,
    width_edge: ArrayLike,
) -> np.ndarray:
    """Build the counter-clockwise perimeter of a rectangle.

    The rectangle is defined by two 1-D coordinate arrays — one along the
    length direction (x) and one along the width direction (y).  The
    returned perimeter starts at the top-right corner and proceeds
    counter-clockwise.

    Parameters
    ----------
    length_edge:
        1-D array of x-axis sample coordinates (at least 2 elements).
    width_edge:
        1-D array of y-axis sample coordinates (at least 2 elements).
        Must have the same length as *length_edge*.

    Returns
    -------
    ndarray of shape ``(2, 4 * (n - 1) + 1)``
        Row 0 = x-coordinates, row 1 = y-coordinates of the perimeter
        loop (the last point closes back to the start).

    Raises
    ------
    ValueError
        If either edge has fewer than 2 elements, or their sizes differ.

    Examples
    --------
    >>> import numpy as np
    >>> from surfmesh.geometry.curves import rectangle_perimeter
    >>> length_edge = np.array([0.0, 1.0])
    >>> width_edge  = np.array([0.0, 1.0])
    >>> perim = rectangle_perimeter(length_edge, width_edge)
    >>> perim.shape
    (2, 5)
    """
    length_edge = np.asarray(length_edge, dtype=float)
    width_edge = np.asarray(width_edge, dtype=float)

    if length_edge.size < 2 or width_edge.size < 2:
        msg = (
            f"length_edge and width_edge must each have at least 2 points. "
            f"Got length_edge: {length_edge.size}, width_edge: {width_edge.size}."
        )
        raise ValueError(msg)

    if length_edge.shape[1:] != width_edge.shape[1:]:
        msg = (
            f"length_edge and width_edge must have matching shapes. "
            f"Got {length_edge.shape} vs {width_edge.shape}."
        )
        raise ValueError(msg)

    length_min, length_max = length_edge[0], length_edge[-1]
    width_min, width_max = width_edge[0], width_edge[-1]

    # Strips excluding the last point (to avoid repeated corners)
    length_strip = length_edge[:-1]
    width_strip = width_edge[:-1]
    length_rev_strip = np.flip(length_edge)[:-1]
    width_rev_strip = np.flip(width_edge)[:-1]

    length_ones = np.ones_like(length_strip)
    width_ones = np.ones_like(width_strip)

    return np.hstack([
        [length_rev_strip,            width_max * length_ones],   # top edge  (right→left)
        [length_min * width_ones,     width_rev_strip],            # left edge (top→bottom)
        [length_strip,                width_min * length_ones],    # bottom edge (left→right)
        [length_max * width_ones,     width_strip],                # right edge (bottom→top)
        [[length_max],                [width_max]],                # close at top-right corner
    ])
