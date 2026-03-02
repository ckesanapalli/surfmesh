"""Tests for surfmesh.geometry.curves — circumference_edges, arc_edges, rectangle_perimeter."""

import numpy as np
import pytest

from surfmesh.geometry.curves import arc_edges, circumference_edges, rectangle_perimeter


# ---------------------------------- #
# circumference_edges
# ---------------------------------- #

def test_circumference_edges_basic() -> None:
    pts = circumference_edges(1.0, 12)
    assert pts.shape == (2, 12)


def test_circumference_edges_values() -> None:
    circ = circumference_edges(1.0, 4, start_angle=0)
    expected = np.array(
        [[ 1. , -0.5, -0.5,  1. ],
         [ 0. ,  0.9, -0.9, -0. ]]
    )
    np.testing.assert_allclose(circ.round(1), expected)


def test_circumference_edges_zero_radius() -> None:
    pts = circumference_edges(0.0, 8)
    assert np.allclose(pts, 0.0)


def test_circumference_edges_negative_radius() -> None:
    with pytest.raises(ValueError, match="radius must be non-negative"):
        circumference_edges(-1.0, 12)


def test_circumference_edges_zero_segment_resolution() -> None:
    with pytest.raises(ValueError, match="segment_resolution must be a positive integer"):
        circumference_edges(1.0, 0)


def test_circumference_edges_clockwise() -> None:
    cw  = circumference_edges(1.0, 8, counter_clockwise=False)
    ccw = circumference_edges(1.0, 8, counter_clockwise=True)
    # clockwise path runs opposite direction — x values should differ
    assert not np.allclose(cw, ccw)


# ---------------------------------- #
# arc_edges
# ---------------------------------- #

def test_arc_edges_basic_shape() -> None:
    import math
    pts = arc_edges(1.0, 0.0, math.pi, 5)
    assert pts.shape == (2, 5)


def test_arc_edges_endpoints() -> None:
    import math
    pts = arc_edges(2.0, 0.0, math.pi / 2, 3)
    np.testing.assert_allclose(pts[:, 0], [2.0, 0.0], atol=1e-12)   # start at angle=0
    np.testing.assert_allclose(pts[:, -1], [0.0, 2.0], atol=1e-12)  # end at angle=π/2


def test_arc_edges_invalid_radius() -> None:
    with pytest.raises(ValueError, match="radius must be strictly positive"):
        arc_edges(0.0, 0.0, 1.0, 4)


def test_arc_edges_too_few_points() -> None:
    with pytest.raises(ValueError, match="n_points must be at least 2"):
        arc_edges(1.0, 0.0, 1.0, 1)


# ---------------------------------- #
# rectangle_perimeter
# ---------------------------------- #

def test_rectangle_perimeter_basic() -> None:
    length_edge = np.array([0.0, 1.0])
    width_edge = np.array([0.0, 1.0])
    result = rectangle_perimeter(length_edge, width_edge)
    assert result.shape == (2, 5)


def test_rectangle_perimeter_shape_multi_element() -> None:
    l_edge = np.linspace(0.0, 1.0, 5)
    w_edge = np.linspace(0.0, 1.0, 5)
    result = rectangle_perimeter(l_edge, w_edge)
    # 4 * (n-1) + 1 = 4*4 + 1 = 17
    assert result.shape == (2, 17)


def test_rectangle_perimeter_mismatched_shapes() -> None:
    length_edge = np.array([[0.0, 1.0]])  # 2D row vs 1D col
    width_edge = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="must have matching shapes"):
        rectangle_perimeter(length_edge, width_edge)


def test_rectangle_perimeter_too_few_points() -> None:
    length_edge = np.array([0.0])
    width_edge = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="at least 2 points"):
        rectangle_perimeter(length_edge, width_edge)
