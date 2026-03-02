"""Tests for surfmesh.geometry.grid — quad_faces_from_edges, mesh_between_edges."""

import numpy as np
import pytest

from surfmesh.geometry.grid import mesh_between_edges, quad_faces_from_edges


# ---------------------------------- #
# quad_faces_from_edges
# ---------------------------------- #

def test_quad_faces_from_edges_basic() -> None:
    u = np.array([0.0, 1.0])
    v = np.array([0.0, 1.0])
    result = quad_faces_from_edges(u, v)
    expected = np.array([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]])
    np.testing.assert_array_equal(result, expected)


def test_quad_faces_from_edges_rectangular_grid() -> None:
    u = np.array([0.0, 1.0, 2.0])
    v = np.array([0.0, 1.0])
    result = quad_faces_from_edges(u, v)
    assert result.shape == (2, 4, 2)


def test_quad_faces_from_edges_zero_area() -> None:
    u = np.array([0.0])
    v = np.array([0.0])
    result = quad_faces_from_edges(u, v)
    assert result.shape == (0, 4, 2)


def test_quad_faces_from_edges_counter_clockwise_order() -> None:
    """Vertices should be in CCW order: BL → BR → TR → TL."""
    u = np.array([0.0, 1.0])
    v = np.array([0.0, 1.0])
    quads = quad_faces_from_edges(u, v)
    bl, br, tr, tl = quads[0]
    assert bl[0] < tr[0]   # x increases from left to right
    assert bl[1] < tr[1]   # y increases from bottom to top


# ---------------------------------- #
# mesh_between_edges
# ---------------------------------- #

def test_mesh_between_edges_basic() -> None:
    edge_start = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]).T  # shape (2, 4)
    edge_end   = np.array([[0., .5], [.5, 0.], [1., .5], [.5, 1.]]).T
    edges = np.stack([edge_start, edge_end])  # (2, 2, 4)
    result = mesh_between_edges(edges, radial_resolution=3)
    # (n_vertices-1) * radial_resolution = 3 * 3 = 9 faces
    assert result.shape == (9, 4, 2)


def test_mesh_between_edges_single_layer() -> None:
    edge_start = np.array([[0., 0.], [1., 0.]]).T  # (2, 2)
    edge_end   = np.array([[0., 1.], [1., 1.]]).T
    edges = np.stack([edge_start, edge_end])
    result = mesh_between_edges(edges, radial_resolution=1)
    assert result.shape == (1, 4, 2)


def test_mesh_between_edges_3d_edges() -> None:
    edge_start = np.array([[0., 0., 0.], [1., 0., 0.]]).T  # (3, 2)
    edge_end   = np.array([[0., 1., 0.], [1., 1., 0.]]).T
    edges = np.stack([edge_start, edge_end])
    result = mesh_between_edges(edges, radial_resolution=2)
    assert result.shape == (2, 4, 3)


def test_mesh_between_edges_invalid_ndim() -> None:
    bad_edges = np.zeros((3, 4))  # 2D, not 3D
    with pytest.raises(ValueError, match="edges must have shape"):
        mesh_between_edges(bad_edges, radial_resolution=2)


def test_mesh_between_edges_invalid_first_dim() -> None:
    bad_edges = np.zeros((3, 2, 4))  # first dim is 3, not 2
    with pytest.raises(ValueError, match="exactly 2 edges"):
        mesh_between_edges(bad_edges, radial_resolution=2)


def test_mesh_between_edges_zero_resolution() -> None:
    edges = np.zeros((2, 2, 4))
    with pytest.raises(ValueError, match="radial_resolution must be a positive integer"):
        mesh_between_edges(edges, radial_resolution=0)
