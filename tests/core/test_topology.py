"""Tests for surfmesh.core.topology — extract_vertices_faces."""

import numpy as np
import pytest

from surfmesh.core.topology import extract_vertices_faces


def test_extract_vertices_faces_quad_mesh() -> None:
    """Basic extraction for a quad mesh."""
    mesh = np.array([
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    ])
    vertices, faces = extract_vertices_faces(mesh)
    assert vertices.shape == (8, 3)
    assert faces.shape == (2, 4)
    assert np.all(faces.max() < len(vertices))


def test_extract_vertices_faces_triangle_mesh() -> None:
    """Extraction for a triangle mesh (3 vertices per face)."""
    mesh = np.array([
        [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
    ])
    vertices, faces = extract_vertices_faces(mesh)
    assert vertices.shape == (6, 3)
    assert faces.shape == (2, 3)
    assert np.all(faces.max() < len(vertices))


def test_extract_vertices_faces_duplicate_faces() -> None:
    """Multiple faces that share vertices."""
    mesh = np.array([
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]],
    ])
    vertices, faces = extract_vertices_faces(mesh)
    assert vertices.shape[1] == 3
    assert faces.shape == (2, 4)
    assert np.all(faces.max() < len(vertices))


def test_extract_vertices_faces_invalid_input_shape() -> None:
    """ValueError raised for a 2-D input (missing face dimension)."""
    mesh = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(ValueError, match="must be 3-dimensional"):
        extract_vertices_faces(mesh)


def test_extract_vertices_faces_random_mesh() -> None:
    """Random mesh: output indices are valid."""
    mesh = np.random.rand(10, 4, 3)
    vertices, faces = extract_vertices_faces(mesh)
    assert vertices.shape[1] == 3
    assert faces.shape == (10, 4)
    assert np.all(faces.max() < len(vertices))


def test_extract_vertices_faces_preserves_structure() -> None:
    """Face indices correctly reconstruct original vertices."""
    mesh = np.array([
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],  # same face repeated
    ])
    vertices, faces = extract_vertices_faces(mesh)
    assert np.array_equal(faces[0], faces[1])


@pytest.mark.parametrize("mesh_shape", [(5, 4, 3), (20, 3, 3), (1, 4, 2)])
def test_extract_vertices_faces_various_shapes(mesh_shape: tuple[int, ...]) -> None:
    """Round-trip: vertices[faces] must equal the original mesh."""
    mesh = np.random.random(mesh_shape)
    vertices, faces = extract_vertices_faces(mesh)
    assert vertices.ndim == 2
    assert faces.shape[0] == mesh_shape[0]
    assert np.all(faces.max() < len(vertices))
    np.testing.assert_array_equal(mesh, vertices[faces])
