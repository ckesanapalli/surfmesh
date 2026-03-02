"""Tests for surfmesh.core.mesh — QuadMesh."""

import numpy as np
import pytest

from surfmesh.core.mesh import QuadMesh
from surfmesh.primitives.sphere import sphere_mesher_from_projection


def _unit_cube_panel() -> np.ndarray:
    """Single quad panel on the XY-plane (z=0), CCW vertices."""
    return np.array([[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]]])


def test_quad_mesh_basic_construction() -> None:
    panels = _unit_cube_panel()
    mesh = QuadMesh(panels)
    assert mesh.n_panels == 1


def test_quad_mesh_repr() -> None:
    mesh = QuadMesh(_unit_cube_panel())
    r = repr(mesh)
    assert "QuadMesh" in r
    assert "n_panels=1" in r


def test_quad_mesh_invalid_ndim() -> None:
    with pytest.raises(ValueError, match="must be 3-dimensional"):
        QuadMesh(np.ones((4, 3)))


def test_quad_mesh_invalid_verts_per_face() -> None:
    with pytest.raises(ValueError, match="exactly 4 vertices"):
        QuadMesh(np.ones((2, 3, 3)))  # triangles, not quads


def test_quad_mesh_read_only_panels() -> None:
    mesh = QuadMesh(_unit_cube_panel())
    with pytest.raises((ValueError, TypeError)):
        mesh.panels[0, 0, 0] = 99.0


def test_quad_mesh_centroids_shape() -> None:
    panels = sphere_mesher_from_projection(1.0, 3)
    mesh = QuadMesh(panels)
    assert mesh.centroids.shape == (mesh.n_panels, 3)


def test_quad_mesh_normals_unit_length() -> None:
    panels = sphere_mesher_from_projection(1.0, 4)
    mesh = QuadMesh(panels)
    norms = np.linalg.norm(mesh.normals, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_quad_mesh_areas_positive() -> None:
    panels = sphere_mesher_from_projection(1.0, 4)
    mesh = QuadMesh(panels)
    assert np.all(mesh.areas > 0)


def test_quad_mesh_total_area_sphere_approx() -> None:
    """Total area of a high-res projected sphere should be close to 4π."""
    panels = sphere_mesher_from_projection(1.0, 20)
    mesh = QuadMesh(panels)
    np.testing.assert_allclose(mesh.total_area, 4.0 * np.pi, rtol=1e-2)


def test_quad_mesh_vertices_faces_roundtrip() -> None:
    panels = sphere_mesher_from_projection(1.0, 3)
    mesh = QuadMesh(panels)
    reconstructed = mesh.vertices[mesh.faces]
    np.testing.assert_array_equal(panels, reconstructed)


def test_quad_mesh_n_vertices() -> None:
    mesh = QuadMesh(_unit_cube_panel())
    assert mesh.n_vertices == 4


def test_quad_mesh_to_dict_keys() -> None:
    mesh = QuadMesh(_unit_cube_panel())
    d = mesh.to_dict()
    expected_keys = {"panels", "vertices", "faces", "centroids", "normals",
                     "areas", "n_panels", "n_vertices", "total_area"}
    assert expected_keys == set(d.keys())


def test_quad_mesh_flat_panel_normal_direction() -> None:
    """A CCW quad on the XY-plane should have a +Z normal."""
    mesh = QuadMesh(_unit_cube_panel())
    assert mesh.normals[0, 2] > 0  # z-component should be positive
