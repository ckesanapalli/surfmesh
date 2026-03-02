"""Tests for surfmesh.primitives.sphere."""

import numpy as np
import pytest

from surfmesh.primitives.sphere import sphere_mesher_from_projection, sphere_mesher_from_radial


# ---------------------------------- #
# sphere_mesher_from_projection
# ---------------------------------- #

def test_sphere_projection_basic() -> None:
    mesh = sphere_mesher_from_projection(radius=1.0, resolution=10)
    assert isinstance(mesh, np.ndarray)
    assert mesh.ndim == 3
    assert mesh.shape[1:] == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(mesh, axis=2), 1.0, rtol=1e-10)


def test_sphere_projection_invalid_radius() -> None:
    with pytest.raises(ValueError, match="radius must be strictly positive"):
        sphere_mesher_from_projection(0, 10)
    with pytest.raises(ValueError, match="radius must be strictly positive"):
        sphere_mesher_from_projection(-1, 10)


def test_sphere_projection_invalid_resolution() -> None:
    with pytest.raises(ValueError, match="resolution must be a positive integer"):
        sphere_mesher_from_projection(1.0, 0)
    with pytest.raises(ValueError, match="resolution must be a positive integer"):
        sphere_mesher_from_projection(1.0, -5)


def test_sphere_projection_radius_scaling() -> None:
    mesh = sphere_mesher_from_projection(radius=3.5, resolution=5)
    np.testing.assert_allclose(np.linalg.norm(mesh, axis=2), 3.5, rtol=1e-10)


def test_sphere_projection_panel_count() -> None:
    mesh = sphere_mesher_from_projection(radius=1.0, resolution=5)
    assert mesh.shape[0] == 6 * 5 ** 2


# ---------------------------------- #
# sphere_mesher_from_radial
# ---------------------------------- #

def test_sphere_radial_basic() -> None:
    mesh = sphere_mesher_from_radial(radius=1.0, radial_resolution=10, segment_resolution=10)
    assert isinstance(mesh, np.ndarray)
    assert mesh.ndim == 3
    assert mesh.shape == (100, 4, 3)


def test_sphere_radial_partial_angle() -> None:
    mesh = sphere_mesher_from_radial(
        radius=1.0, radial_resolution=5, segment_resolution=8,
        start_angle=0.0, end_angle=np.pi,
    )
    assert mesh.shape == (5 * 8, 4, 3)


def test_sphere_radial_invalid_radius() -> None:
    with pytest.raises(ValueError, match="radius must be strictly positive"):
        sphere_mesher_from_radial(0, 10, 10)


def test_sphere_radial_zero_radial_resolution() -> None:
    with pytest.raises(ValueError, match="radial_resolution must be a positive integer"):
        sphere_mesher_from_radial(1.0, 0, 10)


def test_sphere_radial_zero_segment_resolution() -> None:
    with pytest.raises(ValueError, match="segment_resolution must be a positive integer"):
        sphere_mesher_from_radial(1.0, 10, 0)


@pytest.mark.parametrize("radius", [0.5, 1.0, 10.0])
def test_sphere_radial_radius_variations(radius: float) -> None:
    mesh = sphere_mesher_from_radial(radius, 5, 8)
    np.testing.assert_allclose(np.linalg.norm(mesh, axis=2), radius, rtol=1e-10)


# ---------------------------------- #
# Randomised stress tests
# ---------------------------------- #

def test_sphere_projection_randomised() -> None:
    rng = np.random.default_rng(0)
    radius = float(rng.uniform(0.5, 5.0))
    resolution = int(rng.integers(3, 20))
    mesh = sphere_mesher_from_projection(radius, resolution)
    assert mesh.shape[1:] == (4, 3)
    assert np.isfinite(mesh).all()


def test_sphere_radial_randomised() -> None:
    rng = np.random.default_rng(1)
    radius = float(rng.uniform(0.5, 5.0))
    radial_res = int(rng.integers(3, 15))
    segment_res = int(rng.integers(5, 20))
    mesh = sphere_mesher_from_radial(radius, radial_res, segment_res)
    assert mesh.shape == (radial_res * segment_res, 4, 3)
    assert np.isfinite(mesh).all()
