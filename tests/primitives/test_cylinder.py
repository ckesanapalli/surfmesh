"""Tests for surfmesh.primitives.cylinder."""

import numpy as np
import pytest

from surfmesh.primitives.cylinder import cylinder_mesher_radial, cylinder_mesher_square_centered


# ---------------------------------- #
# cylinder_mesher_radial
# ---------------------------------- #

def test_cylinder_mesher_radial_basic_shape() -> None:
    mesh = cylinder_mesher_radial(1.0, 2.0, 5, 8, 6)
    assert isinstance(mesh, np.ndarray)
    assert mesh.ndim == 3
    assert mesh.shape[1:] == (4, 3)


def test_cylinder_mesher_radial_faces_count() -> None:
    radial_res, segment_res, height_res = 5, 8, 6
    expected = 2 * radial_res * segment_res + height_res * segment_res
    mesh = cylinder_mesher_radial(1.0, 2.0, radial_res, segment_res, height_res)
    assert mesh.shape[0] == expected


def test_cylinder_mesher_radial_coordinate_bounds() -> None:
    radius, height = 1.0, 2.0
    mesh = cylinder_mesher_radial(radius, height, 4, 8, 5)
    r = np.linalg.norm(mesh[..., :2], axis=-1)
    assert np.all(r <= radius + 1e-9)
    z = mesh[..., 2]
    assert np.all(z >= -height / 2 - 1e-9)
    assert np.all(z <=  height / 2 + 1e-9)


@pytest.mark.parametrize("radial, segment, height", [(2, 2, 2), (4, 8, 5), (6, 12, 10)])
def test_cylinder_mesher_radial_various_resolutions(radial: int, segment: int, height: int) -> None:
    mesh = cylinder_mesher_radial(1.0, 2.0, radial, segment, height)
    assert mesh.shape[1:] == (4, 3)


@pytest.mark.parametrize("radius", [-1.0, 0.0])
def test_cylinder_mesher_radial_invalid_radius(radius: float) -> None:
    with pytest.raises(ValueError, match="radius must be strictly positive"):
        cylinder_mesher_radial(radius, 2.0, 5, 8, 6)


@pytest.mark.parametrize("height", [-1.0, 0.0])
def test_cylinder_mesher_radial_invalid_height(height: float) -> None:
    with pytest.raises(ValueError, match="height must be strictly positive"):
        cylinder_mesher_radial(1.0, height, 5, 8, 6)


@pytest.mark.parametrize("bad_res", [0, -1])
def test_cylinder_mesher_radial_invalid_resolutions(bad_res: int) -> None:
    with pytest.raises(ValueError):
        cylinder_mesher_radial(1.0, 2.0, bad_res, 8, 6)
    with pytest.raises(ValueError):
        cylinder_mesher_radial(1.0, 2.0, 5, bad_res, 6)
    with pytest.raises(ValueError):
        cylinder_mesher_radial(1.0, 2.0, 5, 8, bad_res)


# ---------------------------------- #
# cylinder_mesher_square_centered
# ---------------------------------- #

def test_cylinder_mesher_square_centered_basic_shape() -> None:
    mesh = cylinder_mesher_square_centered(1.0, 2.0, 5, 4, 8)
    assert isinstance(mesh, np.ndarray)
    assert mesh.ndim == 3
    assert mesh.shape[1:] == (4, 3)


def test_cylinder_mesher_square_centered_faces_count() -> None:
    radial, half_sq, height_res = 4, 3, 5
    sq = 2 * half_sq
    expected_disk = sq * sq + sq * 4 * radial
    expected_lateral = height_res * sq * 4
    expected = 2 * expected_disk + expected_lateral
    mesh = cylinder_mesher_square_centered(1.0, 2.0, radial, half_sq, height_res)
    assert mesh.shape[0] == expected


def test_cylinder_mesher_square_centered_coordinate_bounds() -> None:
    radius, height = 1.0, 2.0
    mesh = cylinder_mesher_square_centered(radius, height, 5, 4, 8)
    r = np.linalg.norm(mesh[..., :2], axis=-1)
    assert np.all(r <= radius + 1e-9)
    z = mesh[..., 2]
    assert np.all(z >= -height / 2 - 1e-9)
    assert np.all(z <=  height / 2 + 1e-9)


@pytest.mark.parametrize("radial, half_sq, height", [(2, 2, 2), (4, 3, 5), (6, 5, 8)])
def test_cylinder_mesher_square_centered_various_resolutions(
    radial: int, half_sq: int, height: int
) -> None:
    mesh = cylinder_mesher_square_centered(1.0, 2.0, radial, half_sq, height)
    assert mesh.shape[1:] == (4, 3)


@pytest.mark.parametrize("radius", [-1.0, 0.0])
def test_cylinder_mesher_square_centered_invalid_radius(radius: float) -> None:
    with pytest.raises(ValueError, match="radius must be strictly positive"):
        cylinder_mesher_square_centered(radius, 2.0, 5, 4, 8)


@pytest.mark.parametrize("height", [-1.0, 0.0])
def test_cylinder_mesher_square_centered_invalid_height(height: float) -> None:
    with pytest.raises(ValueError, match="height must be strictly positive"):
        cylinder_mesher_square_centered(1.0, height, 5, 4, 8)


@pytest.mark.parametrize("bad_res", [0, -1])
def test_cylinder_mesher_square_centered_invalid_resolutions(bad_res: int) -> None:
    with pytest.raises(ValueError):
        cylinder_mesher_square_centered(1.0, 2.0, bad_res, 4, 8)
    with pytest.raises(ValueError):
        cylinder_mesher_square_centered(1.0, 2.0, 5, bad_res, 8)
    with pytest.raises(ValueError):
        cylinder_mesher_square_centered(1.0, 2.0, 5, 4, bad_res)
