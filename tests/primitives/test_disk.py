"""Tests for surfmesh.primitives.disk."""

import numpy as np
import pytest

from surfmesh.primitives.disk import disk_mesher_radial, disk_mesher_square_centered


# ---------------------------------- #
# disk_mesher_radial
# ---------------------------------- #

def test_disk_mesher_radial_basic_shape() -> None:
    mesh = disk_mesher_radial(1.0, radial_resolution=5, segment_resolution=8)
    assert isinstance(mesh, np.ndarray)
    assert mesh.shape == (5 * 8, 4, 2)


def test_disk_mesher_radial_dtype_is_float() -> None:
    mesh = disk_mesher_radial(1.0, 3, 3)
    assert np.issubdtype(mesh.dtype, np.floating)


def test_disk_mesher_radial_radius_limit() -> None:
    mesh = disk_mesher_radial(1.0, 10, 10)
    r = np.linalg.norm(mesh, axis=-1)
    assert np.all(r <= 1.0 + 1e-9)


def test_disk_mesher_radial_zero_radius() -> None:
    """radius=0 is permitted and produces an all-zero mesh."""
    mesh = disk_mesher_radial(0.0, 5, 8)
    assert np.allclose(mesh, 0.0)


@pytest.mark.parametrize("radial_res, segment_res", [(1, 1), (2, 4), (5, 10)])
def test_disk_mesher_radial_various_resolutions(radial_res: int, segment_res: int) -> None:
    mesh = disk_mesher_radial(1.0, radial_res, segment_res)
    assert mesh.shape == (radial_res * segment_res, 4, 2)


def test_disk_mesher_radial_negative_radius() -> None:
    with pytest.raises(ValueError, match="radius must be non-negative"):
        disk_mesher_radial(-1.0, 5, 5)


def test_disk_mesher_radial_zero_radial_resolution() -> None:
    with pytest.raises(ValueError, match="radial_resolution must be a positive integer"):
        disk_mesher_radial(1.0, 0, 5)


def test_disk_mesher_radial_zero_segment_resolution() -> None:
    with pytest.raises(ValueError, match="segment_resolution must be a positive integer"):
        disk_mesher_radial(1.0, 5, 0)


# ---------------------------------- #
# disk_mesher_square_centered
# ---------------------------------- #

def test_disk_mesher_square_centered_basic_shape() -> None:
    mesh = disk_mesher_square_centered(1.0, 5, 5)
    assert mesh.shape[1:] == (4, 2)
    assert mesh.ndim == 3


def test_disk_mesher_square_centered_larger_has_more_panels() -> None:
    mesh_small = disk_mesher_square_centered(1.0, 2, 2)
    mesh_large = disk_mesher_square_centered(1.0, 5, 5)
    assert mesh_large.shape[0] > mesh_small.shape[0]


def test_disk_mesher_square_centered_with_rotation() -> None:
    mesh_rot   = disk_mesher_square_centered(1.0, 5, 5, square_disk_rotation=np.pi / 8)
    mesh_norot = disk_mesher_square_centered(1.0, 5, 5, square_disk_rotation=0.0)
    assert mesh_rot.shape == mesh_norot.shape


def test_disk_mesher_square_centered_zero_radius() -> None:
    with pytest.raises(ValueError, match="radius must be strictly positive"):
        disk_mesher_square_centered(0.0, 5, 5)


def test_disk_mesher_square_centered_zero_square_resolution() -> None:
    with pytest.raises(ValueError, match="square_resolution must be a positive integer"):
        disk_mesher_square_centered(1.0, 0, 5)


def test_disk_mesher_square_centered_zero_radial_resolution() -> None:
    with pytest.raises(ValueError, match="radial_resolution must be a positive integer"):
        disk_mesher_square_centered(1.0, 5, 0)


def test_disk_mesher_square_centered_invalid_ratio_zero() -> None:
    with pytest.raises(ValueError, match="square_side_radius_ratio must be in"):
        disk_mesher_square_centered(1.0, 5, 5, square_side_radius_ratio=0.0)


def test_disk_mesher_square_centered_invalid_ratio_too_large() -> None:
    with pytest.raises(ValueError, match="square_side_radius_ratio must be in"):
        disk_mesher_square_centered(1.0, 5, 5, square_side_radius_ratio=1.5)
