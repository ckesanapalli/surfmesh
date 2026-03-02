"""Tests for surfmesh.primitives.cuboid."""

import numpy as np
import pytest

from surfmesh.primitives.cuboid import cuboid_mesher, cuboid_mesher_with_resolution


# ---------------------------------- #
# cuboid_mesher
# ---------------------------------- #

def test_cuboid_mesher_basic() -> None:
    faces = cuboid_mesher([0., 1.], [0., 1.], [0., 1.])
    assert faces.shape == (6, 4, 3)


def test_cuboid_mesher_multiple_cells() -> None:
    faces = cuboid_mesher([0., 1., 2.], [0., .5, 1.], [0., .5, 1.])
    assert faces.shape == (24, 4, 3)


@pytest.mark.parametrize("container", [list, tuple, np.array])
def test_cuboid_mesher_accepts_arraylike(container: type) -> None:
    faces = cuboid_mesher(container([0., 1., 2.]),
                          container([0., 1.]),
                          container([0., .5, 1.]))
    assert faces.shape == (16, 4, 3)


def test_cuboid_mesher_invalid_dimensions() -> None:
    x = np.array([[0., 1.]])  # 2D
    with pytest.raises(ValueError, match="x_coords must be 1-dimensional"):
        cuboid_mesher(x, [0., 1.], [0., 1.])


def test_cuboid_mesher_too_few_values() -> None:
    with pytest.raises(ValueError, match="x_coords must have at least 2 element"):
        cuboid_mesher([0.], [0., 1.], [0., 1.])


def test_cuboid_mesher_non_strictly_increasing() -> None:
    with pytest.raises(ValueError, match="x_coords must be strictly increasing"):
        cuboid_mesher([0., 1., 0.5], [0., 1.], [0., 1.])


def test_cuboid_mesher_quad_shape() -> None:
    faces = cuboid_mesher([0., 1.], [0., 1.], [0., 1.])
    assert faces.shape[1:] == (4, 3)


def test_cuboid_mesher_large_x_axis() -> None:
    faces = cuboid_mesher(np.linspace(0, 1, 10), [0., 1.], [0., 1.])
    assert faces.shape[0] == 38


def test_cuboid_mesher_degenerate_single_cell() -> None:
    faces = cuboid_mesher([0., 1.], [0., 1.], [0., 1.])
    assert faces.shape[0] == 6
    for quad in faces:
        assert quad.shape == (4, 3)


# ---------------------------------- #
# cuboid_mesher_with_resolution
# ---------------------------------- #

def test_cuboid_mesher_with_resolution_scalar() -> None:
    faces = cuboid_mesher_with_resolution(2., 1., 1., resolution=2)
    assert faces.shape == (24, 4, 3)


def test_cuboid_mesher_with_resolution_arraylike() -> None:
    faces = cuboid_mesher_with_resolution(2., 1., 1., resolution=[2, 1, 2])
    assert isinstance(faces, np.ndarray)
    assert faces.shape[1:] == (4, 3)


def test_cuboid_mesher_with_resolution_invalid_shape() -> None:
    with pytest.raises(ValueError, match="resolution must be a single int or a 3-element tuple"):
        cuboid_mesher_with_resolution(2., 1., 1., resolution=[2, 2])


def test_cuboid_mesher_with_resolution_zero_value() -> None:
    with pytest.raises(ValueError):
        cuboid_mesher_with_resolution(2., 1., 1., resolution=[2, 0, 2])
