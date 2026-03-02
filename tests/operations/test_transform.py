"""Tests for surfmesh.operations.transform."""

import numpy as np
import pytest

from surfmesh.operations.transform import (
    convert_2d_face_to_3d,
    flip_normals,
    rotate_z,
    scale,
    translate,
)


# ---------------------------------- #
# convert_2d_face_to_3d
# ---------------------------------- #

def test_convert_2d_face_to_3d_basic() -> None:
    quad_2d = np.array([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]])
    result = convert_2d_face_to_3d(quad_2d, axis=2, offset=5.0)
    expected = np.array([[[0., 0., 5.], [1., 0., 5.], [1., 1., 5.], [0., 1., 5.]]])
    np.testing.assert_array_equal(result, expected)


def test_convert_2d_face_to_3d_all_axes() -> None:
    quad_2d = np.array([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]])
    for axis in (0, 1, 2):
        result = convert_2d_face_to_3d(quad_2d, axis=axis, offset=3.5)
        assert result.shape == (1, 4, 3)
        np.testing.assert_allclose(result[:, :, axis], 3.5)


def test_convert_2d_face_to_3d_invalid_axis() -> None:
    quad_2d = np.array([[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]])
    with pytest.raises(ValueError, match="axis must be 0"):
        convert_2d_face_to_3d(quad_2d, axis=3, offset=0.0)


# ---------------------------------- #
# translate
# ---------------------------------- #

def test_translate_basic() -> None:
    mesh = np.zeros((2, 4, 3))
    result = translate(mesh, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(result[0, 0], [1.0, 2.0, 3.0])


def test_translate_invalid_translation() -> None:
    mesh = np.zeros((1, 4, 3))
    with pytest.raises(ValueError, match="translation must have shape"):
        translate(mesh, [1.0, 2.0])


# ---------------------------------- #
# scale
# ---------------------------------- #

def test_scale_uniform() -> None:
    mesh = np.ones((1, 4, 3))
    result = scale(mesh, 3.0)
    np.testing.assert_allclose(result, 3.0)


def test_scale_per_axis() -> None:
    mesh = np.ones((1, 4, 3))
    result = scale(mesh, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(result[0, 0], [1.0, 2.0, 3.0])


# ---------------------------------- #
# rotate_z
# ---------------------------------- #

def test_rotate_z_90_degrees() -> None:
    mesh = np.array([[[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [-1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0]]])
    result = rotate_z(mesh, np.pi / 2)
    # [1, 0, 0] → [0, 1, 0]
    np.testing.assert_allclose(result[0, 0], [0.0, 1.0, 0.0], atol=1e-10)


def test_rotate_z_full_circle() -> None:
    mesh = np.random.rand(3, 4, 3)
    result = rotate_z(mesh, 2 * np.pi)
    np.testing.assert_allclose(result, mesh, atol=1e-10)


# ---------------------------------- #
# flip_normals
# ---------------------------------- #

def test_flip_normals_reverses_vertices() -> None:
    mesh = np.arange(12).reshape(1, 4, 3).astype(float)
    flipped = flip_normals(mesh)
    np.testing.assert_array_equal(flipped[0], mesh[0, ::-1])


def test_flip_normals_double_flip_identity() -> None:
    mesh = np.random.rand(5, 4, 3)
    np.testing.assert_array_equal(flip_normals(flip_normals(mesh)), mesh)
