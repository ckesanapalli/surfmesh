"""Tests for surfmesh.operations.revolve."""

import numpy as np
import pytest

from surfmesh.operations.revolve import circular_revolve, revolve_curve_along_path


# ----------------------------- #
# revolve_curve_along_path
# ----------------------------- #

def test_revolve_curve_along_path_basic() -> None:
    curve = np.array([[1.0, 2.0], [3.0, 4.0]])
    path  = np.array([[0.0, 1.0], [np.pi / 2, 2.0]])
    mesh  = revolve_curve_along_path(curve, path)
    assert isinstance(mesh, np.ndarray)
    assert mesh.shape == (1, 4, 3)


def test_revolve_curve_along_path_multiple_segments() -> None:
    curve = np.array([[0., 0.], [1., 1.], [2., 0.]])
    path  = np.array([[0., 1.], [np.pi / 2, 2.], [np.pi, 3.]])
    mesh  = revolve_curve_along_path(curve, path)
    assert mesh.shape == ((3 - 1) * (3 - 1), 4, 3)


def test_revolve_curve_along_path_invalid_curve() -> None:
    with pytest.raises(ValueError, match="curve must have shape"):
        revolve_curve_along_path(np.array([1.0, 2.0, 3.0]),
                                 np.array([[0., 1.], [np.pi / 2, 2.]]))


def test_revolve_curve_along_path_invalid_path() -> None:
    with pytest.raises(ValueError, match="revolve_path must have shape"):
        revolve_curve_along_path(np.array([[1., 2.], [3., 4.]]),
                                 np.array([0.0, 1.0]))


# ----------------------------- #
# circular_revolve
# ----------------------------- #

def test_circular_revolve_basic() -> None:
    curve = np.array([[1.0, 0.0], [2.0, 1.0]])
    mesh  = circular_revolve(curve, segment_resolution=8)
    assert isinstance(mesh, np.ndarray)
    assert mesh.shape == (8, 4, 3)


def test_circular_revolve_full_circle() -> None:
    curve = np.array([[1.0, 0.0], [2.0, 1.0], [2.5, 2.0]])
    mesh  = circular_revolve(curve, segment_resolution=12)
    assert mesh.shape == (12 * 2, 4, 3)


def test_circular_revolve_partial_revolve() -> None:
    curve = np.array([[1.0, 0.0], [2.0, 1.0]])
    mesh  = circular_revolve(curve, segment_resolution=4, start_angle=0, end_angle=np.pi)
    assert mesh.shape == (4, 4, 3)


def test_circular_revolve_invalid_curve_shape() -> None:
    with pytest.raises(ValueError, match="curve must have shape"):
        circular_revolve(np.array([1.0, 2.0]), segment_resolution=8)


def test_circular_revolve_zero_segment_resolution() -> None:
    with pytest.raises(ValueError, match="segment_resolution must be a positive integer"):
        circular_revolve(np.array([[1., 0.], [2., 1.]]), segment_resolution=0)


@pytest.mark.parametrize("segments", [4, 8, 16])
def test_circular_revolve_randomised(segments: int) -> None:
    np.random.seed(42)
    curve = np.random.rand(5, 2)
    mesh  = circular_revolve(curve, segments)
    assert mesh.shape == (segments * 4, 4, 3)
    assert np.isfinite(mesh).all()
