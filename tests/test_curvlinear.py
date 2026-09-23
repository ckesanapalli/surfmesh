from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from surfmesh.curvlinear import (
    CornerMismatchError,
    CurveDimensionMismatchError,
    CurveFn,
    CurveOutputError,
    MeshDivisionError,
    coons_patch,
)

Curves = tuple[CurveFn, CurveFn, CurveFn, CurveFn]

# --------------------------------------------------------------------- #
# Curve builders
# --------------------------------------------------------------------- #


def _unit_square_curves(dim: int) -> Curves:
    """Straight-edge unit-square boundary curves in `dim` dimensions
    (extra coordinates beyond x, y held at 0)."""
    extra = [np.zeros(1)] * (dim - 2)

    def pad(t: np.ndarray, *comps: np.ndarray) -> np.ndarray:
        cols = [*comps, *(np.zeros_like(t) for _ in extra)]
        return np.stack(cols, axis=-1)

    def bottom(t: np.ndarray) -> np.ndarray:
        return pad(t, t, np.zeros_like(t))

    def top(t: np.ndarray) -> np.ndarray:
        return pad(t, t, np.ones_like(t))

    def left(t: np.ndarray) -> np.ndarray:
        return pad(t, np.zeros_like(t), t)

    def right(t: np.ndarray) -> np.ndarray:
        return pad(t, np.ones_like(t), t)

    return bottom, top, left, right


def _warped_curves(dim: int) -> Curves:
    """Curved-edge boundary curves in `dim` dimensions (sinusoidal bumps on
    the non-x/y coordinate(s) too, when dim == 3)."""
    if dim == 2:

        def bottom(t: np.ndarray) -> np.ndarray:
            return np.c_[t, 0.15 * np.sin(np.pi * t)]

        def top(t: np.ndarray) -> np.ndarray:
            return np.c_[t, 1.0 + 0.25 * np.sin(np.pi * t)]

        def left(t: np.ndarray) -> np.ndarray:
            return np.c_[np.zeros_like(t), t]

        def right(t: np.ndarray) -> np.ndarray:
            return np.c_[1.0 + 0.2 * np.sin(np.pi * t), t]

        return bottom, top, left, right

    else:
        def bottom(t: np.ndarray) -> np.ndarray:
            return np.c_[t, np.zeros_like(t), 0.3 * np.sin(np.pi * t)]
    
        def top(t: np.ndarray) -> np.ndarray:
            return np.c_[t, np.ones_like(t), 0.5 * np.sin(np.pi * t) + 0.4]
    
        def left(t: np.ndarray) -> np.ndarray:
            return np.c_[np.zeros_like(t), t, 0.4 * t]
    
        def right(t: np.ndarray) -> np.ndarray:
            return np.c_[np.ones_like(t), t, 0.3 * np.sin(np.pi * t) + 0.4 * t]
    
        return bottom, top, left, right


def _tuple_style_curves(dim: int) -> Curves:
    """Same warped panel as `_warped_curves`, but each curve returns a bare
    component tuple instead of a pre-stacked ``np.c_`` array."""
    if dim == 2:

        def bottom(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return (t, 0.15 * np.sin(np.pi * t))

        def top(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return (t, 1.0 + 0.25 * np.sin(np.pi * t))

        def left(t: np.ndarray) -> tuple[float, np.ndarray]:
            return (0.0, t)

        def right(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return (1.0 + 0.2 * np.sin(np.pi * t), t)

        return bottom, top, left, right

    def bottom3(t: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        return (t, 0.0, 0.3 * np.sin(np.pi * t))

    def top3(t: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
        return (t, 1.0, 0.5 * np.sin(np.pi * t) + 0.4)

    def left3(t: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        return (0.0, t, 0.4 * t)

    def right3(t: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        return (1.0, t, 0.3 * np.sin(np.pi * t) + 0.4 * t)

    return bottom3, top3, left3, right3


# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #


@pytest.fixture(params=[2, 3], ids=["dim=2", "dim=3"])
def dim(request: pytest.FixtureRequest) -> int:
    return int(request.param)


@pytest.fixture
def unit_square_curves(dim: int) -> Curves:
    return _unit_square_curves(dim)


@pytest.fixture
def warped_curves(dim: int) -> Curves:
    return _warped_curves(dim)


@pytest.fixture
def tuple_style_curves(dim: int) -> Curves:
    return _tuple_style_curves(dim)


@pytest.fixture
def incompatible_curves(dim: int) -> Curves:
    """Same shape family as `unit_square_curves`, but `left`'s corners don't
    actually meet `bottom`/`top` -- should fail the corner check."""
    bottom, top, _left, right = _unit_square_curves(dim)

    def bad_left(t: np.ndarray) -> np.ndarray:
        cols = [np.full_like(t, 5.0), t, *(np.zeros_like(t) for _ in range(dim - 2))]
        return np.stack(cols, axis=-1)

    return bottom, top, bad_left, right


# --------------------------------------------------------------------- #
# Shape and grid correctness (parametrized over dim)
# --------------------------------------------------------------------- #


class TestShapesAndGrid:
    def test_grid_and_face_shapes(self, unit_square_curves: Curves, dim: int) -> None:
        bottom, top, left, right = unit_square_curves
        grid, faces = coons_patch(bottom, top, left, right, nu=8, nv=6)
        assert grid.shape == (9, 7, dim)
        assert faces.shape == (48, 4, dim)

    def test_default_division_counts(self, unit_square_curves: Curves, dim: int) -> None:
        bottom, top, left, right = unit_square_curves
        grid, faces = coons_patch(bottom, top, left, right)
        assert grid.shape == (21, 21, dim)
        assert faces.shape == (400, 4, dim)

    def test_flat_square_reduces_to_bilinear_interpolation(
        self, unit_square_curves: Curves, dim: int
    ) -> None:
        bottom, top, left, right = unit_square_curves
        grid, _faces = coons_patch(bottom, top, left, right, nu=10, nv=10)
        u = np.linspace(0, 1, 11)
        v = np.linspace(0, 1, 11)
        expected_xy = np.stack(np.meshgrid(u, v, indexing="ij"), axis=-1)
        np.testing.assert_allclose(grid[..., :2], expected_xy, atol=1e-12)
        if dim == 3:
            np.testing.assert_allclose(grid[..., 2], 0.0, atol=1e-12)

    def test_corners_match_input_curve_endpoints(self, warped_curves: Curves, dim: int) -> None:
        bottom, top, left, right = warped_curves
        grid, _faces = coons_patch(bottom, top, left, right, nu=12, nv=10)
        zero, one = np.array([0.0]), np.array([1.0])
        np.testing.assert_allclose(grid[0, 0], np.asarray(bottom(zero)).reshape(dim), atol=1e-10)
        np.testing.assert_allclose(grid[-1, 0], np.asarray(bottom(one)).reshape(dim), atol=1e-10)
        np.testing.assert_allclose(grid[0, -1], np.asarray(top(zero)).reshape(dim), atol=1e-10)
        np.testing.assert_allclose(grid[-1, -1], np.asarray(top(one)).reshape(dim), atol=1e-10)

    def test_face_connectivity_matches_grid(self, warped_curves: Curves) -> None:
        bottom, top, left, right = warped_curves
        grid, faces = coons_patch(bottom, top, left, right, nu=5, nv=4)
        np.testing.assert_allclose(faces[0, 0], grid[0, 0])
        np.testing.assert_allclose(faces[0, 1], grid[1, 0])
        np.testing.assert_allclose(faces[0, 2], grid[1, 1])
        np.testing.assert_allclose(faces[0, 3], grid[0, 1])
        # last face is at (nu-1, nv-1)
        np.testing.assert_allclose(faces[-1, 2], grid[-1, -1])

    def test_boundary_curves_are_matched_exactly(self, warped_curves: Curves) -> None:
        bottom, top, left, right = warped_curves
        nu, nv = 16, 12
        grid, _faces = coons_patch(bottom, top, left, right, nu=nu, nv=nv)
        u = np.linspace(0, 1, nu + 1)
        v = np.linspace(0, 1, nv + 1)
        np.testing.assert_allclose(grid[:, 0], np.asarray(bottom(u)), atol=1e-10)
        np.testing.assert_allclose(grid[:, -1], np.asarray(top(u)), atol=1e-10)
        np.testing.assert_allclose(grid[0, :], np.asarray(left(v)), atol=1e-10)
        np.testing.assert_allclose(grid[-1, :], np.asarray(right(v)), atol=1e-10)


# --------------------------------------------------------------------- #
# Tuple-style curve outputs
# --------------------------------------------------------------------- #


class TestTupleStyleCurves:
    def test_matches_array_style_result(
        self, warped_curves: Curves, tuple_style_curves: Curves, dim: int
    ) -> None:
        grid_array, faces_array = coons_patch(*warped_curves, nu=10, nv=8)
        grid_tuple, faces_tuple = coons_patch(*tuple_style_curves, nu=10, nv=8)
        np.testing.assert_allclose(grid_array, grid_tuple, atol=1e-10)
        np.testing.assert_allclose(faces_array, faces_tuple, atol=1e-10)
        assert grid_tuple.shape[-1] == dim

    def test_all_constant_corner_tuple(self, dim: int) -> None:
        # every curve returns a fully constant point -- degenerate panel,
        # but should not error, and every grid point should equal that point
        point = tuple(float(i) for i in range(dim))

        def const_curve(t: np.ndarray) -> tuple[float, ...]:
            return point

        grid, _faces = coons_patch(const_curve, const_curve, const_curve, const_curve, nu=3, nv=3)
        np.testing.assert_allclose(grid, np.broadcast_to(point, grid.shape), atol=1e-10)


# --------------------------------------------------------------------- #
# Curve output validation errors
# --------------------------------------------------------------------- #


class TestCurveOutputValidation:
    def test_wrong_last_axis_length_array_raises(self, dim: int) -> None:
        def bad(t: np.ndarray) -> np.ndarray:
            return np.c_[t, t, t, t]  # shape (N, 4) -- not 2 or 3

        bottom, top, left, right = _unit_square_curves(dim)
        with pytest.raises(CurveOutputError, match="expected"):
            coons_patch(bad, top, left, right)

    def test_wrong_ndim_array_raises(self, dim: int) -> None:
        def bad(t: np.ndarray) -> np.ndarray:
            return t  # 1-D, not (N, dim)

        bottom, top, left, right = _unit_square_curves(dim)
        with pytest.raises(CurveOutputError):
            coons_patch(bad, top, left, right)

    def test_wrong_length_tuple_raises(self, dim: int) -> None:
        def bad(t: np.ndarray) -> tuple[float, float, float, float]:
            return (0.0, 0.0, 0.0, 0.0)  # length 4 -- not 2 or 3

        bottom, top, left, right = _unit_square_curves(dim)
        with pytest.raises(CurveOutputError):
            coons_patch(bad, top, left, right)

    def test_non_array_non_sequence_return_raises(self, dim: int) -> None:
        def bad(t: np.ndarray) -> float:
            return 1.0

        bottom, top, left, right = _unit_square_curves(dim)
        with pytest.raises(CurveOutputError):
            coons_patch(bad, top, left, right)


# --------------------------------------------------------------------- #
# Cross-dimension mismatch
# --------------------------------------------------------------------- #


class TestCurveDimensionMismatch:
    def test_2d_and_3d_curves_mixed_raises(self) -> None:
        bottom2, top2, left2, _right2 = _unit_square_curves(2)
        _bottom3, _top3, _left3, right3 = _unit_square_curves(3)
        with pytest.raises(CurveDimensionMismatchError, match="same dimension"):
            coons_patch(bottom2, top2, left2, right3, nu=2, nv=2)

    def test_error_message_names_the_offending_curve(self) -> None:
        bottom2, top2, left2, _right2 = _unit_square_curves(2)
        _bottom3, _top3, _left3, right3 = _unit_square_curves(3)
        with pytest.raises(CurveDimensionMismatchError) as exc_info:
            coons_patch(bottom2, top2, left2, right3, nu=2, nv=2)
        assert "'right': 3" in str(exc_info.value)
        assert "'bottom': 2" in str(exc_info.value)


# --------------------------------------------------------------------- #
# nu / nv validation
# --------------------------------------------------------------------- #


class TestMeshDivisionValidation:
    @pytest.mark.parametrize("bad_nu", [0, -1, 1.5, "4", None])
    def test_rejects_invalid_nu(self, unit_square_curves: Curves, bad_nu: object) -> None:
        bottom, top, left, right = unit_square_curves
        with pytest.raises(MeshDivisionError, match="nu"):
            coons_patch(bottom, top, left, right, nu=bad_nu, nv=4)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_nv", [0, -1, 1.5, "4", None])
    def test_rejects_invalid_nv(self, unit_square_curves: Curves, bad_nv: object) -> None:
        bottom, top, left, right = unit_square_curves
        with pytest.raises(MeshDivisionError, match="nv"):
            coons_patch(bottom, top, left, right, nu=4, nv=bad_nv)  # type: ignore[arg-type]

    def test_accepts_numpy_integer(self, unit_square_curves: Curves, dim: int) -> None:
        bottom, top, left, right = unit_square_curves
        grid, _faces = coons_patch(bottom, top, left, right, nu=np.int64(3), nv=np.int32(2))
        assert grid.shape == (4, 3, dim)


# --------------------------------------------------------------------- #
# Corner mismatch validation
# --------------------------------------------------------------------- #


class TestCornerMismatchValidation:
    def test_raises_by_default(self, incompatible_curves: Curves) -> None:
        with pytest.raises(CornerMismatchError):
            coons_patch(*incompatible_curves)

    def test_can_be_disabled(self, incompatible_curves: Curves, dim: int) -> None:
        grid, faces = coons_patch(*incompatible_curves, nu=4, nv=4, check_corners=False)
        assert grid.shape == (5, 5, dim)
        assert faces.shape == (16, 4, dim)

    def test_atol_widens_tolerance(self, dim: int) -> None:
        bottom, top, left, right = _unit_square_curves(dim)

        def nearly_left(t: np.ndarray) -> np.ndarray:
            cols = [np.full_like(t, 1e-4), t, *(np.zeros_like(t) for _ in range(dim - 2))]
            return np.stack(cols, axis=-1)

        with pytest.raises(CornerMismatchError):
            coons_patch(bottom, top, nearly_left, right, atol=1e-8)
        # should pass with a looser tolerance
        coons_patch(bottom, top, nearly_left, right, atol=1e-2)

