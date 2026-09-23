from typing import Any

import numpy as np
import pytest

from surfmesh import CurveFn, coons_patch

#: Convenience alias for the (bottom, top, left, right) fixture tuples below.
Curves = tuple[CurveFn, CurveFn, CurveFn, CurveFn]


# --------------------------------------------------------------------- #
# Reusable curve fixtures
# --------------------------------------------------------------------- #


@pytest.fixture
def flat_square_curves() -> Curves:
    """Straight-edged unit square in the z=0 plane (corners compatible)."""

    def bottom(t: np.ndarray) -> np.ndarray:
        return np.c_[t, np.zeros_like(t), np.zeros_like(t)]

    def top(t: np.ndarray) -> np.ndarray:
        return np.c_[t, np.ones_like(t), np.zeros_like(t)]

    def left(t: np.ndarray) -> np.ndarray:
        return np.c_[np.zeros_like(t), t, np.zeros_like(t)]

    def right(t: np.ndarray) -> np.ndarray:
        return np.c_[np.ones_like(t), t, np.zeros_like(t)]

    return bottom, top, left, right


@pytest.fixture
def warped_curves() -> Curves:
    """Curved panel used in the module's __main__ example."""

    def bottom(t: np.ndarray) -> np.ndarray:
        return np.c_[t, np.zeros_like(t), 0.3 * np.sin(np.pi * t)]

    def top(t: np.ndarray) -> np.ndarray:
        return np.c_[t, np.ones_like(t), 0.5 * np.sin(np.pi * t) + 0.4]

    def left(t: np.ndarray) -> np.ndarray:
        return np.c_[np.zeros_like(t), t, 0.4 * t]

    def right(t: np.ndarray) -> np.ndarray:
        return np.c_[np.ones_like(t), t, 0.3 * np.sin(np.pi * t) + 0.4 * t]

    return bottom, top, left, right


@pytest.fixture
def incompatible_curves() -> Curves:
    """Curves whose corners do NOT match (right(0) != bottom(1))."""

    def bottom(t: np.ndarray) -> np.ndarray:
        return np.c_[t, np.zeros_like(t), np.zeros_like(t)]

    def top(t: np.ndarray) -> np.ndarray:
        return np.c_[t, np.ones_like(t), np.zeros_like(t)]

    def left(t: np.ndarray) -> np.ndarray:
        return np.c_[np.zeros_like(t), t, np.zeros_like(t)]

    def right(t: np.ndarray) -> np.ndarray:
        return np.c_[np.ones_like(t) + 5.0, t, np.zeros_like(t)]  # shifted -> mismatch

    return bottom, top, left, right


# --------------------------------------------------------------------- #
# Shape / structure
# --------------------------------------------------------------------- #


class TestOutputShapes:
    @pytest.mark.parametrize("nu,nv", [(1, 1), (4, 4), (24, 16), (3, 9)])
    def test_grid_and_face_shapes(self, flat_square_curves: Curves, nu: int, nv: int) -> None:
        grid, faces = coons_patch(*flat_square_curves, nu=nu, nv=nv)
        assert grid.shape == (nu + 1, nv + 1, 3)
        assert faces.shape == (nu * nv, 4, 3)

    def test_default_divisions(self, flat_square_curves: Curves) -> None:
        grid, faces = coons_patch(*flat_square_curves)
        assert grid.shape == (21, 21, 3)
        assert faces.shape == (400, 4, 3)

    def test_returns_numpy_arrays(self, flat_square_curves: Curves) -> None:
        grid, faces = coons_patch(*flat_square_curves, nu=3, nv=3)
        assert isinstance(grid, np.ndarray)
        assert isinstance(faces, np.ndarray)
        assert grid.dtype.kind == "f"
        assert faces.dtype.kind == "f"


# --------------------------------------------------------------------- #
# Correctness of the interpolation
# --------------------------------------------------------------------- #


class TestFlatSquareMatchesBilinear:
    def test_grid_equals_uv_bilinear(self, flat_square_curves: Curves) -> None:
        nu, nv = 10, 7
        grid, _ = coons_patch(*flat_square_curves, nu=nu, nv=nv)
        u = np.linspace(0, 1, nu + 1)
        v = np.linspace(0, 1, nv + 1)
        u_grid, v_grid = np.meshgrid(u, v, indexing="ij")
        expected = np.stack([u_grid, v_grid, np.zeros_like(u_grid)], axis=-1)
        np.testing.assert_allclose(grid, expected, atol=1e-12)

    def test_midpoint_value(self, flat_square_curves: Curves) -> None:
        grid, _ = coons_patch(*flat_square_curves, nu=4, nv=4)
        np.testing.assert_allclose(grid[2, 3], [0.5, 0.75, 0.0], atol=1e-12)


class TestCornersAndEdgesMatchInputCurves:
    def test_corners(self, warped_curves: Curves) -> None:
        bottom, top, left, right = warped_curves
        grid, _ = coons_patch(*warped_curves, nu=12, nv=8)
        np.testing.assert_allclose(grid[0, 0], bottom(np.array([0.0]))[0], atol=1e-10)
        np.testing.assert_allclose(grid[-1, 0], bottom(np.array([1.0]))[0], atol=1e-10)
        np.testing.assert_allclose(grid[0, -1], top(np.array([0.0]))[0], atol=1e-10)
        np.testing.assert_allclose(grid[-1, -1], top(np.array([1.0]))[0], atol=1e-10)

    def test_bottom_and_top_edges(self, warped_curves: Curves) -> None:
        bottom, top, left, right = warped_curves
        nu, nv = 16, 5
        grid, _ = coons_patch(*warped_curves, nu=nu, nv=nv)
        u = np.linspace(0, 1, nu + 1)
        np.testing.assert_allclose(grid[:, 0], bottom(u), atol=1e-10)
        np.testing.assert_allclose(grid[:, -1], top(u), atol=1e-10)

    def test_left_and_right_edges(self, warped_curves: Curves) -> None:
        bottom, top, left, right = warped_curves
        nu, nv = 5, 16
        grid, _ = coons_patch(*warped_curves, nu=nu, nv=nv)
        v = np.linspace(0, 1, nv + 1)
        np.testing.assert_allclose(grid[0, :], left(v), atol=1e-10)
        np.testing.assert_allclose(grid[-1, :], right(v), atol=1e-10)


# --------------------------------------------------------------------- #
# Face connectivity
# --------------------------------------------------------------------- #


class TestFaceConnectivity:
    def test_face_count(self, flat_square_curves: Curves) -> None:
        for nu, nv in [(1, 1), (5, 3), (24, 16)]:
            _, faces = coons_patch(*flat_square_curves, nu=nu, nv=nv)
            assert len(faces) == nu * nv

    def test_first_face_matches_grid_corners(self, flat_square_curves: Curves) -> None:
        grid, faces = coons_patch(*flat_square_curves, nu=3, nv=3)
        expected = np.array([grid[0, 0], grid[1, 0], grid[1, 1], grid[0, 1]])
        np.testing.assert_allclose(faces[0], expected)

    def test_last_face_matches_grid_corners(self, flat_square_curves: Curves) -> None:
        nu, nv = 3, 3
        grid, faces = coons_patch(*flat_square_curves, nu=nu, nv=nv)
        expected = np.array([grid[nu - 1, nv - 1], grid[nu, nv - 1], grid[nu, nv], grid[nu - 1, nv]])
        np.testing.assert_allclose(faces[-1], expected)

    def test_every_face_vertex_is_a_grid_point(self, warped_curves: Curves) -> None:
        nu, nv = 6, 4
        grid, faces = coons_patch(*warped_curves, nu=nu, nv=nv)
        grid_pts = grid.reshape(-1, 3)
        for quad in faces:
            for vertex in quad:
                assert np.any(np.all(np.isclose(grid_pts, vertex), axis=1))

    def test_single_quad_mesh(self, flat_square_curves: Curves) -> None:
        grid, faces = coons_patch(*flat_square_curves, nu=1, nv=1)
        assert grid.shape == (2, 2, 3)
        assert faces.shape == (1, 4, 3)
        np.testing.assert_allclose(faces[0], [grid[0, 0], grid[1, 0], grid[1, 1], grid[0, 1]])


# --------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------- #


class TestValidation:
    @pytest.mark.parametrize("nu", [0, -1, -10])
    def test_non_positive_nu_raises(self, flat_square_curves: Curves, nu: int) -> None:
        with pytest.raises(ValueError, match="nu"):
            coons_patch(*flat_square_curves, nu=nu, nv=4)

    @pytest.mark.parametrize("nv", [0, -1, -10])
    def test_non_positive_nv_raises(self, flat_square_curves: Curves, nv: int) -> None:
        with pytest.raises(ValueError, match="nv"):
            coons_patch(*flat_square_curves, nu=4, nv=nv)

    @pytest.mark.parametrize("nu", [1.5, "4", None])
    def test_non_integer_nu_raises(self, flat_square_curves: Curves, nu: Any) -> None:
        with pytest.raises(ValueError, match="nu"):
            coons_patch(*flat_square_curves, nu=nu, nv=4)

    def test_incompatible_corners_raise_by_default(self, incompatible_curves: Curves) -> None:
        with pytest.raises(ValueError, match="Corner mismatch"):
            coons_patch(*incompatible_curves, nu=4, nv=4)

    def test_incompatible_corners_allowed_when_check_disabled(self, incompatible_curves: Curves) -> None:
        # Should not raise, and should still return arrays of the right shape.
        grid, faces = coons_patch(*incompatible_curves, nu=4, nv=4, check_corners=False)
        assert grid.shape == (5, 5, 3)
        assert faces.shape == (16, 4, 3)

    def test_atol_controls_corner_tolerance(self) -> None:
        eps = 1e-4

        def bottom(t: np.ndarray) -> np.ndarray:
            return np.c_[t, np.zeros_like(t), np.zeros_like(t)]

        def top(t: np.ndarray) -> np.ndarray:
            return np.c_[t, np.ones_like(t), np.zeros_like(t)]

        def left(t: np.ndarray) -> np.ndarray:
            return np.c_[np.zeros_like(t), t, np.zeros_like(t)]

        def right(t: np.ndarray) -> np.ndarray:
            return np.c_[np.ones_like(t) + eps, t, np.zeros_like(t)]  # off from bottom(1) by `eps`

        # Tight tolerance -> mismatch detected.
        with pytest.raises(ValueError, match="Corner mismatch"):
            coons_patch(bottom, top, left, right, nu=2, nv=2, atol=1e-8)

        # Loose tolerance -> accepted.
        grid, _ = coons_patch(bottom, top, left, right, nu=2, nv=2, atol=1e-3)
        assert grid.shape == (3, 3, 3)


# --------------------------------------------------------------------- #
# Symmetry / sanity checks
# --------------------------------------------------------------------- #


class TestSanity:
    def test_mesh_is_finite(self, warped_curves: Curves) -> None:
        grid, faces = coons_patch(*warped_curves, nu=20, nv=20)
        assert np.all(np.isfinite(grid))
        assert np.all(np.isfinite(faces))

    def test_increasing_resolution_preserves_boundary(self, warped_curves: Curves) -> None:
        bottom, top, left, right = warped_curves
        for nu, nv in [(4, 4), (40, 40)]:
            grid, _ = coons_patch(*warped_curves, nu=nu, nv=nv)
            np.testing.assert_allclose(grid[0, 0], bottom(np.array([0.0]))[0], atol=1e-10)
            np.testing.assert_allclose(grid[-1, -1], top(np.array([1.0]))[0], atol=1e-10)

    def test_non_square_division_counts(self, warped_curves: Curves) -> None:
        # nu != nv should not raise and should give independently-sized axes.
        grid, faces = coons_patch(*warped_curves, nu=30, nv=5)
        assert grid.shape == (31, 6, 3)
        assert faces.shape == (150, 4, 3)
