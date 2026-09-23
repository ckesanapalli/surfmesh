"""
coons_patch
===========

Generate a structured quadrilateral mesh over a 4-sided curvilinear panel
using a bilinearly-blended Coons patch (transfinite interpolation) --
dimension-agnostic: works unchanged for 2D (x, y) or 3D (x, y, z) points,
inferred from what the boundary curve functions return.

Given four boundary curves -- bottom, top, left, right -- each parameterized
by t in [0, 1], this module fills in the interior surface points and the
quad face connectivity needed to render the panel (e.g. with
``matplotlib.collections.PolyCollection`` in 2D or
``mpl_toolkits.mplot3d.art3d.Poly3DCollection`` in 3D) or to hand off to a
structural/hydro mesh (e.g. OrcaFlex, a panel/BEM solver, or an FE mesh
generator).

Mathematical background
------------------------
For parameters (u, v) in [0, 1] x [0, 1], the Coons patch blends two ruled
surfaces (one per curve pair) and subtracts their shared bilinear
interpolation of the four corners, so the result matches all four boundary
curves exactly:

    S(u, v) = [(1-v) C0(u) + v C1(u)]              ruled surface (u-direction)
            + [(1-u) D0(v) + u D1(v)]              ruled surface (v-direction)
            - [(1-u)(1-v) P00 + u(1-v) P10
               + (1-u) v P01 + u v P11]             shared bilinear term

where C0/C1 = bottom/top curves, D0/D1 = left/right curves, and
P00, P10, P01, P11 are the four corner points (shared between adjacent
curves by construction). None of this arithmetic depends on the
dimensionality of a "point" -- it's just linear combinations of whatever
``(N, dim)`` arrays the curve functions produce -- so the same code path
handles planar (dim=2) and spatial (dim=3) panels alike.

Curve function contract
------------------------
Each curve function must accept a 1-D array of parameter values ``t`` of
shape ``(N,)`` (vectorized -- no internal Python loop over ``t``) and return
one of:

  * an array of shape ``(N, dim)`` of points, e.g. built with
    ``np.c_[x, y]`` / ``np.c_[x, y, z]`` or ``np.column_stack([...])``; or
  * a plain ``dim``-tuple/list of components, each either a scalar (for a
    constant coordinate, e.g. ``0.0``) or an array broadcastable to ``t``'s
    shape -- e.g. ``return (t, 0.3 * np.sin(np.pi * t))`` for a 2D curve.

``dim`` is not declared anywhere -- it's inferred the first time a curve is
evaluated (from the array's last axis, or the tuple's length) as either 2 or
3, and every other curve passed to the same :func:`coons_patch` call must
agree with it. Both forms are normalized internally to ``(N, dim)``. The
four corners must be shared between adjacent curves:

    bottom(0) == left(0)     -> P00
    bottom(1) == right(0)    -> P10
    top(0)    == left(1)     -> P01
    top(1)    == right(1)    -> P11
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

#: What a curve function may return for a given ``t``: either a pre-stacked
#: (N, dim) array, or a bare dim-tuple/list of scalars/arrays, dim in {2, 3}.
type CurveOutput = np.ndarray | Sequence[ArrayLike]

#: A curve function: maps an array of parameters ``t`` of shape (N,) to a
#: :data:`CurveOutput`. See "Curve function contract" above.
type CurveFn = Callable[[np.ndarray], CurveOutput]


class MeshDivisionError(ValueError):
    """Raised when ``nu`` or ``nv`` passed to :func:`coons_patch` is not a positive integer."""


class CornerMismatchError(ValueError):
    """Raised when two adjacent boundary curves disagree on a shared corner point."""


class CurveOutputError(TypeError):
    """Raised when a curve function's return value doesn't match the documented contract."""


class CurveDimensionMismatchError(ValueError):
    """Raised when the four boundary curves don't all produce points of the same dimension."""


def _as_points(value: CurveOutput, t: np.ndarray) -> np.ndarray:
    """Normalize a curve function's return value to an (N, dim) points array,
    dim in {2, 3}.

    Accepts either an array already shaped (N, 2) or (N, 3), or a 2- or
    3-tuple/list of x, y[, z] components that are scalars or arrays
    broadcastable to ``t``. Dispatches on the *type* of ``value`` (array vs.
    tuple/list) rather than its shape, so an N-point curve with N == 2 or
    N == 3 is never mistaken for a bare component tuple, or vice versa.
    """
    if isinstance(value, np.ndarray):
        if value.ndim == 2 and value.shape[-1] in (2, 3):
            return value
        msg = f"Curve function returned an array of shape {value.shape}, expected (N, 2) or (N, 3)"
        raise CurveOutputError(msg)

    if isinstance(value, (tuple, list)) and len(value) in (2, 3):
        raw = [np.asarray(component, dtype=float) for component in value]
        broadcasted = np.broadcast_arrays(*raw)
        components = [np.broadcast_to(component, t.shape) for component in broadcasted]
        return np.stack(components, axis=-1)

    msg = f"Curve function must return an (N, 2)/(N, 3) array or a 2/3-component tuple/list, got {value!r}"
    raise CurveOutputError(msg)


def _evaluate(curve: CurveFn, t: np.ndarray) -> np.ndarray:
    """Call ``curve(t)`` and normalize its result to an (N, dim) points array."""
    return _as_points(curve(t), t)


def coons_patch(
    bottom: CurveFn,
    top: CurveFn,
    left: CurveFn,
    right: CurveFn,
    nu: int = 20,
    nv: int = 20,
    check_corners: bool = True,
    atol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a structured quad mesh over a 4-sided patch via Coons
    (transfinite) interpolation. Dimension-agnostic: works for 2D or 3D
    boundary curves alike (see the module docstring).

    Parameters
    ----------
    bottom, top : callable
        Curves running in the u-direction, evaluated at ``t = 0`` (left
        end) and ``t = 1`` (right end). Each must map an array of shape
        ``(N,)`` to an array of shape ``(N, dim)``, dim in {2, 3}.
    left, right : callable
        Curves running in the v-direction, with the same signature as
        ``bottom``/``top``, and the same ``dim``.
    nu, nv : int, optional
        Number of mesh divisions along u and v. The resulting grid has
        ``(nu + 1) x (nv + 1)`` vertices and ``nu * nv`` quad faces.
        Must be positive integers (default 20).
    check_corners : bool, optional
        If True (default), verify that adjacent curves agree on shared
        corners to within ``atol`` and raise ``CornerMismatchError`` if not.
    atol : float, optional
        Absolute tolerance used for the corner-compatibility check.

    Returns
    -------
    grid : numpy.ndarray, shape (nu + 1, nv + 1, dim)
        Mesh vertices indexed ``grid[i, j]`` for ``u = u_i``, ``v = v_j``.
    faces : numpy.ndarray, shape (nu * nv, 4, dim)
        Quad faces as (4, dim) vertex loops in CCW order within each quad
        (``grid[i,j] -> grid[i+1,j] -> grid[i+1,j+1] -> grid[i,j+1]``),
        ready to pass directly to ``matplotlib.collections.PolyCollection``
        (dim=2) or ``mpl_toolkits.mplot3d.art3d.Poly3DCollection`` (dim=3).

    Raises
    ------
    MeshDivisionError
        If ``nu`` or ``nv`` is not a positive integer.
    CornerMismatchError
        If ``check_corners`` is True and adjacent curves disagree on a
        shared corner by more than ``atol``.
    CurveOutputError
        If a curve function's return value doesn't match the documented
        contract (see "Curve function contract" in the module docstring).
    CurveDimensionMismatchError
        If the four curves don't all produce points of the same dimension.

    Notes
    -----
    Because the blend is bilinear in (u, v), a patch bounded by four
    straight-line edges reduces exactly to a flat bilinear (ruled) surface
    -- see the Examples below.

    Examples
    --------
    A flat unit-square patch in 2D (straight edges) reduces to a plain
    bilinear grid:

    >>> import numpy as np
    >>> bottom = lambda t: np.c_[t, np.zeros_like(t)]
    >>> top    = lambda t: np.c_[t, np.ones_like(t)]
    >>> left   = lambda t: np.c_[np.zeros_like(t), t]
    >>> right  = lambda t: np.c_[np.ones_like(t),  t]
    >>> grid, faces = coons_patch(bottom, top, left, right, nu=4, nv=4)
    >>> grid.shape
    (5, 5, 2)
    >>> faces.shape
    (16, 4, 2)
    >>> np.allclose(grid[2, 3], [0.5, 0.75])
    True

    The same flat unit-square patch, but with 3D curves (z = 0 everywhere)
    -- identical code path, dim inferred as 3 instead of 2:

    >>> bottom = lambda t: np.c_[t, np.zeros_like(t), np.zeros_like(t)]
    >>> top    = lambda t: np.c_[t, np.ones_like(t),  np.zeros_like(t)]
    >>> left   = lambda t: np.c_[np.zeros_like(t), t, np.zeros_like(t)]
    >>> right  = lambda t: np.c_[np.ones_like(t),  t, np.zeros_like(t)]
    >>> grid, faces = coons_patch(bottom, top, left, right, nu=4, nv=4)
    >>> grid.shape
    (5, 5, 3)
    >>> np.allclose(grid[2, 3], [0.5, 0.75, 0.0])
    True

    A warped 3D panel with curved edges, written with the bare-tuple
    (x, y, z) component form instead of ``np.c_``-stacked arrays -- both
    forms are accepted:

    >>> def bottom(t): return (t, 0.0, 0.3 * np.sin(np.pi * t))
    >>> def top(t): return (t, 1.0, 0.5 * np.sin(np.pi * t) + 0.4)
    >>> def left(t): return (0.0, t, 0.4 * t)
    >>> def right(t): return (1.0, t, 0.3 * np.sin(np.pi * t) + 0.4 * t)
    >>> grid, faces = coons_patch(bottom, top, left, right, nu=24, nv=16)
    >>> grid.shape
    (25, 17, 3)
    """
    if not (isinstance(nu, (int, np.integer)) and nu > 0):
        msg = f"nu must be a positive integer, got {nu!r}"
        raise MeshDivisionError(msg)
    if not (isinstance(nv, (int, np.integer)) and nv > 0):
        msg = f"nv must be a positive integer, got {nv!r}"
        raise MeshDivisionError(msg)

    u, v = np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1)
    u_grid, v_grid = (g[..., None] for g in np.meshgrid(u, v, indexing="ij"))

    zero, one = np.array([0.0]), np.array([1.0])
    c_bottom, c_top = _evaluate(bottom, u)[:, None, :], _evaluate(top, u)[:, None, :]  # v-edges
    d_left, d_right = _evaluate(left, v)[None, :, :], _evaluate(right, v)[None, :, :]  # u-edges

    dims = {
        "bottom": c_bottom.shape[-1],
        "top": c_top.shape[-1],
        "left": d_left.shape[-1],
        "right": d_right.shape[-1],
    }
    if len(set(dims.values())) != 1:
        msg = f"all four curves must return points of the same dimension, got {dims}"
        raise CurveDimensionMismatchError(msg)

    p00, p10 = _evaluate(bottom, zero)[0], _evaluate(bottom, one)[0]
    p01, p11 = _evaluate(top, zero)[0], _evaluate(top, one)[0]

    if check_corners:
        pairs = {
            "bottom(0) vs left(0)": (p00, _evaluate(left, zero)[0]),
            "bottom(1) vs right(0)": (p10, _evaluate(right, zero)[0]),
            "top(0) vs left(1)": (p01, _evaluate(left, one)[0]),
            "top(1) vs right(1)": (p11, _evaluate(right, one)[0]),
        }
        for name, (a, b) in pairs.items():
            if not np.allclose(a, b, atol=atol):
                msg = f"Corner mismatch [{name}]: {a} != {b} (atol={atol})"
                raise CornerMismatchError(msg)

    ruled = (1 - v_grid) * c_bottom + v_grid * c_top + (1 - u_grid) * d_left + u_grid * d_right
    bilinear = (1 - u_grid) * (1 - v_grid) * p00 + u_grid * (1 - v_grid) * p10 + (1 - u_grid) * v_grid * p01 + u_grid * v_grid * p11
    grid = ruled - bilinear  # (nu+1, nv+1, dim)

    dim = grid.shape[-1]
    faces = np.stack([grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]], axis=-2).reshape(-1, 4, dim)
    return grid, faces
