"""
coons_quad_mesh
================

Generate a structured quadrilateral mesh over a 4-sided curvilinear panel
using a bilinearly-blended Coons patch (transfinite interpolation).

Given four boundary curves -- bottom, top, left, right -- each parameterized
by t in [0, 1], this module fills in the interior surface points and the
quad face connectivity needed to render the panel (e.g. with
``mpl_toolkits.mplot3d.art3d.Poly3DCollection``) or to hand off to a
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
curves by construction).

Curve function contract
------------------------
Each curve function must accept a 1-D array of parameter values ``t`` of
shape ``(N,)`` and return an array of shape ``(N, 3)`` of (x, y, z) points
(vectorized -- no internal Python loop over ``t``). The four corners must
be shared between adjacent curves:

    bottom(0) == left(0)     -> P00
    bottom(1) == right(0)    -> P10
    top(0)    == left(1)     -> P01
    top(1)    == right(1)    -> P11
"""

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

#: A curve function: maps an array of parameters ``t`` of shape (N,) to an
#: array of (x, y, z) points of shape (N, 3).
CurveFn = Callable[[np.ndarray], np.ndarray]


class MeshDivisionError(ValueError):
    """Raised when ``nu`` or ``nv`` passed to :func:`coons_patch` is not a positive integer."""


class CornerMismatchError(ValueError):
    """Raised when two adjacent boundary curves disagree on a shared corner point."""


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
    (transfinite) interpolation.

    Parameters
    ----------
    bottom, top : callable
        Curves running in the u-direction, evaluated at ``t = 0`` (left
        end) and ``t = 1`` (right end). Each must map an array of shape
        ``(N,)`` to an array of shape ``(N, 3)``.
    left, right : callable
        Curves running in the v-direction, with the same signature as
        ``bottom``/``top``.
    nu, nv : int, optional
        Number of mesh divisions along u and v. The resulting grid has
        ``(nu + 1) x (nv + 1)`` vertices and ``nu * nv`` quad faces.
        Must be positive integers (default 20).
    check_corners : bool, optional
        If True (default), verify that adjacent curves agree on shared
        corners to within ``atol`` and raise ``ValueError`` if not.
    atol : float, optional
        Absolute tolerance used for the corner-compatibility check.

    Returns
    -------
    grid : numpy.ndarray, shape (nu + 1, nv + 1, 3)
        Mesh vertices indexed ``grid[i, j]`` for ``u = u_i``, ``v = v_j``.
    faces : numpy.ndarray, shape (nu * nv, 4, 3)
        Quad faces as (4, 3) vertex loops in CCW order within each quad
        (``grid[i,j] -> grid[i+1,j] -> grid[i+1,j+1] -> grid[i,j+1]``),
        ready to pass directly to
        ``mpl_toolkits.mplot3d.art3d.Poly3DCollection``.

    Raises
    ------
    MeshDivisionError
        If ``nu`` or ``nv`` is not a positive integer.
    CornerMismatchError
        If ``check_corners`` is True and adjacent curves disagree on a
        shared corner by more than ``atol``.

    Notes
    -----
    Because the blend is bilinear in (u, v), a patch bounded by four
    straight-line edges reduces exactly to a flat bilinear (ruled) surface
    -- see the Examples below.

    Examples
    --------
    A flat unit-square patch (straight edges) reduces to a plain bilinear
    grid in x and y, with z = 0 everywhere:

    >>> import numpy as np
    >>> bottom = lambda t: np.c_[t, np.zeros_like(t), np.zeros_like(t)]
    >>> top    = lambda t: np.c_[t, np.ones_like(t),  np.zeros_like(t)]
    >>> left   = lambda t: np.c_[np.zeros_like(t), t, np.zeros_like(t)]
    >>> right  = lambda t: np.c_[np.ones_like(t),  t, np.zeros_like(t)]
    >>> grid, faces = coons_patch(bottom, top, left, right, nu=4, nv=4)
    >>> grid.shape
    (5, 5, 3)
    >>> faces.shape
    (16, 4, 3)
    >>> np.allclose(grid[2, 3], [0.5, 0.75, 0.0])
    True
    """
    if not (isinstance(nu, (int, np.integer)) and nu > 0):
        msg = f"nu must be a positive integer, got {nu!r}"
        raise MeshDivisionError(msg)
    if not (isinstance(nv, (int, np.integer)) and nv > 0):
        msg = f"nv must be a positive integer, got {nv!r}"
        raise MeshDivisionError(msg)

    u, v = np.linspace(0, 1, nu + 1), np.linspace(0, 1, nv + 1)
    u_grid, v_grid = (g[..., None] for g in np.meshgrid(u, v, indexing="ij"))

    c_bottom, c_top = bottom(u)[:, None, :], top(u)[:, None, :]  # v-edges, broadcast over v
    d_left, d_right = left(v)[None, :, :], right(v)[None, :, :]  # u-edges, broadcast over u
    p00, p10 = bottom(np.array([0.0]))[0], bottom(np.array([1.0]))[0]
    p01, p11 = top(np.array([0.0]))[0], top(np.array([1.0]))[0]

    if check_corners:
        pairs = {
            "bottom(0) vs left(0)": (p00, left(np.array([0.0]))[0]),
            "bottom(1) vs right(0)": (p10, right(np.array([0.0]))[0]),
            "top(0) vs left(1)": (p01, left(np.array([1.0]))[0]),
            "top(1) vs right(1)": (p11, right(np.array([1.0]))[0]),
        }
        for name, (a, b) in pairs.items():
            if not np.allclose(a, b, atol=atol):
                msg = f"Corner mismatch [{name}]: {a} != {b} (atol={atol})"
                raise CornerMismatchError(msg)

    ruled = (1 - v_grid) * c_bottom + v_grid * c_top + (1 - u_grid) * d_left + u_grid * d_right
    bilinear = (1 - u_grid) * (1 - v_grid) * p00 + u_grid * (1 - v_grid) * p10 + (1 - u_grid) * v_grid * p01 + u_grid * v_grid * p11
    grid = ruled - bilinear  # (nu+1, nv+1, 3)

    faces = np.stack([grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]], axis=-2).reshape(-1, 4, 3)
    return grid, faces
