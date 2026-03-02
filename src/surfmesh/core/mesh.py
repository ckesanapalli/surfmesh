"""QuadMesh — the central data structure for surfmesh.

A ``QuadMesh`` wraps the raw ``(n_panels, 4, 3)`` panel array produced by
every mesher in this library and exposes the derived quantities that
Boundary Element Method (BEM) solvers and mesh-quality tools need:
centroids, outward unit normals, panel areas, and the standard
``(vertices, faces)`` connectivity representation.

All secondary properties are computed lazily on first access and cached so
that repeated look-ups carry no extra cost.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
from numpy.typing import ArrayLike

from surfmesh.core.topology import extract_vertices_faces
from surfmesh.core.validate import validate_mesh_3d

__all__ = ["QuadMesh"]


class QuadMesh:
    """Quadrilateral surface mesh container with BEM-oriented derived quantities.

    Parameters
    ----------
    panels:
        Raw panel array of shape ``(n_panels, 4, 3)``.  Each panel is
        defined by four 3-D vertices ordered **counter-clockwise** when
        viewed from the *outside* of the surface so that normals point
        outward by convention.

    Attributes
    ----------
    panels : ndarray of shape ``(n_panels, 4, 3)``
        The underlying raw panel data (immutable view).
    n_panels : int
        Number of panels.
    n_vertices : int
        Number of unique vertices after de-duplication.
    vertices : ndarray of shape ``(n_vertices, 3)``
        De-duplicated vertex coordinate array.
    faces : ndarray of shape ``(n_panels, 4)``
        Integer connectivity array (indices into ``vertices``).
    centroids : ndarray of shape ``(n_panels, 3)``
        Arithmetic mean of the four panel corner coordinates.
    normals : ndarray of shape ``(n_panels, 3)``
        Outward unit normal vectors, computed via the cross product of the
        two panel diagonals.
    areas : ndarray of shape ``(n_panels,)``
        Panel areas, equal to ``0.5 * |d1 × d2|`` where *d1* and *d2* are
        the two diagonals.  Exact for planar quads.
    total_area : float
        Sum of all panel areas.

    Examples
    --------
    >>> import numpy as np
    >>> from surfmesh import sphere_mesher_from_projection, QuadMesh
    >>> panels = sphere_mesher_from_projection(radius=1.0, resolution=4)
    >>> mesh = QuadMesh(panels)
    >>> mesh.n_panels
    96
    >>> mesh.normals.shape
    (96, 3)
    """

    def __init__(self, panels: ArrayLike) -> None:
        panels = np.asarray(panels, dtype=float)
        validate_mesh_3d(panels, name="panels")
        if panels.shape[1] != 4:
            msg = (
                f"panels must have exactly 4 vertices per face (shape[1] == 4), "
                f"got shape {panels.shape}."
            )
            raise ValueError(msg)
        # Store as read-only to prevent accidental mutation that would
        # silently invalidate cached properties.
        panels.flags.writeable = False
        self._panels = panels

    # ------------------------------------------------------------------
    # Raw data
    # ------------------------------------------------------------------

    @property
    def panels(self) -> np.ndarray:
        """Raw ``(n_panels, 4, 3)`` panel array (read-only)."""
        return self._panels

    # ------------------------------------------------------------------
    # Size information
    # ------------------------------------------------------------------

    @cached_property
    def n_panels(self) -> int:
        """Number of panels."""
        return int(self._panels.shape[0])

    @cached_property
    def n_vertices(self) -> int:
        """Number of unique vertices after de-duplication."""
        return int(self.vertices.shape[0])

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------

    @cached_property
    def _topology(self) -> tuple[np.ndarray, np.ndarray]:
        """Cached ``(vertices, faces)`` pair."""
        return extract_vertices_faces(self._panels)

    @cached_property
    def vertices(self) -> np.ndarray:
        """De-duplicated vertex coordinates, shape ``(n_vertices, 3)``."""
        verts, _ = self._topology
        return verts

    @cached_property
    def faces(self) -> np.ndarray:
        """Integer connectivity array, shape ``(n_panels, 4)``."""
        _, faces = self._topology
        return faces

    # ------------------------------------------------------------------
    # BEM panel quantities
    # ------------------------------------------------------------------

    @cached_property
    def centroids(self) -> np.ndarray:
        """Panel centroids (mean of four corners), shape ``(n_panels, 3)``."""
        return self._panels.mean(axis=1)

    @cached_property
    def normals(self) -> np.ndarray:
        """Outward unit normal vectors, shape ``(n_panels, 3)``.

        Computed as the normalised cross product of the two panel diagonals
        ``(v2 - v0)`` and ``(v3 - v1)``.  This formula is robust for
        non-planar (warped) quads and reduces to the exact geometric normal
        for planar panels.
        """
        d1 = self._panels[:, 2] - self._panels[:, 0]  # diagonal v0 → v2
        d2 = self._panels[:, 3] - self._panels[:, 1]  # diagonal v1 → v3
        cross = np.cross(d1, d2)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        return cross / norms

    @cached_property
    def areas(self) -> np.ndarray:
        """Panel areas, shape ``(n_panels,)``.

        Uses ``0.5 * |d1 × d2|``, which is exact for planar quadrilaterals
        and gives a reasonable approximation for mildly warped panels.
        """
        d1 = self._panels[:, 2] - self._panels[:, 0]
        d2 = self._panels[:, 3] - self._panels[:, 1]
        cross = np.cross(d1, d2)
        return 0.5 * np.linalg.norm(cross, axis=1)

    @cached_property
    def total_area(self) -> float:
        """Scalar sum of all panel areas."""
        return float(self.areas.sum())

    # ------------------------------------------------------------------
    # Convenience / export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, np.ndarray | int | float]:
        """Return all mesh quantities as a plain Python dictionary.

        Keys
        ----
        ``"panels"``, ``"vertices"``, ``"faces"``, ``"centroids"``,
        ``"normals"``, ``"areas"``, ``"n_panels"``, ``"n_vertices"``,
        ``"total_area"``.
        """
        return {
            "panels": self.panels,
            "vertices": self.vertices,
            "faces": self.faces,
            "centroids": self.centroids,
            "normals": self.normals,
            "areas": self.areas,
            "n_panels": self.n_panels,
            "n_vertices": self.n_vertices,
            "total_area": self.total_area,
        }

    def __repr__(self) -> str:
        return (
            f"QuadMesh("
            f"n_panels={self.n_panels}, "
            f"n_vertices={self.n_vertices}, "
            f"total_area={self.total_area:.6g})"
        )
