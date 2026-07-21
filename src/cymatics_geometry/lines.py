"""Reconnect a displaced point grid into a continuous line geometry."""

from __future__ import annotations

import numpy as np
import pyvista as pv


def serpentine_order(n: int) -> np.ndarray:
    """Row-wise serpentine index order for an n×n grid (flat row-major layout).

    Even rows left→right, odd rows right→left, so consecutive samples share an edge.
    """
    order: list[int] = []
    for row in range(n):
        cols = range(n) if row % 2 == 0 else range(n - 1, -1, -1)
        for col in cols:
            order.append(row * n + col)
    return np.asarray(order, dtype=int)


def row_major_order(n: int) -> np.ndarray:
    """Simple left-to-right, bottom-to-top row-major order."""
    return np.arange(n * n, dtype=int)


def ordered_polyline_points(
    points: np.ndarray,
    grid_size: int,
    *,
    pattern: str = "serpentine",
) -> np.ndarray:
    """Return points reordered into a single continuous polyline path."""
    pts = np.asarray(points, dtype=float)
    n = int(grid_size)
    if pts.shape[0] != n * n:
        raise ValueError(
            f"Expected {n * n} points for a {n}×{n} grid, got {pts.shape[0]}"
        )
    if pattern == "serpentine":
        order = serpentine_order(n)
    elif pattern == "row_major":
        order = row_major_order(n)
    else:
        raise ValueError(f"Unknown line_pattern: {pattern!r}")
    return pts[order]


def polyline_to_polydata(polyline: np.ndarray) -> pv.PolyData:
    """Build a PyVista line PolyData from an ordered (M, 3) point array."""
    pts = np.asarray(polyline, dtype=float)
    if len(pts) < 2:
        return pv.PolyData()
    # VTK line cell: [n_points, i0, i1, ..., i_{n-1}]
    lines = np.hstack([[len(pts)], np.arange(len(pts), dtype=np.int64)])
    mesh = pv.PolyData()
    mesh.points = pts
    mesh.lines = lines
    return mesh


def polyline_length(polyline: np.ndarray) -> float:
    pts = np.asarray(polyline, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def build_line_geometry(
    displaced_points: np.ndarray,
    grid_size: int,
    *,
    pattern: str = "serpentine",
) -> tuple[np.ndarray, pv.PolyData]:
    """Reconnect displaced grid points into a continuous line mesh."""
    polyline = ordered_polyline_points(
        displaced_points, grid_size, pattern=pattern
    )
    mesh = polyline_to_polydata(polyline)
    return polyline, mesh
