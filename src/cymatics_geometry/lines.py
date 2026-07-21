"""Reconnect a displaced point grid into a continuous line geometry."""

from __future__ import annotations

import numpy as np
import pyvista as pv


def serpentine_order(nx: int, ny: int | None = None) -> np.ndarray:
    """Row-wise serpentine index order for an nx×ny grid (flat row-major layout).

    Even rows left→right, odd rows right→left, so consecutive samples share an edge.
    """
    if ny is None:
        ny = nx
    nx_i, ny_i = int(nx), int(ny)
    order: list[int] = []
    for row in range(ny_i):
        cols = range(nx_i) if row % 2 == 0 else range(nx_i - 1, -1, -1)
        for col in cols:
            order.append(row * nx_i + col)
    return np.asarray(order, dtype=int)


def row_major_order(nx: int, ny: int | None = None) -> np.ndarray:
    """Simple left-to-right, bottom-to-top row-major order."""
    if ny is None:
        ny = nx
    return np.arange(int(nx) * int(ny), dtype=int)


def ordered_polyline_points(
    points: np.ndarray,
    grid_size_x: int,
    grid_size_y: int | None = None,
    *,
    pattern: str = "serpentine",
) -> np.ndarray:
    """Return points reordered into a single continuous polyline path."""
    pts = np.asarray(points, dtype=float)
    nx = int(grid_size_x)
    ny = int(grid_size_y) if grid_size_y is not None else nx
    expected = nx * ny
    if pts.shape[0] != expected:
        raise ValueError(
            f"Expected {expected} points for a {nx}×{ny} grid, got {pts.shape[0]}"
        )
    if pattern == "serpentine":
        order = serpentine_order(nx, ny)
    elif pattern == "row_major":
        order = row_major_order(nx, ny)
    else:
        raise ValueError(f"Unknown line_pattern: {pattern!r}")
    return pts[order]


def polyline_to_polydata(polyline: np.ndarray) -> pv.PolyData:
    """Build a PyVista line PolyData from an ordered (M, 3) point array."""
    pts = np.asarray(polyline, dtype=float)
    if len(pts) < 2:
        return pv.PolyData()
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
    grid_size_x: int,
    grid_size_y: int | None = None,
    *,
    pattern: str = "serpentine",
) -> tuple[np.ndarray, pv.PolyData]:
    """Reconnect displaced grid points into a continuous line mesh."""
    polyline = ordered_polyline_points(
        displaced_points,
        grid_size_x,
        grid_size_y,
        pattern=pattern,
    )
    mesh = polyline_to_polydata(polyline)
    return polyline, mesh
