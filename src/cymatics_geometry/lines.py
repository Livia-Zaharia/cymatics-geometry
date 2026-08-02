"""Reconnect a displaced point grid into line geometry (grid or serpentine)."""

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


def _nan_break() -> np.ndarray:
    return np.full((1, 3), np.nan, dtype=float)


def grid_line_segments(
    points: np.ndarray,
    nx: int,
    ny: int,
    *,
    lines_x: bool = True,
    lines_y: bool = True,
) -> list[np.ndarray]:
    """Collect parallel X-row and/or Y-column polylines from a displaced grid.

    - ``lines_x``: one polyline per grid row (runs parallel to the X axis)
    - ``lines_y``: one polyline per grid column (runs parallel to the Y axis)
    """
    pts = np.asarray(points, dtype=float).reshape(int(ny), int(nx), 3)
    segments: list[np.ndarray] = []
    if lines_x:
        for row in range(int(ny)):
            segments.append(pts[row, :, :].copy())
    if lines_y:
        for col in range(int(nx)):
            segments.append(pts[:, col, :].copy())
    return segments


def stride_indices_with_boundary(
    count: int,
    stride: int,
    boundary: int,
) -> list[int]:
    """Index set: every ``stride``-th line, with signed end treatment.

    ``boundary``:
      - ``> 0`` — always **keep** first N and last N lines
      - ``< 0`` — **remove** first |N| and last |N| lines (from the strided set)
      - ``0`` — stride only
    """
    m = int(count)
    if m <= 0:
        return []
    stride_i = max(1, int(stride))
    keep: set[int] = set(range(0, m, stride_i))
    b = int(boundary)
    if b > 0:
        for i in range(min(b, m)):
            keep.add(i)
            keep.add(m - 1 - i)
    elif b < 0:
        remove_n = min(abs(b), m)
        drop: set[int] = set()
        for i in range(remove_n):
            drop.add(i)
            drop.add(m - 1 - i)
        keep -= drop
    return sorted(keep)


def select_segments_strided(
    segments: list[np.ndarray],
    *,
    stride: int = 1,
    boundary: int = 0,
) -> list[np.ndarray]:
    """Thin a parallel line group with signed keep/remove end lines."""
    idxs = stride_indices_with_boundary(len(segments), stride, boundary)
    return [segments[i] for i in idxs]


def segments_to_nan_polyline(segments: list[np.ndarray]) -> np.ndarray:
    """Join segments with NaN breaks for Plotly multi-line Scatter3d."""
    if not segments:
        return np.zeros((0, 3), dtype=float)
    chunks: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        chunks.append(np.asarray(seg, dtype=float))
        if i + 1 < len(segments):
            chunks.append(_nan_break())
    return np.vstack(chunks)


def segments_to_polydata(segments: list[np.ndarray]) -> pv.PolyData:
    """Build a PyVista mesh with one line cell per segment."""
    if not segments:
        return pv.PolyData()
    all_pts: list[np.ndarray] = []
    lines: list[int] = []
    offset = 0
    for seg in segments:
        s = np.asarray(seg, dtype=float)
        n = len(s)
        if n < 2:
            continue
        all_pts.append(s)
        lines.append(n)
        lines.extend(range(offset, offset + n))
        offset += n
    if not all_pts:
        return pv.PolyData()
    mesh = pv.PolyData()
    mesh.points = np.vstack(all_pts)
    mesh.lines = np.asarray(lines, dtype=np.int64)
    return mesh


def polyline_to_polydata(polyline: np.ndarray) -> pv.PolyData:
    """Build a PyVista line PolyData from an ordered (M, 3) point array.

    NaN rows (segment breaks) are stripped; remaining points form one line.
    Prefer :func:`segments_to_polydata` for multi-line grids.
    """
    pts = np.asarray(polyline, dtype=float)
    if len(pts) == 0:
        return pv.PolyData()
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) < 2:
        return pv.PolyData()
    lines = np.hstack([[len(pts)], np.arange(len(pts), dtype=np.int64)])
    mesh = pv.PolyData()
    mesh.points = pts
    mesh.lines = lines
    return mesh


def polyline_length(polyline: np.ndarray) -> float:
    """Total length of a polyline; NaN breaks split independent segments."""
    pts = np.asarray(polyline, dtype=float)
    if len(pts) < 2:
        return 0.0
    total = 0.0
    start = 0
    for i in range(len(pts) + 1):
        at_break = i == len(pts) or not np.isfinite(pts[i]).all()
        if at_break:
            chunk = pts[start:i]
            if len(chunk) >= 2:
                total += float(np.sum(np.linalg.norm(np.diff(chunk, axis=0), axis=1)))
            start = i + 1
    return total


def build_line_geometry(
    displaced_points: np.ndarray,
    grid_size_x: int,
    grid_size_y: int | None = None,
    *,
    pattern: str = "grid",
    lines_x: bool = True,
    lines_y: bool = True,
    line_stride: int = 1,
    boundary_lines_x: int = 0,
    boundary_lines_y: int = 0,
    boundary_lines: int | None = None,
) -> tuple[np.ndarray, pv.PolyData]:
    """Reconnect displaced grid points into line geometry.

    Patterns
    --------
    ``grid``
        Parallel X-row and/or Y-column lines (default). Toggle with
        ``lines_x`` / ``lines_y``. ``line_stride`` keeps every N-th line.
        ``boundary_lines_x`` / ``boundary_lines_y`` are signed: positive keeps
        first/last N in that direction, negative removes first/last |N|.
        Legacy ``boundary_lines`` (if given) applies the same value to both.
    ``serpentine`` / ``row_major``
        Legacy single continuous polyline.
    """
    nx = int(grid_size_x)
    ny = int(grid_size_y) if grid_size_y is not None else nx
    pts = np.asarray(displaced_points, dtype=float)
    stride = max(1, int(line_stride))
    if boundary_lines is not None:
        bx = by = int(boundary_lines)
    else:
        bx = int(boundary_lines_x)
        by = int(boundary_lines_y)

    if pattern == "grid":
        if not lines_x and not lines_y:
            return np.zeros((0, 3), dtype=float), pv.PolyData()
        raw = grid_line_segments(
            pts, nx, ny, lines_x=bool(lines_x), lines_y=bool(lines_y)
        )
        n_x = ny if lines_x else 0
        n_y = nx if lines_y else 0
        x_segs = raw[:n_x]
        y_segs = raw[n_x : n_x + n_y]
        segments = select_segments_strided(
            x_segs, stride=stride, boundary=bx
        ) + select_segments_strided(y_segs, stride=stride, boundary=by)
        polyline = segments_to_nan_polyline(segments)
        mesh = segments_to_polydata(segments)
        return polyline, mesh

    polyline = ordered_polyline_points(pts, nx, ny, pattern=pattern)
    mesh = polyline_to_polydata(polyline)
    return polyline, mesh
