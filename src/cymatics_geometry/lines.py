"""Reconnect a displaced point grid into line geometry (grid or serpentine)."""

from __future__ import annotations

from collections.abc import Callable

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


def _pt_key(point: np.ndarray, *, ndigits: int = 7) -> tuple[float, float, float]:
    r = np.round(np.asarray(point, dtype=float).reshape(-1)[:3], ndigits)
    return (float(r[0]), float(r[1]), float(r[2]))


def join_segments_to_polylines(segments: list[np.ndarray]) -> list[np.ndarray]:
    """Connect segments that share endpoints into maximal chains / loops."""
    pieces: list[np.ndarray] = []
    for seg in segments:
        pts = np.asarray(seg, dtype=float)
        finite = np.isfinite(pts).all(axis=1)
        pts = pts[finite]
        if len(pts) < 2:
            continue
        pieces.append(pts)
    if not pieces:
        return []

    unused = set(range(len(pieces)))
    start_of: dict[tuple[float, float, float], list[int]] = {}
    end_of: dict[tuple[float, float, float], list[int]] = {}
    for i, pts in enumerate(pieces):
        start_of.setdefault(_pt_key(pts[0]), []).append(i)
        end_of.setdefault(_pt_key(pts[-1]), []).append(i)

    def _pop_match(
        table: dict[tuple[float, float, float], list[int]],
        key: tuple[float, float, float],
    ) -> int | None:
        ids = table.get(key)
        if not ids:
            return None
        while ids:
            idx = ids.pop()
            if idx in unused:
                return idx
        return None

    loops: list[np.ndarray] = []
    while unused:
        idx = unused.pop()
        chain = pieces[idx]
        extended = True
        while extended:
            extended = False
            head = _pt_key(chain[0])
            tail = _pt_key(chain[-1])
            nxt = _pop_match(start_of, tail)
            if nxt is not None:
                unused.discard(nxt)
                add = pieces[nxt]
                chain = np.vstack([chain, add[1:]])
                extended = True
                continue
            nxt = _pop_match(end_of, tail)
            if nxt is not None:
                unused.discard(nxt)
                add = pieces[nxt][::-1]
                chain = np.vstack([chain, add[1:]])
                extended = True
                continue
            prev = _pop_match(end_of, head)
            if prev is not None:
                unused.discard(prev)
                add = pieces[prev]
                chain = np.vstack([add[:-1], chain])
                extended = True
                continue
            prev = _pop_match(start_of, head)
            if prev is not None:
                unused.discard(prev)
                add = pieces[prev][::-1]
                chain = np.vstack([add[:-1], chain])
                extended = True
        if len(chain) >= 2:
            loops.append(chain)
    loops.sort(key=lambda s: -float(np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1))))
    return loops


def _rectangular_index_loop(nx: int, ny: int) -> list[tuple[int, int]]:
    """Lattice (row, col) around the plane rectangle, CW from NW.

    Row 0 is south (min Y), row ``ny-1`` is north. Walks NW→NE→SE→SW.
    """
    if nx < 2 or ny < 2:
        return []
    ordered: list[tuple[int, int]] = []
    for col in range(nx):
        ordered.append((ny - 1, col))
    for row in range(ny - 2, -1, -1):
        ordered.append((row, nx - 1))
    for col in range(nx - 2, -1, -1):
        ordered.append((0, col))
    for row in range(1, ny - 1):
        ordered.append((row, 0))
    return ordered


_STRETCH = 2.2


def _close_xyz_loop(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[:1]])
    return pts




def alive_lattice_mask(
    moved_points: np.ndarray,
    nx: int,
    ny: int,
    *,
    inside: np.ndarray | None = None,
) -> np.ndarray:
    """True where a lattice point is finite and still shown (inside the clip)."""
    moved = np.asarray(moved_points, dtype=float).reshape(int(ny), int(nx), 3)
    finite = np.isfinite(moved).all(axis=-1)
    if inside is None:
        return finite
    return finite & np.asarray(inside, dtype=bool).reshape(int(ny), int(nx))


def stretch_break_length(moved: np.ndarray, alive: np.ndarray) -> float:
    """Max kept edge length: a multiple of the median in-bounds neighbor step."""
    return _STRETCH * median_alive_edge_length(moved, alive)


def median_alive_edge_length(moved: np.ndarray, alive: np.ndarray) -> float:
    """Median 4-connected spacing among points that are still in-bounds."""
    ny, nx = int(alive.shape[0]), int(alive.shape[1])
    lengths: list[float] = []
    for row in range(ny):
        for col in range(nx):
            if not alive[row, col]:
                continue
            if col + 1 < nx and alive[row, col + 1]:
                lengths.append(float(np.linalg.norm(moved[row, col] - moved[row, col + 1])))
            if row + 1 < ny and alive[row + 1, col]:
                lengths.append(float(np.linalg.norm(moved[row, col] - moved[row + 1, col])))
    if not lengths:
        return 1.0
    return float(np.median(np.asarray(lengths, dtype=float)))


def split_polyline_edges(
    points: np.ndarray,
    *,
    alive_along: np.ndarray | None = None,
    max_edge: float | None = None,
) -> list[np.ndarray]:
    """Break a polyline where points are missing or an edge is stretched."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 2:
        return []
    flags = (
        np.ones(n, dtype=bool)
        if alive_along is None
        else np.asarray(alive_along, dtype=bool)
    )
    chunks: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for i in range(n):
        if not flags[i] or not np.isfinite(pts[i]).all():
            if len(current) >= 2:
                chunks.append(np.vstack(current))
            current = []
            continue
        if current and max_edge is not None:
            if float(np.linalg.norm(pts[i] - current[-1])) > float(max_edge):
                if len(current) >= 2:
                    chunks.append(np.vstack(current))
                current = [pts[i]]
                continue
        current.append(pts[i])
    if len(current) >= 2:
        chunks.append(np.vstack(current))
    return chunks




def _segment_exit_point(
    p_in: np.ndarray,
    p_out: np.ndarray,
    *,
    z_lim: float | None = None,
    exit_fn: Callable[[np.ndarray, np.ndarray], np.ndarray | None] | None = None,
) -> np.ndarray:
    """Point on ``p_in→p_out`` that sits on the clip (stub end)."""
    a = np.asarray(p_in, dtype=float).reshape(3)
    b = np.asarray(p_out, dtype=float).reshape(3)
    if exit_fn is not None:
        hit = exit_fn(a, b)
        if hit is not None:
            return np.asarray(hit, dtype=float).reshape(3)
    if z_lim is not None:
        z0, z1 = float(a[2]), float(b[2])
        span = z1 - z0
        if abs(span) > 1e-15:
            for face in (float(z_lim), -float(z_lim)):
                t = (face - z0) / span
                if 0.0 <= t <= 1.0:
                    return a + t * (b - a)
    return 0.5 * (a + b)


# Marching-squares edge pairs. Corners: SW=1, SE=2, NE=4, NW=8.
# Edges: 0=S (SW-SE), 1=E (SE-NE), 2=N (NE-NW), 3=W (NW-SW).
_MS_PAIRS: dict[int, tuple[tuple[int, int], ...]] = {
    1: ((3, 0),),
    2: ((0, 1),),
    3: ((3, 1),),
    4: ((1, 2),),
    5: ((3, 2), (0, 1)),
    6: ((0, 2),),
    7: ((3, 2),),
    8: ((2, 3),),
    9: ((0, 2),),
    10: ((0, 3), (1, 2)),
    11: ((1, 2),),
    12: ((1, 3),),
    13: ((0, 1),),
    14: ((0, 3),),
}


def _align_loop_to_start(
    loop: np.ndarray,
    start_xyz: np.ndarray,
    toward_xyz: np.ndarray,
) -> np.ndarray:
    """Rotate/reverse a closed loop so it starts at ``start_xyz`` heading toward ``toward``."""
    pts = np.asarray(loop, dtype=float)
    body = pts[:-1] if len(pts) >= 2 and np.allclose(pts[0], pts[-1], atol=1e-8) else pts
    if len(body) < 3:
        return _close_xyz_loop(pts)
    i = int(np.argmin(np.sum((body - np.asarray(start_xyz, dtype=float).reshape(1, 3)) ** 2, axis=1)))
    rot = np.vstack([body[i:], body[:i]])
    toward = np.asarray(toward_xyz, dtype=float).reshape(3)
    d_fwd = float(np.sum((rot[1] - toward) ** 2))
    d_rev = float(np.sum((rot[-1] - toward) ** 2))
    if d_rev < d_fwd:
        rot = np.vstack([rot[0:1], rot[:0:-1]])
    return _close_xyz_loop(rot)


def clip_interface_loops(
    moved: np.ndarray,
    alive: np.ndarray,
    *,
    z_lim: float | None = None,
    exit_fn: Callable[[np.ndarray, np.ndarray], np.ndarray | None] | None = None,
) -> list[np.ndarray]:
    """Closed intersection polylines of the visible / clipped lattice.

    Marching squares on the alive mask. A crossing between two in-grid points
    is the clip hit (the short-stub end). A crossing against the grid frame
    uses the living lattice point, so an intact outline stays on the original
    vertices. Interior holes become their own loops.
    """
    grid = np.asarray(moved, dtype=float)
    mask = np.asarray(alive, dtype=bool)
    ny, nx = int(mask.shape[0]), int(mask.shape[1])
    if ny < 1 or nx < 1:
        return []

    def _inside(pr: int, pc: int) -> bool:
        r, c = pr - 1, pc - 1
        return 0 <= r < ny and 0 <= c < nx and bool(mask[r, c])

    def _xyz(pr: int, pc: int) -> np.ndarray | None:
        r, c = pr - 1, pc - 1
        if not (0 <= r < ny and 0 <= c < nx):
            return None
        return grid[r, c]

    def _crossing(a: tuple[int, int], b: tuple[int, int]) -> np.ndarray | None:
        in_a, in_b = _inside(*a), _inside(*b)
        if in_a == in_b:
            return None
        pa, pb = _xyz(*a), _xyz(*b)
        if pa is None and pb is None:
            return None
        if pa is None:
            return np.asarray(pb, dtype=float)
        if pb is None:
            return np.asarray(pa, dtype=float)
        if in_a:
            return _segment_exit_point(pa, pb, z_lim=z_lim, exit_fn=exit_fn)
        return _segment_exit_point(pb, pa, z_lim=z_lim, exit_fn=exit_fn)

    # Quad SW corner in padded coords: (i, j), i = row+1, j = col+1
    edge_nodes = (
        ((0, 0), (0, 1)),
        ((0, 1), (1, 1)),
        ((1, 1), (1, 0)),
        ((1, 0), (0, 0)),
    )
    segments: list[np.ndarray] = []
    for i in range(ny + 1):
        for j in range(nx + 1):
            corners = ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j))
            case = 0
            if _inside(*corners[0]):
                case |= 1
            if _inside(*corners[1]):
                case |= 2
            if _inside(*corners[2]):
                case |= 4
            if _inside(*corners[3]):
                case |= 8
            pairs = _MS_PAIRS.get(case)
            if not pairs:
                continue
            hits: list[np.ndarray | None] = []
            for (da, db) in edge_nodes:
                hits.append(
                    _crossing(
                        (corners[0][0] + da[0], corners[0][1] + da[1]),
                        (corners[0][0] + db[0], corners[0][1] + db[1]),
                    )
                )
            chosen = pairs
            if case in {5, 10} and all(h is not None for h in hits):
                alt = ((3, 0), (1, 2)) if case == 5 else ((0, 1), (2, 3))
                def _plen(ps: tuple[tuple[int, int], ...]) -> float:
                    total = 0.0
                    for e0, e1 in ps:
                        total += float(np.linalg.norm(hits[e0] - hits[e1]))
                    return total
                if _plen(alt) < _plen(pairs):
                    chosen = alt
            for e0, e1 in chosen:
                p0, p1 = hits[e0], hits[e1]
                if p0 is None or p1 is None:
                    continue
                if float(np.linalg.norm(p0 - p1)) < 1e-12:
                    continue
                segments.append(np.vstack([p0, p1]))

    loops = join_segments_to_polylines(segments)
    closed: list[np.ndarray] = []
    for loop in loops:
        if len(loop) < 3:
            continue
        closed.append(_close_xyz_loop(loop))
    return closed


def tracked_boundary_loops(
    neutral_points: np.ndarray,
    moved_points: np.ndarray,
    nx: int,
    ny: int,
    *,
    outline_rings: list[np.ndarray] | None = None,
    region: object | None = None,
    alive: np.ndarray | None = None,
    z_lim: float | None = None,
    exit_fn: Callable[[np.ndarray, np.ndarray], np.ndarray | None] | None = None,
) -> list[np.ndarray]:
    """Closed boundary plus every viewer-clip intersection loop.

    Intact outline edges keep the original lattice points. Where the viewer
    cuts a grid edge, the polyline connects those clip hits (the short-stub
    ends) instead of stair-stepping through lattice cells. Interior cuts are
    their own closed gold loops.
    """
    moved = np.asarray(moved_points, dtype=float).reshape(int(ny), int(nx), 3)
    if alive is None:
        alive = alive_lattice_mask(moved_points, nx, ny)
    else:
        alive = np.asarray(alive, dtype=bool).reshape(int(ny), int(nx))

    loops = clip_interface_loops(moved, alive, z_lim=z_lim, exit_fn=exit_fn)
    if not loops:
        return []

    if outline_rings is None and region is None:
        order = _rectangular_index_loop(int(nx), int(ny))
        start = next((rc for rc in order if alive[rc[0], rc[1]]), None)
        if start is not None:
            toward = order[(order.index(start) + 1) % len(order)]
            loops[0] = _align_loop_to_start(loops[0], moved[start], moved[toward])
        return loops

    return loops


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
      - ``> 0`` — **keep only** first N and last N (the middle is dropped)
      - ``< 0`` — **remove** first |N| and last |N| from the strided set
      - ``0`` — stride only
    """
    m = int(count)
    if m <= 0:
        return []
    stride_i = max(1, int(stride))
    b = int(boundary)
    if b > 0:
        keep: set[int] = set()
        for i in range(min(b, m)):
            keep.add(i)
            keep.add(m - 1 - i)
        return sorted(keep)
    keep = set(range(0, m, stride_i))
    if b < 0:
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


def split_nan_polyline(polyline: np.ndarray) -> list[np.ndarray]:
    """Split a NaN-separated polyline into finite segments."""
    pts = np.asarray(polyline, dtype=float)
    if len(pts) == 0:
        return []
    segments: list[np.ndarray] = []
    start = 0
    finite = np.isfinite(pts).all(axis=1)
    for i in range(len(pts) + 1):
        at_break = i == len(pts) or not finite[i]
        if at_break:
            chunk = pts[start:i]
            if len(chunk) >= 2:
                segments.append(chunk.copy())
            start = i + 1
    return segments


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
    alive: np.ndarray | None = None,
    max_edge: float | None = None,
) -> tuple[np.ndarray, pv.PolyData]:
    """Reconnect displaced grid points into line geometry.

    Patterns
    --------
    ``grid``
        Parallel X-row and/or Y-column lines (default). Toggle with
        ``lines_x`` / ``lines_y``. ``line_stride`` keeps every N-th line.
        ``boundary_lines_x`` / ``boundary_lines_y`` are signed: positive keeps
        only first/last N (drops the middle); negative removes first/last |N|
        from the strided set. Legacy ``boundary_lines`` applies both axes.
        The shape outline is a separate naked-boundary curve, not these lines.
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
        if alive is not None or max_edge is not None:
            mask = (
                None
                if alive is None
                else np.asarray(alive, dtype=bool).reshape(ny, nx)
            )
            split: list[np.ndarray] = []
            kept_x = stride_indices_with_boundary(n_x, stride, bx) if lines_x else []
            kept_y = stride_indices_with_boundary(n_y, stride, by) if lines_y else []
            grid = pts.reshape(ny, nx, 3)
            for row in kept_x:
                flags = None if mask is None else mask[row, :]
                split.extend(
                    split_polyline_edges(grid[row, :, :], alive_along=flags, max_edge=max_edge)
                )
            for col in kept_y:
                flags = None if mask is None else mask[:, col]
                split.extend(
                    split_polyline_edges(grid[:, col, :], alive_along=flags, max_edge=max_edge)
                )
            segments = split
        if not segments:
            return np.zeros((0, 3), dtype=float), pv.PolyData()
        polyline = segments_to_nan_polyline(segments)
        mesh = segments_to_polydata(segments)
        return polyline, mesh

    polyline = ordered_polyline_points(pts, nx, ny, pattern=pattern)
    mesh = polyline_to_polydata(polyline)
    return polyline, mesh
