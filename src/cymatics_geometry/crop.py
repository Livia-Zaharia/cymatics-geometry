"""Crop polylines and points to a 2D outline and/or an oriented section box.

The custom-shape crop is a 2D XY mask (scale-to-bbox, then drop anything
outside the imported silhouette). The section box is a 3D oriented cube that
clips the already-mapped geometry.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union

from cymatics_geometry.lines import (
    segments_to_nan_polyline,
    segments_to_polydata,
    split_nan_polyline,
)


@dataclass(frozen=True)
class SectionBox:
    """Axis-aligned cube in its own local frame, placed by center + XYZ Euler.

    Rotation is applied in degrees, intrinsic XYZ (rotate around local X, then
    Y, then Z). A point is inside when its local coordinates satisfy
    ``|p'| <= size/2`` on every axis.
    """

    size_x: float = 120.0
    size_y: float = 120.0
    size_z: float = 120.0
    center_x: float = 50.0
    center_y: float = 50.0
    center_z: float = 0.0
    rot_x: float = 0.0
    rot_y: float = 0.0
    rot_z: float = 0.0
    enabled: bool = False

    @property
    def size(self) -> np.ndarray:
        return np.array(
            [max(float(self.size_x), 1e-9), max(float(self.size_y), 1e-9), max(float(self.size_z), 1e-9)],
            dtype=float,
        )

    @property
    def center(self) -> np.ndarray:
        return np.array(
            [float(self.center_x), float(self.center_y), float(self.center_z)],
            dtype=float,
        )

    @property
    def rotation_matrix(self) -> np.ndarray:
        return rotation_matrix_xyz(self.rot_x, self.rot_y, self.rot_z)

    def half_extents(self) -> np.ndarray:
        return 0.5 * self.size


def rotation_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Intrinsic XYZ Euler rotation matrix (degrees)."""
    rx, ry, rz = np.deg2rad([float(rx_deg), float(ry_deg), float(rz_deg)])
    cx, sx = float(np.cos(rx)), float(np.sin(rx))
    cy, sy = float(np.cos(ry)), float(np.sin(ry))
    cz, sz = float(np.cos(rz)), float(np.sin(rz))
    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=float)
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=float)
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz_m @ ry_m @ rx_m


def world_to_box_local(points: np.ndarray, box: SectionBox) -> np.ndarray:
    """Map world XYZ into the section-box local frame."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=float)
    rot = box.rotation_matrix
    return (pts - box.center) @ rot


def box_local_to_world(local: np.ndarray, box: SectionBox) -> np.ndarray:
    """Map section-box local coordinates back to world XYZ."""
    loc = np.asarray(local, dtype=float)
    if loc.size == 0:
        return np.zeros((0, 3), dtype=float)
    return loc @ box.rotation_matrix.T + box.center


def section_box_corners(box: SectionBox) -> np.ndarray:
    """Eight corners of the section box in world coordinates, shape (8, 3)."""
    hx, hy, hz = box.half_extents()
    local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=float,
    )
    return box_local_to_world(local, box)


def section_box_wireframe(box: SectionBox) -> np.ndarray:
    """12 cube edges as a NaN-separated polyline (for Plotly / PyVista)."""
    corners = section_box_corners(box)
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    chunks: list[np.ndarray] = []
    nan = np.full((1, 3), np.nan, dtype=float)
    for i, (a, b) in enumerate(edges):
        chunks.append(np.vstack([corners[a], corners[b]]))
        if i + 1 < len(edges):
            chunks.append(nan)
    return np.vstack(chunks)


def points_inside_section_box(points: np.ndarray, box: SectionBox) -> np.ndarray:
    """Boolean mask: True when the point is inside or on the section box."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)
    local = world_to_box_local(pts, box)
    half = box.half_extents()
    return np.all(np.abs(local) <= half + 1e-9, axis=1)


def clip_segment_aabb(
    p0: np.ndarray,
    p1: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Liang–Barsky clip of a 3D segment against an axis-aligned box."""
    p0 = np.asarray(p0, dtype=float).reshape(3)
    p1 = np.asarray(p1, dtype=float).reshape(3)
    lo = np.asarray(lo, dtype=float).reshape(3)
    hi = np.asarray(hi, dtype=float).reshape(3)
    direction = p1 - p0
    t0 = 0.0
    t1 = 1.0
    for i in range(3):
        if abs(float(direction[i])) < 1e-15:
            if float(p0[i]) < float(lo[i]) - 1e-12 or float(p0[i]) > float(hi[i]) + 1e-12:
                return None
            continue
        inv = 1.0 / float(direction[i])
        t_near = (float(lo[i]) - float(p0[i])) * inv
        t_far = (float(hi[i]) - float(p0[i])) * inv
        if t_near > t_far:
            t_near, t_far = t_far, t_near
        t0 = max(t0, t_near)
        t1 = min(t1, t_far)
        if t0 > t1:
            return None
    return p0 + t0 * direction, p0 + t1 * direction


def clip_polyline_to_section_box(points: np.ndarray, box: SectionBox) -> list[np.ndarray]:
    """Clip one polyline to the oriented section box; may return several pieces."""
    pts = np.asarray(points, dtype=float)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) < 2:
        return []
    local = world_to_box_local(pts, box)
    half = box.half_extents()
    lo = -half
    hi = half
    pieces: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    for i in range(len(local) - 1):
        clipped = clip_segment_aabb(local[i], local[i + 1], lo, hi)
        if clipped is None:
            if len(current) >= 2:
                pieces.append(current)
            current = []
            continue
        a_loc, b_loc = clipped
        if not current:
            current = [a_loc, b_loc]
            continue
        if np.allclose(current[-1], a_loc, atol=1e-9):
            current.append(b_loc)
        else:
            if len(current) >= 2:
                pieces.append(current)
            current = [a_loc, b_loc]
    if len(current) >= 2:
        pieces.append(current)
    return [box_local_to_world(np.vstack(piece), box) for piece in pieces]


def polygons_from_rings(rings: list[np.ndarray]) -> Polygon | MultiPolygon | None:
    """Build a shapely (multi)polygon from closed XY rings."""
    polys: list[Polygon] = []
    for ring in rings:
        xy = np.asarray(ring, dtype=float)
        if xy.ndim != 2 or xy.shape[0] < 3:
            continue
        if xy.shape[1] > 2:
            xy = xy[:, :2]
        if not np.allclose(xy[0], xy[-1]):
            xy = np.vstack([xy, xy[0]])
        if len(xy) < 4:
            continue
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        if poly.geom_type == "Polygon" and poly.area > 1e-12:
            polys.append(poly)
        elif poly.geom_type == "MultiPolygon":
            polys.extend([g for g in poly.geoms if g.area > 1e-12])
    if not polys:
        return None
    merged = unary_union(polys)
    if merged.is_empty:
        return None
    return merged


def points_inside_polygons(points: np.ndarray, region: Polygon | MultiPolygon) -> np.ndarray:
    """Boolean mask for XY-inside (Z ignored). Boundary counts as inside."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)
    finite = np.isfinite(pts).all(axis=1)
    mask = np.zeros(len(pts), dtype=bool)
    if not np.any(finite):
        return mask
    xy = pts[finite][:, :2]
    from shapely import contains_xy

    inside = contains_xy(region, xy[:, 0], xy[:, 1])
    mask[finite] = np.asarray(inside, dtype=bool)
    return mask


def _line_geoms(geom: object) -> list[LineString]:
    if geom is None or getattr(geom, "is_empty", True):
        return []
    gtype = geom.geom_type
    if gtype == "LineString":
        return [geom]  # type: ignore[list-item]
    if gtype == "MultiLineString":
        return [g for g in geom.geoms if not g.is_empty]  # type: ignore[union-attr]
    if gtype == "GeometryCollection":
        out: list[LineString] = []
        for child in geom.geoms:  # type: ignore[union-attr]
            out.extend(_line_geoms(child))
        return out
    return []


def _lift_xy_to_3d(xy: np.ndarray, pts3: np.ndarray) -> np.ndarray:
    """Place 2D clip vertices back onto the original 3D polyline by arc length."""
    pts3 = np.asarray(pts3, dtype=float)
    xy_src = pts3[:, :2]
    seg = np.diff(xy_src, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(seg_len.sum())
    if total < 1e-12:
        z = float(pts3[0, 2]) if pts3.shape[1] > 2 else 0.0
        out = np.zeros((len(xy), 3), dtype=float)
        out[:, :2] = xy
        out[:, 2] = z
        return out
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    out = np.zeros((len(xy), 3), dtype=float)
    for i, q in enumerate(np.asarray(xy, dtype=float)):
        d = np.linalg.norm(xy_src - q.reshape(1, 2), axis=1)
        nearest = int(np.argmin(d))
        if d[nearest] < 1e-8:
            out[i] = pts3[nearest]
            continue
        # Distance along 2D polyline of the closest projection onto edges
        best_t = 0.0
        best_dist = float("inf")
        acc = 0.0
        for e in range(len(seg)):
            a = xy_src[e]
            b = xy_src[e + 1]
            ab = b - a
            ab2 = float(np.dot(ab, ab))
            if ab2 < 1e-18:
                acc += float(seg_len[e])
                continue
            t = float(np.clip(np.dot(q - a, ab) / ab2, 0.0, 1.0))
            proj = a + t * ab
            dist = float(np.linalg.norm(q - proj))
            if dist < best_dist:
                best_dist = dist
                best_t = acc + t * float(seg_len[e])
            acc += float(seg_len[e])
        # Interpolate 3D at the same 2D arc-length fraction
        target = best_t
        e = int(np.searchsorted(cum, target, side="right") - 1)
        e = int(np.clip(e, 0, len(seg) - 1))
        span = float(seg_len[e])
        local_t = 0.0 if span < 1e-12 else (target - float(cum[e])) / span
        local_t = float(np.clip(local_t, 0.0, 1.0))
        out[i] = pts3[e] + local_t * (pts3[e + 1] - pts3[e])
        out[i, :2] = q
    return out


def clip_polyline_to_polygons(
    points: np.ndarray,
    region: Polygon | MultiPolygon,
) -> list[np.ndarray]:
    """Clip a 3D polyline to a 2D XY region (Z interpolated at crossings)."""
    pts = np.asarray(points, dtype=float)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) < 2:
        return []
    line = LineString([(float(p[0]), float(p[1])) for p in pts])
    inter = line.intersection(region)
    pieces: list[np.ndarray] = []
    for geom in _line_geoms(inter):
        coords = np.asarray(geom.coords, dtype=float)
        if len(coords) < 2:
            continue
        pieces.append(_lift_xy_to_3d(coords[:, :2], pts))
    return pieces


def clip_segments(
    segments: list[np.ndarray],
    *,
    region: Polygon | MultiPolygon | None = None,
    box: SectionBox | None = None,
) -> list[np.ndarray]:
    """Clip a list of polylines by optional 2D region then optional section box."""
    out: list[np.ndarray] = []
    for seg in segments:
        pieces = [np.asarray(seg, dtype=float)]
        if region is not None:
            cropped: list[np.ndarray] = []
            for piece in pieces:
                cropped.extend(clip_polyline_to_polygons(piece, region))
            pieces = cropped
        if box is not None and box.enabled:
            boxed: list[np.ndarray] = []
            for piece in pieces:
                boxed.extend(clip_polyline_to_section_box(piece, box))
            pieces = boxed
        for piece in pieces:
            finite = np.isfinite(piece).all(axis=1)
            kept = piece[finite]
            if len(kept) >= 2:
                out.append(kept)
    return out


def rebuild_line_geometry(
    segments: list[np.ndarray],
) -> tuple[np.ndarray, object]:
    """NaN polyline + PyVista mesh from already-clipped segments."""
    return segments_to_nan_polyline(segments), segments_to_polydata(segments)


def crop_points_mask(
    points: np.ndarray,
    *,
    region: Polygon | MultiPolygon | None = None,
    box: SectionBox | None = None,
) -> np.ndarray:
    """Per-point inside mask after 2D crop and/or section box."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros(0, dtype=bool)
    mask = np.isfinite(pts).all(axis=1)
    if region is not None:
        mask = mask & points_inside_polygons(pts, region)
    if box is not None and box.enabled:
        mask = mask & points_inside_section_box(pts, box)
    return mask


def section_box_from_config(config: object) -> SectionBox:
    """Build a :class:`SectionBox` from a ``PipelineConfig`` (or similar object)."""
    return SectionBox(
        size_x=float(getattr(config, "section_box_size_x", 120.0)),
        size_y=float(getattr(config, "section_box_size_y", 120.0)),
        size_z=float(getattr(config, "section_box_size_z", 120.0)),
        center_x=float(getattr(config, "section_box_center_x", 50.0)),
        center_y=float(getattr(config, "section_box_center_y", 50.0)),
        center_z=float(getattr(config, "section_box_center_z", 0.0)),
        rot_x=float(getattr(config, "section_box_rot_x", 0.0)),
        rot_y=float(getattr(config, "section_box_rot_y", 0.0)),
        rot_z=float(getattr(config, "section_box_rot_z", 0.0)),
        enabled=bool(getattr(config, "section_box_enabled", False)),
    )


def crop_line_segments(
    segments: list[np.ndarray],
    config: object,
    *,
    region: Polygon | MultiPolygon | None = None,
) -> list[np.ndarray]:
    """Clip display/voxel segments using the config section box and optional 2D region."""
    box = section_box_from_config(config)
    use_box = box if box.enabled else None
    return clip_segments(segments, region=region, box=use_box)


def crop_polyline(
    polyline: np.ndarray,
    config: object,
    *,
    region: Polygon | MultiPolygon | None = None,
) -> list[np.ndarray]:
    """Split a NaN polyline, then clip each piece."""
    return crop_line_segments(split_nan_polyline(polyline), config, region=region)


def section_box_from_points(
    points: np.ndarray,
    *,
    pad: float = 1.0,
) -> SectionBox:
    """Axis-aligned section box fitted to point bounds (no rotation)."""
    pts = np.asarray(points, dtype=float)
    finite = pts[np.isfinite(pts).all(axis=1)]
    if len(finite) == 0:
        return SectionBox(enabled=True)
    lo = finite.min(axis=0)
    hi = finite.max(axis=0)
    size = np.maximum(hi - lo, 1e-6) + 2.0 * float(pad)
    center = 0.5 * (lo + hi)
    return SectionBox(
        size_x=float(size[0]),
        size_y=float(size[1]),
        size_z=float(max(size[2], 2.0 * float(pad))),
        center_x=float(center[0]),
        center_y=float(center[1]),
        center_z=float(center[2]),
        rot_x=0.0,
        rot_y=0.0,
        rot_z=0.0,
        enabled=True,
    )
