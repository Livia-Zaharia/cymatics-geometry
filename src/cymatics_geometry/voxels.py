"""Voxel-modulated piping of cymatics grid lines into printable solids.

Uses **PicoPie** (Python bindings for LEAP 71 / PicoGK OpenVDB voxels) to sweep
solid rods or hollow pipes along each polyline spine, optionally with surface
modulation, then mesh + export STL (same pattern as enhancement-geometry).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh

from cymatics_geometry.crop import crop_line_segments
from cymatics_geometry.custom_shape import load_and_place_shape
from cymatics_geometry.lines import (
    grid_line_segments,
    select_segments_strided,
    split_nan_polyline,
    stride_indices_with_boundary,
)

logger = logging.getLogger(__name__)

# Practical notes shown in the notebook UI (keep in sync with field docs below).
VOXEL_PARAM_HELP: dict[str, str] = {
    "voxel_size": (
        "Edge length of one voxel in world units (same as your grid). "
        "Smaller → smoother surface and sharper modulation, but much slower and "
        "heavier STL. Start ~0.5–1.0 for previews; drop to 0.2–0.4 for print."
    ),
    "pipe_radius": (
        "Outer radius of the solid tube swept along each line. "
        "Must be large enough for your printer nozzle / wall thickness "
        "(e.g. ≥ 0.6–1.0 mm after you scale units)."
    ),
    "inner_radius": (
        "Hollow bore radius. 0 = solid rod (fastest, strongest). "
        "Set below pipe_radius for a true pipe; keep wall ≥ ~2× voxel_size."
    ),
    "modulation_amp": (
        "How far the outer radius ripples in/out along the spine (world units). "
        "0 = smooth constant tube. Larger values carve decorative waves into the "
        "surface (voxel-modulated geometry)."
    ),
    "modulation_freq": (
        "Number of full radius waves along each line’s length. "
        "Higher → tighter corrugation along the pipe."
    ),
    "modulation_lobes": (
        "Angular flutes around the circumference (0 = none). "
        "E.g. 6 ≈ hexagonal cross-section ripple."
    ),
    "line_stride": (
        "Keep every N-th grid line (rows and columns). "
        "1 = all lines; 2 = every other → fewer tubes, faster, less dense lattice."
    ),
    "boundary_lines_x": (
        "X-rows end treatment (signed): >0 keep only first/last N (drops the "
        "middle); <0 remove first/last |N|, 0 = stride only."
    ),
    "boundary_lines_y": (
        "Y-cols end treatment (signed): >0 keep only first/last N (drops the "
        "middle); <0 remove first/last |N|, 0 = stride only."
    ),
    "boundary_lines": (
        "Legacy single end treatment applied to both axes when per-axis fields "
        "are absent. Prefer boundary_lines_x / boundary_lines_y."
    ),
    "point_stride": (
        "Keep every N-th sample along each polyline before building the spine. "
        "Higher → coarser curve following, much faster voxelization."
    ),
    "spine_samples": (
        "Cubic-spline arc-length samples for each polyline spine (preview + STL). "
        "More samples → smoother bends; cost grows with sample count × lines."
    ),
    "spine_smooth": (
        "Line-smoothing strength before piping (0 = interpolate through samples, "
        "higher → rounder bends that ease sharp kinks from the grid). "
        "Applies to preview tubes and PicoPie spines."
    ),
}


@dataclass(frozen=True)
class VoxelPipeConfig:
    """Parameters for piping grid lines into a voxel solid.

    All lengths are in the same world units as the cymatics grid / shape map.
    Polyline spines are cubic-spline resampled to ``spine_samples`` before
    PicoPie frames / tube preview (linear fallback for short segments).
    """

    voxel_size: float = 0.8
    pipe_radius: float = 1.2
    inner_radius: float = 0.0
    modulation_amp: float = 0.0
    modulation_freq: float = 2.0
    modulation_lobes: int = 0
    line_stride: int = 2
    # Signed end treatment per direction (>0 keep only ends, <0 remove, 0 = stride)
    boundary_lines_x: int = 0
    boundary_lines_y: int = 0
    # Legacy alias (from_dict maps onto both axes when new fields absent)
    boundary_lines: int = 0
    point_stride: int = 2
    spine_samples: int = 40
    # Relative smoothing for cubic-spline spines (0 = exact interpolation)
    spine_smooth: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> VoxelPipeConfig:
        """Build from a JSON-like mapping (unknown keys ignored)."""
        known = {f.name: f for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for name, field_info in known.items():
            if name not in raw:
                continue
            value = raw[name]
            default = field_info.default
            if isinstance(default, bool):
                kwargs[name] = bool(value)
            elif isinstance(default, int) and not isinstance(default, bool):
                kwargs[name] = int(value)
            elif isinstance(default, float):
                kwargs[name] = float(value)
            else:
                kwargs[name] = value
        if "boundary_lines" in raw:
            legacy = int(raw["boundary_lines"])
            if "boundary_lines_x" not in raw:
                kwargs["boundary_lines_x"] = legacy
            if "boundary_lines_y" not in raw:
                kwargs["boundary_lines_y"] = legacy
        return cls(**kwargs)

    def __post_init__(self) -> None:
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be > 0")
        if self.pipe_radius <= 0:
            raise ValueError("pipe_radius must be > 0")
        if self.inner_radius < 0:
            raise ValueError("inner_radius must be ≥ 0")
        if self.inner_radius >= self.pipe_radius:
            raise ValueError("inner_radius must be < pipe_radius")
        if self.modulation_amp < 0:
            raise ValueError("modulation_amp must be ≥ 0")
        if self.modulation_freq < 0:
            raise ValueError("modulation_freq must be ≥ 0")
        if self.modulation_lobes < 0:
            raise ValueError("modulation_lobes must be ≥ 0")
        if self.line_stride < 1:
            raise ValueError("line_stride must be ≥ 1")
        if self.point_stride < 1:
            raise ValueError("point_stride must be ≥ 1")
        if self.spine_samples < 4:
            raise ValueError("spine_samples must be ≥ 4")
        if self.spine_smooth < 0:
            raise ValueError("spine_smooth must be ≥ 0")


@dataclass
class VoxelPipeResult:
    """Meshed solid produced by :func:`pipe_lines_to_voxels`."""

    config: VoxelPipeConfig
    trimesh_result: trimesh.Trimesh
    volume: float
    segment_count: int
    stats: dict[str, Any]

    @property
    def is_watertight(self) -> bool:
        return bool(self.trimesh_result.is_watertight)

    @property
    def is_valid_volume(self) -> bool:
        mesh = self.trimesh_result
        return bool(mesh.is_watertight and mesh.is_volume and abs(float(mesh.volume)) > 0.0)


# Back-compat aliases (selection lives in lines.py — used by display + voxels)
_stride_indices_with_boundary = stride_indices_with_boundary
_select_segments_strided = select_segments_strided


def polyline_segments_from_result(
    result: Any,
    *,
    line_stride: int = 1,
    point_stride: int = 1,
    boundary_lines_x: int = 0,
    boundary_lines_y: int = 0,
    boundary_lines: int | None = None,
) -> list[np.ndarray]:
    """Extract subsampled polylines from a :class:`PipelineResult`.

    Prefers the same grid topology as stage 5 (X-rows / Y-cols). Falls back to
    splitting ``result.polyline`` on NaN breaks.

    ``boundary_lines_x`` / ``boundary_lines_y`` are signed (keep only ends vs
    remove ends). Legacy ``boundary_lines`` applies the same value to both axes.
    """
    stride_line = max(1, int(line_stride))
    stride_pt = max(1, int(point_stride))
    if boundary_lines is not None:
        bx = by = int(boundary_lines)
    else:
        bx = int(boundary_lines_x)
        by = int(boundary_lines_y)
    cfg = result.config
    nx = int(cfg.grid_size_x)
    ny = int(cfg.grid_size_y)
    pts = np.asarray(result.shape_points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] != nx * ny:
        pts = np.asarray(result.displaced_points, dtype=float)

    if cfg.line_pattern == "grid" and pts.shape[0] == nx * ny:
        raw = grid_line_segments(
            pts,
            nx,
            ny,
            lines_x=bool(cfg.lines_x),
            lines_y=bool(cfg.lines_y),
        )
        selected: list[np.ndarray] = []
        n_x = ny if cfg.lines_x else 0
        n_y = nx if cfg.lines_y else 0
        x_segs = raw[:n_x]
        y_segs = raw[n_x : n_x + n_y]
        selected.extend(
            select_segments_strided(x_segs, stride=stride_line, boundary=bx)
        )
        selected.extend(
            select_segments_strided(y_segs, stride=stride_line, boundary=by)
        )
    else:
        selected = split_nan_polyline(np.asarray(result.polyline, dtype=float))
        selected = select_segments_strided(
            selected, stride=stride_line, boundary=bx
        )

    region = None
    if str(getattr(cfg, "shape", "")).lower() == "custom" and str(
        getattr(cfg, "custom_shape_path", "")
    ).strip():
        placed = load_and_place_shape(
            str(cfg.custom_shape_path),
            size=float(cfg.custom_shape_size),
        )
        region = placed.region()
    selected = crop_line_segments(selected, cfg, region=region)
    if bool(getattr(cfg, "boundary_curve", True)):
        selected.extend(
            split_nan_polyline(np.asarray(getattr(result, "boundary_polyline", []), dtype=float))
        )

    out: list[np.ndarray] = []
    for seg in selected:
        s = np.asarray(seg, dtype=float)
        finite = np.isfinite(s).all(axis=1)
        s = s[finite]
        if len(s) < 2:
            continue
        s = s[::stride_pt]
        if len(s) < 2:
            s = np.asarray(seg, dtype=float)
            s = s[np.isfinite(s).all(axis=1)]
            if len(s) < 2:
                continue
            s = np.vstack([s[0], s[-1]])
        # Drop zero-length consecutive duplicates
        diffs = np.linalg.norm(np.diff(s, axis=0), axis=1)
        keep = np.ones(len(s), dtype=bool)
        keep[1:] = diffs > 1e-9
        s = s[keep]
        if len(s) >= 2:
            out.append(s)
    return out


def _split_nan_polyline(polyline: np.ndarray) -> list[np.ndarray]:
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


def _preview_radial_segments(config: VoxelPipeConfig) -> int:
    """Facet count around the tube — larger voxels → coarser preview rings."""
    radius = max(float(config.pipe_radius), 1e-6)
    voxel = max(float(config.voxel_size), 1e-6)
    # Circumference / voxel_size ≈ how many facets OpenVDB can resolve
    n = int(round(2.0 * np.pi * radius / voxel))
    return int(np.clip(n, 4, 48))


def _preview_spine_samples(config: VoxelPipeConfig, segment: np.ndarray) -> int:
    """Spine density for preview — denser when voxel_size is smaller."""
    pts = np.asarray(segment, dtype=float)
    if len(pts) < 2:
        return max(4, int(config.spine_samples))
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    voxel = max(float(config.voxel_size), 1e-6)
    from_voxel = int(round(length / voxel)) + 1
    # Honour user spine_samples as an upper preference, but never starve small voxels
    target = max(int(config.spine_samples), from_voxel)
    return int(np.clip(target, 8, 200))


def preview_pipe_mesh(
    result: Any,
    config: VoxelPipeConfig | None = None,
    *,
    radial_segments: int | None = None,
) -> trimesh.Trimesh:
    """Fast approximate tube mesh for interactive preview (no PicoPie / voxels).

    Lofts a continuous circle along cubic-spline-smoothed spines. ``voxel_size``
    drives preview facet density (smaller → rounder rings / denser spines) so the
    voxel slider visibly changes the overlay. Use :func:`pipe_lines_to_voxels`
    for the printable solid.
    """
    cfg = config or VoxelPipeConfig()
    segments = polyline_segments_from_result(
        result,
        line_stride=cfg.line_stride,
        point_stride=cfg.point_stride,
        boundary_lines_x=cfg.boundary_lines_x,
        boundary_lines_y=cfg.boundary_lines_y,
    )
    if not segments:
        raise ValueError("No polyline segments to preview.")

    parts: list[trimesh.Trimesh] = []
    radius = float(cfg.pipe_radius)
    rings = (
        int(radial_segments)
        if radial_segments is not None
        else _preview_radial_segments(cfg)
    )
    for seg in segments:
        n_spine = _preview_spine_samples(cfg, seg)
        spine = _smooth_resample_polyline(
            seg,
            n_spine,
            smooth=float(cfg.spine_smooth),
        )
        tube = _lofted_tube_mesh(spine, radius, radial_segments=rings)
        if tube is not None and len(tube.faces) > 0:
            parts.append(tube)

    if not parts:
        raise ValueError("Tube preview produced an empty mesh.")
    if len(parts) == 1:
        return parts[0]
    return trimesh.util.concatenate(parts)


def _parallel_transport_frames(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (T, N, B) frames along a polyline via parallel transport."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    tangents = np.zeros((n, 3), dtype=float)
    if n >= 2:
        diffs = np.diff(pts, axis=0)
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        unit = diffs / np.maximum(norms, 1e-12)
        tangents[0] = unit[0]
        tangents[-1] = unit[-1]
        if n > 2:
            tangents[1:-1] = unit[:-1] + unit[1:]
            t_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
            tangents = tangents / np.maximum(t_norms, 1e-12)

    # Seed a normal perpendicular to the first tangent
    t0 = tangents[0]
    helper = np.array([0.0, 0.0, 1.0]) if abs(t0[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n0 = np.cross(t0, helper)
    n0_norm = float(np.linalg.norm(n0))
    if n0_norm < 1e-12:
        helper = np.array([0.0, 1.0, 0.0])
        n0 = np.cross(t0, helper)
        n0_norm = float(np.linalg.norm(n0))
    n0 /= max(n0_norm, 1e-12)
    normals = np.zeros((n, 3), dtype=float)
    binormals = np.zeros((n, 3), dtype=float)
    normals[0] = n0
    binormals[0] = np.cross(t0, n0)

    for i in range(1, n):
        t_prev = tangents[i - 1]
        t_cur = tangents[i]
        # Rotate previous normal by the same rotation that maps t_prev → t_cur
        axis = np.cross(t_prev, t_cur)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-12:
            normals[i] = normals[i - 1]
        else:
            axis = axis / axis_norm
            cos_a = float(np.clip(np.dot(t_prev, t_cur), -1.0, 1.0))
            sin_a = axis_norm  # |t_prev × t_cur| = sinθ when both unit
            # Rodrigues on previous normal
            n_prev = normals[i - 1]
            normals[i] = (
                n_prev * cos_a
                + np.cross(axis, n_prev) * sin_a
                + axis * np.dot(axis, n_prev) * (1.0 - cos_a)
            )
            n_norm = float(np.linalg.norm(normals[i]))
            normals[i] /= max(n_norm, 1e-12)
        # Re-orthogonalize against current tangent
        normals[i] = normals[i] - np.dot(normals[i], t_cur) * t_cur
        n_norm = float(np.linalg.norm(normals[i]))
        normals[i] /= max(n_norm, 1e-12)
        binormals[i] = np.cross(t_cur, normals[i])
    return tangents, normals, binormals


def _lofted_tube_mesh(
    points: np.ndarray,
    radius: float,
    *,
    radial_segments: int = 12,
) -> trimesh.Trimesh | None:
    """Continuous tube by lofting circles along parallel-transport frames.

    Avoids the faceted look of stacked short cylinders at every spine sample.
    """
    pts = _dedupe_polyline(np.asarray(points, dtype=float))
    if len(pts) < 2 or radius <= 0:
        return None
    rings = max(4, int(radial_segments))
    _tangents, normals, binormals = _parallel_transport_frames(pts)
    angles = np.linspace(0.0, 2.0 * np.pi, rings, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    ring_verts: list[np.ndarray] = []
    for i, p in enumerate(pts):
        ring = (
            p[None, :]
            + radius
            * (
                cos_a[:, None] * normals[i][None, :]
                + sin_a[:, None] * binormals[i][None, :]
            )
        )
        ring_verts.append(ring)
    verts = np.vstack(ring_verts)

    faces: list[list[int]] = []
    for i in range(len(pts) - 1):
        base_a = i * rings
        base_b = (i + 1) * rings
        for j in range(rings):
            j2 = (j + 1) % rings
            a0, a1 = base_a + j, base_a + j2
            b0, b1 = base_b + j, base_b + j2
            faces.append([a0, b0, b1])
            faces.append([a0, b1, a1])

    # Cap ends with fans to the spine endpoints
    cap_centers = [pts[0], pts[-1]]
    for end_i, center in enumerate(cap_centers):
        c_idx = len(verts) + end_i
        ring_base = 0 if end_i == 0 else (len(pts) - 1) * rings
        for j in range(rings):
            j2 = (j + 1) % rings
            if end_i == 0:
                faces.append([c_idx, ring_base + j2, ring_base + j])
            else:
                faces.append([c_idx, ring_base + j, ring_base + j2])
    verts = np.vstack([verts, np.asarray(cap_centers, dtype=float)])

    return trimesh.Trimesh(
        vertices=verts,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def _polyline_tube_mesh(
    points: np.ndarray,
    radius: float,
    *,
    radial_segments: int = 8,
) -> trimesh.Trimesh | None:
    """Legacy stacked-cylinder tube; prefer :func:`_lofted_tube_mesh`."""
    return _lofted_tube_mesh(points, radius, radial_segments=radial_segments)


def _resample_polyline(points: np.ndarray, n_samples: int) -> np.ndarray:
    """Linear arc-length resample a polyline to ``n_samples`` points."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total < 1e-12:
        return np.vstack([pts[0], pts[-1]])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, int(n_samples))
    out = np.empty((int(n_samples), 3), dtype=float)
    for i, t in enumerate(targets):
        j = int(np.searchsorted(cum, t, side="right") - 1)
        j = max(0, min(j, len(pts) - 2))
        span = cum[j + 1] - cum[j]
        alpha = 0.0 if span < 1e-12 else (t - cum[j]) / span
        out[i] = (1.0 - alpha) * pts[j] + alpha * pts[j + 1]
    return out


def _dedupe_polyline(points: np.ndarray, *, eps: float = 1e-9) -> np.ndarray:
    """Drop consecutive near-duplicate vertices."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts
    keep = [0]
    for i in range(1, len(pts)):
        if float(np.linalg.norm(pts[i] - pts[keep[-1]])) > eps:
            keep.append(i)
    return pts[keep]


def _smooth_resample_polyline(
    points: np.ndarray,
    n_samples: int,
    *,
    smooth: float = 0.0,
) -> np.ndarray:
    """Cubic-spline arc-length resample; falls back to linear if needed.

    ``smooth`` ≥ 0 is a relative fit slack for ``splprep`` (scaled by point
    count and segment length). 0 interpolates through samples; larger values
    round sharp kinks from a coarse grid before piping.
    """
    from scipy.interpolate import splev, splprep

    n = max(2, int(n_samples))
    pts = _dedupe_polyline(np.asarray(points, dtype=float))
    if len(pts) < 2:
        return pts
    if len(pts) < 4:
        return _resample_polyline(pts, n)

    # Scale smoothing with path length so the control stays intuitive in world units
    seg_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    s_val = max(0.0, float(smooth)) * max(seg_len, 1e-6) * 0.02
    try:
        tck, _u = splprep(
            [pts[:, 0], pts[:, 1], pts[:, 2]],
            s=s_val,
            k=min(3, len(pts) - 1),
        )
        u_new = np.linspace(0.0, 1.0, n)
        x, y, z = splev(u_new, tck)
        return np.column_stack([x, y, z])
    except (TypeError, ValueError):
        return _resample_polyline(pts, n)


def _radius_modulation(
    config: VoxelPipeConfig,
) -> float | Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Build a PicoPie surface modulation for the outer radius."""
    base = float(config.pipe_radius)
    amp = float(config.modulation_amp)
    freq = float(config.modulation_freq)
    lobes = int(config.modulation_lobes)
    if amp <= 0.0 and lobes <= 0:
        return base

    def radius(phi: np.ndarray, lr: np.ndarray) -> np.ndarray:
        r = np.full_like(np.asarray(lr, dtype=float), base, dtype=float)
        if amp > 0.0 and freq > 0.0:
            r = r + amp * np.sin(2.0 * np.pi * freq * np.asarray(lr, dtype=float))
        if lobes > 0 and amp > 0.0:
            r = r + 0.5 * amp * np.cos(float(lobes) * np.asarray(phi, dtype=float))
        elif lobes > 0:
            # Angular-only fluting when amp is 0 but lobes requested
            r = r + 0.15 * base * np.cos(float(lobes) * np.asarray(phi, dtype=float))
        return np.maximum(r, 0.05 * base)

    return radius


_ACTIVE_VOXEL_SIZE: float | None = None


def _init_picopie(voxel_size: float) -> None:
    """Initialize PicoGK, re-init when voxel size changes (process-global)."""
    import picopie

    global _ACTIVE_VOXEL_SIZE
    size = float(voxel_size)
    if picopie.is_initialized():
        if _ACTIVE_VOXEL_SIZE is not None and abs(_ACTIVE_VOXEL_SIZE - size) < 1e-12:
            return
        picopie.shutdown()
    picopie.init(voxel_size_mm=size)
    _ACTIVE_VOXEL_SIZE = size


def _segment_to_voxels(segment: np.ndarray, config: VoxelPipeConfig) -> Any:
    from picopie.shapes import Cylinder, Frames, Pipe

    # Denser spines when voxels are small so the solid follows the smoothed curve
    pts = np.asarray(segment, dtype=float)
    length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) >= 2 else 0.0
    voxel = max(float(config.voxel_size), 1e-6)
    n_from_voxel = int(round(length / voxel)) + 1
    n_spine = max(int(config.spine_samples), n_from_voxel, 8)
    n_spine = min(n_spine, 400)
    spine_pts = _smooth_resample_polyline(
        segment,
        n_spine,
        smooth=float(config.spine_smooth),
    )
    frames = Frames.aligned(spine_pts, "min_rotation")
    outer = _radius_modulation(config)
    if float(config.inner_radius) > 0.0:
        return Pipe(
            frames=frames,
            inner_radius=float(config.inner_radius),
            outer_radius=outer,
        ).to_voxels()
    return Cylinder(frames=frames, radius=outer).to_voxels()


def _picopie_mesh_to_trimesh(mesh: Any) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if not tm.is_watertight:
        trimesh.repair.fix_normals(tm)
        trimesh.repair.fill_holes(tm)
    tm.remove_unreferenced_vertices()
    return tm


def pipe_lines_to_voxels(
    result: Any,
    config: VoxelPipeConfig | None = None,
    *,
    verbose: bool = True,
) -> VoxelPipeResult:
    """Sweep voxel pipes along the stage-5 grid lines and return a solid mesh.

    Parameters
    ----------
    result:
        Output of :func:`cymatics_geometry.pipeline.run_pipeline`.
    config:
        Voxel / pipe / modulation parameters. Defaults are preview-friendly.
    """
    cfg = config or VoxelPipeConfig()
    segments = polyline_segments_from_result(
        result,
        line_stride=cfg.line_stride,
        point_stride=cfg.point_stride,
        boundary_lines_x=cfg.boundary_lines_x,
        boundary_lines_y=cfg.boundary_lines_y,
    )
    if not segments:
        raise ValueError(
            "No polyline segments to pipe — enable lines, boundary, or check the grid."
        )

    if verbose:
        print(
            f"Voxel pipe: {len(segments)} segments, "
            f"voxel_size={cfg.voxel_size}, radius={cfg.pipe_radius}, "
            f"inner={cfg.inner_radius}, mod_amp={cfg.modulation_amp}"
        )

    _init_picopie(cfg.voxel_size)
    combined = None
    for i, seg in enumerate(segments):
        part = _segment_to_voxels(seg, cfg)
        combined = part if combined is None else (combined + part)
        if verbose and (i + 1) % 10 == 0:
            print(f"  … unioned {i + 1}/{len(segments)} segments")

    assert combined is not None
    vol, _bbox = combined.calculate_properties()
    picopie_mesh = combined.to_mesh()
    tm = _picopie_mesh_to_trimesh(picopie_mesh)
    stats = {
        "segment_count": len(segments),
        "voxel_size": float(cfg.voxel_size),
        "pipe_radius": float(cfg.pipe_radius),
        "inner_radius": float(cfg.inner_radius),
        "modulation_amp": float(cfg.modulation_amp),
        "modulation_freq": float(cfg.modulation_freq),
        "modulation_lobes": int(cfg.modulation_lobes),
        "line_stride": int(cfg.line_stride),
        "boundary_lines_x": int(cfg.boundary_lines_x),
        "boundary_lines_y": int(cfg.boundary_lines_y),
        "point_stride": int(cfg.point_stride),
        "spine_samples": int(cfg.spine_samples),
        "spine_smooth": float(cfg.spine_smooth),
        "volume_voxels": float(vol),
        "volume_mesh": float(tm.volume) if tm.is_volume else 0.0,
        "faces": int(len(tm.faces)),
        "vertices": int(len(tm.vertices)),
        "watertight": bool(tm.is_watertight),
        "is_volume": bool(tm.is_volume),
        "config": asdict(cfg),
    }
    if verbose:
        print(
            f"Voxel solid: faces={stats['faces']}, "
            f"vol~={stats['volume_voxels']:.2f}, watertight={stats['watertight']}"
        )
    return VoxelPipeResult(
        config=cfg,
        trimesh_result=tm,
        volume=float(vol),
        segment_count=len(segments),
        stats=stats,
    )


def export_stl(
    result: VoxelPipeResult | trimesh.Trimesh,
    export_dir: str | Path,
    *,
    suffix: str = "",
    prefix: str = "cymatics_pipe",
) -> Path:
    """Write a watertight-ish mesh to a timestamped STL (enhancement-geometry style)."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = export_dir / f"{prefix}_{ts}{suffix}.stl"
    mesh = result.trimesh_result if isinstance(result, VoxelPipeResult) else result
    mesh.export(str(path))
    return path


def pipe_and_export_stl(
    pipeline_result: Any,
    export_dir: str | Path = "exports",
    config: VoxelPipeConfig | None = None,
    *,
    suffix: str = "",
    verbose: bool = True,
) -> tuple[VoxelPipeResult, Path]:
    """Convenience: voxel-pipe lines from a pipeline run and write STL."""
    solid = pipe_lines_to_voxels(pipeline_result, config, verbose=verbose)
    path = export_stl(solid, export_dir, suffix=suffix)
    if verbose:
        print(f"STL exported: {path}")
    return solid, path


def voxel_config_from_mapping(data: dict[str, Any]) -> VoxelPipeConfig:
    """Build :class:`VoxelPipeConfig` from a plain dict (ignores unknown keys)."""
    allowed = {f.name for f in fields(VoxelPipeConfig)}
    kwargs = {k: v for k, v in data.items() if k in allowed}
    return VoxelPipeConfig(**kwargs)
