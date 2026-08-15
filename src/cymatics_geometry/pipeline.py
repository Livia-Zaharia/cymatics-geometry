"""End-to-end cymatics plane → line geometry pipeline.

Mirrors the enhancement-geometry style: config in → staged processing →
PipelineResult out, usable from notebook and CLI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np
import pyvista as pv

from cymatics_geometry.config import PipelineConfig
from cymatics_geometry.crop import (
    SectionBox,
    box_local_to_world,
    clip_segment_aabb,
    crop_line_segments,
    crop_points_mask,
    rebuild_line_geometry,
    section_box_from_config,
    section_box_wireframe,
    world_to_box_local,
)
from cymatics_geometry.custom_shape import PlacedShape2D, load_and_place_shape
from cymatics_geometry.grid import (
    SOURCE_LABELS,
    build_square_grid,
    grid_shape,
    source_positions,
)
from cymatics_geometry.lines import (
    build_line_geometry,
    polyline_length,
    segments_to_nan_polyline,
    split_nan_polyline,
    tracked_boundary_loops,
)
from cymatics_geometry.shapes import (
    SHAPE_KINDS,
    ShapeParams,
    map_points_to_shape,
    map_sources_to_shape,
)
from cymatics_geometry.waves import (
    distances_to_sources,
    displace_points,
    interference_field,
    release_xy_offsets,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything produced by a single pipeline run, stage by stage."""

    config: PipelineConfig
    # Stage 1 — flat square grid
    grid_points: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    # Stage 2 — wave sources (corners + mid-edges)
    sources: np.ndarray
    source_labels: tuple[str, ...] = SOURCE_LABELS
    # Backward-compatible aliases (corners = first four sources)
    corners: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    corner_labels: tuple[str, ...] = ("SW", "SE", "NE", "NW")
    # Stage 3 — interference field
    displacement: np.ndarray = field(default_factory=lambda: np.zeros(0))
    contributions: np.ndarray = field(default_factory=lambda: np.zeros((0, 8)))
    # Stage 4 — displaced points on the plane (Z + optional XY release)
    displaced_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    xy_offsets: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    # Stage 4b — same UV lattice + plane offsets mapped onto the target shape
    shape_base_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    shape_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    plane_offsets: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    shape_sources: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # Stage 5 — reconnected line (on the mapped shape, then cropped)
    polyline: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    line_mesh: pv.PolyData = field(default_factory=pv.PolyData)
    boundary_polyline: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    inside_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    custom_outline: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    section_box_wire: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    stats: dict = field(default_factory=dict)


def _placed_custom_shape(config: PipelineConfig) -> PlacedShape2D | None:
    if str(config.shape).lower().strip() != "custom":
        return None
    path = str(config.custom_shape_path).strip()
    if not path:
        raise ValueError("shape='custom' requires custom_shape_path (SVG, DXF, or DWG)")
    return load_and_place_shape(path, size=float(config.custom_shape_size))


def _shape_params_from_config(
    config: PipelineConfig,
    *,
    placed: PlacedShape2D | None = None,
) -> ShapeParams:
    kind = str(config.shape).lower().strip()
    if kind not in SHAPE_KINDS:
        allowed = ", ".join(SHAPE_KINDS)
        raise ValueError(f"Unknown shape {config.shape!r}; expected one of: {allowed}")
    bbox = (0.0, 0.0, float(config.side_length), float(config.side_length))
    if placed is not None:
        bbox = placed.bbox
    return ShapeParams(
        kind=kind,  # type: ignore[arg-type]
        cylinder_diameter=float(config.cylinder_diameter),
        cylinder_length=float(config.cylinder_length),
        cone_height=float(config.cone_height),
        cone_base_radius=float(config.cone_base_radius),
        frustum_height=float(config.frustum_height),
        frustum_base_diameter=float(config.frustum_base_diameter),
        frustum_top_diameter=float(config.frustum_top_diameter),
        variable_cylinder_radius_begin=float(config.variable_cylinder_radius_begin),
        variable_cylinder_radius_middle=float(config.variable_cylinder_radius_middle),
        variable_cylinder_radius_end=float(config.variable_cylinder_radius_end),
        variable_cylinder_length=float(config.variable_cylinder_length),
        variable_cylinder_middle=float(config.variable_cylinder_middle),
        bead_diameter=float(config.bead_diameter),
        bead_bottom_radius=float(config.bead_bottom_radius),
        bead_top_radius=float(config.bead_top_radius),
        custom_bbox_xmin=float(bbox[0]),
        custom_bbox_ymin=float(bbox[1]),
        custom_bbox_xmax=float(bbox[2]),
        custom_bbox_ymax=float(bbox[3]),
    )


def _section_box_exit(
    box: SectionBox,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray | None]:
    """Return the clip hit of an in→out segment on the section box face."""
    half = box.half_extents()

    def _exit(p_in: np.ndarray, p_out: np.ndarray) -> np.ndarray | None:
        a = world_to_box_local(np.asarray(p_in, dtype=float).reshape(1, 3), box)[0]
        b = world_to_box_local(np.asarray(p_out, dtype=float).reshape(1, 3), box)[0]
        clipped = clip_segment_aabb(a, b, -half, half)
        if clipped is None:
            return None
        return box_local_to_world(np.asarray(clipped[1], dtype=float).reshape(1, 3), box)[0]

    return _exit


def run_pipeline(
    config: PipelineConfig,
    *,
    verbose: bool = True,
) -> PipelineResult:
    """Execute the full cymatics plane → line pipeline.

    Stages
    ------
    1. Deploy points on an nx×ny grid
    2. Place wave sources at corners (+ optional mid-edges)
    3. Compute wave interference from active source amplitudes
    4. Displace points in Z; release unlocks XY as a connected surface
    5. Reconnect displaced points into a continuous line
    6. Optional: crop to a custom 2D silhouette and/or clip with a section box
    """
    nx, ny = grid_shape(config)
    if verbose:
        print(f"Grid: {nx}×{ny} (X×Y), side={config.side_length}")
        print(f"Active sources: {config.active_source_labels()}")
        print(
            "Amplitudes SW/SE/NE/NW/S/E/N/W: "
            + " / ".join(f"{a:.3f}" for a in config.amplitudes)
        )
        print(
            f"frequency={config.frequency:.3f}, "
            f"time={config.time:.3f}, decay={config.decay:.3f}"
        )
        print(f"line_pattern={config.line_pattern}")

    # Stage 1
    grid_points, xs, ys = build_square_grid(config)
    if verbose:
        print(f"Stage 1 — flat grid: {len(grid_points)} points")

    # Stage 2
    sources = source_positions(config.side_length)
    corners = sources[:4]
    if verbose:
        print(f"Stage 2 — sources: {list(SOURCE_LABELS)}")

    # Stage 3
    distances = distances_to_sources(grid_points, config.side_length)
    displacement, contributions = interference_field(grid_points, config)
    if verbose:
        print(
            "Stage 3 — interference field: "
            f"z∈[{float(displacement.min()):.4f}, {float(displacement.max()):.4f}]"
        )

    # Stage 4 — Z waves + radial XY shockwave from released sources (always on plane)
    xy_offsets = release_xy_offsets(
        grid_points,
        config,
        distances=distances,
        contributions=contributions,
    )
    displaced = displace_points(grid_points, displacement, xy_offsets=xy_offsets)
    n_released = int(np.count_nonzero(np.linalg.norm(xy_offsets, axis=1) > 1e-9))
    if verbose:
        print(
            f"Stage 4 — displaced points: {len(displaced)} "
            f"(xy released: {n_released})"
        )

    # Stage 4b — map plane UV + local offsets onto the selected shape
    placed = _placed_custom_shape(config)
    crop_region = placed.region() if placed is not None else None
    custom_outline = (
        placed.outline_polyline() if placed is not None else np.zeros((0, 3), dtype=float)
    )
    shape_params = _shape_params_from_config(config, placed=placed)
    shape_base, shape_pts, plane_offs = map_points_to_shape(
        grid_points,
        displaced,
        shape_params,
        side_length=float(config.side_length),
    )
    shape_sources = map_sources_to_shape(
        sources,
        shape_params,
        side_length=float(config.side_length),
    )
    if verbose:
        extra = ""
        if placed is not None:
            extra = (
                f" custom={placed.source_path.name} "
                f"size={config.custom_shape_size:g} "
                f"aspect={placed.aspect_ratio:.3f}"
            )
        print(f"Stage 4b — shape map: kind={shape_params.kind}{extra}")

    # Stage 5 — reconnect. Missing = outside the section box (when enabled)
    # or the custom 2D silhouette. The viewer Z axis is not a crop.
    box = section_box_from_config(config)
    alive = crop_points_mask(shape_pts, region=crop_region, box=box).reshape(ny, nx)
    polyline, line_mesh = build_line_geometry(
        shape_pts,
        nx,
        ny,
        pattern=config.line_pattern,
        lines_x=bool(config.lines_x),
        lines_y=bool(config.lines_y),
        line_stride=int(config.line_stride),
        boundary_lines_x=int(config.boundary_lines_x),
        boundary_lines_y=int(config.boundary_lines_y),
    )
    boundary_segs = (
        tracked_boundary_loops(
            grid_points,
            shape_pts,
            nx,
            ny,
            outline_rings=list(placed.rings) if placed is not None else None,
            region=crop_region,
            alive=alive,
            exit_fn=_section_box_exit(box) if box.enabled else None,
        )
        if bool(config.boundary_curve)
        else []
    )

    # Stage 5b — clip interior lines to the box (short stubs at the Z faces).
    # The tracked boundary already walks only visible points; do not fragment it.
    box_wire = (
        section_box_wireframe(box) if box.enabled else np.zeros((0, 3), dtype=float)
    )
    needs_crop = crop_region is not None or box.enabled
    if needs_crop:
        segments = split_nan_polyline(polyline)
        clipped = crop_line_segments(segments, config, region=crop_region)
        polyline, line_mesh = rebuild_line_geometry(clipped)
    boundary_polyline = (
        segments_to_nan_polyline(boundary_segs)
        if boundary_segs
        else np.zeros((0, 3), dtype=float)
    )
    inside_mask = crop_points_mask(shape_pts, region=crop_region, box=box)
    length = polyline_length(polyline)
    if verbose:
        kept = int(np.count_nonzero(inside_mask)) if len(inside_mask) else len(shape_pts)
        crop_note = ""
        if needs_crop:
            crop_note = f", cropped points={kept}/{len(shape_pts)}"
            if box.enabled:
                crop_note += " (section box on)"
        print(
            f"Stage 5 — line geometry: {len(polyline)} vertices, "
            f"length={length:.3f}, cells={line_mesh.n_cells} "
            f"(pattern={config.line_pattern}, "
            f"X={config.lines_x}, Y={config.lines_y}, "
            f"stride={config.line_stride}, "
            f"keepX={config.boundary_lines_x}, keepY={config.boundary_lines_y}"
            f"{crop_note})"
        )

    stats = {
        "point_count": int(len(grid_points)),
        "grid_size": int(nx),  # legacy
        "grid_size_x": int(nx),
        "grid_size_y": int(ny),
        "displacement_min": float(displacement.min()),
        "displacement_max": float(displacement.max()),
        "displacement_mean": float(displacement.mean()),
        "displacement_std": float(displacement.std()),
        "border_released": n_released,  # legacy key
        "xy_released": n_released,
        "xy_offset_max": float(np.linalg.norm(xy_offsets, axis=1).max())
        if len(xy_offsets)
        else 0.0,
        "polyline_vertices": int(len(polyline)),
        "polyline_length": length,
        "amplitudes": list(config.amplitudes),
        "active": list(config.active_flags),
        "active_labels": config.active_source_labels(),
        "wavelength": float(config.wavelength),
        "frequency": float(config.frequency),
        "time": float(config.time),
        "boundary_tension": float(config.boundary_tension),
        "cloth": float(config.cloth),
        "release_pace": float(config.release_pace),
        "line_pattern": config.line_pattern,
        "lines_x": bool(config.lines_x),
        "lines_y": bool(config.lines_y),
        "boundary_curve": bool(config.boundary_curve),
        "boundary_points": int(
            sum(max(0, len(np.asarray(s)) - 1) for s in boundary_segs)
        ),
        "points_out_of_bounds": int(alive.size - int(np.count_nonzero(alive))),
        "line_stride": int(config.line_stride),
        "boundary_lines_x": int(config.boundary_lines_x),
        "boundary_lines_y": int(config.boundary_lines_y),
        "shape": str(config.shape),
        "custom_shape_path": str(config.custom_shape_path),
        "custom_shape_size": float(config.custom_shape_size),
        "section_box_enabled": bool(config.section_box_enabled),
        "points_kept": int(np.count_nonzero(inside_mask)) if len(inside_mask) else int(len(shape_pts)),
    }

    return PipelineResult(
        config=config,
        grid_points=grid_points,
        xs=xs,
        ys=ys,
        sources=sources,
        source_labels=SOURCE_LABELS,
        corners=corners,
        displacement=displacement,
        contributions=contributions,
        displaced_points=displaced,
        xy_offsets=xy_offsets,
        shape_base_points=shape_base,
        shape_points=shape_pts,
        plane_offsets=plane_offs,
        shape_sources=shape_sources,
        polyline=polyline,
        line_mesh=line_mesh,
        boundary_polyline=boundary_polyline,
        inside_mask=inside_mask,
        custom_outline=custom_outline,
        section_box_wire=box_wire,
        stats=stats,
    )


def export_line_obj(result: PipelineResult, export_dir: str | Path, *, suffix: str = "") -> Path:
    """Write the line mesh to an OBJ file."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = export_dir / f"cymatics_line_{ts}{suffix}.obj"
    result.line_mesh.save(str(path))
    return path


def export_line_ply(result: PipelineResult, export_dir: str | Path, *, suffix: str = "") -> Path:
    """Write the line mesh to a PLY file."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = export_dir / f"cymatics_line_{ts}{suffix}.ply"
    result.line_mesh.save(str(path))
    return path


def export_pipe_stl(
    result: PipelineResult,
    export_dir: str | Path,
    *,
    voxel_config: Any | None = None,
    suffix: str = "",
    verbose: bool = True,
) -> Path:
    """Pipe stage-5 lines into a voxel solid and write a timestamped STL.

    Thin wrapper around :func:`cymatics_geometry.voxels.pipe_and_export_stl`
    (same export style as enhancement-geometry's ``export_stl``).
    """
    from cymatics_geometry.voxels import pipe_and_export_stl

    _solid, path = pipe_and_export_stl(
        result,
        export_dir,
        voxel_config,
        suffix=suffix,
        verbose=verbose,
    )
    return path
