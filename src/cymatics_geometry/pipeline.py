"""End-to-end cymatics plane → line geometry pipeline.

Mirrors the enhancement-geometry style: config in → staged processing →
PipelineResult out, usable from notebook and CLI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pyvista as pv

from cymatics_geometry.config import PipelineConfig
from cymatics_geometry.grid import (
    SOURCE_LABELS,
    build_square_grid,
    grid_shape,
    source_positions,
)
from cymatics_geometry.lines import build_line_geometry, polyline_length
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
    # Stage 4 — displaced points (Z + optional border XY release)
    displaced_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    xy_offsets: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    # Stage 5 — reconnected line
    polyline: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    line_mesh: pv.PolyData = field(default_factory=pv.PolyData)
    stats: dict = field(default_factory=dict)


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

    # Stage 4 — Z waves + radial XY release mobility
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

    # Stage 5
    polyline, line_mesh = build_line_geometry(
        displaced,
        nx,
        ny,
        pattern=config.line_pattern,
    )
    length = polyline_length(polyline)
    if verbose:
        print(
            f"Stage 5 — line geometry: {len(polyline)} vertices, "
            f"length={length:.3f}, cells={line_mesh.n_cells}"
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
        "release_pace": float(config.release_pace),
        "line_pattern": config.line_pattern,
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
        polyline=polyline,
        line_mesh=line_mesh,
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
