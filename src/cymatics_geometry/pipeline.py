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
from cymatics_geometry.grid import CORNER_LABELS, build_square_grid, corner_positions
from cymatics_geometry.lines import build_line_geometry, polyline_length
from cymatics_geometry.waves import displace_points, interference_field

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything produced by a single pipeline run, stage by stage."""

    config: PipelineConfig
    # Stage 1 — flat square grid
    grid_points: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    # Stage 2 — corner wave sources
    corners: np.ndarray
    corner_labels: tuple[str, str, str, str] = CORNER_LABELS
    # Stage 3 — interference field
    displacement: np.ndarray = field(default_factory=lambda: np.zeros(0))
    contributions: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    # Stage 4 — displaced points
    displaced_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
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
    1. Deploy points on a square N×N grid
    2. Place wave-producing outputs at the four corners
    3. Compute wave interference from corner amplitudes
    4. Displace points in Z according to the field
    5. Reconnect displaced points into a continuous line
    """
    if verbose:
        print(f"Grid: {config.grid_size}×{config.grid_size}, side={config.side_length}")
        print(
            "Corner amplitudes SW/SE/NE/NW: "
            f"{config.amplitude_sw:.3f} / {config.amplitude_se:.3f} / "
            f"{config.amplitude_ne:.3f} / {config.amplitude_nw:.3f}"
        )
        print(
            f"wavelength={config.wavelength:.3f}, frequency={config.frequency:.3f}, "
            f"time={config.time:.3f}, decay={config.decay:.3f}"
        )
        print(f"line_pattern={config.line_pattern}")

    # Stage 1
    grid_points, xs, ys = build_square_grid(config)
    if verbose:
        print(f"Stage 1 — flat grid: {len(grid_points)} points")

    # Stage 2
    corners = corner_positions(config.side_length)
    if verbose:
        print(f"Stage 2 — corner sources: {list(CORNER_LABELS)}")

    # Stage 3
    displacement, contributions = interference_field(grid_points, config)
    if verbose:
        print(
            "Stage 3 — interference field: "
            f"z∈[{float(displacement.min()):.4f}, {float(displacement.max()):.4f}]"
        )

    # Stage 4
    displaced = displace_points(grid_points, displacement)
    if verbose:
        print(f"Stage 4 — displaced points: {len(displaced)}")

    # Stage 5
    polyline, line_mesh = build_line_geometry(
        displaced,
        config.grid_size,
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
        "grid_size": int(config.grid_size),
        "displacement_min": float(displacement.min()),
        "displacement_max": float(displacement.max()),
        "displacement_mean": float(displacement.mean()),
        "displacement_std": float(displacement.std()),
        "polyline_vertices": int(len(polyline)),
        "polyline_length": length,
        "amplitudes": list(config.amplitudes),
        "wavelength": float(config.wavelength),
        "frequency": float(config.frequency),
        "time": float(config.time),
        "line_pattern": config.line_pattern,
    }

    return PipelineResult(
        config=config,
        grid_points=grid_points,
        xs=xs,
        ys=ys,
        corners=corners,
        displacement=displacement,
        contributions=contributions,
        displaced_points=displaced,
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
