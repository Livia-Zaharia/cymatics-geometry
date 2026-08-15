"""Voxel piping / STL export smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import json

from cymatics_geometry.config import (
    PipelineConfig,
    load_voxel_params,
    save_model_params,
)
from cymatics_geometry.pipeline import run_pipeline
from cymatics_geometry.voxels import (
    VoxelPipeConfig,
    _resample_polyline,
    _smooth_resample_polyline,
    export_stl,
    pipe_lines_to_voxels,
    polyline_segments_from_result,
    preview_pipe_mesh,
)


def test_smooth_resample_differs_from_linear_on_bends() -> None:
    """Cubic spline leaves the L-shaped polyline near corners."""
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    n = 41
    linear = _resample_polyline(pts, n)
    smooth = _smooth_resample_polyline(pts, n)
    assert smooth.shape == (n, 3)
    assert float(np.linalg.norm(smooth[0] - pts[0])) < 1e-6
    assert float(np.linalg.norm(smooth[-1] - pts[-1])) < 1e-6
    max_diff = float(np.linalg.norm(smooth - linear, axis=1).max())
    assert max_diff > 0.1
    # Extra smoothing rounds the kink further than pure interpolation
    rounder = _smooth_resample_polyline(pts, n, smooth=5.0)
    assert float(np.linalg.norm(rounder - linear, axis=1).max()) > max_diff * 0.5
    # Short polyline falls back to linear
    short = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    fallback = _smooth_resample_polyline(short, 8)
    assert fallback.shape == (8, 3)
    assert np.allclose(fallback, _resample_polyline(short, 8))


def test_preview_respects_voxel_size_facet_density() -> None:
    """Smaller voxel_size must produce a denser preview tube mesh."""
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=10,
            grid_size_y=8,
            amplitude_sw=1.0,
            wavelength_sw=20.0,
            lines_x=True,
            lines_y=False,
        ),
        verbose=False,
    )
    coarse = preview_pipe_mesh(
        result,
        VoxelPipeConfig(
            voxel_size=2.0,
            pipe_radius=1.2,
            line_stride=4,
            point_stride=2,
            spine_samples=16,
            spine_smooth=1.0,
        ),
    )
    fine = preview_pipe_mesh(
        result,
        VoxelPipeConfig(
            voxel_size=0.4,
            pipe_radius=1.2,
            line_stride=4,
            point_stride=2,
            spine_samples=16,
            spine_smooth=1.0,
        ),
    )
    assert len(fine.faces) > len(coarse.faces)


def test_save_model_params_writes_voxel_nested(tmp_path: Path) -> None:
    pipe = PipelineConfig(amplitude_sw=1.0, wavelength_sw=20.0)
    voxel = VoxelPipeConfig(pipe_radius=1.5, spine_samples=32)
    path = save_model_params(pipe, voxel, tmp_path)
    assert path is not None
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["amplitude_sw"] == 1.0
    assert data["voxel"]["pipe_radius"] == 1.5
    assert data["voxel"]["spine_samples"] == 32
    again = save_model_params(pipe, voxel, tmp_path)
    assert again is None
    loaded = load_voxel_params(path)
    assert loaded is not None
    assert VoxelPipeConfig.from_dict(loaded).pipe_radius == 1.5


def test_preview_pipe_mesh_is_fast_and_nonempty() -> None:
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=12,
            grid_size_y=10,
            amplitude_sw=1.0,
            wavelength_sw=20.0,
            lines_x=True,
            lines_y=True,
        ),
        verbose=False,
    )
    mesh = preview_pipe_mesh(
        result,
        VoxelPipeConfig(pipe_radius=1.5, line_stride=2, point_stride=2, spine_samples=12),
    )
    assert len(mesh.faces) > 0
    assert len(mesh.vertices) > 0
    span = mesh.bounds[1] - mesh.bounds[0]
    assert float(np.max(span)) > 5.0


def test_polyline_segments_respect_line_stride() -> None:
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=8,
            grid_size_y=6,
            lines_x=True,
            lines_y=True,
            boundary_curve=False,
            amplitude_sw=1.0,
            wavelength_sw=20.0,
        ),
        verbose=False,
    )
    all_segs = polyline_segments_from_result(result, line_stride=1, point_stride=1)
    half = polyline_segments_from_result(
        result, line_stride=2, point_stride=1, boundary_lines=0
    )
    assert len(all_segs) == 6 + 8
    assert len(half) == 3 + 4
    assert all(len(s) >= 2 for s in half)


def test_boundary_lines_keep_first_and_last() -> None:
    """First/last N lines per direction survive even with a large stride."""
    from cymatics_geometry.lines import stride_indices_with_boundary

    assert stride_indices_with_boundary(6, stride=3, boundary=0) == [0, 3]
    # +N keeps only the ends (stride is ignored — otherwise it looks like a no-op)
    assert stride_indices_with_boundary(6, stride=3, boundary=1) == [0, 5]
    assert stride_indices_with_boundary(6, stride=3, boundary=2) == [0, 1, 4, 5]
    assert stride_indices_with_boundary(6, stride=1, boundary=1) == [0, 5]
    assert stride_indices_with_boundary(6, stride=1, boundary=2) == [0, 1, 4, 5]
    # Negative removes ends from the strided set
    assert stride_indices_with_boundary(6, stride=1, boundary=-1) == [1, 2, 3, 4]

    result = run_pipeline(
        PipelineConfig(
            grid_size_x=8,
            grid_size_y=6,
            lines_x=True,
            lines_y=True,
            boundary_curve=False,
            amplitude_sw=1.0,
            wavelength_sw=20.0,
        ),
        verbose=False,
    )
    rim = polyline_segments_from_result(
        result,
        line_stride=2,
        point_stride=1,
        boundary_lines_x=1,
        boundary_lines_y=1,
    )
    # +N keeps only the ends (stride is ignored)
    assert len(rim) == 2 + 2

    via_cfg = polyline_segments_from_result(
        result,
        line_stride=2,
        point_stride=1,
        boundary_lines_x=VoxelPipeConfig().boundary_lines_x,
        boundary_lines_y=VoxelPipeConfig().boundary_lines_y,
    )
    # Default keep=0 → stride only
    assert len(via_cfg) == 3 + 4


def test_pipe_lines_to_voxels_produces_mesh(tmp_path: Path) -> None:
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=10,
            grid_size_y=8,
            amplitude_sw=1.0,
            amplitude_se=0.8,
            wavelength_sw=25.0,
            wavelength_se=25.0,
            lines_x=True,
            lines_y=False,
            shape="plane",
        ),
        verbose=False,
    )
    vcfg = VoxelPipeConfig(
        voxel_size=1.2,
        pipe_radius=1.5,
        inner_radius=0.0,
        modulation_amp=0.3,
        modulation_freq=2.0,
        modulation_lobes=4,
        line_stride=2,
        point_stride=2,
        spine_samples=16,
    )
    solid = pipe_lines_to_voxels(result, vcfg, verbose=False)
    assert solid.segment_count > 0
    assert len(solid.trimesh_result.faces) > 0
    assert solid.volume > 0.0
    path = export_stl(solid, tmp_path, suffix="_test")
    assert path.exists()
    assert path.suffix == ".stl"
    assert path.stat().st_size > 100


def test_cylinder_shape_pipes(tmp_path: Path) -> None:
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=12,
            grid_size_y=10,
            amplitude_sw=0.8,
            wavelength_sw=22.0,
            shape="cylinder",
            cylinder_diameter=40.0,
            cylinder_length=80.0,
            lines_x=True,
            lines_y=True,
        ),
        verbose=False,
    )
    solid = pipe_lines_to_voxels(
        result,
        VoxelPipeConfig(
            voxel_size=1.5,
            pipe_radius=1.0,
            line_stride=3,
            point_stride=3,
            spine_samples=12,
        ),
        verbose=False,
    )
    path = export_stl(solid, tmp_path)
    assert path.exists()
    assert solid.stats["faces"] > 50
    # Mesh should span roughly cylinder extent
    bounds = solid.trimesh_result.bounds
    assert bounds is not None
    span = bounds[1] - bounds[0]
    assert float(np.max(span)) > 10.0
