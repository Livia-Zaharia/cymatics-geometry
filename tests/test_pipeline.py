"""Basic invariants for the cymatics plane → line pipeline."""

from __future__ import annotations

import numpy as np

from cymatics_geometry.config import PipelineConfig
from cymatics_geometry.pipeline import run_pipeline


def test_default_pipeline_point_count() -> None:
    result = run_pipeline(PipelineConfig(grid_size_x=20, grid_size_y=20), verbose=False)
    assert len(result.grid_points) == 20 * 20
    assert len(result.displaced_points) == 20 * 20
    assert len(result.polyline) == 20 * 20
    assert result.line_mesh.n_points == 20 * 20
    assert len(result.sources) == 8


def test_rectangular_grid() -> None:
    result = run_pipeline(PipelineConfig(grid_size_x=12, grid_size_y=8), verbose=False)
    assert len(result.grid_points) == 12 * 8
    assert result.stats["grid_size_x"] == 12
    assert result.stats["grid_size_y"] == 8


def test_zero_defaults_keep_flat_locked_plane() -> None:
    result = run_pipeline(PipelineConfig(grid_size_x=10, grid_size_y=10), verbose=False)
    assert np.allclose(result.displacement, 0.0)
    assert np.allclose(result.displaced_points[:, :2], result.grid_points[:, :2])
    assert result.stats["xy_released"] == 0


def test_amplitude_changes_field() -> None:
    base = run_pipeline(
        PipelineConfig(
            grid_size_x=16,
            grid_size_y=16,
            amplitude_sw=1.0,
            wavelength_sw=25.0,
        ),
        verbose=False,
    )
    boosted = run_pipeline(
        PipelineConfig(
            grid_size_x=16,
            grid_size_y=16,
            amplitude_sw=2.0,
            wavelength_sw=25.0,
        ),
        verbose=False,
    )
    assert not np.allclose(base.displacement, boosted.displacement)


def test_inactive_source_does_not_contribute() -> None:
    off = run_pipeline(
        PipelineConfig(
            grid_size_x=12,
            grid_size_y=12,
            active_sw=False,
            active_se=False,
            active_ne=False,
            active_nw=False,
            active_s=True,
            amplitude_s=2.0,
            wavelength_s=20.0,
        ),
        verbose=False,
    )
    assert off.config.active_source_labels() == ["S"]
    assert not np.allclose(off.displacement, 0.0)


def test_serpentine_connectivity_starts_at_sw() -> None:
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=5,
            grid_size_y=5,
            side_length=10.0,
            amplitude_sw=1.0,
            wavelength_sw=25.0,
            release_sw=0.0,
        ),
        verbose=False,
    )
    assert np.allclose(result.polyline[0], [0.0, 0.0, result.polyline[0, 2]])


def test_per_source_wavelength_changes_field() -> None:
    base = run_pipeline(
        PipelineConfig(
            grid_size_x=16,
            grid_size_y=16,
            amplitude_sw=1.0,
            amplitude_se=1.0,
            wavelength_sw=25.0,
            wavelength_se=25.0,
        ),
        verbose=False,
    )
    tight = run_pipeline(
        PipelineConfig(
            grid_size_x=16,
            grid_size_y=16,
            amplitude_sw=1.0,
            amplitude_se=1.0,
            wavelength_sw=10.0,
            wavelength_se=25.0,
        ),
        verbose=False,
    )
    assert not np.allclose(base.displacement, tight.displacement)


def test_release_zero_locks_xy() -> None:
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=12,
            grid_size_y=12,
            amplitude_sw=2.0,
            wavelength_sw=20.0,
            release_sw=0.0,
        ),
        verbose=False,
    )
    assert np.allclose(result.displaced_points[:, :2], result.grid_points[:, :2])
    assert result.stats["xy_released"] == 0


def test_release_moves_surface_including_center() -> None:
    locked = run_pipeline(
        PipelineConfig(
            grid_size_x=21,
            grid_size_y=21,
            side_length=100.0,
            amplitude_sw=2.0,
            wavelength_sw=25.0,
            release_sw=0.0,
        ),
        verbose=False,
    )
    freed = run_pipeline(
        PipelineConfig(
            grid_size_x=21,
            grid_size_y=21,
            side_length=100.0,
            amplitude_sw=2.0,
            wavelength_sw=25.0,
            release_sw=2.0,
        ),
        verbose=False,
    )
    assert locked.stats["xy_released"] == 0
    assert freed.stats["xy_released"] > 0
    assert freed.stats["xy_offset_max"] > 0.0

    # Center of an odd grid should move once release is on (surface coupling)
    center_idx = (21 // 2) * 21 + (21 // 2)
    center_move = np.linalg.norm(
        freed.displaced_points[center_idx, :2] - freed.grid_points[center_idx, :2]
    )
    assert center_move > 1e-6
