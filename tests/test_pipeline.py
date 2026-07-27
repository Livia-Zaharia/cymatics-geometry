"""Basic invariants for the cymatics plane → line pipeline."""

from __future__ import annotations

import numpy as np

from cymatics_geometry.config import PipelineConfig
from cymatics_geometry.pipeline import run_pipeline


def test_default_pipeline_point_count() -> None:
    result = run_pipeline(PipelineConfig(grid_size_x=20, grid_size_y=20), verbose=False)
    assert len(result.grid_points) == 20 * 20
    assert len(result.displaced_points) == 20 * 20
    # Grid pattern: 20 X-rows + 20 Y-cols; mesh duplicates points per direction
    assert result.line_mesh.n_cells == 40
    assert result.line_mesh.n_points == 2 * 20 * 20
    assert len(result.sources) == 8
    assert result.config.line_pattern == "grid"


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
    """Zero amp/λ/release = idle; only sources with values contribute."""
    off = run_pipeline(
        PipelineConfig(
            grid_size_x=12,
            grid_size_y=12,
            amplitude_s=2.0,
            wavelength_s=20.0,
        ),
        verbose=False,
    )
    assert off.config.active_source_labels() == ["S"]
    assert not np.allclose(off.displacement, 0.0)


def test_release_unit_strength_is_linear() -> None:
    """Raising release max must not dilute per-unit shove (cloth off)."""
    low = run_pipeline(
        PipelineConfig(
            grid_size_x=20,
            grid_size_y=20,
            release_sw=10.0,
            cloth=0.0,
        ),
        verbose=False,
    )
    high = run_pipeline(
        PipelineConfig(
            grid_size_x=20,
            grid_size_y=20,
            release_sw=20.0,
            cloth=0.0,
        ),
        verbose=False,
    )
    ratio = high.stats["xy_offset_max"] / max(low.stats["xy_offset_max"], 1e-9)
    assert abs(ratio - 2.0) < 0.08


def test_serpentine_connectivity_starts_at_sw() -> None:
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=5,
            grid_size_y=5,
            side_length=10.0,
            amplitude_sw=1.0,
            wavelength_sw=25.0,
            release_sw=0.0,
            line_pattern="serpentine",
        ),
        verbose=False,
    )
    assert np.allclose(result.polyline[0], [0.0, 0.0, result.polyline[0, 2]])


def test_grid_lines_have_both_directions() -> None:
    both = run_pipeline(
        PipelineConfig(grid_size_x=5, grid_size_y=4, line_pattern="grid"),
        verbose=False,
    )
    # 4 X-rows + 5 Y-cols = 9 segments → 8 NaN breaks
    n_nan = int(np.isnan(both.polyline).all(axis=1).sum())
    assert n_nan == 8
    assert both.line_mesh.n_cells == 9

    x_only = run_pipeline(
        PipelineConfig(
            grid_size_x=5,
            grid_size_y=4,
            line_pattern="grid",
            lines_x=True,
            lines_y=False,
        ),
        verbose=False,
    )
    assert x_only.line_mesh.n_cells == 4

    y_only = run_pipeline(
        PipelineConfig(
            grid_size_x=5,
            grid_size_y=4,
            line_pattern="grid",
            lines_x=False,
            lines_y=True,
        ),
        verbose=False,
    )
    assert y_only.line_mesh.n_cells == 5


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


def test_release_alone_shoves_xy_without_amplitude() -> None:
    """Release is a radial force — amp is optional for XY shockwave."""
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=25,
            grid_size_y=25,
            side_length=100.0,
            amplitude_sw=0.0,
            wavelength_sw=0.0,
            release_sw=20.0,
            cloth=40.0,
        ),
        verbose=False,
    )
    assert result.stats["xy_offset_max"] > 1.0
    assert result.stats["xy_offset_max"] <= 150.0 + 1e-6
    corner_move = np.linalg.norm(
        result.displaced_points[0, :2] - result.grid_points[0, :2]
    )
    assert corner_move > 0.5
    xy = result.displaced_points[:, :2]
    assert float(xy[:, 0].max()) > 100.0 or float(xy[:, 0].min()) < 0.0
    assert float(xy[:, 1].max()) > 100.0 or float(xy[:, 1].min()) < 0.0


def test_release_peak_capped_at_one_fifty_units() -> None:
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=21,
            grid_size_y=21,
            side_length=100.0,
            amplitude_sw=0.0,
            release_sw=150.0,
            cloth=0.0,
        ),
        verbose=False,
    )
    assert result.stats["xy_offset_max"] <= 150.0 + 1e-6
    assert result.stats["xy_offset_max"] > 50.0


def test_release_shockwave_is_stronger_near_source() -> None:
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=31,
            grid_size_y=31,
            side_length=100.0,
            amplitude_sw=0.0,
            release_sw=15.0,
            cloth=40.0,
        ),
        verbose=False,
    )
    d = np.linalg.norm(
        result.displaced_points[:, :2] - result.grid_points[:, :2], axis=1
    )
    ne_idx = 31 * 30 + 30
    assert float(d[0]) > float(d[ne_idx])


def test_cloth_keeps_neighbour_edges_from_exploding() -> None:
    """Internal springs: mean edge stretch stays bounded under release shove."""
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=20,
            grid_size_y=20,
            side_length=100.0,
            amplitude_sw=0.0,
            release_sw=40.0,
            cloth=80.0,
        ),
        verbose=False,
    )
    nx, ny = 20, 20
    xy = result.displaced_points[:, :2].reshape(ny, nx, 2)
    rest = result.grid_points[:, :2].reshape(ny, nx, 2)
    h_cur = np.linalg.norm(xy[:, 1:] - xy[:, :-1], axis=-1)
    h_rest = np.linalg.norm(rest[:, 1:] - rest[:, :-1], axis=-1)
    mean_stretch = float(np.mean(np.abs(h_cur - h_rest) / np.maximum(h_rest, 1e-9)))
    assert mean_stretch < 0.35
    assert result.stats["xy_offset_max"] > 1.0


def test_cloth_gradient_core_stiffer_than_edge() -> None:
    from cymatics_geometry.waves import _cloth_stiffness_field

    field = _cloth_stiffness_field(41, 41, cloth=70.0)
    core = float(field[20, 20])
    corner = float(field[0, 0])
    edge_mid = float(field[0, 20])
    assert core > edge_mid > 0.0
    assert core > corner
    assert corner / core < 0.35


def test_cloth_zero_skips_springs() -> None:
    free = run_pipeline(
        PipelineConfig(
            grid_size_x=15,
            grid_size_y=15,
            release_sw=20.0,
            cloth=0.0,
        ),
        verbose=False,
    )
    tight = run_pipeline(
        PipelineConfig(
            grid_size_x=15,
            grid_size_y=15,
            release_sw=20.0,
            cloth=100.0,
        ),
        verbose=False,
    )
    assert tight.stats["xy_offset_max"] < free.stats["xy_offset_max"]
