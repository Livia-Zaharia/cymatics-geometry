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
            boundary_curve=False,
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
            boundary_curve=False,
        ),
        verbose=False,
    )
    assert y_only.line_mesh.n_cells == 5


def test_line_stride_and_boundary_lines_thin_display() -> None:
    """line_stride thins; signed keep X/Y keep or remove ends per direction."""
    full = run_pipeline(
        PipelineConfig(grid_size_x=8, grid_size_y=6, line_pattern="grid"),
        verbose=False,
    )
    assert full.line_mesh.n_cells == 6 + 8

    thinned = run_pipeline(
        PipelineConfig(
            grid_size_x=8,
            grid_size_y=6,
            line_pattern="grid",
            line_stride=2,
            boundary_lines_x=0,
            boundary_lines_y=0,
            boundary_curve=False,
        ),
        verbose=False,
    )
    # X-rows 6 → 3; Y-cols 8 → 4
    assert thinned.line_mesh.n_cells == 3 + 4

    only_ends = run_pipeline(
        PipelineConfig(
            grid_size_x=8,
            grid_size_y=6,
            line_pattern="grid",
            line_stride=2,
            boundary_lines_x=1,
            boundary_lines_y=1,
        ),
        verbose=False,
    )
    # +N ignores stride and keeps only the ends
    assert only_ends.line_mesh.n_cells == 2 + 2
    assert only_ends.line_mesh.n_cells < thinned.line_mesh.n_cells

    # Negative keep X removes first/last from X-rows only
    drop_x = run_pipeline(
        PipelineConfig(
            grid_size_x=8,
            grid_size_y=6,
            line_pattern="grid",
            line_stride=1,
            boundary_lines_x=-1,
            boundary_lines_y=0,
            boundary_curve=False,
        ),
        verbose=False,
    )
    # X-rows 6 − 2 ends = 4; Y-cols unchanged 8
    assert drop_x.line_mesh.n_cells == 4 + 8


def test_boundary_curve_is_continuous_network_loop() -> None:
    """Tracked rectangle stays one closed loop of the original edge points."""
    from cymatics_geometry.lines import split_nan_polyline

    result = run_pipeline(
        PipelineConfig(
            grid_size_x=8,
            grid_size_y=6,
            line_pattern="grid",
            line_stride=3,
            boundary_lines_x=0,
            boundary_lines_y=0,
            amplitude_sw=2.0,
            wavelength_sw=20.0,
        ),
        verbose=False,
    )
    loops = split_nan_polyline(result.boundary_polyline)
    assert len(loops) == 1
    loop = loops[0]
    assert len(loop) >= 5
    np.testing.assert_allclose(loop[0], loop[-1], atol=1e-8)

    pts = result.shape_points.reshape(6, 8, 3)
    corners = [pts[0, 0], pts[0, -1], pts[-1, -1], pts[-1, 0]]
    for corner in corners:
        dist = np.linalg.norm(loop - corner.reshape(1, 3), axis=1).min()
        assert dist < 1e-6

    # Interior thinning does not add extra perimeter lines to the grid mesh
    thinned_cells = result.line_mesh.n_cells
    assert thinned_cells == 2 + 3  # stride 3: X-rows {0,3} Y-cols {0,3,6}

    frame_only = run_pipeline(
        PipelineConfig(
            grid_size_x=8,
            grid_size_y=6,
            line_pattern="grid",
            lines_x=False,
            lines_y=False,
            boundary_curve=True,
        ),
        verbose=False,
    )
    assert frame_only.line_mesh.n_cells == 0
    assert len(split_nan_polyline(frame_only.boundary_polyline)) == 1


def test_negative_amplitude_flips_displacement() -> None:
    pos = run_pipeline(
        PipelineConfig(
            grid_size_x=12,
            grid_size_y=12,
            amplitude_sw=1.5,
            wavelength_sw=20.0,
        ),
        verbose=False,
    )
    neg = run_pipeline(
        PipelineConfig(
            grid_size_x=12,
            grid_size_y=12,
            amplitude_sw=-1.5,
            wavelength_sw=20.0,
        ),
        verbose=False,
    )
    assert np.allclose(neg.displacement, -pos.displacement)
    assert neg.config.active_source_labels() == ["SW"]


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


def test_plane_shape_is_identity_map() -> None:
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=12,
            grid_size_y=10,
            amplitude_sw=1.5,
            wavelength_sw=20.0,
            release_sw=5.0,
            shape="plane",
        ),
        verbose=False,
    )
    assert np.allclose(result.shape_points, result.displaced_points)
    assert np.allclose(result.shape_base_points, result.grid_points)
    assert np.allclose(
        result.plane_offsets, result.displaced_points - result.grid_points
    )


def test_cylinder_preserves_point_count_and_line_topology() -> None:
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=16,
            grid_size_y=12,
            amplitude_sw=1.0,
            wavelength_sw=25.0,
            shape="cylinder",
            cylinder_diameter=40.0,
            cylinder_length=80.0,
        ),
        verbose=False,
    )
    n = 16 * 12
    assert len(result.shape_points) == n
    assert len(result.shape_base_points) == n
    assert result.line_mesh.n_cells == 12 + 16
    # Undeformed cylinder sits on the radius shell (before wave offsets)
    radii = np.linalg.norm(result.shape_base_points[:, [0, 2]], axis=1)
    assert np.allclose(radii, 20.0, atol=1e-6)
    assert float(result.shape_base_points[:, 1].min()) == 0.0
    assert float(result.shape_base_points[:, 1].max()) == 80.0


def test_cone_and_frustum_map_offsets_in_local_frame() -> None:
    flat = run_pipeline(
        PipelineConfig(
            grid_size_x=11,
            grid_size_y=11,
            amplitude_sw=2.0,
            wavelength_sw=30.0,
            shape="plane",
        ),
        verbose=False,
    )
    cone = run_pipeline(
        PipelineConfig(
            grid_size_x=11,
            grid_size_y=11,
            amplitude_sw=2.0,
            wavelength_sw=30.0,
            shape="cone",
            cone_height=90.0,
            cone_base_radius=25.0,
        ),
        verbose=False,
    )
    frust = run_pipeline(
        PipelineConfig(
            grid_size_x=11,
            grid_size_y=11,
            amplitude_sw=2.0,
            wavelength_sw=30.0,
            shape="frustum",
            frustum_height=90.0,
            frustum_base_diameter=50.0,
            frustum_top_diameter=10.0,
        ),
        verbose=False,
    )
    assert np.allclose(flat.plane_offsets, cone.plane_offsets)
    assert np.allclose(flat.plane_offsets, frust.plane_offsets)
    # Same offset magnitude in local frames (orthonormal)
    cone_travel = np.linalg.norm(cone.shape_points - cone.shape_base_points, axis=1)
    frust_travel = np.linalg.norm(frust.shape_points - frust.shape_base_points, axis=1)
    plane_travel = np.linalg.norm(flat.plane_offsets, axis=1)
    assert np.allclose(cone_travel, plane_travel, atol=1e-5)
    assert np.allclose(frust_travel, plane_travel, atol=1e-5)
    # Tip of cone (v=1) collapses to axis
    tip_mask = cone.shape_base_points[:, 1] >= 90.0 - 1e-9
    assert tip_mask.any()
    tip_r = np.linalg.norm(cone.shape_base_points[tip_mask][:, [0, 2]], axis=1)
    assert np.allclose(tip_r, 0.0, atol=1e-6)


def test_bead_slice_radii_independent() -> None:
    """Sphere with different top/bottom slice radii maps onto a bead surface."""
    from cymatics_geometry.shapes import _bead_slice_heights

    r, r0, r1, z0, z1 = _bead_slice_heights(
        diameter=40.0, bottom_radius=8.0, top_radius=15.0
    )
    assert abs(r - 20.0) < 1e-9
    assert abs(r0 - 8.0) < 1e-9
    assert abs(r1 - 15.0) < 1e-9
    assert z0 < 0.0 < z1
    assert abs(r0 * r0 + z0 * z0 - r * r) < 1e-6
    assert abs(r1 * r1 + z1 * z1 - r * r) < 1e-6

    result = run_pipeline(
        PipelineConfig(
            grid_size_x=12,
            grid_size_y=10,
            amplitude_sw=1.0,
            wavelength_sw=25.0,
            shape="bead",
            bead_diameter=40.0,
            bead_bottom_radius=8.0,
            bead_top_radius=15.0,
        ),
        verbose=False,
    )
    assert len(result.shape_points) == 12 * 10
    y = result.shape_base_points[:, 1]
    assert float(y.min()) == 0.0
    # Bottom / top rings match the requested slice radii
    begin = result.shape_base_points[np.isclose(y, 0.0)]
    end = result.shape_base_points[np.isclose(y, float(y.max()))]
    begin_r = np.linalg.norm(begin[:, [0, 2]], axis=1)
    end_r = np.linalg.norm(end[:, [0, 2]], axis=1)
    assert np.allclose(begin_r, 8.0, atol=1e-4)
    assert np.allclose(end_r, 15.0, atol=1e-4)
    # Bulge of the sphere exceeds both slice radii
    all_r = np.linalg.norm(result.shape_base_points[:, [0, 2]], axis=1)
    assert float(all_r.max()) > max(8.0, 15.0) + 0.5


def test_variable_cylinder_three_radii_and_middle_clamp() -> None:
    from cymatics_geometry.shapes import (
        _clamp_variable_cylinder_middle,
        _variable_cylinder_radius_profile,
    )

    assert _clamp_variable_cylinder_middle(0.0) == 0.1
    assert _clamp_variable_cylinder_middle(1.0) == 0.9
    assert _clamp_variable_cylinder_middle(0.5) == 0.5

    v = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    radius, _dr = _variable_cylinder_radius_profile(
        v,
        radius_begin=10.0,
        radius_middle=30.0,
        radius_end=5.0,
        middle_t=0.5,
    )
    assert abs(float(radius[0]) - 10.0) < 1e-9
    assert abs(float(radius[2]) - 30.0) < 1e-9
    assert abs(float(radius[-1]) - 5.0) < 1e-9

    result = run_pipeline(
        PipelineConfig(
            grid_size_x=12,
            grid_size_y=10,
            amplitude_sw=1.0,
            wavelength_sw=25.0,
            shape="variable_cylinder",
            variable_cylinder_radius_begin=10.0,
            variable_cylinder_radius_middle=25.0,
            variable_cylinder_radius_end=8.0,
            variable_cylinder_length=80.0,
            variable_cylinder_middle=0.05,  # clamped to 0.1 at map time
        ),
        verbose=False,
    )
    assert len(result.shape_points) == 12 * 10
    assert result.line_mesh.n_cells == 10 + 12
    y = result.shape_base_points[:, 1]
    assert float(y.min()) == 0.0
    assert float(y.max()) == 80.0
    # Begin / end rings match the requested radii
    begin = result.shape_base_points[np.isclose(y, 0.0)]
    end = result.shape_base_points[np.isclose(y, 80.0)]
    begin_r = np.linalg.norm(begin[:, [0, 2]], axis=1)
    end_r = np.linalg.norm(end[:, [0, 2]], axis=1)
    assert np.allclose(begin_r, 10.0, atol=1e-5)
    assert np.allclose(end_r, 8.0, atol=1e-5)


def test_teardrop_five_circles_and_tip() -> None:
    from cymatics_geometry.shapes import ShapeParams, surface_base_and_frames

    params = ShapeParams(
        kind="teardrop",
        teardrop_height=80.0,
        teardrop_radius_0=22.0,
        teardrop_radius_1=20.0,
        teardrop_radius_2=16.0,
        teardrop_radius_3=8.0,
        teardrop_radius_4=0.0,
        teardrop_station_1=0.20,
        teardrop_station_2=0.45,
        teardrop_station_3=0.70,
    )
    v = np.array([0.0, 0.20, 0.45, 0.70, 1.0])
    u = np.zeros_like(v)
    base, _, _, _ = surface_base_and_frames(u, v, params, side_length=100.0)
    radii = np.linalg.norm(base[:, [0, 2]], axis=1)
    assert abs(float(radii[0]) - 22.0) < 1e-6
    assert abs(float(radii[1]) - 20.0) < 1e-6
    assert abs(float(radii[2]) - 16.0) < 1e-6
    assert abs(float(radii[3]) - 8.0) < 1e-6
    assert abs(float(radii[4]) - 0.0) < 1e-6
    assert abs(float(base[-1, 1]) - 80.0) < 1e-6
    assert abs(float(base[0, 1]) - 0.0) < 1e-6

    result = run_pipeline(
        PipelineConfig(
            grid_size_x=12,
            grid_size_y=10,
            amplitude_sw=1.0,
            wavelength_sw=25.0,
            shape="teardrop",
            teardrop_height=80.0,
            teardrop_radius_4=0.0,
        ),
        verbose=False,
    )
    assert len(result.shape_points) == 12 * 10
    y = result.shape_base_points[:, 1]
    assert float(y.min()) == 0.0
    assert abs(float(y.max()) - 80.0) < 1e-6
    tip = result.shape_base_points[np.isclose(y, float(y.max()))]
    tip_r = np.linalg.norm(tip[:, [0, 2]], axis=1)
    assert np.allclose(tip_r, 0.0, atol=1e-5)


def test_bead_five_circle_override() -> None:
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=12,
            grid_size_y=11,
            amplitude_sw=1.0,
            wavelength_sw=25.0,
            shape="bead",
            bead_diameter=40.0,
            bead_bottom_radius=12.0,
            bead_top_radius=12.0,
            bead_height=32.0,
            bead_radius_0=12.0,
            bead_radius_1=18.0,
            bead_radius_2=30.0,
            bead_radius_3=18.0,
            bead_radius_4=12.0,
            bead_station_1=0.25,
            bead_station_2=0.50,
            bead_station_3=0.75,
        ),
        verbose=False,
    )
    y = result.shape_base_points[:, 1]
    begin = result.shape_base_points[np.isclose(y, 0.0)]
    end = result.shape_base_points[np.isclose(y, float(y.max()))]
    begin_r = np.linalg.norm(begin[:, [0, 2]], axis=1)
    end_r = np.linalg.norm(end[:, [0, 2]], axis=1)
    all_r = np.linalg.norm(result.shape_base_points[:, [0, 2]], axis=1)
    assert np.allclose(begin_r, 12.0, atol=1e-4)
    assert np.allclose(end_r, 12.0, atol=1e-4)
    assert float(all_r.max()) > 28.0


def test_interior_stations_clamp_ordered() -> None:
    from cymatics_geometry.shapes import _clamp_interior_stations

    t1, t2, t3 = _clamp_interior_stations(0.8, 0.1, 0.05)
    assert t1 < t2 < t3
    assert t1 >= 0.05
    assert t3 <= 0.95
    assert t2 - t1 >= 0.05 - 1e-12
    assert t3 - t2 >= 0.05 - 1e-12

    t1, t2, t3 = _clamp_interior_stations(0.5, 0.5, 0.5)
    assert t1 < t2 < t3
    assert t2 - t1 >= 0.05 - 1e-12
    assert t3 - t2 >= 0.05 - 1e-12
