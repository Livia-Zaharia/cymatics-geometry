"""Original 2D outline tracked through the wave by lattice index order."""

from __future__ import annotations

import numpy as np

from cymatics_geometry.config import PipelineConfig, load_pipeline_config
from cymatics_geometry.lines import (
    _rectangular_index_loop,
    split_nan_polyline,
    tracked_boundary_loops,
)
from cymatics_geometry.pipeline import run_pipeline


def test_plane_boundary_follows_neutral_edge_order() -> None:
    nx, ny = 8, 6
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=nx,
            grid_size_y=ny,
            amplitude_sw=2.0,
            wavelength_sw=20.0,
            cloth=4.0,
            line_pattern="grid",
        ),
        verbose=False,
    )
    loops = split_nan_polyline(result.boundary_polyline)
    assert len(loops) == 1
    loop = loops[0]
    np.testing.assert_allclose(loop[0], loop[-1], atol=1e-8)

    moved = result.shape_points.reshape(ny, nx, 3)
    order = _rectangular_index_loop(nx, ny)
    expected = np.asarray([moved[r, c] for r, c in order], dtype=float)
    expected = np.vstack([expected, expected[:1]])
    np.testing.assert_allclose(loop, expected, atol=1e-8)


def test_interior_motion_does_not_reorder_boundary() -> None:
    nx, ny = 7, 5
    neutral = np.zeros((ny, nx, 3), dtype=float)
    for row in range(ny):
        for col in range(nx):
            neutral[row, col] = [float(col), float(row), 0.0]
    moved = neutral.copy()
    moved[1:-1, 1:-1, 2] = 9.0
    loops = tracked_boundary_loops(neutral.reshape(-1, 3), moved.reshape(-1, 3), nx, ny)
    assert len(loops) == 1
    assert np.allclose(loops[0][:, 2], 0.0)


def _cells_of(loop: np.ndarray, moved: np.ndarray) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for p in loop[:-1]:
        d2 = np.sum((moved - p.reshape(1, 1, 3)) ** 2, axis=-1)
        row, col = np.unravel_index(int(np.argmin(d2)), d2.shape)
        cells.append((int(row), int(col)))
    return cells


def _assert_local_steps(loop: np.ndarray, max_xy: float) -> None:
    for a, b in zip(loop[:-1], loop[1:]):
        assert float(np.linalg.norm(a[:2] - b[:2])) <= max_xy + 1e-6


def test_boundary_detours_inward_when_z_clipped_by_box() -> None:
    nx, ny = 8, 6
    neutral = np.zeros((ny, nx, 3), dtype=float)
    for row in range(ny):
        for col in range(nx):
            neutral[row, col] = [float(col), float(row), 0.0]
    moved = neutral.copy()
    # South-edge middle is clipped by the representation Z box.
    moved[0, 2:6, 2] = 40.0
    alive = np.ones((ny, nx), dtype=bool)
    alive[0, 2:6] = False
    loops = tracked_boundary_loops(
        neutral.reshape(-1, 3),
        moved.reshape(-1, 3),
        nx,
        ny,
        alive=alive,
        z_lim=10.0,
    )
    assert len(loops) == 1
    loop = loops[0]
    np.testing.assert_allclose(loop[0], loop[-1], atol=1e-8)
    for p in loop:
        assert float(p[2]) < 20.0
    _assert_local_steps(loop, 2.0)
    assert any(abs(float(p[1]) - 1.0) < 1.1 for p in loop)


def test_boundary_hugs_north_canyon_nw_to_ne() -> None:
    """Gap on NW→NE: turn sideways, walk the canyon floor, resume the edge."""
    nx, ny = 9, 7
    neutral = np.zeros((ny, nx, 3), dtype=float)
    for row in range(ny):
        for col in range(nx):
            neutral[row, col] = [float(col), float(row), 0.0]
    moved = neutral.copy()
    alive = np.ones((ny, nx), dtype=bool)
    alive[ny - 1, 3:6] = False
    alive[ny - 2, 3:6] = False
    loops = tracked_boundary_loops(
        neutral.reshape(-1, 3),
        moved.reshape(-1, 3),
        nx,
        ny,
        alive=alive,
        z_lim=10.0,
    )
    assert len(loops) == 1
    loop = loops[0]
    np.testing.assert_allclose(loop[0], loop[-1], atol=1e-8)
    _assert_local_steps(loop, 2.0)
    floor = moved[ny - 3, 4]
    assert float(np.min(np.linalg.norm(loop - floor.reshape(1, 3), axis=1))) < 1.5
    assert float(np.min(np.linalg.norm(loop - moved[ny - 1, 0].reshape(1, 3), axis=1))) < 1e-6
    assert float(np.min(np.linalg.norm(loop - moved[ny - 1, 6].reshape(1, 3), axis=1))) < 1e-6


def test_grid_lines_break_where_z_is_clipped() -> None:
    from cymatics_geometry.lines import build_line_geometry, split_nan_polyline

    nx, ny = 8, 6
    moved = np.zeros((ny, nx, 3), dtype=float)
    for row in range(ny):
        for col in range(nx):
            moved[row, col] = [float(col), float(row), 0.0]
    moved[3, 5, 2] = 40.0
    alive = np.ones((ny, nx), dtype=bool)
    alive[3, 5] = False
    polyline, _mesh = build_line_geometry(moved.reshape(-1, 3), nx, ny, alive=alive)
    pieces = split_nan_polyline(polyline)
    row3 = [
        p
        for p in pieces
        if abs(float(p[0, 1]) - 3.0) < 1e-6 and abs(float(p[-1, 1]) - 3.0) < 1e-6
    ]
    assert row3
    assert all(len(p) < nx for p in row3)


def test_saved_config_section_box_closes_boundary() -> None:
    from dataclasses import replace
    from pathlib import Path

    cfg = load_pipeline_config(
        Path(__file__).resolve().parents[1]
        / "notebooks"
        / "configs"
        / "20260815_233948.json"
    )
    cfg = replace(
        cfg,
        section_box_enabled=True,
        section_box_size_x=200.0,
        section_box_size_y=200.0,
        section_box_size_z=8.0,
        section_box_center_x=50.0,
        section_box_center_y=50.0,
        section_box_center_z=0.0,
    )
    result = run_pipeline(cfg, verbose=False)
    loops = split_nan_polyline(result.boundary_polyline)
    assert loops
    for loop in loops:
        np.testing.assert_allclose(loop[0], loop[-1], atol=1e-8)
        assert float(np.nanmax(np.abs(loop[:, 2]))) <= 4.0 + 1e-6
        _assert_local_steps(loop, 8.0)
    assert int(result.stats.get("points_out_of_bounds", 0)) > 0


def test_interior_clip_hole_gets_gold_loop() -> None:
    nx, ny = 9, 9
    neutral = np.zeros((ny, nx, 3), dtype=float)
    for row in range(ny):
        for col in range(nx):
            neutral[row, col] = [float(col), float(row), 0.0]
    moved = neutral.copy()
    moved[3:6, 3:6, 2] = 40.0
    alive = np.ones((ny, nx), dtype=bool)
    alive[3:6, 3:6] = False
    loops = tracked_boundary_loops(
        neutral.reshape(-1, 3),
        moved.reshape(-1, 3),
        nx,
        ny,
        alive=alive,
        z_lim=10.0,
    )
    assert len(loops) >= 2
    for loop in loops:
        np.testing.assert_allclose(loop[0], loop[-1], atol=1e-8)
    inner = min(loops, key=len)
    assert float(np.mean(np.abs(inner[:, 2]))) > 5.0
    centre = np.array([4.0, 4.0, 10.0])
    assert float(np.min(np.linalg.norm(inner - centre.reshape(1, 3), axis=1))) < 3.0


def test_pipeline_section_box_z_keeps_closed_boundary() -> None:
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=12,
            grid_size_y=10,
            amplitude_sw=8.0,
            wavelength_sw=20.0,
            section_box_enabled=True,
            section_box_size_x=200.0,
            section_box_size_y=200.0,
            section_box_size_z=8.0,
            section_box_center_x=50.0,
            section_box_center_y=50.0,
            section_box_center_z=0.0,
        ),
        verbose=False,
    )
    loops = split_nan_polyline(result.boundary_polyline)
    assert loops
    assert int(result.stats.get("points_out_of_bounds", 0)) > 0
    for loop in loops:
        np.testing.assert_allclose(loop[0], loop[-1], atol=1e-8)
        for p in loop:
            assert abs(float(p[2])) <= 4.0 + 1e-6


def test_custom_boundary_is_closed_and_uses_import_rim() -> None:
    from pathlib import Path

    from shapely.geometry import Point

    from cymatics_geometry.custom_shape import load_and_place_shape

    fixture = Path(__file__).parent / "fixtures" / "l_shape.svg"
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=21,
            grid_size_y=21,
            side_length=100.0,
            shape="custom",
            custom_shape_path=str(fixture),
            custom_shape_size=100.0,
            boundary_curve=True,
        ),
        verbose=False,
    )
    loops = split_nan_polyline(result.boundary_polyline)
    assert loops
    loop = max(loops, key=len)
    assert len(loop) >= 4
    np.testing.assert_allclose(loop[0], loop[-1], atol=1e-5)

    region = load_and_place_shape(str(fixture), size=100.0).region()
    outline = region.boundary
    # Neutral positions of the same lattice points stay on the import edge.
    moved = result.shape_points.reshape(21, 21, 3)
    neutral = result.grid_points.reshape(21, 21, 3)
    for p in loop[:: max(1, len(loop) // 20)]:
        d = np.linalg.norm(moved.reshape(-1, 3) - p.reshape(1, 3), axis=1)
        idx = int(np.argmin(d))
        row, col = divmod(idx, 21)
        nxy = neutral[row, col, :2]
        assert float(outline.distance(Point(float(nxy[0]), float(nxy[1])))) < 16.0
