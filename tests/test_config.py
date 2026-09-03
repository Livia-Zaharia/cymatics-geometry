"""Pipeline config JSON save/load, including older files missing new fields."""

from __future__ import annotations

import json
from pathlib import Path

from cymatics_geometry.config import (
    PipelineConfig,
    load_pipeline_config,
    save_model_params,
    save_pipeline_config,
)
from cymatics_geometry.voxels import VoxelPipeConfig


def test_save_stores_grid_and_line_thinning(tmp_path: Path) -> None:
    cfg = PipelineConfig(
        grid_size_x=48,
        grid_size_y=36,
        line_stride=4,
        boundary_lines_x=-2,
        boundary_lines_y=1,
    )
    path = save_pipeline_config(cfg, tmp_path, allow_duplicates=True)
    assert path is not None
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["grid_size_x"] == 48
    assert raw["grid_size_y"] == 36
    assert raw["line_stride"] == 4
    assert raw["boundary_lines_x"] == -2
    assert raw["boundary_lines_y"] == 1
    loaded = load_pipeline_config(path)
    assert loaded.grid_size_x == 48
    assert loaded.grid_size_y == 36
    assert loaded.line_stride == 4
    assert loaded.boundary_lines_x == -2
    assert loaded.boundary_lines_y == 1


def test_old_config_uses_voxel_line_stride_when_pipeline_field_absent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "grid_size_x": 60,
                "grid_size_y": 80,
                "line_pattern": "grid",
                "voxel": {"line_stride": 2, "boundary_lines": 1},
            }
        ),
        encoding="utf-8",
    )
    loaded = load_pipeline_config(path)
    assert loaded.grid_size_x == 60
    assert loaded.grid_size_y == 80
    assert loaded.line_stride == 2
    assert loaded.boundary_lines_x == 1
    assert loaded.boundary_lines_y == 1


def test_old_config_without_thinning_keys_uses_era_defaults(tmp_path: Path) -> None:
    path = tmp_path / "older.json"
    path.write_text(
        json.dumps({"grid_size": 40, "line_pattern": "grid"}),
        encoding="utf-8",
    )
    loaded = load_pipeline_config(path)
    assert loaded.grid_size_x == 40
    assert loaded.grid_size_y == 40
    assert loaded.line_stride == 1
    assert loaded.boundary_lines_x == 0
    assert loaded.boundary_lines_y == 0
    assert loaded.boundary_curve is True


def test_save_model_params_keeps_pipeline_thinning(tmp_path: Path) -> None:
    path = save_model_params(
        PipelineConfig(grid_size_x=22, grid_size_y=18, line_stride=3),
        VoxelPipeConfig(line_stride=5),
        tmp_path,
        allow_duplicates=True,
    )
    assert path is not None
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["grid_size_x"] == 22
    assert raw["grid_size_y"] == 18
    assert raw["line_stride"] == 3
    assert raw["voxel"]["line_stride"] == 5
    loaded = load_pipeline_config(path)
    assert loaded.line_stride == 3


def test_old_bead_config_fills_five_circle_profile(tmp_path: Path) -> None:
    from cymatics_geometry.shapes import bead_profile_from_sphere

    path = tmp_path / "old_bead.json"
    path.write_text(
        json.dumps(
            {
                "shape": "bead",
                "bead_diameter": 50.0,
                "bead_bottom_radius": 8.0,
                "bead_top_radius": 15.0,
            }
        ),
        encoding="utf-8",
    )
    loaded = load_pipeline_config(path)
    height, radii, stations = bead_profile_from_sphere(50.0, 8.0, 15.0)
    assert abs(loaded.bead_height - height) < 1e-6
    assert abs(loaded.bead_radius_0 - radii[0]) < 1e-6
    assert abs(loaded.bead_radius_2 - radii[2]) < 1e-6
    assert abs(loaded.bead_radius_4 - radii[4]) < 1e-6
    assert abs(loaded.bead_station_1 - stations[0]) < 1e-9
    assert abs(loaded.bead_station_2 - stations[1]) < 1e-9
    assert abs(loaded.bead_station_3 - stations[2]) < 1e-9
