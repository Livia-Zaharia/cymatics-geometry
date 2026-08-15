"""Cropping, custom 2D shapes, and section-box clipping."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from cymatics_geometry.config import PipelineConfig
from cymatics_geometry.crop import (
    SectionBox,
    clip_polyline_to_section_box,
    crop_points_mask,
    points_inside_section_box,
    polygons_from_rings,
    section_box_corners,
    section_box_from_points,
)
from cymatics_geometry.custom_shape import load_and_place_shape, load_shape_2d
from cymatics_geometry.pipeline import run_pipeline

FIXTURE = Path(__file__).parent / "fixtures" / "l_shape.svg"


def test_section_box_axis_aligned_contains_center() -> None:
    box = SectionBox(
        size_x=10.0,
        size_y=10.0,
        size_z=10.0,
        center_x=0.0,
        center_y=0.0,
        center_z=0.0,
        enabled=True,
    )
    pts = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.9, 0.0, 0.0],
            [5.1, 0.0, 0.0],
            [0.0, 0.0, 6.0],
        ]
    )
    mask = points_inside_section_box(pts, box)
    assert bool(mask[0]) and bool(mask[1])
    assert not bool(mask[2])
    assert not bool(mask[3])


def test_section_box_rotation_moves_corners() -> None:
    box = SectionBox(size_x=20.0, size_y=4.0, size_z=4.0, enabled=True, rot_z=90.0)
    corners = section_box_corners(box)
    # After 90° about Z, the long axis lies along Y
    y_span = float(corners[:, 1].max() - corners[:, 1].min())
    x_span = float(corners[:, 0].max() - corners[:, 0].min())
    assert y_span > 15.0
    assert x_span < 6.0


def test_clip_polyline_to_section_box_cuts_segment() -> None:
    box = SectionBox(
        size_x=10.0,
        size_y=10.0,
        size_z=10.0,
        center_x=0.0,
        center_y=0.0,
        center_z=0.0,
        enabled=True,
    )
    line = np.array([[-20.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
    pieces = clip_polyline_to_section_box(line, box)
    assert len(pieces) == 1
    xs = pieces[0][:, 0]
    assert float(xs.min()) >= -5.01
    assert float(xs.max()) <= 5.01


def test_load_svg_keeps_aspect_ratio_on_uniform_scale() -> None:
    loaded = load_shape_2d(FIXTURE)
    assert loaded.native_width == 100.0
    assert loaded.native_height == 80.0
    assert abs(loaded.aspect_ratio - 1.25) < 1e-9
    placed = load_and_place_shape(FIXTURE, size=50.0)
    assert abs(placed.width - 50.0) < 1e-6
    assert abs(placed.height - 40.0) < 1e-6
    assert abs(placed.aspect_ratio - loaded.aspect_ratio) < 1e-9


def test_polygon_crop_drops_outside_corner() -> None:
    region = polygons_from_rings(
        [np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]])]
    )
    assert region is not None
    pts = np.array(
        [
            [5.0, 5.0, 1.0],
            [12.0, 5.0, 1.0],
            [5.0, -1.0, 0.0],
        ]
    )
    mask = crop_points_mask(pts, region=region)
    assert bool(mask[0])
    assert not bool(mask[1])
    assert not bool(mask[2])


def test_pipeline_custom_shape_crops_square_corners() -> None:
    result = run_pipeline(
        PipelineConfig(
            grid_size_x=21,
            grid_size_y=21,
            side_length=100.0,
            shape="custom",
            custom_shape_path=str(FIXTURE),
            custom_shape_size=100.0,
            line_stride=1,
        ),
        verbose=False,
    )
    assert result.config.shape == "custom"
    assert len(result.custom_outline) > 4
    kept = int(np.count_nonzero(result.inside_mask))
    assert kept < 21 * 21
    assert kept > 50
    # NE corner of the square bbox sits in the missing L notch
    assert result.line_mesh.n_cells >= 1
    assert result.stats["points_kept"] == kept


def test_custom_shape_boundary_follows_silhouette() -> None:
    from shapely.geometry import Point

    from cymatics_geometry.custom_shape import load_and_place_shape
    from cymatics_geometry.lines import split_nan_polyline

    result = run_pipeline(
        PipelineConfig(
            grid_size_x=21,
            grid_size_y=21,
            side_length=100.0,
            shape="custom",
            custom_shape_path=str(FIXTURE),
            custom_shape_size=100.0,
            line_stride=2,
            boundary_curve=True,
        ),
        verbose=False,
    )
    loops = split_nan_polyline(result.boundary_polyline)
    assert loops
    loop = max(loops, key=len)
    assert len(loop) > 8
    np.testing.assert_allclose(loop[0], loop[-1], atol=1e-5)

    region = load_and_place_shape(str(FIXTURE), size=100.0).region()
    outline = region.boundary
    moved = result.shape_points.reshape(21, 21, 3)
    neutral = result.grid_points.reshape(21, 21, 3)
    step = max(1, len(loop) // 24)
    for p in loop[::step]:
        idx = int(np.linalg.norm(moved.reshape(-1, 3) - p.reshape(1, 3), axis=1).argmin())
        row, col = divmod(idx, 21)
        nxy = neutral[row, col, :2]
        dist = float(outline.distance(Point(float(nxy[0]), float(nxy[1]))))
        assert dist < 16.0


def test_pipeline_section_box_clips_lines() -> None:
    full = run_pipeline(
        PipelineConfig(grid_size_x=12, grid_size_y=12, side_length=100.0),
        verbose=False,
    )
    clipped = run_pipeline(
        PipelineConfig(
            grid_size_x=12,
            grid_size_y=12,
            side_length=100.0,
            section_box_enabled=True,
            section_box_size_x=40.0,
            section_box_size_y=40.0,
            section_box_size_z=40.0,
            section_box_center_x=50.0,
            section_box_center_y=50.0,
            section_box_center_z=0.0,
        ),
        verbose=False,
    )
    assert clipped.line_mesh.n_cells > 0
    assert clipped.line_mesh.n_cells <= full.line_mesh.n_cells
    assert int(np.count_nonzero(clipped.inside_mask)) < 12 * 12
    assert len(clipped.section_box_wire) > 8


def test_load_dwg_without_external_converter() -> None:
    """A real DWG is read in-process — no ODA File Converter."""
    from ezdwg import raw

    from cymatics_geometry.custom_shape import load_shape_2d

    tmp = Path(tempfile.mkdtemp())
    dwg = tmp / "circle.dwg"
    raw.write_ac1015_dwg(
        str(dwg),
        [],
        [],
        [(10, 0.0, 0.0, 0.0, 10.0)],
        [],
        [],
        [],
    )
    loaded = load_shape_2d(dwg)
    assert len(loaded.rings) >= 1
    ring = loaded.rings[0]
    radii = np.linalg.norm(ring, axis=1)
    assert float(np.mean(radii)) == pytest.approx(10.0, abs=0.15)
    assert loaded.aspect_ratio == pytest.approx(1.0, abs=0.05)


def test_misnamed_dxf_loads_as_dwg_suffix() -> None:
    """A DXF saved with a .dwg suffix still loads (no converter)."""
    import ezdxf

    from cymatics_geometry.custom_shape import load_shape_2d

    tmp = Path(tempfile.mkdtemp()) / "square.dwg"
    doc = ezdxf.new()
    doc.modelspace().add_lwpolyline(
        [(0, 0), (20, 0), (20, 10), (0, 10)],
        close=True,
    )
    doc.saveas(tmp.with_suffix(".dxf"))
    # Copy DXF bytes to a .dwg name
    dwg = tmp
    dwg.write_bytes(tmp.with_suffix(".dxf").read_bytes())
    loaded = load_shape_2d(dwg)
    assert loaded.native_width == pytest.approx(20.0, abs=1e-6)
    assert loaded.native_height == pytest.approx(10.0, abs=1e-6)


def test_section_box_from_points_encloses_cloud() -> None:
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 4.0, 2.0], [3.0, 8.0, -1.0]])
    box = section_box_from_points(pts, pad=1.0)
    assert np.all(points_inside_section_box(pts, box))
