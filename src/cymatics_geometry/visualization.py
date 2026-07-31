"""Visualization helpers for every pipeline stage (notebook + interactive)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pyvista as pv
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection
from plotly.graph_objs import FigureWidget

from cymatics_geometry.config import PipelineConfig, save_model_params
from cymatics_geometry.grid import SOURCE_LABELS, grid_shape
from cymatics_geometry.pipeline import PipelineResult, run_pipeline
from cymatics_geometry.shapes import ShapeParams, shape_boundary_polyline, shape_bounds
from cymatics_geometry.voxels import (
    VOXEL_PARAM_HELP,
    VoxelPipeConfig,
    pipe_and_export_stl,
    preview_pipe_mesh,
)

_SOLID_MESH_LIGHTING: dict[str, float] = {
    "ambient": 0.35,
    "diffuse": 0.95,
    "specular": 0.45,
    "roughness": 0.4,
    "fresnel": 0.15,
}
_SOLID_MESH_LIGHTPOS: dict[str, float] = {"x": 1200, "y": -800, "z": 900}


def _source_arrays(result: PipelineResult) -> tuple[np.ndarray, tuple[str, ...], tuple[bool, ...]]:
    """Return source XYZ, labels, and engaged flags for plotting."""
    sources = getattr(result, "sources", None)
    engaged = set(result.config.engaged_source_labels())
    if sources is None or len(sources) == 0:
        sources = result.corners
        labels: tuple[str, ...] = ("SW", "SE", "NE", "NW")
    else:
        labels = tuple(result.source_labels)
    active = tuple(label in engaged for label in labels)
    return np.asarray(sources, dtype=float), labels, active


def _bounds_from_points(
    points: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    pts = np.asarray(points, dtype=float)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    # Pad Z a little when the field is flat so the camera still has depth
    if abs(float(hi[2] - lo[2])) < 1e-9:
        lo = lo.copy()
        hi = hi.copy()
        lo[2] -= 1.0
        hi[2] += 1.0
    return (float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]), float(lo[2]), float(hi[2]))


def camera_position_from_bounds(
    bounds: tuple[float, float, float, float, float, float],
    target: np.ndarray | list[float] | tuple[float, float, float],
) -> list[list[float]]:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    max_span = max(xmax - xmin, ymax - ymin, zmax - zmin, 1.0)
    target_array = np.asarray(target, dtype=float)
    camera_position = target_array + np.array(
        [1.05 * max_span, -1.45 * max_span, 0.78 * max_span], dtype=float
    )
    return [camera_position.tolist(), target_array.tolist(), [0.0, 0.0, 1.0]]


def show_stage_grid(result: PipelineResult, *, point_size: float = 3.0) -> None:
    """Stage 1 — flat square grid of points."""
    plotter = pv.Plotter(notebook=False)
    plotter.set_background("#1a1a2e")
    plotter.add_points(
        result.grid_points,
        color="#9b8cff",
        point_size=point_size,
        render_points_as_spheres=True,
    )
    plotter.add_axes()
    plotter.add_text("Stage 1 — Flat square grid", font_size=12, color="white")
    bounds = _bounds_from_points(result.grid_points)
    target = np.array(
        [
            0.5 * (bounds[0] + bounds[1]),
            0.5 * (bounds[2] + bounds[3]),
            0.5 * (bounds[4] + bounds[5]),
        ]
    )
    plotter.camera_position = camera_position_from_bounds(bounds, target)
    plotter.show()


def show_stage_corners(result: PipelineResult, *, point_size: float = 3.0) -> None:
    """Stage 2 — grid + wave sources (corners and mid-edges)."""
    sources, labels, active = _source_arrays(result)
    plotter = pv.Plotter(notebook=False)
    plotter.set_background("#1a1a2e")
    plotter.add_points(
        result.grid_points,
        color="#64748b",
        point_size=point_size,
        render_points_as_spheres=True,
        opacity=0.55,
    )
    if np.any(active):
        plotter.add_points(
            sources[np.asarray(active)],
            color="#f59e0b",
            point_size=18,
            render_points_as_spheres=True,
        )
    if np.any(~np.asarray(active)):
        plotter.add_points(
            sources[~np.asarray(active)],
            color="#64748b",
            point_size=12,
            render_points_as_spheres=True,
        )
    plotter.add_point_labels(
        sources,
        list(labels),
        font_size=12,
        text_color="white",
        point_color="#f59e0b",
        point_size=8,
    )
    plotter.add_axes()
    plotter.add_text("Stage 2 — Wave sources", font_size=12, color="white")
    bounds = _bounds_from_points(np.vstack([result.grid_points, sources]))
    target = np.mean(sources, axis=0)
    plotter.camera_position = camera_position_from_bounds(bounds, target)
    plotter.show()


def show_stage_field_heatmap(result: PipelineResult, *, figsize: tuple[int, int] = (8, 7)) -> None:
    """Stage 3 — top-down interference field heatmap (matplotlib)."""
    nx, ny = grid_shape(result.config)
    field = result.displacement.reshape(ny, nx)
    sources, labels, active = _source_arrays(result)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        field,
        origin="lower",
        extent=[0, result.config.side_length, 0, result.config.side_length],
        cmap="coolwarm",
        aspect="equal",
    )
    colors = ["#f59e0b" if on else "#94a3b8" for on in active]
    ax.scatter(
        sources[:, 0],
        sources[:, 1],
        c=colors,
        s=80,
        edgecolors="black",
        zorder=5,
    )
    amp_map = dict(zip(labels, result.config.amplitudes))
    for label, source, on in zip(labels, sources, active):
        tag = "on" if on else "off"
        ax.annotate(
            f"{label} [{tag}]\nA={amp_map[label]:.2f}",
            (source[0], source[1]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            color="#111827",
        )
    ax.set_title("Stage 3 — Wave interference field")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    fig.colorbar(im, ax=ax, fraction=0.046, label="displacement Z")
    fig.tight_layout()
    plt.show()


def show_stage_displaced_points(
    result: PipelineResult,
    *,
    point_size: float = 3.0,
) -> None:
    """Stage 4 — points moved in space by the interference field."""
    plotter = pv.Plotter(notebook=False)
    plotter.set_background("#1a1a2e")
    cloud = pv.PolyData(result.displaced_points)
    cloud["displacement"] = result.displacement
    plotter.add_mesh(
        cloud,
        scalars="displacement",
        cmap="coolwarm",
        point_size=point_size,
        render_points_as_spheres=True,
        show_scalar_bar=True,
    )
    sources, _, active = _source_arrays(result)
    if np.any(active):
        plotter.add_points(
            sources[np.asarray(active)],
            color="#f59e0b",
            point_size=16,
            render_points_as_spheres=True,
        )
    plotter.add_axes()
    plotter.add_text("Stage 4 — Displaced points", font_size=12, color="white")
    bounds = _bounds_from_points(result.displaced_points)
    target = np.mean(result.displaced_points, axis=0)
    plotter.camera_position = camera_position_from_bounds(bounds, target)
    plotter.show()


def show_stage_line(
    result: PipelineResult,
    *,
    line_width: float = 1.5,
    color: str = "#ffffff",
    line_color_mode: str = "difference",
) -> None:
    """Stage 5 — reconnected line geometry on the mapped shape.

    ``line_color_mode``:
      - ``difference`` — colour by point Z (coolwarm)
      - ``white`` — solid ``color`` (default white)
    """
    plotter = pv.Plotter(notebook=False)
    plotter.set_background("#1a1a2e")
    mesh = result.line_mesh
    mode = str(line_color_mode).lower().strip()
    if mode == "white":
        plotter.add_mesh(
            mesh,
            color=color,
            line_width=line_width,
            render_lines_as_tubes=True,
        )
    else:
        coloured = mesh.copy(deep=True)
        coloured["z"] = np.asarray(coloured.points, dtype=float)[:, 2]
        plotter.add_mesh(
            coloured,
            scalars="z",
            cmap="coolwarm",
            line_width=line_width,
            render_lines_as_tubes=True,
            show_scalar_bar=True,
        )
    display_sources = np.asarray(result.shape_sources, dtype=float)
    if len(display_sources) == 0:
        display_sources, _, active = _source_arrays(result)
    else:
        _, _, active = _source_arrays(result)
    if np.any(active):
        plotter.add_points(
            display_sources[np.asarray(active)],
            color="#f59e0b",
            point_size=14,
            render_points_as_spheres=True,
        )
    plotter.add_axes()
    title = f"Stage 5 — {result.config.shape} lines ({mode})"
    plotter.add_text(title, font_size=12, color="white")
    poly = np.asarray(result.polyline, dtype=float)
    finite = poly[np.isfinite(poly).all(axis=1)]
    bounds_pts = finite if len(finite) else np.asarray(result.shape_points, dtype=float)
    bounds = _bounds_from_points(bounds_pts)
    target = np.mean(bounds_pts, axis=0) if len(bounds_pts) else np.zeros(3)
    plotter.camera_position = camera_position_from_bounds(bounds, target)
    plotter.show()


def show_all_stages_matplotlib(result: PipelineResult, *, figsize: tuple[int, int] = (14, 10)) -> None:
    """Notebook-friendly multi-panel overview of every stage (no native VTK window)."""
    nx, ny = grid_shape(result.config)
    field = result.displacement.reshape(ny, nx)
    sources, labels, active = _source_arrays(result)
    colors = ["#f59e0b" if on else "#94a3b8" for on in active]
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(result.grid_points[:, 0], result.grid_points[:, 1], s=1, c="#6366f1")
    ax1.scatter(sources[:, 0], sources[:, 1], c=colors, s=60, edgecolors="k", zorder=5)
    for label, source in zip(labels, sources):
        ax1.annotate(label, (source[0], source[1]), xytext=(4, 4), textcoords="offset points")
    ax1.set_aspect("equal")
    ax1.set_title("1–2  Grid + sources")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax2 = fig.add_subplot(2, 2, 2)
    im = ax2.imshow(
        field,
        origin="lower",
        extent=[0, result.config.side_length, 0, result.config.side_length],
        cmap="coolwarm",
        aspect="equal",
    )
    ax2.scatter(sources[:, 0], sources[:, 1], c=colors, s=50, edgecolors="k")
    ax2.set_title("3  Interference field")
    fig.colorbar(im, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    step_x = max(1, nx // 40)
    step_y = max(1, ny // 40)
    pts = result.displaced_points.reshape(ny, nx, 3)[::step_y, ::step_x].reshape(-1, 3)
    disp = result.displacement.reshape(ny, nx)[::step_y, ::step_x].ravel()
    ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=disp, cmap="coolwarm", s=6)
    ax3.set_title("4  Displaced points (+ surface release)")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    line = np.asarray(result.polyline, dtype=float)
    # Matplotlib breaks the stroke on NaNs — perfect for grid segments
    ax4.plot(line[:, 0], line[:, 1], line[:, 2], color="#7c3aed", linewidth=0.6)
    finite = np.isfinite(line).all(axis=1)
    pts = line[finite]
    if len(pts):
        step = max(1, len(pts) // 200)
        norm = Normalize(vmin=pts[:, 2].min(), vmax=pts[:, 2].max())
        ax4.scatter(
            pts[::step, 0],
            pts[::step, 1],
            pts[::step, 2],
            c=pts[::step, 2],
            cmap="coolwarm",
            s=4,
            norm=norm,
        )
    dirs = []
    if result.config.lines_x:
        dirs.append("X")
    if result.config.lines_y:
        dirs.append("Y")
    ax4.set_title(f"5  Grid lines ({'+'.join(dirs) or 'none'})")
    ax4.set_xlabel("X")
    ax4.set_ylabel("Y")
    ax4.set_zlabel("Z")

    active_txt = ",".join(result.config.active_source_labels()) or "none"
    fig.suptitle(
        "Cymatics geometry stages — "
        f"active=[{active_txt}]  λ={result.config.wavelength:.1f}  "
        f"tension={result.config.boundary_tension:.2f}",
        fontsize=12,
    )
    fig.tight_layout()
    plt.show()


def save_line_screenshot(
    result: PipelineResult,
    path: str | Path,
    *,
    title: str = "Cymatics line",
    window_size: tuple[int, int] = (1300, 950),
    line_color_mode: str = "difference",
    color: str = "#ffffff",
) -> Path:
    """Render the final line mesh to a PNG screenshot."""
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("#1a1a2e")
    mesh = result.line_mesh
    mode = str(line_color_mode).lower().strip()
    if mode == "white":
        plotter.add_mesh(
            mesh,
            color=color,
            line_width=2.0,
            render_lines_as_tubes=True,
        )
    else:
        coloured = mesh.copy(deep=True)
        coloured["z"] = np.asarray(coloured.points, dtype=float)[:, 2]
        plotter.add_mesh(
            coloured,
            scalars="z",
            cmap="coolwarm",
            line_width=2.0,
            render_lines_as_tubes=True,
            show_scalar_bar=True,
        )
    display_sources = np.asarray(result.shape_sources, dtype=float)
    if len(display_sources) == 0:
        sources, _, active = _source_arrays(result)
        display_sources = sources
    else:
        _, _, active = _source_arrays(result)
    if np.any(active):
        plotter.add_points(
            display_sources[np.asarray(active)],
            color="#f59e0b",
            point_size=14,
            render_points_as_spheres=True,
        )
    plotter.add_axes()
    plotter.add_text(title, position="upper_left", font_size=12, color="white")
    poly = np.asarray(result.polyline, dtype=float)
    finite = poly[np.isfinite(poly).all(axis=1)]
    bounds_pts = finite if len(finite) else np.asarray(result.shape_points, dtype=float)
    bounds = _bounds_from_points(bounds_pts)
    target = np.mean(bounds_pts, axis=0) if len(bounds_pts) else np.zeros(3)
    plotter.camera_position = camera_position_from_bounds(bounds, target)
    image_path = Path(tempfile.gettempdir()) / f"{uuid4().hex}.png"
    try:
        plotter.screenshot(str(image_path))
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(image_path.read_bytes())
        return out
    finally:
        plotter.close()
        image_path.unlink(missing_ok=True)


def _subsample_indices(n: int, max_vertices: int) -> np.ndarray:
    if n <= max_vertices:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, max_vertices, dtype=int)


def _subsample_polyline(polyline: np.ndarray, max_vertices: int) -> np.ndarray:
    """Keep interactive redraws light; preserve NaN segment breaks for grid lines."""
    pts = np.asarray(polyline, dtype=float)
    if len(pts) == 0:
        return pts
    if not np.isnan(pts).any():
        return pts[_subsample_indices(len(pts), max_vertices)]

    # Split on NaN rows, subsample each segment, rejoin with NaN breaks
    chunks: list[np.ndarray] = []
    start = 0
    finite = np.isfinite(pts).all(axis=1)
    for i in range(len(pts) + 1):
        at_break = i == len(pts) or not finite[i]
        if at_break:
            seg = pts[start:i]
            if len(seg) > 0:
                # Budget vertices roughly evenly across segments later; cap per seg
                cap = max(8, max_vertices // 8)
                chunks.append(seg[_subsample_indices(len(seg), min(len(seg), cap))])
            start = i + 1
    if not chunks:
        return np.zeros((0, 3), dtype=float)
    out: list[np.ndarray] = []
    for i, seg in enumerate(chunks):
        out.append(seg)
        if i + 1 < len(chunks):
            out.append(np.full((1, 3), np.nan))
    joined = np.vstack(out)
    if len(joined) > max_vertices * 2:
        # Hard fallback if too many segments
        return joined[_subsample_indices(len(joined), max_vertices)]
    return joined


def _plotly_json_float(value: float) -> float | None:
    """JSON-safe float for FigureWidget / jupyter-client (NaN/Inf → None)."""
    v = float(value)
    if not np.isfinite(v):
        return None
    return v


def _plotly_xyz(points: np.ndarray) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """FigureWidget XYZ lists — NaN segment breaks become None (JSON null)."""
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return [], [], []
    x = [_plotly_json_float(v) for v in pts[:, 0]]
    y = [_plotly_json_float(v) for v in pts[:, 1]]
    z = [_plotly_json_float(v) for v in pts[:, 2]]
    return x, y, z


def _plotly_floats(values: np.ndarray, *, fill: float = 0.0) -> list[float]:
    """Finite float list for Plotly scalar colors (None is invalid on line.color)."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    out: list[float] = []
    fill_f = float(fill)
    for v in arr:
        fv = float(v)
        out.append(fill_f if not np.isfinite(fv) else fv)
    return out


def _source_marker_colors(active: tuple[bool, ...] | list[bool]) -> list[str]:
    return ["#f59e0b" if on else "#64748b" for on in active]


def _fixed_display_scales(
    side_length: float,
    *,
    z_display_max: float,
) -> tuple[float, float, float]:
    """Absolute scene limits — never autoscale to the current frame.

    Slider amp maps 1:1 into Z. Keeping ±z_display_max fixed means amp=1 is
    always 1/z_display_max of full height/color, not remapped to "full red".
    XY pad is generous so release shockwaves leaving the square stay in frame.
    """
    s = float(side_length)
    z_lim = max(float(z_display_max), 1e-6)
    # Tight pad so the square fills the view (still room for modest release)
    xy_pad = 0.28 * s
    travel_max = 150.0
    return z_lim, xy_pad, travel_max


def _shape_params_from_result(result: PipelineResult) -> ShapeParams:
    cfg = result.config
    return ShapeParams(
        kind=str(cfg.shape).lower(),  # type: ignore[arg-type]
        cylinder_diameter=float(cfg.cylinder_diameter),
        cylinder_length=float(cfg.cylinder_length),
        cone_height=float(cfg.cone_height),
        cone_base_radius=float(cfg.cone_base_radius),
        frustum_height=float(cfg.frustum_height),
        frustum_base_diameter=float(cfg.frustum_base_diameter),
        frustum_top_diameter=float(cfg.frustum_top_diameter),
    )


def _scene_ranges_for_shape(
    result: PipelineResult,
    *,
    z_display_max: float,
) -> tuple[list[float], list[float], list[float], dict[str, float]]:
    """Return (x_range, y_range, z_range, aspectratio) for the current shape."""
    z_lim = max(float(z_display_max), 1e-6)
    travel_max = 150.0
    kind = str(result.config.shape).lower()
    if kind == "plane":
        s = float(result.config.side_length)
        xy_pad = 0.28 * s
        return (
            [-xy_pad, s + xy_pad],
            [-xy_pad, s + xy_pad],
            [-z_lim, z_lim],
            {"x": 1.0, "y": 1.0, "z": 0.45},
        )

    params = _shape_params_from_result(result)
    xmin, xmax, ymin, ymax, zmin, zmax = shape_bounds(
        params, side_length=float(result.config.side_length)
    )
    # Room for normal / tangential wave offsets (amp maps 1:1 into local frame)
    pad = z_lim + 0.15 * max(
        xmax - xmin, ymax - ymin, zmax - zmin, float(result.config.side_length), 1.0
    )
    x_range = [xmin - pad, xmax + pad]
    y_range = [ymin - pad, ymax + pad]
    z_range = [zmin - pad, zmax + pad]
    spans = [
        max(x_range[1] - x_range[0], 1e-6),
        max(y_range[1] - y_range[0], 1e-6),
        max(z_range[1] - z_range[0], 1e-6),
    ]
    m = max(spans)
    aspect = {"x": spans[0] / m, "y": spans[1] / m, "z": spans[2] / m}
    _ = travel_max  # kept for callers that still read fixed travel scale
    return x_range, y_range, z_range, aspect


def _ghost_footprint(result: PipelineResult) -> np.ndarray:
    """Undeformed shape outline for the dashed guide stroke."""
    params = _shape_params_from_result(result)
    return shape_boundary_polyline(
        params, side_length=float(result.config.side_length), samples=72
    )


def _line_color_style(
    z_values: np.ndarray,
    *,
    mode: str,
    z_display_max: float,
) -> dict:
    """Plotly Scatter3d ``line`` kwargs for difference coloring or solid white."""
    z_lim = max(float(z_display_max), 1e-6)
    if mode == "white":
        return {"width": 2, "color": "#ffffff"}
    return {
        "width": 2,
        "color": _plotly_floats(z_values),
        "colorscale": "RdBu",
        "cmin": -z_lim,
        "cmax": z_lim,
        "colorbar": {
            "title": "mapped Z",
            "thickness": 12,
            "len": 0.35,
            "y": 0.72,
        },
    }


def _set_line_trace_style(fig: FigureWidget, line_style: dict, *, mode: str) -> None:
    """Apply line color mode onto scatter trace index 2."""
    fig.data[2].line.width = line_style["width"]
    fig.data[2].line.color = line_style["color"]
    if mode == "white":
        return
    fig.data[2].line.colorscale = line_style["colorscale"]
    fig.data[2].line.cmin = line_style["cmin"]
    fig.data[2].line.cmax = line_style["cmax"]
    fig.data[2].line.colorbar = line_style["colorbar"]


def _line_figure_widget(
    result: PipelineResult,
    *,
    z_display_max: float = 10.0,
    max_vertices: int = 4000,
    line_color_mode: str = "difference",
) -> FigureWidget:
    """Persistent Plotly 3D scene with fixed absolute color units."""
    sources, labels, active = _source_arrays(result)
    # Display the mapped shape (plane identity when shape=plane)
    display_pts = np.asarray(result.shape_points, dtype=float)
    if len(display_pts) == 0:
        display_pts = np.asarray(result.displaced_points, dtype=float)
    display_sources = np.asarray(result.shape_sources, dtype=float)
    if len(display_sources) == 0:
        display_sources = sources

    polyline = np.asarray(result.polyline, dtype=float)
    line = _subsample_polyline(polyline, max_vertices)
    z = line[:, 2]
    travel_max = 150.0

    ghost = _ghost_footprint(result)
    ghost_x, ghost_y, ghost_z = _plotly_xyz(ghost)

    if len(display_pts) > 0:
        idx = _subsample_indices(len(display_pts), min(1200, max_vertices))
        cloud = display_pts[idx]
        base = np.asarray(result.shape_base_points, dtype=float)
        if len(base) == len(display_pts):
            travel = np.linalg.norm(cloud - base[idx], axis=1)
        else:
            offs = np.asarray(result.plane_offsets, dtype=float)
            travel = (
                np.linalg.norm(offs[idx], axis=1)
                if len(offs) == len(display_pts)
                else np.abs(cloud[:, 2])
            )
    else:
        cloud = line
        travel = np.abs(z)

    cloud_x, cloud_y, cloud_z = _plotly_xyz(cloud)
    line_x, line_y, line_z = _plotly_xyz(line)
    src_x, src_y, src_z = _plotly_xyz(display_sources)
    travel_list = _plotly_floats(travel)
    line_style = _line_color_style(
        z, mode=str(line_color_mode), z_display_max=z_display_max
    )
    z_lim = max(float(z_display_max), 1e-6)

    x_range, y_range, z_range, aspect = _scene_ranges_for_shape(
        result, z_display_max=z_display_max
    )

    fig = FigureWidget(
        data=[
            go.Scatter3d(
                x=ghost_x,
                y=ghost_y,
                z=ghost_z,
                mode="lines",
                name="undeformed shape",
                line={"width": 4, "color": "#64748b", "dash": "dash"},
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=cloud_x,
                y=cloud_y,
                z=cloud_z,
                mode="markers",
                name="points",
                marker={
                    "size": 2.5,
                    "color": travel_list,
                    "colorscale": "Viridis",
                    "cmin": 0.0,
                    "cmax": travel_max,
                    "opacity": 0.85,
                    "colorbar": {
                        "title": "|offset|",
                        "thickness": 12,
                        "len": 0.35,
                        "y": 0.22,
                    },
                },
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=line_x,
                y=line_y,
                z=line_z,
                mode="lines",
                name="line",
                line=line_style,
                hoverinfo="skip",
            ),
            go.Scatter3d(
                x=src_x,
                y=src_y,
                z=src_z,
                mode="markers+text",
                name="sources",
                text=list(labels),
                textposition="top center",
                marker={"size": 7, "color": _source_marker_colors(active)},
                hoverinfo="text",
            ),
        ]
    )
    kind = str(result.config.shape)
    color_label = (
        "white"
        if line_color_mode == "white"
        else f"difference ∈ [−{z_lim:g}, {z_lim:g}]"
    )
    fig.update_layout(
        height=900,
        width=900,
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        paper_bgcolor="#111827",
        font={"color": "#e5e7eb", "size": 11},
        title={
            "text": f"Shape · {kind} · lines {color_label}",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 11},
        },
        scene={
            "xaxis": {
                "title": "X",
                "range": x_range,
                "autorange": False,
                "backgroundcolor": "#1f2937",
                "gridcolor": "#374151",
            },
            "yaxis": {
                "title": "Y",
                "range": y_range,
                "autorange": False,
                "backgroundcolor": "#1f2937",
                "gridcolor": "#374151",
            },
            "zaxis": {
                "title": "Z",
                "range": z_range,
                "autorange": False,
                "backgroundcolor": "#1f2937",
                "gridcolor": "#374151",
            },
            "aspectmode": "manual",
            "aspectratio": aspect,
            "bgcolor": "#111827",
            "camera": {"eye": {"x": 1.05, "y": -1.15, "z": 0.85}},
        },
        showlegend=False,
        uirevision="cymatics-line",
    )
    return fig


def _apply_result_to_figure(
    fig: FigureWidget,
    result: PipelineResult,
    *,
    z_display_max: float = 10.0,
    max_vertices: int = 4000,
    line_color_mode: str = "difference",
) -> None:
    """Mutate an existing FigureWidget in place — camera preserved via uirevision."""
    line = _subsample_polyline(result.polyline, max_vertices)
    if len(line) == 0:
        line = np.full((1, 3), np.nan, dtype=float)
    z = line[:, 2]
    sources, labels, active = _source_arrays(result)
    display_sources = np.asarray(result.shape_sources, dtype=float)
    if len(display_sources) == 0:
        display_sources = sources

    display_pts = np.asarray(result.shape_points, dtype=float)
    if len(display_pts) == 0:
        display_pts = np.asarray(result.displaced_points, dtype=float)
    base = np.asarray(result.shape_base_points, dtype=float)
    idx = _subsample_indices(len(display_pts), min(1200, max_vertices))
    cloud = display_pts[idx]
    if len(base) == len(display_pts):
        travel = np.linalg.norm(cloud - base[idx], axis=1)
    else:
        travel = np.abs(cloud[:, 2])

    z_lim = max(float(z_display_max), 1e-6)
    travel_max = 150.0
    lx = "X" if result.config.lines_x else ""
    ly = "Y" if result.config.lines_y else ""
    dirs = f"{lx}+{ly}".strip("+") or "none"
    kind = str(result.config.shape)
    line_style = _line_color_style(
        z, mode=str(line_color_mode), z_display_max=z_display_max
    )

    ghost = _ghost_footprint(result)
    ghost_x, ghost_y, ghost_z = _plotly_xyz(ghost)
    cloud_x, cloud_y, cloud_z = _plotly_xyz(cloud)
    line_x, line_y, line_z = _plotly_xyz(line)
    src_x, src_y, src_z = _plotly_xyz(display_sources)
    x_range, y_range, z_range, aspect = _scene_ranges_for_shape(
        result, z_display_max=z_display_max
    )
    color_label = (
        "white"
        if line_color_mode == "white"
        else f"difference ∈ [−{z_lim:g}, {z_lim:g}]"
    )

    with fig.batch_update():
        # 0 = undeformed outline, 1 = points, 2 = line, 3 = sources
        fig.data[0].x = ghost_x
        fig.data[0].y = ghost_y
        fig.data[0].z = ghost_z
        fig.data[1].x = cloud_x
        fig.data[1].y = cloud_y
        fig.data[1].z = cloud_z
        fig.data[1].marker.color = _plotly_floats(travel)
        fig.data[1].marker.cmin = 0.0
        fig.data[1].marker.cmax = travel_max
        fig.data[2].x = line_x
        fig.data[2].y = line_y
        fig.data[2].z = line_z
        _set_line_trace_style(fig, line_style, mode=str(line_color_mode))
        fig.data[3].x = src_x
        fig.data[3].y = src_y
        fig.data[3].z = src_z
        fig.data[3].text = list(labels)
        fig.data[3].marker.color = _source_marker_colors(active)
        fig.layout.scene.xaxis.range = x_range
        fig.layout.scene.yaxis.range = y_range
        fig.layout.scene.zaxis.range = z_range
        fig.layout.scene.aspectratio = aspect
        fig.layout.title.text = (
            f"Shape · {kind} · grid {dirs} · lines {color_label}"
        )


_SOURCE_ROLES: dict[str, str] = {
    "SW": "corner · south-west",
    "SE": "corner · south-east",
    "NE": "corner · north-east",
    "NW": "corner · north-west",
    "S": "mid-edge · south",
    "E": "mid-edge · east",
    "N": "mid-edge · north",
    "W": "mid-edge · west",
}


def show_interactive_line_viewer(
    config: PipelineConfig | None = None,
    *,
    max_vertices: int = 4000,
    amplitude_max: float = 10.0,
):
    """Notebook UI: all 8 sources + globals in one scrollable panel, view on the left."""
    from IPython.display import display
    from ipywidgets import (
        Button,
        Checkbox,
        Dropdown,
        FloatSlider,
        HTML,
        HBox,
        IntSlider,
        Layout,
        VBox,
    )

    base = config or PipelineConfig(grid_size_x=60, grid_size_y=60)

    src_style = {"description_width": "58px"}
    global_style = {"description_width": "72px"}
    full_slider = Layout(width="96%", height="32px")

    source_widgets: dict[str, dict] = {}
    for label in SOURCE_LABELS:
        key = label.lower()
        role = _SOURCE_ROLES[label]
        link_cb = Checkbox(
            value=False,
            description="sync",
            indent=False,
            layout=Layout(width="72px", height="28px"),
            tooltip="Share amp / λ / release with other synced sources",
        )
        amp = FloatSlider(
            value=0.0,
            min=0.0,
            max=float(amplitude_max),
            step=0.01,
            description="amp",
            continuous_update=True,
            readout=True,
            readout_format=".2f",
            style=src_style,
            layout=full_slider,
        )
        wl = FloatSlider(
            value=0.0,
            min=0.0,
            max=80.0,
            step=0.5,
            description="λ",
            continuous_update=True,
            readout=True,
            readout_format=".1f",
            style=src_style,
            layout=full_slider,
        )
        rel = FloatSlider(
            value=0.0,
            min=0.0,
            max=150.0,
            step=0.1,
            description="release",
            continuous_update=True,
            readout=True,
            readout_format=".1f",
            style=src_style,
            layout=full_slider,
        )
        header = HBox(
            [
                HTML(
                    "<div style='flex:1;line-height:1.2'>"
                    f"<div style='font-size:20px;font-weight:800;"
                    f"letter-spacing:0.02em'>{label}</div>"
                    f"<div style='font-size:11px;color:#6b7280'>{role}</div>"
                    "</div>"
                ),
                link_cb,
            ],
            layout=Layout(
                width="100%",
                align_items="center",
                margin="0 0 4px 0",
            ),
        )
        block = VBox(
            [header, amp, wl, rel],
            layout=Layout(
                width="100%",
                border="1px solid #d1d5db",
                padding="8px 8px 6px 8px",
                margin="0 0 10px 0",
            ),
        )
        source_widgets[key] = {
            "label": label,
            "link": link_cb,
            "amp": amp,
            "wavelength": wl,
            "release": rel,
            "block": block,
        }

    time_sl = FloatSlider(
        value=0.0,
        min=0.0,
        max=6.28,
        step=0.05,
        description="time",
        style=global_style,
        layout=full_slider,
        readout_format=".2f",
    )
    decay_sl = FloatSlider(
        value=0.0,
        min=0.0,
        max=0.05,
        step=0.001,
        description="decay",
        readout_format=".3f",
        style=global_style,
        layout=full_slider,
    )
    cloth_sl = FloatSlider(
        value=0.0,
        min=0.0,
        max=100.0,
        step=0.1,
        description="cloth",
        continuous_update=True,
        readout_format=".1f",
        style=global_style,
        layout=full_slider,
    )
    grid_x = IntSlider(
        value=min(int(base.grid_size_x), 80),
        min=20,
        max=100,
        step=10,
        description="grid X",
        style=global_style,
        layout=full_slider,
    )
    grid_y = IntSlider(
        value=min(int(base.grid_size_y), 80),
        min=20,
        max=100,
        step=10,
        description="grid Y",
        style=global_style,
        layout=full_slider,
    )
    # Checkboxes are reliable; ToggleButtons(bool) often fail to fire in notebooks
    lines_x_cb = Checkbox(
        value=True,
        description="X lines (rows)",
        indent=False,
        layout=Layout(width="160px"),
    )
    lines_y_cb = Checkbox(
        value=True,
        description="Y lines (cols)",
        indent=False,
        layout=Layout(width="160px"),
    )
    line_color_dd = Dropdown(
        options=[
            ("difference (by Z)", "difference"),
            ("uniform white", "white"),
        ],
        value="difference",
        description="line color",
        style=global_style,
        layout=Layout(width="96%"),
    )
    shape_dd = Dropdown(
        options=[
            ("plane", "plane"),
            ("cylinder", "cylinder"),
            ("cone", "cone"),
            ("frustum (truncated cone)", "frustum"),
        ],
        value=str(getattr(base, "shape", "plane") or "plane"),
        description="shape",
        style=global_style,
        layout=Layout(width="96%"),
    )
    cyl_diam = FloatSlider(
        value=float(base.cylinder_diameter),
        min=5.0,
        max=120.0,
        step=1.0,
        description="cyl Ø",
        readout_format=".0f",
        style=global_style,
        layout=full_slider,
    )
    cyl_len = FloatSlider(
        value=float(base.cylinder_length),
        min=10.0,
        max=200.0,
        step=1.0,
        description="cyl len",
        readout_format=".0f",
        style=global_style,
        layout=full_slider,
    )
    cone_h = FloatSlider(
        value=float(base.cone_height),
        min=10.0,
        max=200.0,
        step=1.0,
        description="cone H",
        readout_format=".0f",
        style=global_style,
        layout=full_slider,
    )
    cone_r = FloatSlider(
        value=float(base.cone_base_radius),
        min=1.0,
        max=80.0,
        step=0.5,
        description="cone R",
        readout_format=".1f",
        style=global_style,
        layout=full_slider,
    )
    fr_h = FloatSlider(
        value=float(base.frustum_height),
        min=10.0,
        max=200.0,
        step=1.0,
        description="frust H",
        readout_format=".0f",
        style=global_style,
        layout=full_slider,
    )
    fr_base = FloatSlider(
        value=float(base.frustum_base_diameter),
        min=1.0,
        max=160.0,
        step=1.0,
        description="base Ø",
        readout_format=".0f",
        style=global_style,
        layout=full_slider,
    )
    fr_top = FloatSlider(
        value=float(base.frustum_top_diameter),
        min=0.0,
        max=160.0,
        step=1.0,
        description="top Ø",
        readout_format=".0f",
        style=global_style,
        layout=full_slider,
    )
    # --- Voxel pipe (3D-printable solid) ---
    voxel_size_sl = FloatSlider(
        value=0.8,
        min=0.2,
        max=2.5,
        step=0.05,
        description="voxel",
        readout_format=".2f",
        style=global_style,
        layout=full_slider,
        tooltip=VOXEL_PARAM_HELP["voxel_size"],
    )
    pipe_radius_sl = FloatSlider(
        value=1.2,
        min=0.2,
        max=6.0,
        step=0.1,
        description="radius",
        readout_format=".1f",
        style=global_style,
        layout=full_slider,
        tooltip=VOXEL_PARAM_HELP["pipe_radius"],
    )
    inner_radius_sl = FloatSlider(
        value=0.0,
        min=0.0,
        max=4.0,
        step=0.1,
        description="inner r",
        readout_format=".1f",
        style=global_style,
        layout=full_slider,
        tooltip=VOXEL_PARAM_HELP["inner_radius"],
    )
    mod_amp_sl = FloatSlider(
        value=0.0,
        min=0.0,
        max=2.0,
        step=0.05,
        description="mod amp",
        readout_format=".2f",
        style=global_style,
        layout=full_slider,
        tooltip=VOXEL_PARAM_HELP["modulation_amp"],
    )
    mod_freq_sl = FloatSlider(
        value=2.0,
        min=0.0,
        max=12.0,
        step=0.5,
        description="mod freq",
        readout_format=".1f",
        style=global_style,
        layout=full_slider,
        tooltip=VOXEL_PARAM_HELP["modulation_freq"],
    )
    mod_lobes_sl = IntSlider(
        value=0,
        min=0,
        max=12,
        step=1,
        description="lobes",
        style=global_style,
        layout=full_slider,
        tooltip=VOXEL_PARAM_HELP["modulation_lobes"],
    )
    line_stride_sl = IntSlider(
        value=2,
        min=1,
        max=10,
        step=1,
        description="line step",
        style=global_style,
        layout=full_slider,
        tooltip=VOXEL_PARAM_HELP["line_stride"],
    )
    point_stride_sl = IntSlider(
        value=2,
        min=1,
        max=10,
        step=1,
        description="pt step",
        style=global_style,
        layout=full_slider,
        tooltip=VOXEL_PARAM_HELP["point_stride"],
    )
    spine_samples_sl = IntSlider(
        value=40,
        min=8,
        max=120,
        step=4,
        description="spine n",
        style=global_style,
        layout=full_slider,
        tooltip=VOXEL_PARAM_HELP["spine_samples"],
    )
    voxel_help = HTML(
        value=(
            "<div style='font-size:11px;color:#4b5563;line-height:1.35;margin:4px 0 8px 0'>"
            "<b>Preview solid</b> — opaque shaded tubes in the 3D view "
            "(hides the wireframe).<br>"
            "<b>Export STL</b> — real PicoPie voxels + file write (slower).<br>"
            "<b>Save params</b> — write pipeline + voxel settings to "
            "<code>configs/</code>.<br>"
            "<b>voxel</b> — cell size for STL (smaller = smoother/slower).<br>"
            "<b>radius</b> — outer tube thickness along each line.<br>"
            "<b>inner r</b> — hollow bore for STL (0 = solid rod).<br>"
            "<b>mod amp / freq / lobes</b> — STL surface ripples.<br>"
            "<b>line/pt step</b> — skip lines/points for a lighter lattice.<br>"
            "<b>spine n</b> — cubic-spline samples along each bend."
            "</div>"
        ),
        layout=Layout(width="98%"),
    )
    voxel_status = HTML(value="", layout=Layout(width="98%"))
    shape_param_box = VBox(
        [],
        layout=Layout(width="100%", margin="0 0 6px 0"),
    )
    status = HTML(value="", layout=Layout(width="98%"))

    def _sync_shape_param_visibility(_change: object | None = None) -> None:
        kind = str(shape_dd.value)
        if kind == "cylinder":
            shape_param_box.children = (cyl_diam, cyl_len)
        elif kind == "cone":
            shape_param_box.children = (cone_h, cone_r)
        elif kind == "frustum":
            shape_param_box.children = (fr_h, fr_base, fr_top)
        else:
            shape_param_box.children = ()

    _sync_shape_param_visibility()

    _loading = {"on": False}

    def _linked_keys() -> list[str]:
        # Preserve SOURCE_LABELS order so "first" is well-defined
        return [
            label.lower()
            for label in SOURCE_LABELS
            if bool(source_widgets[label.lower()]["link"].value)
        ]

    def _read_source(key: str) -> dict[str, float]:
        w = source_widgets[key]
        return {
            "amp": float(w["amp"].value),
            "wavelength": float(w["wavelength"].value),
            "release": float(w["release"].value),
        }

    def _write_source(key: str, values: dict[str, float]) -> None:
        w = source_widgets[key]
        w["amp"].value = float(values["amp"])
        w["wavelength"].value = float(values["wavelength"])
        w["release"].value = float(values["release"])

    def _config_from_state() -> PipelineConfig:
        kwargs: dict = {
            "grid_size_x": int(grid_x.value),
            "grid_size_y": int(grid_y.value),
            "side_length": base.side_length,
            "frequency": base.frequency,
            "time": float(time_sl.value),
            "decay": float(decay_sl.value),
            "cloth": float(cloth_sl.value),
            "boundary_tension": 0.0,
            "release_pace": 0.0,
            "line_pattern": "grid",
            "lines_x": bool(lines_x_cb.value),
            "lines_y": bool(lines_y_cb.value),
            "shape": str(shape_dd.value),
            "cylinder_diameter": float(cyl_diam.value),
            "cylinder_length": float(cyl_len.value),
            "cone_height": float(cone_h.value),
            "cone_base_radius": float(cone_r.value),
            "frustum_height": float(fr_h.value),
            "frustum_base_diameter": float(fr_base.value),
            "frustum_top_diameter": float(fr_top.value),
        }
        for key, w in source_widgets.items():
            kwargs[f"active_{key}"] = True
            kwargs[f"amplitude_{key}"] = float(w["amp"].value)
            kwargs[f"wavelength_{key}"] = float(w["wavelength"].value)
            kwargs[f"release_{key}"] = float(w["release"].value)
        kwargs["wavelength"] = float(source_widgets["sw"]["wavelength"].value)
        return PipelineConfig(**kwargs)

    def _status_text(live: PipelineResult) -> str:
        lx = "X" if live.config.lines_x else ""
        ly = "Y" if live.config.lines_y else ""
        dirs = f"{lx}+{ly}".strip("+") or "none"
        return (
            "<div style='font-size:11px;color:#4b5563;margin-top:6px'>"
            f"shape=<b>{live.config.shape}</b> · lines=<b>{dirs}</b> · "
            f"engaged={live.stats['active_labels']} · "
            f"z∈[{live.stats['displacement_min']:.2f}, {live.stats['displacement_max']:.2f}] · "
            f"XY max={live.stats.get('xy_offset_max', 0.0):.1f}"
            "</div>"
        )

    def _voxel_config_from_state() -> VoxelPipeConfig:
        return VoxelPipeConfig(
            voxel_size=float(voxel_size_sl.value),
            pipe_radius=float(pipe_radius_sl.value),
            inner_radius=min(float(inner_radius_sl.value), float(pipe_radius_sl.value) * 0.85),
            modulation_amp=float(mod_amp_sl.value),
            modulation_freq=float(mod_freq_sl.value),
            modulation_lobes=int(mod_lobes_sl.value),
            line_stride=int(line_stride_sl.value),
            point_stride=int(point_stride_sl.value),
            spine_samples=int(spine_samples_sl.value),
        )

    def _set_line_opacity(opacity: float) -> None:
        """Dim / restore wireframe traces so a solid overlay is obvious."""
        op = float(np.clip(opacity, 0.0, 1.0))
        with fig.batch_update():
            # 0 = ghost outline, 1 = points, 2 = lines, 3 = sources
            fig.data[0].opacity = op
            fig.data[1].opacity = min(op, 0.35)
            fig.data[2].opacity = op

    def _clear_solid_overlay() -> None:
        if len(fig.data) <= 4:
            return
        # Tiny dummy triangle keeps Mesh3d alive; empty arrays often break updates
        with fig.batch_update():
            fig.data[4].update(
                x=[0.0, 0.0, 0.0],
                y=[0.0, 0.0, 0.0],
                z=[0.0, 0.0, 0.0],
                i=[0],
                j=[1],
                k=[2],
                visible=False,
            )
        _set_line_opacity(1.0)

    def _overlay_solid_mesh(mesh, *, label: str = "pipe solid") -> None:
        """Show an opaque shaded Mesh3d overlay and hide the wireframe."""
        verts = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if len(verts) == 0 or len(faces) == 0:
            _clear_solid_overlay()
            return
        # Cap for browser; subsample faces only (verts stay referenced)
        if len(faces) > 60_000:
            idx = np.linspace(0, len(faces) - 1, 60_000, dtype=int)
            faces = faces[idx]
        with fig.batch_update():
            fig.data[4].update(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                visible=True,
                opacity=1.0,
                color="#38bdf8",
                name=label,
                flatshading=False,
                lighting=dict(_SOLID_MESH_LIGHTING),
                lightposition=dict(_SOLID_MESH_LIGHTPOS),
                hoverinfo="skip",
                showscale=False,
            )
        _set_line_opacity(0.0)

    initial = run_pipeline(_config_from_state(), verbose=False)
    fig = _line_figure_widget(
        initial,
        z_display_max=float(amplitude_max),
        max_vertices=max_vertices,
        line_color_mode=str(line_color_dd.value),
    )
    # Placeholder Mesh3d — must start with a real triangle (empty arrays
    # break later FigureWidget updates of i/j/k on some Plotly versions).
    fig.add_trace(
        go.Mesh3d(
            x=[0.0, 0.0, 0.0],
            y=[0.0, 0.0, 0.0],
            z=[0.0, 0.0, 0.0],
            i=[0],
            j=[1],
            k=[2],
            color="#38bdf8",
            opacity=1.0,
            name="pipe solid",
            hoverinfo="skip",
            flatshading=False,
            visible=False,
            showscale=False,
            lighting=dict(_SOLID_MESH_LIGHTING),
            lightposition=dict(_SOLID_MESH_LIGHTPOS),
        )
    )
    status.value = _status_text(initial)
    _last_live: dict[str, PipelineResult] = {"result": initial}

    def _rerun(_change: object | None = None) -> None:
        if _loading["on"]:
            return
        live = run_pipeline(_config_from_state(), verbose=False)
        _last_live["result"] = live
        _apply_result_to_figure(
            fig,
            live,
            z_display_max=float(amplitude_max),
            max_vertices=max_vertices,
            line_color_mode=str(line_color_dd.value),
        )
        status.value = _status_text(live)
        _clear_solid_overlay()

    def _mirror_from_first(linked: list[str]) -> None:
        """Copy first selected source (SOURCE_LABELS order) onto the rest."""
        if len(linked) < 2:
            return
        values = _read_source(linked[0])
        _loading["on"] = True
        for key in linked[1:]:
            _write_source(key, values)
        _loading["on"] = False

    def _on_source_slider(change: dict) -> None:
        if _loading["on"] or change.get("name") != "value":
            return
        owner = change.get("owner")
        src_key = None
        for key, w in source_widgets.items():
            if owner in (w["amp"], w["wavelength"], w["release"]):
                src_key = key
                break
        if src_key is None:
            return
        linked = _linked_keys()
        if src_key in linked and len(linked) > 1:
            values = _read_source(src_key)
            _loading["on"] = True
            for key in linked:
                if key != src_key:
                    _write_source(key, values)
            _loading["on"] = False
        _rerun()

    def _select_and_mirror(keys: set[str]) -> None:
        """Link ``keys``, zero deselected sources, mirror from first linked."""
        _loading["on"] = True
        zero = {"amp": 0.0, "wavelength": 0.0, "release": 0.0}
        for key, w in source_widgets.items():
            w["link"].value = key in keys
            if key not in keys:
                _write_source(key, zero)
        _loading["on"] = False
        linked = _linked_keys()
        _mirror_from_first(linked)
        _rerun()

    def _link_corners(_btn: object = None) -> None:
        _select_and_mirror({"sw", "se", "ne", "nw"})

    def _link_mids(_btn: object = None) -> None:
        _select_and_mirror({"s", "e", "n", "w"})

    def _link_all(_btn: object = None) -> None:
        _select_and_mirror(set(source_widgets.keys()))

    def _link_clear(_btn: object = None) -> None:
        _loading["on"] = True
        zero = {"amp": 0.0, "wavelength": 0.0, "release": 0.0}
        for key, w in source_widgets.items():
            w["link"].value = False
            _write_source(key, zero)
        _loading["on"] = False
        _rerun()

    def _on_preview_solid(_btn: object = None) -> None:
        """Instant tube overlay (no PicoPie) so the display clearly changes."""
        live = _last_live.get("result") or run_pipeline(_config_from_state(), verbose=False)
        vcfg = _voxel_config_from_state()
        voxel_status.value = (
            "<div style='font-size:11px;color:#b45309'>"
            f"Building tube preview (radius={vcfg.pipe_radius}, "
            f"line step={vcfg.line_stride})…"
            "</div>"
        )
        try:
            mesh = preview_pipe_mesh(live, vcfg)
            _overlay_solid_mesh(mesh, label="tube preview")
            voxel_status.value = (
                "<div style='font-size:11px;color:#065f46'>"
                f"Preview tubes · faces={len(mesh.faces)} · "
                "fast approx (Export STL runs real voxels)"
                "</div>"
            )
        except Exception as exc:  # noqa: BLE001 — surface errors in the widget
            voxel_status.value = (
                f"<div style='font-size:11px;color:#b91c1c'>Preview failed: {exc}</div>"
            )

    def _on_export_stl(_btn: object = None) -> None:
        from pathlib import Path as _Path

        live = _last_live.get("result") or run_pipeline(_config_from_state(), verbose=False)
        vcfg = _voxel_config_from_state()
        voxel_status.value = (
            "<div style='font-size:11px;color:#b45309'>"
            "Building PicoPie voxel solid + STL (can take a while)…"
            "</div>"
        )
        try:
            # Immediate tube overlay so the UI responds while voxels bake
            preview = preview_pipe_mesh(live, vcfg)
            _overlay_solid_mesh(preview, label="tube preview")
            solid, path = pipe_and_export_stl(
                live,
                _Path("exports"),
                vcfg,
                verbose=False,
            )
            _overlay_solid_mesh(solid.trimesh_result, label="voxel solid")
            voxel_status.value = (
                "<div style='font-size:11px;color:#065f46'>"
                f"STL → <b>{path.name}</b> · faces={solid.stats['faces']} · "
                f"vol~={solid.volume:.1f} · watertight={solid.is_watertight}"
                "</div>"
            )
        except Exception as exc:  # noqa: BLE001 — surface errors in the widget
            voxel_status.value = (
                f"<div style='font-size:11px;color:#b91c1c'>STL export failed: {exc}</div>"
            )

    def _on_save_params(_btn: object = None) -> None:
        path = save_model_params(
            _config_from_state(),
            _voxel_config_from_state(),
            Path("configs"),
        )
        if path is None:
            voxel_status.value = (
                "<div style='font-size:11px;color:#6b7280'>"
                "Params identical to an existing file — not saved."
                "</div>"
            )
            return
        voxel_status.value = (
            "<div style='font-size:11px;color:#065f46'>"
            f"Params → <b>{path.as_posix()}</b>"
            "</div>"
        )

    btn_style = Layout(width="88px", height="28px", margin="2px")
    btn_corners = Button(description="corners", layout=btn_style)
    btn_mids = Button(description="mids", layout=btn_style)
    btn_all = Button(description="all", layout=btn_style)
    btn_clear = Button(description="clear", layout=btn_style)
    btn_corners.on_click(_link_corners)
    btn_mids.on_click(_link_mids)
    btn_all.on_click(_link_all)
    btn_clear.on_click(_link_clear)
    btn_preview_solid = Button(
        description="Preview solid",
        layout=Layout(width="140px", height="30px", margin="2px"),
        tooltip="Fast tube overlay along current lines (instant; not the voxel STL)",
    )
    btn_export_stl = Button(
        description="Export STL",
        button_style="success",
        layout=Layout(width="120px", height="30px", margin="2px"),
        tooltip="PicoPie voxels along lines → exports/cymatics_pipe_*.stl",
    )
    btn_save_params = Button(
        description="Save params",
        layout=Layout(width="120px", height="30px", margin="2px"),
        tooltip="Write pipeline + voxel settings JSON to configs/",
    )
    btn_preview_solid.on_click(_on_preview_solid)
    btn_export_stl.on_click(_on_export_stl)
    btn_save_params.on_click(_on_save_params)

    for w in source_widgets.values():
        for sl in (w["amp"], w["wavelength"], w["release"]):
            sl.observe(_on_source_slider, names="value")

    for w in (
        time_sl,
        decay_sl,
        cloth_sl,
        grid_x,
        grid_y,
        lines_x_cb,
        lines_y_cb,
        line_color_dd,
        shape_dd,
        cyl_diam,
        cyl_len,
        cone_h,
        cone_r,
        fr_h,
        fr_base,
        fr_top,
    ):
        w.observe(_rerun, names="value")
    shape_dd.observe(_sync_shape_param_visibility, names="value")

    source_blocks = [source_widgets[label.lower()]["block"] for label in SOURCE_LABELS]
    controls = VBox(
        [
            HTML(
                "<style>"
                ".cym-panel { overflow-x: hidden !important; }"
                ".cym-panel .jupyter-widgets { font-size: 12px !important; }"
                ".cym-panel .widget-label { font-size: 12px !important; }"
                ".cym-panel .widget-readout { font-size: 12px !important; }"
                ".cym-panel .widget-box { overflow: visible !important; }"
                "</style>"
                "<div style='font-size:13px;margin:0 0 6px 0'><b>Display</b></div>"
            ),
            line_color_dd,
            HBox(
                [lines_x_cb, lines_y_cb],
                layout=Layout(width="100%", margin="4px 0 8px 0"),
            ),
            HTML(
                "<div style='font-size:13px;margin:10px 0 4px 0'><b>Shape map</b> "
                "<span style='color:#6b7280;font-size:11px'>"
                "waves on plane → same UV + local offsets on target"
                "</span></div>"
            ),
            shape_dd,
            shape_param_box,
            HTML(
                "<div style='font-size:13px;margin:14px 0 10px 0'>"
                "<b>Sources</b> "
                "<span style='color:#6b7280;font-size:11px'>"
                "— titled blocks · values on the right · sync to mirror"
                "</span></div>"
            ),
            *source_blocks,
            HTML(
                "<div style='font-size:13px;margin:14px 0 4px 0'><b>Mirroring</b> "
                "<span style='color:#6b7280;font-size:11px'>"
                "copies from the <b>first</b> selected point in order "
                "(SW→SE→NE→NW→S→E→N→W); deselected → 0"
                "</span></div>"
            ),
            HBox(
                [btn_corners, btn_mids, btn_all, btn_clear],
                layout=Layout(width="100%", flex_flow="row wrap"),
            ),
            HTML(
                "<div style='font-size:13px;margin:14px 0 4px 0'><b>Global</b></div>"
            ),
            time_sl,
            decay_sl,
            cloth_sl,
            grid_x,
            grid_y,
            status,
            HTML(
                "<div style='font-size:13px;margin:16px 0 4px 0'>"
                "<b>Voxel print</b> "
                "<span style='color:#6b7280;font-size:11px'>"
                "pipe lines → PicoPie voxels → STL"
                "</span></div>"
            ),
            voxel_help,
            voxel_size_sl,
            pipe_radius_sl,
            inner_radius_sl,
            mod_amp_sl,
            mod_freq_sl,
            mod_lobes_sl,
            line_stride_sl,
            point_stride_sl,
            spine_samples_sl,
            HBox(
                [btn_preview_solid, btn_export_stl, btn_save_params],
                layout=Layout(width="100%", flex_flow="row wrap", margin="6px 0 0 0"),
            ),
            voxel_status,
        ],
        layout=Layout(
            width="380px",
            height="900px",
            overflow_x="hidden",
            overflow_y="scroll",
            border="1px solid #374151",
            padding="10px",
            margin="0 0 0 8px",
        ),
    )
    controls.add_class("cym-panel")
    ui = HBox(
        [fig, controls],
        layout=Layout(width="100%", align_items="flex-start"),
    )
    display(ui)
    return None
