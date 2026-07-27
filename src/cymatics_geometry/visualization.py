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

from cymatics_geometry.config import PipelineConfig
from cymatics_geometry.grid import SOURCE_LABELS, grid_shape
from cymatics_geometry.pipeline import PipelineResult, run_pipeline


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
    color: str = "#9b8cff",
) -> None:
    """Stage 5 — reconnected line geometry."""
    plotter = pv.Plotter(notebook=False)
    plotter.set_background("#1a1a2e")
    plotter.add_mesh(
        result.line_mesh,
        color=color,
        line_width=line_width,
        render_lines_as_tubes=True,
    )
    sources, _, active = _source_arrays(result)
    if np.any(active):
        plotter.add_points(
            sources[np.asarray(active)],
            color="#f59e0b",
            point_size=14,
            render_points_as_spheres=True,
        )
    plotter.add_axes()
    plotter.add_text("Stage 5 — Reconnected line geometry", font_size=12, color="white")
    bounds = _bounds_from_points(result.polyline)
    target = np.mean(result.polyline, axis=0)
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
) -> Path:
    """Render the final line mesh to a PNG screenshot."""
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("#1a1a2e")
    plotter.add_mesh(
        result.line_mesh,
        color="#9b8cff",
        line_width=2.0,
        render_lines_as_tubes=True,
    )
    sources, _, active = _source_arrays(result)
    if np.any(active):
        plotter.add_points(
            sources[np.asarray(active)],
            color="#f59e0b",
            point_size=14,
            render_points_as_spheres=True,
        )
    plotter.add_axes()
    plotter.add_text(title, position="upper_left", font_size=12, color="white")
    bounds = _bounds_from_points(result.polyline)
    target = np.mean(result.polyline, axis=0)
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


def _plotly_xyz(points: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    """FigureWidget needs Python lists — numpy arrays crash trait sync (truthiness)."""
    pts = np.asarray(points, dtype=float)
    return pts[:, 0].tolist(), pts[:, 1].tolist(), pts[:, 2].tolist()


def _plotly_floats(values: np.ndarray) -> list[float]:
    return np.asarray(values, dtype=float).reshape(-1).tolist()


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


def _line_figure_widget(
    polyline: np.ndarray,
    sources: np.ndarray,
    labels: tuple[str, ...],
    active: tuple[bool, ...],
    *,
    grid_points: np.ndarray | None = None,
    displaced_points: np.ndarray | None = None,
    side_length: float = 100.0,
    z_display_max: float = 10.0,
    max_vertices: int = 4000,
) -> FigureWidget:
    """Persistent Plotly 3D scene with fixed absolute Z / color units."""
    line = _subsample_polyline(polyline, max_vertices)
    z = line[:, 2]

    s = float(side_length)
    z_lim, xy_pad, travel_max = _fixed_display_scales(
        s, z_display_max=z_display_max
    )
    ghost_x = [0.0, s, s, 0.0, 0.0]
    ghost_y = [0.0, 0.0, s, s, 0.0]
    ghost_z = [0.0, 0.0, 0.0, 0.0, 0.0]

    if displaced_points is not None and len(displaced_points) > 0:
        disp = np.asarray(displaced_points, dtype=float)
        idx = _subsample_indices(len(disp), min(1200, max_vertices))
        cloud = disp[idx]
        if grid_points is not None and len(grid_points) == len(disp):
            base = np.asarray(grid_points, dtype=float)[idx]
            travel = np.linalg.norm(cloud[:, :2] - base[:, :2], axis=1)
        else:
            travel = np.abs(cloud[:, 2])
    else:
        cloud = line
        travel = np.abs(z)

    cloud_x, cloud_y, cloud_z = _plotly_xyz(cloud)
    line_x, line_y, line_z = _plotly_xyz(line)
    src_x, src_y, src_z = _plotly_xyz(sources)
    travel_list = _plotly_floats(travel)
    z_list = _plotly_floats(z)

    fig = FigureWidget(
        data=[
            go.Scatter3d(
                x=ghost_x,
                y=ghost_y,
                z=ghost_z,
                mode="lines",
                name="original footprint",
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
                        "title": "XY travel",
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
                line={
                    "width": 2,
                    "color": z_list,
                    "colorscale": "RdBu",
                    "cmin": -z_lim,
                    "cmax": z_lim,
                    "colorbar": {
                        "title": "Z (fixed)",
                        "thickness": 12,
                        "len": 0.35,
                        "y": 0.72,
                    },
                },
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
    fig.update_layout(
        height=900,
        width=900,
        margin={"l": 0, "r": 0, "t": 36, "b": 0},
        paper_bgcolor="#111827",
        font={"color": "#e5e7eb", "size": 11},
        title={
            "text": (
                f"Fixed scale · Z ∈ [−{z_lim:g}, {z_lim:g}] "
                "(amp maps 1:1 — low amp stays visually small)"
            ),
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 11},
        },
        scene={
            "xaxis": {
                "title": "X",
                "range": [-xy_pad, s + xy_pad],
                "autorange": False,
                "backgroundcolor": "#1f2937",
                "gridcolor": "#374151",
            },
            "yaxis": {
                "title": "Y",
                "range": [-xy_pad, s + xy_pad],
                "autorange": False,
                "backgroundcolor": "#1f2937",
                "gridcolor": "#374151",
            },
            "zaxis": {
                "title": "Z",
                "range": [-z_lim, z_lim],
                "autorange": False,
                "backgroundcolor": "#1f2937",
                "gridcolor": "#374151",
            },
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": 0.45},
            "bgcolor": "#111827",
            # Closer camera = denser / larger subject in the frame
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
) -> None:
    """Mutate an existing FigureWidget in place — fixed units, camera preserved."""
    line = _subsample_polyline(result.polyline, max_vertices)
    if len(line) == 0:
        line = np.full((1, 3), np.nan, dtype=float)
    z = line[:, 2]
    sources, labels, active = _source_arrays(result)

    disp = np.asarray(result.displaced_points, dtype=float)
    grid = np.asarray(result.grid_points, dtype=float)
    idx = _subsample_indices(len(disp), min(1200, max_vertices))
    cloud = disp[idx]
    base = grid[idx]
    travel = np.linalg.norm(cloud[:, :2] - base[:, :2], axis=1)

    s = float(result.config.side_length)
    z_lim, xy_pad, travel_max = _fixed_display_scales(
        s, z_display_max=z_display_max
    )
    lx = "X" if result.config.lines_x else ""
    ly = "Y" if result.config.lines_y else ""
    dirs = f"{lx}+{ly}".strip("+") or "none"

    cloud_x, cloud_y, cloud_z = _plotly_xyz(cloud)
    line_x, line_y, line_z = _plotly_xyz(line)
    src_x, src_y, src_z = _plotly_xyz(sources)

    with fig.batch_update():
        # 0 = original XY footprint, 1 = points, 2 = line, 3 = sources
        fig.data[0].x = [0.0, s, s, 0.0, 0.0]
        fig.data[0].y = [0.0, 0.0, s, s, 0.0]
        fig.data[0].z = [0.0, 0.0, 0.0, 0.0, 0.0]
        fig.data[1].x = cloud_x
        fig.data[1].y = cloud_y
        fig.data[1].z = cloud_z
        fig.data[1].marker.color = _plotly_floats(travel)
        fig.data[1].marker.cmin = 0.0
        fig.data[1].marker.cmax = travel_max
        fig.data[2].x = line_x
        fig.data[2].y = line_y
        fig.data[2].z = line_z
        fig.data[2].line.color = _plotly_floats(z)
        fig.data[2].line.cmin = -z_lim
        fig.data[2].line.cmax = z_lim
        fig.data[3].x = src_x
        fig.data[3].y = src_y
        fig.data[3].z = src_z
        fig.data[3].text = list(labels)
        fig.data[3].marker.color = _source_marker_colors(active)
        fig.layout.scene.xaxis.range = [-xy_pad, s + xy_pad]
        fig.layout.scene.yaxis.range = [-xy_pad, s + xy_pad]
        fig.layout.scene.zaxis.range = [-z_lim, z_lim]
        fig.layout.title.text = (
            f"Grid lines {dirs} · Z ∈ [−{z_lim:g}, {z_lim:g}]"
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
    status = HTML(value="", layout=Layout(width="98%"))

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
            f"lines=<b>{dirs}</b> · engaged={live.stats['active_labels']} · "
            f"z∈[{live.stats['displacement_min']:.2f}, {live.stats['displacement_max']:.2f}] · "
            f"XY max={live.stats.get('xy_offset_max', 0.0):.1f}"
            "</div>"
        )

    initial = run_pipeline(_config_from_state(), verbose=False)
    sources, labels, active = _source_arrays(initial)
    fig = _line_figure_widget(
        initial.polyline,
        sources,
        labels,
        active,
        grid_points=initial.grid_points,
        displaced_points=initial.displaced_points,
        side_length=float(initial.config.side_length),
        z_display_max=float(amplitude_max),
        max_vertices=max_vertices,
    )
    status.value = _status_text(initial)

    def _rerun(_change: object | None = None) -> None:
        if _loading["on"]:
            return
        live = run_pipeline(_config_from_state(), verbose=False)
        _apply_result_to_figure(
            fig,
            live,
            z_display_max=float(amplitude_max),
            max_vertices=max_vertices,
        )
        status.value = _status_text(live)

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
        _loading["on"] = True
        for key, w in source_widgets.items():
            w["link"].value = key in keys
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
        for w in source_widgets.values():
            w["link"].value = False
        _loading["on"] = False

    btn_style = Layout(width="88px", height="28px", margin="2px")
    btn_corners = Button(description="corners", layout=btn_style)
    btn_mids = Button(description="mids", layout=btn_style)
    btn_all = Button(description="all", layout=btn_style)
    btn_clear = Button(description="clear", layout=btn_style)
    btn_corners.on_click(_link_corners)
    btn_mids.on_click(_link_mids)
    btn_all.on_click(_link_all)
    btn_clear.on_click(_link_clear)

    for w in source_widgets.values():
        for sl in (w["amp"], w["wavelength"], w["release"]):
            sl.observe(_on_source_slider, names="value")

    for w in (time_sl, decay_sl, cloth_sl, grid_x, grid_y, lines_x_cb, lines_y_cb):
        w.observe(_rerun, names="value")

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
                "<div style='font-size:13px;margin:0 0 10px 0'>"
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
                "(SW→SE→NE→NW→S→E→N→W)"
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
            HTML(
                "<div style='font-size:12px;margin:8px 0 4px 0'>"
                "<b>Grid lines</b> "
                "<span style='color:#6b7280;font-size:11px'>"
                "uncheck to hide that direction"
                "</span></div>"
            ),
            HBox(
                [lines_x_cb, lines_y_cb],
                layout=Layout(width="100%"),
            ),
            status,
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
