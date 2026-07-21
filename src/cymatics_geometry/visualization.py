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
    """Return source XYZ, labels, and active flags for plotting."""
    sources = getattr(result, "sources", None)
    if sources is None or len(sources) == 0:
        sources = result.corners
        labels: tuple[str, ...] = ("SW", "SE", "NE", "NW")
        active = result.config.active_flags[: len(sources)]
    else:
        labels = tuple(result.source_labels)
        active = result.config.active_flags
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
    # Subsample the polyline for a readable plot
    line = result.polyline[:: max(1, len(result.polyline) // 4000)]
    norm = Normalize(vmin=line[:, 2].min(), vmax=line[:, 2].max())
    ax4.plot(line[:, 0], line[:, 1], line[:, 2], color="#7c3aed", linewidth=0.6)
    ax4.scatter(line[::20, 0], line[::20, 1], line[::20, 2], c=line[::20, 2], cmap="coolwarm", s=4, norm=norm)
    ax4.set_title("5  Reconnected line")
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
    """Keep interactive redraws light while preserving the path shape."""
    pts = np.asarray(polyline, dtype=float)
    return pts[_subsample_indices(len(pts), max_vertices)]


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
    """
    s = float(side_length)
    z_lim = max(float(z_display_max), 1e-6)
    # Room for release motion outside the square without reframing each drag
    xy_pad = 0.25 * s
    travel_max = 0.35 * s
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
                x=cloud[:, 0],
                y=cloud[:, 1],
                z=cloud[:, 2],
                mode="markers",
                name="points",
                marker={
                    "size": 2.5,
                    "color": travel,
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
                x=line[:, 0],
                y=line[:, 1],
                z=line[:, 2],
                mode="lines",
                name="line",
                line={
                    "width": 2,
                    "color": z,
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
                x=sources[:, 0],
                y=sources[:, 1],
                z=sources[:, 2],
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
        height=780,
        width=780,
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        paper_bgcolor="#111827",
        font={"color": "#e5e7eb"},
        title={
            "text": (
                f"Fixed scale · Z ∈ [−{z_lim:g}, {z_lim:g}] "
                "(amp maps 1:1 — low amp stays visually small)"
            ),
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 12},
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
            "camera": {"eye": {"x": 1.55, "y": -1.75, "z": 1.15}},
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

    with fig.batch_update():
        # 0 = original XY footprint, 1 = points, 2 = line, 3 = sources
        fig.data[0].x = [0.0, s, s, 0.0, 0.0]
        fig.data[0].y = [0.0, 0.0, s, s, 0.0]
        fig.data[0].z = [0.0, 0.0, 0.0, 0.0, 0.0]
        fig.data[1].x = cloud[:, 0]
        fig.data[1].y = cloud[:, 1]
        fig.data[1].z = cloud[:, 2]
        fig.data[1].marker.color = travel
        fig.data[1].marker.cmin = 0.0
        fig.data[1].marker.cmax = travel_max
        fig.data[2].x = line[:, 0]
        fig.data[2].y = line[:, 1]
        fig.data[2].z = line[:, 2]
        fig.data[2].line.color = z
        fig.data[2].line.cmin = -z_lim
        fig.data[2].line.cmax = z_lim
        fig.data[3].x = sources[:, 0]
        fig.data[3].y = sources[:, 1]
        fig.data[3].z = sources[:, 2]
        fig.data[3].text = list(labels)
        fig.data[3].marker.color = _source_marker_colors(active)
        fig.layout.scene.xaxis.range = [-xy_pad, s + xy_pad]
        fig.layout.scene.yaxis.range = [-xy_pad, s + xy_pad]
        fig.layout.scene.zaxis.range = [-z_lim, z_lim]
        fig.layout.title.text = (
            f"Fixed scale · Z ∈ [−{z_lim:g}, {z_lim:g}] "
            "(amp maps 1:1 — low amp stays visually small)"
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
    """Notebook UI: source dropdown + sliders, orbitable 3D view on the left."""
    from IPython.display import display
    from ipywidgets import (
        Dropdown,
        FloatSlider,
        HTML,
        HBox,
        IntSlider,
        Label,
        Layout,
        ToggleButtons,
        VBox,
    )

    base = config or PipelineConfig(grid_size_x=60, grid_size_y=60)

    slider_layout = Layout(width="95%")
    desc_style = {"description_width": "140px"}

    # All numeric controls start at 0; active flags follow config (corners on).
    state: dict[str, dict[str, float | bool]] = {}
    for label in SOURCE_LABELS:
        key = label.lower()
        state[key] = {
            "active": bool(getattr(base, f"active_{key}")),
            "amp": 0.0,
            "wavelength": 0.0,
            "release": 0.0,
        }

    source_dd = Dropdown(
        options=[
            (f"{label} — {_SOURCE_ROLES[label]}", label) for label in SOURCE_LABELS
        ],
        value="SW",
        description="source",
        style={"description_width": "70px"},
        layout=Layout(width="95%"),
    )
    source_header = HTML(value="")
    active_tb = ToggleButtons(
        options=[("off", False), ("on", True)],
        value=True,
        description="active",
        style={"description_width": "70px"},
        layout=Layout(width="100%"),
    )
    amp_sl = FloatSlider(
        value=0.0,
        min=0.0,
        max=float(amplitude_max),
        step=0.01,
        description="amp (=Z height)",
        continuous_update=True,
        readout_format=".2f",
        style=desc_style,
        layout=slider_layout,
    )
    wl_sl = FloatSlider(
        value=0.0,
        min=0.0,
        max=80.0,
        step=0.5,
        description="λ (wavelength)",
        continuous_update=True,
        style=desc_style,
        layout=slider_layout,
    )
    release_sl = FloatSlider(
        value=0.0,
        min=0.0,
        max=10.0,
        step=0.05,
        description="release (0–10)",
        continuous_update=True,
        style=desc_style,
        layout=slider_layout,
        readout_format=".2f",
    )
    time_sl = FloatSlider(
        value=0.0,
        min=0.0,
        max=6.28,
        step=0.05,
        description="time (global)",
        style=desc_style,
        layout=slider_layout,
    )
    decay_sl = FloatSlider(
        value=0.0,
        min=0.0,
        max=0.05,
        step=0.001,
        description="decay (global)",
        readout_format=".3f",
        style=desc_style,
        layout=slider_layout,
    )
    grid_x = IntSlider(
        value=min(int(base.grid_size_x), 80),
        min=20,
        max=100,
        step=10,
        description="grid X",
        style=desc_style,
        layout=slider_layout,
    )
    grid_y = IntSlider(
        value=min(int(base.grid_size_y), 80),
        min=20,
        max=100,
        step=10,
        description="grid Y",
        style=desc_style,
        layout=slider_layout,
    )
    status = Label(value="", layout=Layout(width="95%"))

    _loading = {"on": False}

    def _current_key() -> str:
        return str(source_dd.value).lower()

    def _refresh_header() -> None:
        label = str(source_dd.value)
        source_header.value = (
            f"<div style='margin:8px 0;padding:8px;"
            f"background:#1f2937;border-left:3px solid #f59e0b;color:#f3f4f6'>"
            f"<b>{label}</b> — {_SOURCE_ROLES[label]}</div>"
        )

    def _load_source_into_widgets() -> None:
        _loading["on"] = True
        st = state[_current_key()]
        active_tb.value = bool(st["active"])
        amp_sl.value = float(st["amp"])
        wl_sl.value = float(st["wavelength"])
        release_sl.value = float(st["release"])
        on = bool(st["active"])
        amp_sl.disabled = not on
        wl_sl.disabled = not on
        release_sl.disabled = not on
        _refresh_header()
        _loading["on"] = False

    def _save_widgets_into_state() -> None:
        key = _current_key()
        state[key] = {
            "active": bool(active_tb.value),
            "amp": float(amp_sl.value),
            "wavelength": float(wl_sl.value),
            "release": float(release_sl.value),
        }

    def _config_from_state() -> PipelineConfig:
        kwargs: dict = {
            "grid_size_x": int(grid_x.value),
            "grid_size_y": int(grid_y.value),
            "side_length": base.side_length,
            "frequency": base.frequency,
            "time": float(time_sl.value),
            "decay": float(decay_sl.value),
            "boundary_tension": 0.0,
            "release_pace": 0.0,
            "line_pattern": base.line_pattern,
        }
        for label in SOURCE_LABELS:
            key = label.lower()
            st = state[key]
            kwargs[f"active_{key}"] = bool(st["active"])
            kwargs[f"amplitude_{key}"] = float(st["amp"])
            kwargs[f"wavelength_{key}"] = float(st["wavelength"])
            kwargs[f"release_{key}"] = float(st["release"])
        kwargs["wavelength"] = float(state["sw"]["wavelength"])
        return PipelineConfig(**kwargs)

    def _status_text(live: PipelineResult) -> str:
        return (
            f"active={live.stats['active_labels']}  "
            f"points={live.stats['point_count']}  "
            f"z∈[{live.stats['displacement_min']:.3f}, {live.stats['displacement_max']:.3f}]  "
            f"XY travel max={live.stats.get('xy_offset_max', 0.0):.2f}  "
            f"xy_moved={live.stats.get('xy_released', 0)}"
        )

    _load_source_into_widgets()
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
        _save_widgets_into_state()
        on = bool(active_tb.value)
        amp_sl.disabled = not on
        wl_sl.disabled = not on
        release_sl.disabled = not on
        live = run_pipeline(_config_from_state(), verbose=False)
        _apply_result_to_figure(
            fig,
            live,
            z_display_max=float(amplitude_max),
            max_vertices=max_vertices,
        )
        status.value = _status_text(live)

    # Dropdown change: persist the previous source's widgets, then load the next
    _prev = {"label": str(source_dd.value)}

    def _on_dropdown(change: dict) -> None:
        if change.get("name") != "value":
            return
        old = str(_prev["label"]).lower()
        state[old] = {
            "active": bool(active_tb.value),
            "amp": float(amp_sl.value),
            "wavelength": float(wl_sl.value),
            "release": float(release_sl.value),
        }
        _prev["label"] = str(change["new"])
        _load_source_into_widgets()
        _rerun()

    source_dd.observe(_on_dropdown, names="value")
    for w in (active_tb, amp_sl, wl_sl, release_sl, time_sl, decay_sl, grid_x, grid_y):
        w.observe(_rerun, names="value")

    controls = VBox(
        [
            HTML(
                "<b>Source controls</b><br>"
                "<span style='color:#9ca3af;font-size:12px'>"
                "Need <b>amp + λ + release</b> together. "
                "Display axes/colors are <b>fixed</b> to slider max "
                f"(Z ∈ [−{amplitude_max:g}, {amplitude_max:g}]) — "
                "amp=1 is 1 unit tall, not remapped to full red. "
                "release moves real XY (leave the dashed square)."
                "</span>"
            ),
            source_dd,
            source_header,
            active_tb,
            amp_sl,
            wl_sl,
            release_sl,
            HTML("<div style='margin-top:12px'><b>Global</b></div>"),
            time_sl,
            decay_sl,
            grid_x,
            grid_y,
            status,
        ],
        layout=Layout(
            width="440px",
            height="780px",
            border="1px solid #374151",
            padding="8px",
            margin="0 0 0 8px",
        ),
    )
    ui = HBox(
        [fig, controls],
        layout=Layout(width="100%", align_items="flex-start"),
    )
    display(ui)
    return None
