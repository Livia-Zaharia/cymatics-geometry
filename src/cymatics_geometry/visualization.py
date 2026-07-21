"""Visualization helpers for every pipeline stage (notebook + interactive)."""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import uuid4

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

from cymatics_geometry.grid import CORNER_LABELS
from cymatics_geometry.pipeline import PipelineResult


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
    """Stage 2 — grid + corner wave sources."""
    plotter = pv.Plotter(notebook=False)
    plotter.set_background("#1a1a2e")
    plotter.add_points(
        result.grid_points,
        color="#64748b",
        point_size=point_size,
        render_points_as_spheres=True,
        opacity=0.55,
    )
    plotter.add_points(
        result.corners,
        color="#f59e0b",
        point_size=18,
        render_points_as_spheres=True,
    )
    plotter.add_point_labels(
        result.corners,
        list(CORNER_LABELS),
        font_size=14,
        text_color="white",
        point_color="#f59e0b",
        point_size=10,
    )
    plotter.add_axes()
    plotter.add_text("Stage 2 — Corner wave sources", font_size=12, color="white")
    bounds = _bounds_from_points(np.vstack([result.grid_points, result.corners]))
    target = np.mean(result.corners, axis=0)
    plotter.camera_position = camera_position_from_bounds(bounds, target)
    plotter.show()


def show_stage_field_heatmap(result: PipelineResult, *, figsize: tuple[int, int] = (8, 7)) -> None:
    """Stage 3 — top-down interference field heatmap (matplotlib)."""
    n = result.config.grid_size
    field = result.displacement.reshape(n, n)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        field,
        origin="lower",
        extent=[0, result.config.side_length, 0, result.config.side_length],
        cmap="coolwarm",
        aspect="equal",
    )
    ax.scatter(
        result.corners[:, 0],
        result.corners[:, 1],
        c="#f59e0b",
        s=80,
        edgecolors="black",
        zorder=5,
    )
    for label, corner in zip(CORNER_LABELS, result.corners):
        ax.annotate(
            f"{label}\nA={dict(zip(CORNER_LABELS, result.config.amplitudes))[label]:.2f}",
            (corner[0], corner[1]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
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
    plotter.add_points(
        result.corners,
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
    plotter.add_points(
        result.corners,
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
    n = result.config.grid_size
    field = result.displacement.reshape(n, n)
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(result.grid_points[:, 0], result.grid_points[:, 1], s=1, c="#6366f1")
    ax1.scatter(result.corners[:, 0], result.corners[:, 1], c="#f59e0b", s=60, zorder=5)
    for label, corner in zip(CORNER_LABELS, result.corners):
        ax1.annotate(label, (corner[0], corner[1]), xytext=(4, 4), textcoords="offset points")
    ax1.set_aspect("equal")
    ax1.set_title("1–2  Grid + corners")
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
    ax2.scatter(result.corners[:, 0], result.corners[:, 1], c="#f59e0b", s=50, edgecolors="k")
    ax2.set_title("3  Interference field")
    fig.colorbar(im, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    # Subsample for readability in matplotlib
    step = max(1, n // 40)
    pts = result.displaced_points.reshape(n, n, 3)[::step, ::step].reshape(-1, 3)
    disp = result.displacement.reshape(n, n)[::step, ::step].ravel()
    ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=disp, cmap="coolwarm", s=6)
    ax3.set_title("4  Displaced points")
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

    fig.suptitle(
        "Cymatics geometry stages — "
        f"A=({result.config.amplitude_sw:.2f}, {result.config.amplitude_se:.2f}, "
        f"{result.config.amplitude_ne:.2f}, {result.config.amplitude_nw:.2f})  "
        f"λ={result.config.wavelength:.1f}",
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
    plotter.add_points(
        result.corners,
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
