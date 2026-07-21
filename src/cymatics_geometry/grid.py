"""Square point-grid construction and corner source placement."""

from __future__ import annotations

import numpy as np

from cymatics_geometry.config import PipelineConfig

# Corner order: SW, SE, NE, NW (counter-clockwise from bottom-left)
CORNER_LABELS: tuple[str, str, str, str] = ("SW", "SE", "NE", "NW")


def corner_positions(side_length: float) -> np.ndarray:
    """Return (4, 3) XYZ positions of the square corners (Z=0)."""
    s = float(side_length)
    return np.array(
        [
            [0.0, 0.0, 0.0],  # SW
            [s, 0.0, 0.0],  # SE
            [s, s, 0.0],  # NE
            [0.0, s, 0.0],  # NW
        ],
        dtype=float,
    )


def build_square_grid(config: PipelineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a flat N×N square grid of points in the XY plane.

    Returns
    -------
    points : (N*N, 3) array of XYZ coordinates (Z=0)
    xs, ys : 1D coordinate axes of length N
    """
    n = int(config.grid_size)
    if n < 2:
        raise ValueError("grid_size must be >= 2")
    side = float(config.side_length)
    xs = np.linspace(0.0, side, n, dtype=float)
    ys = np.linspace(0.0, side, n, dtype=float)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = np.zeros_like(xx)
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return points, xs, ys


def grid_shape(config: PipelineConfig) -> tuple[int, int]:
    n = int(config.grid_size)
    return n, n
