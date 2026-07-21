"""Point-grid construction and wave-source placement."""

from __future__ import annotations

import numpy as np

from cymatics_geometry.config import PipelineConfig

# Corners first (CCW from SW), then mid-edge sources (S, E, N, W).
SOURCE_LABELS: tuple[str, ...] = ("SW", "SE", "NE", "NW", "S", "E", "N", "W")
CORNER_LABELS: tuple[str, str, str, str] = ("SW", "SE", "NE", "NW")
MID_EDGE_LABELS: tuple[str, str, str, str] = ("S", "E", "N", "W")


def source_positions(side_length: float) -> np.ndarray:
    """Return (8, 3) XYZ positions: corners then mid-edge centers (Z=0)."""
    s = float(side_length)
    half = 0.5 * s
    return np.array(
        [
            [0.0, 0.0, 0.0],  # SW
            [s, 0.0, 0.0],  # SE
            [s, s, 0.0],  # NE
            [0.0, s, 0.0],  # NW
            [half, 0.0, 0.0],  # S mid-edge
            [s, half, 0.0],  # E mid-edge
            [half, s, 0.0],  # N mid-edge
            [0.0, half, 0.0],  # W mid-edge
        ],
        dtype=float,
    )


def corner_positions(side_length: float) -> np.ndarray:
    """Return (4, 3) XYZ positions of the square corners (Z=0)."""
    return source_positions(side_length)[:4]


def grid_shape(config: PipelineConfig) -> tuple[int, int]:
    """Return (nx, ny) point counts along X and Y."""
    nx = int(config.grid_size_x)
    ny = int(config.grid_size_y)
    if nx < 2 or ny < 2:
        raise ValueError("grid_size_x and grid_size_y must be >= 2")
    return nx, ny


def build_square_grid(config: PipelineConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a flat nx×ny grid of points in the XY plane.

    Returns
    -------
    points : (nx*ny, 3) array of XYZ coordinates (Z=0), row-major in Y then X
    xs, ys : 1D coordinate axes of length nx and ny
    """
    nx, ny = grid_shape(config)
    side = float(config.side_length)
    xs = np.linspace(0.0, side, nx, dtype=float)
    ys = np.linspace(0.0, side, ny, dtype=float)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")  # shapes (ny, nx)
    zz = np.zeros_like(xx)
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return points, xs, ys


def border_point_mask(nx: int, ny: int | None = None) -> np.ndarray:
    """Boolean mask (nx*ny,) for points on the outer boundary."""
    if ny is None:
        ny = nx
    nx_i, ny_i = int(nx), int(ny)
    mask = np.zeros(nx_i * ny_i, dtype=bool)
    for row in range(ny_i):
        for col in range(nx_i):
            if row == 0 or row == ny_i - 1 or col == 0 or col == nx_i - 1:
                mask[row * nx_i + col] = True
    return mask
