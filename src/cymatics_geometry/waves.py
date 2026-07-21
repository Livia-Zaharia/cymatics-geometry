"""Four-corner circular-wave interference (cymatics displacement)."""

from __future__ import annotations

import numpy as np

from cymatics_geometry.config import PipelineConfig
from cymatics_geometry.grid import corner_positions


def distances_to_corners(points: np.ndarray, side_length: float) -> np.ndarray:
    """Euclidean XY distance from each point to each of the 4 corners.

    Returns shape (n_points, 4).
    """
    corners = corner_positions(side_length)[:, :2]
    xy = np.asarray(points, dtype=float)[:, :2]
    # (n, 1, 2) - (1, 4, 2) → (n, 4, 2)
    delta = xy[:, None, :] - corners[None, :, :]
    return np.linalg.norm(delta, axis=2)


def corner_wave_contributions(
    distances: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """Per-corner wave field values at each point.

    Each corner emits a circular travelling wave:
        A_i * sin(k * r_i - ω t + φ_i) / (1 + decay * r_i)

    Returns shape (n_points, 4).
    """
    amplitudes = np.asarray(config.amplitudes, dtype=float)
    phases = np.asarray(config.phases, dtype=float)
    k = config.wave_number
    omega = config.angular_frequency
    t = float(config.time)
    decay = float(config.decay)

    envelope = 1.0 / (1.0 + decay * distances)
    phase = k * distances - omega * t + phases[None, :]
    return amplitudes[None, :] * np.sin(phase) * envelope


def interference_field(
    points: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scalar interference displacement and per-corner contributions.

    Returns
    -------
    displacement : (n_points,) summed Z displacement
    contributions : (n_points, 4) per-corner fields
    """
    distances = distances_to_corners(points, config.side_length)
    contributions = corner_wave_contributions(distances, config)
    displacement = contributions.sum(axis=1)
    return displacement, contributions


def displace_points(
    points: np.ndarray,
    displacement: np.ndarray,
) -> np.ndarray:
    """Move points in Z according to the interference displacement field."""
    out = np.asarray(points, dtype=float).copy()
    out[:, 2] = displacement
    return out
