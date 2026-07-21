"""Multi-source circular-wave interference + surface-wide XY release."""

from __future__ import annotations

import numpy as np

from cymatics_geometry.config import PipelineConfig
from cymatics_geometry.grid import grid_shape, source_positions


def distances_to_sources(points: np.ndarray, side_length: float) -> np.ndarray:
    """Euclidean XY distance from each point to each of the 8 sources.

    Returns shape (n_points, 8).
    """
    sources = source_positions(side_length)[:, :2]
    xy = np.asarray(points, dtype=float)[:, :2]
    delta = xy[:, None, :] - sources[None, :, :]
    return np.linalg.norm(delta, axis=2)


def distances_to_corners(points: np.ndarray, side_length: float) -> np.ndarray:
    """Backward-compatible alias: distances to the four corner sources only."""
    return distances_to_sources(points, side_length)[:, :4]


def source_wave_contributions(
    distances: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """Per-source wave field values at each point.

    Each active source with λ_i > 0 emits:
        A_i * sin(k_i * r_i − ω t + φ_i) / (1 + decay * r_i)
        k_i = 2π / λ_i

    λ_i ≤ 0 or inactive → contribution 0. Returns shape (n_points, 8).
    """
    amplitudes = np.asarray(config.effective_amplitudes, dtype=float).copy()
    wavelengths = np.asarray(config.wavelengths, dtype=float)
    amplitudes = np.where(wavelengths > 0.0, amplitudes, 0.0)

    phases = np.asarray(config.phases, dtype=float)
    k = np.asarray(config.wave_numbers, dtype=float)
    omega = config.angular_frequency
    t = float(config.time)
    decay = float(config.decay)

    envelope = 1.0 / (1.0 + decay * distances)
    phase = k[None, :] * distances - omega * t + phases[None, :]
    return amplitudes[None, :] * np.sin(phase) * envelope


def corner_wave_contributions(
    distances: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """Backward-compatible wrapper when distances are (n, 4) corner-only."""
    if distances.shape[1] == 4:
        pad = np.full((distances.shape[0], 4), 1e9, dtype=float)
        distances = np.concatenate([distances, pad], axis=1)
    return source_wave_contributions(distances, config)


def interference_field(
    points: np.ndarray,
    config: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scalar interference displacement and per-source contributions."""
    distances = distances_to_sources(points, config.side_length)
    contributions = source_wave_contributions(distances, config)
    displacement = contributions.sum(axis=1)
    return displacement, contributions


def _radial_unit_from_sources(
    points: np.ndarray,
    side_length: float,
    distances: np.ndarray,
) -> np.ndarray:
    """Unit vectors from each source toward each point. Shape (n, 8, 2)."""
    sources = source_positions(side_length)[:, :2]
    xy = np.asarray(points, dtype=float)[:, :2]
    delta = xy[:, None, :] - sources[None, :, :]
    r = np.asarray(distances, dtype=float)
    r_safe = np.maximum(r, 1e-9)
    unit = delta / r_safe[:, :, None]

    center = 0.5 * float(side_length)
    inward = np.asarray([center, center], dtype=float) - xy
    inward_norm = np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-9)
    inward_u = inward / inward_norm
    at_source = r < 1e-6
    for i in range(unit.shape[1]):
        mask = at_source[:, i]
        if np.any(mask):
            unit[mask, i, :] = inward_u[mask]
    return unit


def _diffuse_xy_field(
    offsets: np.ndarray,
    nx: int,
    ny: int,
    *,
    iterations: int,
    blend: float = 0.55,
) -> np.ndarray:
    """Spread XY offsets across the lattice like a connected surface membrane."""
    if iterations <= 0:
        return offsets
    field = np.asarray(offsets, dtype=float).reshape(ny, nx, 2)
    alpha = float(np.clip(blend, 0.0, 1.0))
    for _ in range(int(iterations)):
        padded = np.pad(field, ((1, 1), (1, 1), (0, 0)), mode="edge")
        avg = 0.25 * (
            padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )
        field = (1.0 - alpha) * field + alpha * avg
    return field.reshape(-1, 2)


def release_xy_offsets(
    points: np.ndarray,
    config: PipelineConfig,
    *,
    distances: np.ndarray | None = None,
    contributions: np.ndarray | None = None,
) -> np.ndarray:
    """XY offsets: whole plane acts as a surface pulled by released sources.

    ``release`` per source is a float in ``[0, 10]``:
      - 0 → locked in original XY
      - 1 → free under wave influence at base scale
      - up to 10 → multiplier of that influence

    Primary drive uses a **wide** radial kernel so the centre of the plane is
    affected, then a diffusion pass spreads motion to neighbours — when a
    corner loosens, the surface around it follows.
    """
    pts = np.asarray(points, dtype=float)
    if distances is None:
        distances = distances_to_sources(pts, config.side_length)

    amplitudes = np.asarray(config.effective_amplitudes, dtype=float).copy()
    wavelengths = np.asarray(config.wavelengths, dtype=float)
    amplitudes = np.where(wavelengths > 0.0, amplitudes, 0.0)
    phases = np.asarray(config.phases, dtype=float)
    k = np.asarray(config.wave_numbers, dtype=float)
    omega = config.angular_frequency
    t = float(config.time)
    decay = float(config.decay)
    envelope = 1.0 / (1.0 + decay * distances)
    phase = k[None, :] * distances - omega * t + phases[None, :]
    lateral_wave = amplitudes[None, :] * np.cos(phase) * envelope

    releases = np.asarray(config.releases, dtype=float)
    side = float(config.side_length)
    # Wide soft length: influence reaches across most of the plane
    soft = max(0.85 * side, 1e-6)
    # Gaussian + floor so even far / central points keep some coupling
    radial = 0.20 + 0.80 * np.exp(-((distances / soft) ** 2))

    strength = releases[None, :] * lateral_wave * radial
    unit = _radial_unit_from_sources(pts, config.side_length, distances)
    primary = np.sum(strength[:, :, None] * unit, axis=1)

    nx, ny = grid_shape(config)
    iterations = max(6, int(0.35 * max(nx, ny)))
    diffused = _diffuse_xy_field(primary, nx, ny, iterations=iterations, blend=0.6)
    # Dimensionless field → world XY (side-length units).
    # release=1, amp=1 → order ~6% of the side — clearly visible vs Z waves.
    world_scale = 0.06 * side
    offset = (0.35 * primary + 0.65 * diffused) * world_scale

    tension = float(config.boundary_tension)
    if tension > 0.0:
        offset = offset / (1.0 + tension)

    _ = contributions
    return offset


def boundary_release_offsets(
    points: np.ndarray,
    config: PipelineConfig,
    distances: np.ndarray | None = None,
) -> np.ndarray:
    """Alias for :func:`release_xy_offsets` (legacy name)."""
    return release_xy_offsets(points, config, distances=distances)


def displace_points(
    points: np.ndarray,
    displacement: np.ndarray,
    *,
    xy_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """Move points in Z (and optionally XY under release mobility)."""
    out = np.asarray(points, dtype=float).copy()
    if xy_offsets is not None:
        offsets = np.asarray(xy_offsets, dtype=float)
        out[:, 0] += offsets[:, 0]
        out[:, 1] += offsets[:, 1]
    out[:, 2] = displacement
    return out
