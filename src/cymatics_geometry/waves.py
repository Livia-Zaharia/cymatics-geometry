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
    amplitudes = np.asarray(config.amplitudes, dtype=float).copy()
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


# Release slider max. Per-unit shove is independent of this max:
# raising the ceiling adds headroom; it does NOT dilute unit strength.
_RELEASE_MAX = 150.0
# World-units of peak XY shove per 1.0 of release at the source seat.
_RELEASE_UNIT_STRENGTH = 1.0
_MAX_XY_DISPLACEMENT = _RELEASE_MAX * _RELEASE_UNIT_STRENGTH
# Rim springs get this fraction of core stiffness (edges more flexible).
_CLOTH_EDGE_RATIO = 0.22
# Cloth slider span (finer 0–100 scale).
_CLOTH_MAX = 100.0


def _cloth_stiffness_field(nx: int, ny: int, cloth: float) -> np.ndarray:
    """Per-point spring strength: core stiff → edge soft, scaled by ``cloth``.

    ``cloth`` in ``[0, 100]`` sets the overall coefficient. The spatial gradient
    always keeps the rim more flexible than the centre at any non-zero value.
    Returns shape ``(ny, nx)`` with values in ``[0, ~0.95]``.
    """
    base = float(np.clip(cloth, 0.0, _CLOTH_MAX)) / _CLOTH_MAX
    if base <= 0.0:
        return np.zeros((ny, nx), dtype=float)

    yy, xx = np.mgrid[0:ny, 0:nx]
    cx = 0.5 * max(nx - 1, 1)
    cy = 0.5 * max(ny - 1, 1)
    dx = (xx.astype(float) - cx) / cx
    dy = (yy.astype(float) - cy) / cy
    # 0 at core, 1 at corners
    rim = np.clip(np.sqrt(dx * dx + dy * dy) / np.sqrt(2.0), 0.0, 1.0)
    spatial = 1.0 - (1.0 - _CLOTH_EDGE_RATIO) * rim
    # Map cloth into a usable Jakobsen stiffness band
    return spatial * (0.08 + 0.87 * base)


def _cloth_relax_xy(
    rest_xy: np.ndarray,
    xy: np.ndarray,
    nx: int,
    ny: int,
    *,
    iterations: int,
    stiffness_field: np.ndarray,
) -> np.ndarray:
    """Neighbor spring pass — cloth / mesh internal resistance.

    Structural (edge) + shear (diagonal) springs restore rest lengths so the
    plane stays one connected surface. Per-edge stiffness uses the softer of
    the two endpoints so the rim stays more flexible than the core.
    """
    field = np.asarray(stiffness_field, dtype=float)
    if iterations <= 0 or float(field.max()) <= 0.0:
        return np.asarray(xy, dtype=float)

    pos = np.asarray(xy, dtype=float).reshape(ny, nx, 2).copy()
    rest = np.asarray(rest_xy, dtype=float).reshape(ny, nx, 2)
    kmap = field.reshape(ny, nx)

    def _relax_edges(
        a: np.ndarray,
        b: np.ndarray,
        ra: np.ndarray,
        rb: np.ndarray,
        ka: np.ndarray,
        kb: np.ndarray,
    ) -> None:
        # Softer endpoint wins — edge/edge springs stay flexible
        k_edge = np.minimum(ka, kb)[..., None]
        diff = b - a
        cur = np.linalg.norm(diff, axis=-1, keepdims=True)
        rest_len = np.linalg.norm(rb - ra, axis=-1, keepdims=True)
        safe = np.maximum(cur, 1e-9)
        corr = k_edge * 0.5 * (cur - rest_len) / safe * diff
        a += corr
        b -= corr

    for _ in range(int(iterations)):
        _relax_edges(
            pos[:, :-1], pos[:, 1:], rest[:, :-1], rest[:, 1:],
            kmap[:, :-1], kmap[:, 1:],
        )
        _relax_edges(
            pos[:-1, :], pos[1:, :], rest[:-1, :], rest[1:, :],
            kmap[:-1, :], kmap[1:, :],
        )
        _relax_edges(
            pos[:-1, :-1], pos[1:, 1:], rest[:-1, :-1], rest[1:, 1:],
            kmap[:-1, :-1], kmap[1:, 1:],
        )
        _relax_edges(
            pos[:-1, 1:], pos[1:, :-1], rest[:-1, 1:], rest[1:, :-1],
            kmap[:-1, 1:], kmap[1:, :-1],
        )

    return pos.reshape(-1, 2)


def release_xy_offsets(
    points: np.ndarray,
    config: PipelineConfig,
    *,
    distances: np.ndarray | None = None,
    contributions: np.ndarray | None = None,
) -> np.ndarray:
    """Radial XY shockwave + cloth resistance across the whole plane.

    ``release`` per source is in ``[0, 150]`` with fixed unit strength:
      - 0 → locked in original XY
      - each +1 of release ≈ +1 world-unit peak shove near the source
        (raising the max does not dilute that rate)
      - 150 → 150 world-unit peak shove near the source

    ``cloth`` (global, 0–100) sets membrane spring strength with a fixed
    core-stiff / edge-soft gradient. External force radiates from the source;
    cloth springs keep points grouped as one connected surface.
    """
    pts = np.asarray(points, dtype=float)
    rest_xy = pts[:, :2]
    if distances is None:
        distances = distances_to_sources(pts, config.side_length)

    releases = np.asarray(config.releases, dtype=float)
    if not np.any(releases > 0.0):
        return np.zeros((len(pts), 2), dtype=float)

    side = float(config.side_length)
    amplitudes = np.asarray(config.amplitudes, dtype=float)
    wavelengths = np.asarray(config.wavelengths, dtype=float)
    phases = np.asarray(config.phases, dtype=float)
    k = np.asarray(config.wave_numbers, dtype=float)
    omega = float(config.angular_frequency)
    t = float(config.time)

    soft = max(0.55 * side, 1e-6)
    r = np.asarray(distances, dtype=float)
    falloff = 1.0 / (1.0 + (r / soft) ** 2)

    # Linear unit strength — max only clips the slider, never rescales the unit
    release_drive = (
        np.clip(releases, 0.0, _RELEASE_MAX)[None, :] * _RELEASE_UNIT_STRENGTH
    )
    base_push = release_drive * falloff

    has_wave = wavelengths > 0.0
    ripple = np.ones_like(r)
    if np.any(has_wave):
        phase = k[None, :] * r - omega * t + phases[None, :]
        ring = np.cos(phase)
        ring_gain = 0.25 + 0.35 * np.maximum(amplitudes, 0.0)
        ripple = np.where(
            has_wave[None, :],
            1.0 + ring_gain[None, :] * ring,
            1.0,
        )

    strength = base_push * ripple
    unit = _radial_unit_from_sources(pts, config.side_length, distances)
    primary = np.sum(strength[:, :, None] * unit, axis=1)

    mag = np.linalg.norm(primary, axis=1, keepdims=True)
    peak = float(mag.max()) if len(mag) else 0.0
    if peak > _MAX_XY_DISPLACEMENT:
        primary = primary * (_MAX_XY_DISPLACEMENT / peak)

    nx, ny = grid_shape(config)
    cloth = float(getattr(config, "cloth", 0.0))
    # Legacy: boundary_tension adds into cloth if cloth was left at 0
    tension = float(config.boundary_tension)
    if cloth <= 0.0 and tension > 0.0:
        cloth = min(_CLOTH_MAX, tension * _CLOTH_MAX)

    stiffness_field = _cloth_stiffness_field(nx, ny, cloth)
    shoved = rest_xy + primary
    if float(stiffness_field.max()) <= 0.0:
        offset = primary
    else:
        iterations = max(12, int(0.45 * max(nx, ny)))
        relaxed = _cloth_relax_xy(
            rest_xy,
            shoved,
            nx,
            ny,
            iterations=iterations,
            stiffness_field=stiffness_field,
        )
        offset = relaxed - rest_xy

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
