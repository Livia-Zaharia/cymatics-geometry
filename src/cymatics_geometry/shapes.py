"""Map plane UV lattice + local offsets onto cylinder / cone / frustum surfaces.

Waves always run on the flat plane. Each grid point keeps its (u, v) ∈ [0, 1]²
and its local offset δ = (dx, dy, dz) from the plane frame:

    Tu_plane = (1, 0, 0),  Tv_plane = (0, 1, 0),  N_plane = (0, 0, 1)
    displaced_plane = original_plane + dx·Tu + dy·Tv + dz·N

On a target shape the same (u, v) sample a base point P and an orthonormal
frame (Tu, Tv, N). The mapped point is:

    P' = P + dx·Tu + dy·Tv + dz·N

So the same Cartesian offset components ride the surface's tangent / normal
basis instead of world XYZ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ShapeKind = Literal["plane", "cylinder", "cone", "frustum"]
SHAPE_KINDS: tuple[ShapeKind, ...] = ("plane", "cylinder", "cone", "frustum")


@dataclass(frozen=True)
class ShapeParams:
    """Geometric parameters for the target surface (ignored when kind=plane)."""

    kind: ShapeKind = "plane"
    # Cylinder: diameter + length (axis along V / former Y)
    cylinder_diameter: float = 40.0
    cylinder_length: float = 100.0
    # Cone: height + base radius (tip at v=1)
    cone_height: float = 100.0
    cone_base_radius: float = 30.0
    # Frustum (truncated cone): height + base/top diameters
    frustum_height: float = 100.0
    frustum_base_diameter: float = 60.0
    frustum_top_diameter: float = 20.0


def plane_uv(points: np.ndarray, side_length: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (u, v) ∈ [0, 1] from flat plane XY coordinates."""
    s = max(float(side_length), 1e-12)
    xy = np.asarray(points, dtype=float)[:, :2]
    u = np.clip(xy[:, 0] / s, 0.0, 1.0)
    v = np.clip(xy[:, 1] / s, 0.0, 1.0)
    return u, v


def plane_offsets(
    original: np.ndarray,
    displaced: np.ndarray,
) -> np.ndarray:
    """Per-point (dx, dy, dz) in the plane local frame (= world XYZ on a flat plane)."""
    return np.asarray(displaced, dtype=float) - np.asarray(original, dtype=float)


def _orthonormalize(tu: np.ndarray, tv: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize and re-orthogonalize frames; keep shape (n, 3)."""
    def _unit(vec: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        return vec / np.maximum(norms, 1e-12)

    n_u = _unit(n)
    # Project Tu off N, then Tv off both
    tu_p = tu - np.sum(tu * n_u, axis=1, keepdims=True) * n_u
    tu_u = _unit(tu_p)
    tv_p = tv - np.sum(tv * n_u, axis=1, keepdims=True) * n_u
    tv_p = tv_p - np.sum(tv_p * tu_u, axis=1, keepdims=True) * tu_u
    tv_u = _unit(tv_p)
    # Fix rare degeneracies (e.g. cone tip) with a stable fallback
    bad = (np.linalg.norm(tu_p, axis=1) < 1e-12) | (np.linalg.norm(tv_p, axis=1) < 1e-12)
    if np.any(bad):
        # Build a perpendicular to N via cross with a world axis
        ref = np.tile(np.array([0.0, 1.0, 0.0]), (int(np.count_nonzero(bad)), 1))
        n_bad = n_u[bad]
        alt = np.cross(n_bad, ref)
        weak = np.linalg.norm(alt, axis=1) < 1e-8
        if np.any(weak):
            ref2 = np.tile(np.array([1.0, 0.0, 0.0]), (int(np.count_nonzero(weak)), 1))
            alt[weak] = np.cross(n_bad[weak], ref2)
        tu_u[bad] = _unit(alt)
        tv_u[bad] = _unit(np.cross(n_u[bad], tu_u[bad]))
    return tu_u, tv_u, n_u


def _cylinder_surface(
    u: np.ndarray,
    v: np.ndarray,
    *,
    diameter: float,
    length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wrap U around circumference, V along length (Y)."""
    r = 0.5 * max(float(diameter), 1e-9)
    length = float(length)
    theta = 2.0 * np.pi * u
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    base = np.column_stack([r * cos_t, v * length, r * sin_t])
    # Circumferential +u, axial +v, outward normal
    tu = np.column_stack([-sin_t, np.zeros_like(u), cos_t])
    tv = np.column_stack([np.zeros_like(u), np.ones_like(u), np.zeros_like(u)])
    n = np.column_stack([cos_t, np.zeros_like(u), sin_t])
    return base, *_orthonormalize(tu, tv, n)


def _cone_surface(
    u: np.ndarray,
    v: np.ndarray,
    *,
    height: float,
    base_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cone: base at v=0, tip at v=1."""
    h = max(float(height), 1e-9)
    r0 = max(float(base_radius), 0.0)
    theta = 2.0 * np.pi * u
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    radius = r0 * (1.0 - v)
    base = np.column_stack([radius * cos_t, v * h, radius * sin_t])
    # ∂P/∂θ direction and ∂P/∂v (slant)
    tu = np.column_stack([-sin_t, np.zeros_like(u), cos_t])
    dr_dv = -r0
    tv = np.column_stack([dr_dv * cos_t, np.full_like(u, h), dr_dv * sin_t])
    # Outward normal ≈ radial in XZ + lean from slant
    # N ∝ (−dr_dv * radial_xz normalized against axis) — use cross(Tu, Tv)
    n = np.cross(tu, tv)
    return base, *_orthonormalize(tu, tv, n)


def _frustum_surface(
    u: np.ndarray,
    v: np.ndarray,
    *,
    height: float,
    base_diameter: float,
    top_diameter: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Truncated cone: base circle at v=0, top circle at v=1."""
    h = max(float(height), 1e-9)
    r0 = 0.5 * max(float(base_diameter), 0.0)
    r1 = 0.5 * max(float(top_diameter), 0.0)
    theta = 2.0 * np.pi * u
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    radius = r0 + (r1 - r0) * v
    base = np.column_stack([radius * cos_t, v * h, radius * sin_t])
    tu = np.column_stack([-sin_t, np.zeros_like(u), cos_t])
    dr_dv = r1 - r0
    tv = np.column_stack([dr_dv * cos_t, np.full_like(u, h), dr_dv * sin_t])
    n = np.cross(tu, tv)
    return base, *_orthonormalize(tu, tv, n)


def surface_base_and_frames(
    u: np.ndarray,
    v: np.ndarray,
    params: ShapeParams,
    *,
    side_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Base points P and orthonormal (Tu, Tv, N) for the chosen shape."""
    u_a = np.asarray(u, dtype=float).reshape(-1)
    v_a = np.asarray(v, dtype=float).reshape(-1)
    kind = params.kind

    if kind == "plane":
        s = float(side_length)
        base = np.column_stack([u_a * s, v_a * s, np.zeros_like(u_a)])
        tu = np.tile(np.array([1.0, 0.0, 0.0]), (len(u_a), 1))
        tv = np.tile(np.array([0.0, 1.0, 0.0]), (len(u_a), 1))
        n = np.tile(np.array([0.0, 0.0, 1.0]), (len(u_a), 1))
        return base, tu, tv, n

    if kind == "cylinder":
        return _cylinder_surface(
            u_a,
            v_a,
            diameter=params.cylinder_diameter,
            length=params.cylinder_length,
        )

    if kind == "cone":
        return _cone_surface(
            u_a,
            v_a,
            height=params.cone_height,
            base_radius=params.cone_base_radius,
        )

    if kind == "frustum":
        return _frustum_surface(
            u_a,
            v_a,
            height=params.frustum_height,
            base_diameter=params.frustum_base_diameter,
            top_diameter=params.frustum_top_diameter,
        )

    raise ValueError(f"Unknown shape kind: {kind!r}")


def apply_local_offsets(
    base: np.ndarray,
    tu: np.ndarray,
    tv: np.ndarray,
    n: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """P' = P + dx·Tu + dy·Tv + dz·N."""
    d = np.asarray(offsets, dtype=float)
    return (
        np.asarray(base, dtype=float)
        + d[:, 0:1] * tu
        + d[:, 1:2] * tv
        + d[:, 2:3] * n
    )


def map_points_to_shape(
    original_plane: np.ndarray,
    displaced_plane: np.ndarray,
    params: ShapeParams,
    *,
    side_length: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map plane samples onto the target shape.

    Returns
    -------
    shape_base : (N, 3) undeformed surface points
    shape_displaced : (N, 3) surface points with plane offsets applied in local frame
    offsets : (N, 3) plane-frame (dx, dy, dz)
    """
    u, v = plane_uv(original_plane, side_length)
    offsets = plane_offsets(original_plane, displaced_plane)
    base, tu, tv, n = surface_base_and_frames(u, v, params, side_length=side_length)
    mapped = apply_local_offsets(base, tu, tv, n, offsets)
    return base, mapped, offsets


def map_sources_to_shape(
    sources_plane: np.ndarray,
    params: ShapeParams,
    *,
    side_length: float,
) -> np.ndarray:
    """Place wave-source markers on the undeformed target surface (no wave offset)."""
    u, v = plane_uv(sources_plane, side_length)
    base, _, _, _ = surface_base_and_frames(u, v, params, side_length=side_length)
    return base


def shape_bounds(params: ShapeParams, *, side_length: float) -> tuple[float, float, float, float, float, float]:
    """Axis-aligned bounds of the undeformed target surface (generous pad later)."""
    # Sample a coarse UV lattice for bounds
    n = 24
    uu, vv = np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n), indexing="xy")
    base, _, _, _ = surface_base_and_frames(
        uu.ravel(), vv.ravel(), params, side_length=side_length
    )
    lo = base.min(axis=0)
    hi = base.max(axis=0)
    return (
        float(lo[0]),
        float(hi[0]),
        float(lo[1]),
        float(hi[1]),
        float(lo[2]),
        float(hi[2]),
    )


def shape_boundary_polyline(
    params: ShapeParams,
    *,
    side_length: float,
    samples: int = 64,
) -> np.ndarray:
    """Undeformed surface silhouette: u=0/1 meridians + v=0/1 parallels, NaN-separated."""
    t = np.linspace(0.0, 1.0, int(samples), dtype=float)
    zero = np.zeros_like(t)
    one = np.ones_like(t)
    chunks: list[np.ndarray] = []
    for u_line, v_line in (
        (t, zero),  # v=0
        (t, one),  # v=1
        (zero, t),  # u=0
        (one, t),  # u=1
    ):
        base, _, _, _ = surface_base_and_frames(
            u_line, v_line, params, side_length=side_length
        )
        chunks.append(base)
        chunks.append(np.full((1, 3), np.nan, dtype=float))
    return np.vstack(chunks[:-1])
