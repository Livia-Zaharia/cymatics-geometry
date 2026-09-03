"""Map plane UV lattice + local offsets onto cylinder / cone / frustum / teardrop / bead surfaces.

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

ShapeKind = Literal[
    "plane",
    "cylinder",
    "cone",
    "frustum",
    "variable_cylinder",
    "teardrop",
    "bead",
    "custom",
]
SHAPE_KINDS: tuple[ShapeKind, ...] = (
    "plane",
    "cylinder",
    "cone",
    "frustum",
    "variable_cylinder",
    "teardrop",
    "bead",
    "custom",
)

Radii5 = tuple[float, float, float, float, float]
Stations3 = tuple[float, float, float]

STATION_MIN_GAP: float = 0.05
STOCK_BEAD_DIAMETER: float = 40.0
STOCK_BEAD_BOTTOM_RADIUS: float = 12.0
STOCK_BEAD_TOP_RADIUS: float = 12.0
# Mid-station radius of the stock sliced sphere (Ø=40, openings 12/12): sqrt(20²−8²)
_STOCK_BEAD_R_MID: float = float(np.sqrt(20.0 * 20.0 - 8.0 * 8.0))
STOCK_BEAD_HEIGHT: float = 32.0
STOCK_BEAD_RADII: Radii5 = (
    12.0,
    _STOCK_BEAD_R_MID,
    20.0,
    _STOCK_BEAD_R_MID,
    12.0,
)
STOCK_BEAD_STATIONS: Stations3 = (0.25, 0.50, 0.75)


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
    # Variable cylinder: three circle radii + length + middle station along V
    # middle_t is clamped to [0.1, 0.9] (never flush with begin/end)
    variable_cylinder_radius_begin: float = 20.0
    variable_cylinder_radius_middle: float = 30.0
    variable_cylinder_radius_end: float = 15.0
    variable_cylinder_length: float = 100.0
    variable_cylinder_middle: float = 0.5
    # Teardrop: five circle radii + three interior stations + height (tip at v=1)
    teardrop_height: float = 100.0
    teardrop_radius_0: float = 22.0
    teardrop_radius_1: float = 20.0
    teardrop_radius_2: float = 16.0
    teardrop_radius_3: float = 8.0
    teardrop_radius_4: float = 0.0
    teardrop_station_1: float = 0.20
    teardrop_station_2: float = 0.45
    teardrop_station_3: float = 0.70
    # Bead: sphere seed (diameter + slice radii) plus five-circle profile
    bead_diameter: float = 40.0
    bead_bottom_radius: float = 12.0
    bead_top_radius: float = 12.0
    bead_height: float = 32.0
    bead_radius_0: float = 12.0
    bead_radius_1: float = _STOCK_BEAD_R_MID
    bead_radius_2: float = 20.0
    bead_radius_3: float = _STOCK_BEAD_R_MID
    bead_radius_4: float = 12.0
    bead_station_1: float = 0.25
    bead_station_2: float = 0.50
    bead_station_3: float = 0.75
    # Custom 2D: axis-aligned bbox the square UV lattice is stretched onto
    custom_bbox_xmin: float = 0.0
    custom_bbox_ymin: float = 0.0
    custom_bbox_xmax: float = 100.0
    custom_bbox_ymax: float = 100.0


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


def _clamp_variable_cylinder_middle(middle_t: float) -> float:
    """Keep the middle station away from the ends (inclusive [0.1, 0.9])."""
    return float(np.clip(float(middle_t), 0.1, 0.9))


def _variable_cylinder_radius_profile(
    v: np.ndarray,
    *,
    radius_begin: float,
    radius_middle: float,
    radius_end: float,
    middle_t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear radius and d(radius)/dv along V.

    Control stations: v=0 → begin, v=middle_t → middle, v=1 → end.
    ``middle_t`` is clamped to [0.1, 0.9].
    """
    t = _clamp_variable_cylinder_middle(middle_t)
    r0 = max(float(radius_begin), 0.0)
    r1 = max(float(radius_middle), 0.0)
    r2 = max(float(radius_end), 0.0)
    vv = np.asarray(v, dtype=float)
    left = vv <= t
    # Left segment [0, t]: r0 → r1; right [t, 1]: r1 → r2
    alpha_l = vv / t
    alpha_r = (vv - t) / max(1.0 - t, 1e-12)
    radius = np.where(left, r0 + (r1 - r0) * alpha_l, r1 + (r2 - r1) * alpha_r)
    dr_dv = np.where(left, (r1 - r0) / t, (r2 - r1) / max(1.0 - t, 1e-12))
    return radius, dr_dv


def _variable_cylinder_surface(
    u: np.ndarray,
    v: np.ndarray,
    *,
    radius_begin: float,
    radius_middle: float,
    radius_end: float,
    length: float,
    middle_t: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cylinder of revolution with three controllable circle radii along V."""
    h = max(float(length), 1e-9)
    theta = 2.0 * np.pi * u
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    radius, dr_dv = _variable_cylinder_radius_profile(
        v,
        radius_begin=radius_begin,
        radius_middle=radius_middle,
        radius_end=radius_end,
        middle_t=middle_t,
    )
    base = np.column_stack([radius * cos_t, v * h, radius * sin_t])
    tu = np.column_stack([-sin_t, np.zeros_like(u), cos_t])
    tv = np.column_stack([dr_dv * cos_t, np.full_like(u, h), dr_dv * sin_t])
    n = np.cross(tu, tv)
    return base, *_orthonormalize(tu, tv, n)


def _clamp_interior_stations(
    t1: float,
    t2: float,
    t3: float,
    *,
    gap: float = STATION_MIN_GAP,
) -> Stations3:
    """Return strictly increasing interior stations with a minimum gap.

    Ends stay fixed at 0 and 1. Inputs are sorted, then pushed so
    ``gap ≤ t1 < t2 < t3 ≤ 1 − gap`` and neighbors differ by at least ``gap``.
    """
    g = float(gap)
    ordered = sorted((float(t1), float(t2), float(t3)))
    s0 = float(np.clip(ordered[0], g, 1.0 - 3.0 * g))
    s1 = float(np.clip(ordered[1], s0 + g, 1.0 - 2.0 * g))
    s2 = float(np.clip(ordered[2], s1 + g, 1.0 - g))
    return s0, s1, s2


def _polyline_radius_profile(
    v: np.ndarray,
    radii: Radii5,
    stations: Stations3,
) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linear radius and d(radius)/dv through five control circles."""
    t1, t2, t3 = _clamp_interior_stations(*stations)
    ts = np.array([0.0, t1, t2, t3, 1.0], dtype=float)
    rs = np.array([max(float(r), 0.0) for r in radii], dtype=float)
    vv = np.asarray(v, dtype=float)
    radius = np.interp(vv, ts, rs)
    dr_dv = np.empty_like(vv)
    for i in range(4):
        span = max(float(ts[i + 1] - ts[i]), 1e-12)
        slope = (rs[i + 1] - rs[i]) / span
        if i < 3:
            mask = (vv >= ts[i]) & (vv < ts[i + 1])
        else:
            mask = vv >= ts[i]
        dr_dv[mask] = slope
    return radius, dr_dv


def _revolution_from_stations(
    u: np.ndarray,
    v: np.ndarray,
    *,
    height: float,
    radii: Radii5,
    stations: Stations3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Surface of revolution around Y from a five-circle radius profile."""
    h = max(float(height), 1e-9)
    theta = 2.0 * np.pi * u
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    radius, dr_dv = _polyline_radius_profile(v, radii, stations)
    base = np.column_stack([radius * cos_t, v * h, radius * sin_t])
    tu = np.column_stack([-sin_t, np.zeros_like(u), cos_t])
    tv = np.column_stack([dr_dv * cos_t, np.full_like(u, h), dr_dv * sin_t])
    n = np.cross(tu, tv)
    return base, *_orthonormalize(tu, tv, n)


def _teardrop_surface(
    u: np.ndarray,
    v: np.ndarray,
    *,
    height: float,
    radii: Radii5,
    stations: Stations3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cone-like teardrop: five movable circles, tip typically at v=1."""
    return _revolution_from_stations(
        u, v, height=height, radii=radii, stations=stations
    )


def _bead_slice_heights(
    *,
    diameter: float,
    bottom_radius: float,
    top_radius: float,
) -> tuple[float, float, float, float, float]:
    """Return (R, r0, r1, z0, z1) for a sphere with planar end caps.

    Bottom uses the southern root ``z0 = -sqrt(R² − r0²)``, top the northern
    root ``z1 = +sqrt(R² − r1²)``. Slice radii are clamped to ``[0, R]``.
    """
    r_sphere = 0.5 * max(float(diameter), 1e-9)
    r0 = float(np.clip(float(bottom_radius), 0.0, r_sphere))
    r1 = float(np.clip(float(top_radius), 0.0, r_sphere))
    z0 = -float(np.sqrt(max(r_sphere * r_sphere - r0 * r0, 0.0)))
    z1 = float(np.sqrt(max(r_sphere * r_sphere - r1 * r1, 0.0)))
    # Degenerate flat cut (both at equator): nudge so V still has a span
    if abs(z1 - z0) < 1e-9:
        z1 = z0 + 1e-6
    return r_sphere, r0, r1, z0, z1


def bead_profile_from_sphere(
    diameter: float,
    bottom_radius: float,
    top_radius: float,
) -> tuple[float, Radii5, Stations3]:
    """Sample a sliced sphere at five equal-v stations.

    Returns ``(height, radii, interior_stations)`` suitable for the 5-circle
    bead profile. Openings match the slice radii; the equator (or near it)
    is the middle circle.
    """
    r_sphere, _r0, _r1, z0, z1 = _bead_slice_heights(
        diameter=diameter,
        bottom_radius=bottom_radius,
        top_radius=top_radius,
    )
    height = float(z1 - z0)
    stations_full = (0.0, 0.25, 0.50, 0.75, 1.0)
    radii_list: list[float] = []
    for t in stations_full:
        z = z0 + (z1 - z0) * t
        rho = float(np.sqrt(max(r_sphere * r_sphere - z * z, 0.0)))
        radii_list.append(rho)
    radii: Radii5 = (
        radii_list[0],
        radii_list[1],
        radii_list[2],
        radii_list[3],
        radii_list[4],
    )
    return height, radii, (0.25, 0.50, 0.75)


def _profile_close(
    height: float,
    radii: Radii5,
    stations: Stations3,
    other: tuple[float, Radii5, Stations3],
    *,
    atol: float = 1e-6,
) -> bool:
    other_h, other_r, other_t = other
    if abs(float(height) - float(other_h)) > atol:
        return False
    if any(abs(float(a) - float(b)) > atol for a, b in zip(radii, other_r)):
        return False
    if any(abs(float(a) - float(b)) > atol for a, b in zip(stations, other_t)):
        return False
    return True


def resolve_bead_profile(
    *,
    diameter: float,
    bottom_radius: float,
    top_radius: float,
    height: float,
    radii: Radii5,
    stations: Stations3,
) -> tuple[float, Radii5, Stations3]:
    """Use stored 5-circle fields, or re-derive from the sphere seed.

    If the stored profile still matches the stock sphere sample (Ø=40,
    openings 12/12) and the three sphere params differ, sample the new
    sphere. Otherwise the stored circles win (user already customized them).
    """
    t1, t2, t3 = _clamp_interior_stations(*stations)
    stored_radii: Radii5 = (
        max(float(radii[0]), 0.0),
        max(float(radii[1]), 0.0),
        max(float(radii[2]), 0.0),
        max(float(radii[3]), 0.0),
        max(float(radii[4]), 0.0),
    )
    stored_h = float(height)
    stock = bead_profile_from_sphere(
        STOCK_BEAD_DIAMETER,
        STOCK_BEAD_BOTTOM_RADIUS,
        STOCK_BEAD_TOP_RADIUS,
    )
    sphere_changed = (
        abs(float(diameter) - STOCK_BEAD_DIAMETER) > 1e-9
        or abs(float(bottom_radius) - STOCK_BEAD_BOTTOM_RADIUS) > 1e-9
        or abs(float(top_radius) - STOCK_BEAD_TOP_RADIUS) > 1e-9
    )
    if _profile_close(stored_h, stored_radii, (t1, t2, t3), stock) and sphere_changed:
        return bead_profile_from_sphere(diameter, bottom_radius, top_radius)
    return stored_h, stored_radii, (t1, t2, t3)


def overlay_bead_profile(
    diameter: float,
    bottom_radius: float,
    top_radius: float,
    *,
    height: float | None = None,
    radius_0: float | None = None,
    radius_1: float | None = None,
    radius_2: float | None = None,
    radius_3: float | None = None,
    radius_4: float | None = None,
    station_1: float | None = None,
    station_2: float | None = None,
    station_3: float | None = None,
) -> tuple[float, Radii5, Stations3]:
    """Sphere-seeded profile with optional per-circle / station overlays (CLI)."""
    h, rs, ts = bead_profile_from_sphere(diameter, bottom_radius, top_radius)
    out_r = [rs[0], rs[1], rs[2], rs[3], rs[4]]
    out_t = [ts[0], ts[1], ts[2]]
    if height is not None:
        h = float(height)
    overlays_r = (radius_0, radius_1, radius_2, radius_3, radius_4)
    for i, val in enumerate(overlays_r):
        if val is not None:
            out_r[i] = max(float(val), 0.0)
    overlays_t = (station_1, station_2, station_3)
    for i, val in enumerate(overlays_t):
        if val is not None:
            out_t[i] = float(val)
    t1, t2, t3 = _clamp_interior_stations(out_t[0], out_t[1], out_t[2])
    radii: Radii5 = (out_r[0], out_r[1], out_r[2], out_r[3], out_r[4])
    return float(h), radii, (t1, t2, t3)


def _bead_surface(
    u: np.ndarray,
    v: np.ndarray,
    *,
    diameter: float,
    bottom_radius: float,
    top_radius: float,
    height: float,
    radii: Radii5,
    stations: Stations3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bead: five-circle loft, optionally seeded from a sliced sphere."""
    h, rs, ts = resolve_bead_profile(
        diameter=diameter,
        bottom_radius=bottom_radius,
        top_radius=top_radius,
        height=height,
        radii=radii,
        stations=stations,
    )
    return _revolution_from_stations(u, v, height=h, radii=rs, stations=ts)


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

    if kind == "custom":
        xmin = float(params.custom_bbox_xmin)
        ymin = float(params.custom_bbox_ymin)
        xmax = float(params.custom_bbox_xmax)
        ymax = float(params.custom_bbox_ymax)
        width = max(xmax - xmin, 1e-12)
        height = max(ymax - ymin, 1e-12)
        base = np.column_stack(
            [xmin + u_a * width, ymin + v_a * height, np.zeros_like(u_a)]
        )
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

    if kind == "variable_cylinder":
        return _variable_cylinder_surface(
            u_a,
            v_a,
            radius_begin=params.variable_cylinder_radius_begin,
            radius_middle=params.variable_cylinder_radius_middle,
            radius_end=params.variable_cylinder_radius_end,
            length=params.variable_cylinder_length,
            middle_t=params.variable_cylinder_middle,
        )

    if kind == "teardrop":
        return _teardrop_surface(
            u_a,
            v_a,
            height=params.teardrop_height,
            radii=(
                params.teardrop_radius_0,
                params.teardrop_radius_1,
                params.teardrop_radius_2,
                params.teardrop_radius_3,
                params.teardrop_radius_4,
            ),
            stations=(
                params.teardrop_station_1,
                params.teardrop_station_2,
                params.teardrop_station_3,
            ),
        )

    if kind == "bead":
        return _bead_surface(
            u_a,
            v_a,
            diameter=params.bead_diameter,
            bottom_radius=params.bead_bottom_radius,
            top_radius=params.bead_top_radius,
            height=params.bead_height,
            radii=(
                params.bead_radius_0,
                params.bead_radius_1,
                params.bead_radius_2,
                params.bead_radius_3,
                params.bead_radius_4,
            ),
            stations=(
                params.bead_station_1,
                params.bead_station_2,
                params.bead_station_3,
            ),
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
