"""Load 2D vector drawings (SVG / DXF / DWG) and place them with uniform scale.

The imported silhouette keeps its aspect ratio. A single size control sets the
longest bounding-box side; the distorted square lattice is then stretched to
that bbox and cropped to the outline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from cymatics_geometry.crop import polygons_from_rings

SUPPORTED_SHAPE_SUFFIXES: tuple[str, ...] = (".svg", ".dxf", ".dwg")


@dataclass(frozen=True)
class LoadedShape2D:
    """Closed rings in native drawing coordinates (XY)."""

    path: Path
    rings: tuple[np.ndarray, ...]
    native_bbox: tuple[float, float, float, float]
    aspect_ratio: float

    @property
    def native_width(self) -> float:
        return float(self.native_bbox[2] - self.native_bbox[0])

    @property
    def native_height(self) -> float:
        return float(self.native_bbox[3] - self.native_bbox[1])


@dataclass(frozen=True)
class PlacedShape2D:
    """Uniformly scaled + translated rings ready to crop / map onto."""

    rings: tuple[np.ndarray, ...]
    bbox: tuple[float, float, float, float]
    scale: float
    aspect_ratio: float
    source_path: Path

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    def region(self) -> Polygon | MultiPolygon:
        region = polygons_from_rings(list(self.rings))
        if region is None:
            raise ValueError(f"No closed 2D region in {self.source_path}")
        return region

    def outline_polyline(self, *, z: float = 0.0) -> np.ndarray:
        """NaN-separated 3D polyline of the placed silhouette (for the ghost stroke)."""
        chunks: list[np.ndarray] = []
        nan = np.full((1, 3), np.nan, dtype=float)
        for i, ring in enumerate(self.rings):
            xy = np.asarray(ring, dtype=float)
            xyz = np.column_stack([xy[:, 0], xy[:, 1], np.full(len(xy), float(z))])
            chunks.append(xyz)
            if i + 1 < len(self.rings):
                chunks.append(nan)
        if not chunks:
            return np.zeros((0, 3), dtype=float)
        return np.vstack(chunks)


def _bbox_of_rings(rings: list[np.ndarray]) -> tuple[float, float, float, float]:
    pts = np.vstack([np.asarray(r, dtype=float)[:, :2] for r in rings])
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]))


def _close_ring(xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(xy, dtype=float)[:, :2]
    if len(pts) == 0:
        return pts
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts


def _dedupe_consecutive(xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(xy, dtype=float)[:, :2]
    if len(pts) < 2:
        return pts
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-12
    kept = pts[keep]
    if len(kept) and not np.allclose(kept[0], kept[-1]):
        kept = np.vstack([kept, kept[0]])
    return kept


def _rings_from_paths(paths: list[np.ndarray]) -> list[np.ndarray]:
    """Keep closed rings; close near-closed paths; hull-fallback if nothing closes."""
    rings: list[np.ndarray] = []
    open_pts: list[np.ndarray] = []
    for raw in paths:
        pts = np.asarray(raw, dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue
        xy = pts[:, :2]
        closed = len(xy) >= 3 and np.allclose(xy[0], xy[-1], atol=1e-6)
        span = float(np.linalg.norm(xy[-1] - xy[0]))
        bbox_diag = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
        near_closed = len(xy) >= 3 and bbox_diag > 1e-9 and span / bbox_diag < 0.02
        if closed or near_closed:
            ring = _dedupe_consecutive(_close_ring(xy))
            if len(ring) >= 4:
                rings.append(ring)
            continue
        open_pts.append(xy)
    if rings:
        return rings
    if not open_pts:
        return []
    # Fallback: convex hull of every vertex so an open drawing still crops.
    from shapely.geometry import MultiPoint

    cloud = np.vstack(open_pts)
    hull = MultiPoint([(float(x), float(y)) for x, y in cloud]).convex_hull
    if hull.geom_type != "Polygon" or hull.area < 1e-12:
        raise ValueError("Imported drawing has no closed outline and no usable hull")
    coords = np.asarray(hull.exterior.coords, dtype=float)[:, :2]
    return [_close_ring(coords)]


def _sample_svg_shape(element: object, *, samples: int = 64) -> np.ndarray | None:
    from svgelements import Path as SvgPath
    from svgelements import Shape

    if not isinstance(element, Shape):
        return None
    path = SvgPath(element)
    if path is None or not len(path):
        return None
    length = float(path.length(error=1e-4))
    n = max(8, int(samples))
    if length < 1e-9:
        start = path.point(0.0)
        return np.array([[float(start.x), float(start.y)]], dtype=float)
    ts = np.linspace(0.0, 1.0, n, dtype=float)
    pts = []
    for t in ts:
        p = path.point(float(t))
        pts.append([float(p.x), float(p.y)])
    return np.asarray(pts, dtype=float)


def load_svg_rings(path: Path) -> list[np.ndarray]:
    from svgelements import SVG, Path as SvgPath, Polygon as SvgPolygon
    from svgelements import Polyline as SvgPolyline
    from svgelements import Rect, Circle, Ellipse, SimpleLine

    svg = SVG.parse(str(path))
    paths: list[np.ndarray] = []
    for element in svg.elements():
        if isinstance(element, (SvgPolygon, SvgPolyline)):
            pts = np.asarray([(float(p.x), float(p.y)) for p in element.points], dtype=float)
            if isinstance(element, SvgPolygon) and len(pts) >= 3:
                pts = _close_ring(pts)
            if len(pts) >= 2:
                paths.append(pts)
            continue
        if isinstance(element, (Rect, Circle, Ellipse, SvgPath, SimpleLine)):
            sampled = _sample_svg_shape(element)
            if sampled is not None and len(sampled) >= 2:
                paths.append(sampled)
    return _rings_from_paths(paths)


def _dxf_lwpolyline_points(entity: object) -> np.ndarray:
    pts = np.asarray(list(entity.get_points("xy")), dtype=float)  # type: ignore[union-attr]
    if bool(getattr(entity, "closed", False)) and len(pts) >= 3:
        pts = _close_ring(pts)
    return pts


def _sample_dxf_circle(entity: object, *, samples: int = 64) -> np.ndarray:
    cx, cy, _cz = entity.dxf.center  # type: ignore[union-attr]
    r = float(entity.dxf.radius)  # type: ignore[union-attr]
    t = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=True)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def _sample_dxf_arc(entity: object, *, samples: int = 32) -> np.ndarray:
    cx, cy, _cz = entity.dxf.center  # type: ignore[union-attr]
    r = float(entity.dxf.radius)  # type: ignore[union-attr]
    start = np.deg2rad(float(entity.dxf.start_angle))  # type: ignore[union-attr]
    end = np.deg2rad(float(entity.dxf.end_angle))  # type: ignore[union-attr]
    if end < start:
        end += 2.0 * np.pi
    t = np.linspace(start, end, int(samples), endpoint=True)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def _sample_dxf_ellipse(entity: object, *, samples: int = 64) -> np.ndarray:
    pts = []
    for x, y, _z in entity.flattening(distance=max(float(entity.minor_axis.length) / samples, 1e-4)):  # type: ignore[union-attr]
        pts.append([float(x), float(y)])
    arr = np.asarray(pts, dtype=float)
    if bool(getattr(entity, "is_closed", False)) and len(arr) >= 3:
        arr = _close_ring(arr)
    return arr


def _sample_dxf_spline(entity: object) -> np.ndarray:
    pts = []
    for x, y, _z in entity.flattening(distance=0.5):  # type: ignore[union-attr]
        pts.append([float(x), float(y)])
    arr = np.asarray(pts, dtype=float)
    if bool(getattr(entity, "closed", False)) and len(arr) >= 3:
        arr = _close_ring(arr)
    return arr


def _rings_from_dxf_document(doc: object) -> list[np.ndarray]:
    msp = doc.modelspace()  # type: ignore[union-attr]
    paths: list[np.ndarray] = []
    for entity in msp:
        kind = entity.dxftype()
        if kind == "LWPOLYLINE":
            paths.append(_dxf_lwpolyline_points(entity))
        elif kind == "POLYLINE":
            pts = np.asarray([(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices], dtype=float)
            if bool(entity.is_closed) and len(pts) >= 3:
                pts = _close_ring(pts)
            paths.append(pts)
        elif kind == "CIRCLE":
            paths.append(_sample_dxf_circle(entity))
        elif kind == "ARC":
            paths.append(_sample_dxf_arc(entity))
        elif kind == "ELLIPSE":
            paths.append(_sample_dxf_ellipse(entity))
        elif kind == "SPLINE":
            paths.append(_sample_dxf_spline(entity))
        elif kind == "LINE":
            start = entity.dxf.start
            end = entity.dxf.end
            paths.append(
                np.array(
                    [[float(start.x), float(start.y)], [float(end.x), float(end.y)]],
                    dtype=float,
                )
            )
        elif kind == "SOLID" or kind == "3DFACE":
            verts = list(entity.wcs_vertices())  # type: ignore[union-attr]
            pts = np.asarray([(float(v.x), float(v.y)) for v in verts], dtype=float)
            if len(pts) >= 3:
                paths.append(_close_ring(pts))
    return _rings_from_paths(paths)


def load_dxf_rings(path: Path) -> list[np.ndarray]:
    import ezdxf

    return _rings_from_dxf_document(ezdxf.readfile(str(path)))


def _file_looks_like_dxf(path: Path) -> bool:
    data = path.read_bytes()[:96]
    stripped = data.lstrip()
    return (
        stripped.startswith(b"0")
        or b"SECTION" in data
        or data.startswith(b"AutoCAD Binary DXF")
    )


def _file_looks_like_svg(path: Path) -> bool:
    data = path.read_bytes()[:256].lstrip()
    low = data[:64].lower()
    return low.startswith(b"<?xml") or low.startswith(b"<svg") or b"<svg" in data.lower()


def _file_looks_like_dwg(path: Path) -> bool:
    data = path.read_bytes()[:8]
    return data.startswith(b"AC10") or data.startswith(b"AC2")


def sniff_vector_format(path: Path) -> str:
    """Detect svg / dxf / dwg from contents, then fall back to the suffix."""
    if _file_looks_like_dwg(path):
        return "dwg"
    if _file_looks_like_svg(path):
        return "svg"
    if _file_looks_like_dxf(path):
        return "dxf"
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"svg", "dxf", "dwg"}:
        return suffix
    return suffix


def load_dwg_rings(path: Path) -> list[np.ndarray]:
    """Read a DWG file in-process (no external CAD converter).

    Misnamed DXF files are accepted. Real DWG is parsed with ezdwg and
    converted to DXF in memory for the existing entity extractor.
    """
    if _file_looks_like_dxf(path):
        return load_dxf_rings(path)

    import tempfile

    import ezdwg

    with tempfile.TemporaryDirectory() as tmp:
        dxf_path = Path(tmp) / f"{path.stem}.dxf"
        ezdwg.to_dxf(
            str(path),
            str(dxf_path),
            flatten_inserts=True,
            modelspace_only=True,
            explode_dimensions=False,
        )
        if not dxf_path.is_file() or dxf_path.stat().st_size == 0:
            raise ValueError(f"No 2D outline found in {path}")
        return load_dxf_rings(dxf_path)


_SHAPE_CACHE: dict[tuple[str, float], LoadedShape2D] = {}


def load_shape_2d(path: str | Path) -> LoadedShape2D:
    """Load a 2D vector file into native-coordinate closed rings."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Custom shape file not found: {file_path}")
    resolved = file_path.resolve()
    mtime = resolved.stat().st_mtime
    cache_key = (str(resolved), float(mtime))
    cached = _SHAPE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    kind = sniff_vector_format(file_path)
    if kind == "svg":
        rings = load_svg_rings(file_path)
    elif kind == "dxf":
        rings = load_dxf_rings(file_path)
    elif kind == "dwg":
        rings = load_dwg_rings(file_path)
    else:
        allowed = ", ".join(SUPPORTED_SHAPE_SUFFIXES)
        raise ValueError(f"Unsupported vector format {kind!r}; expected {allowed}")
    if not rings:
        raise ValueError(f"No 2D outline found in {file_path}")
    bbox = _bbox_of_rings(rings)
    width = max(bbox[2] - bbox[0], 1e-12)
    height = max(bbox[3] - bbox[1], 1e-12)
    loaded = LoadedShape2D(
        path=resolved,
        rings=tuple(rings),
        native_bbox=bbox,
        aspect_ratio=float(width / height),
    )
    _SHAPE_CACHE[cache_key] = loaded
    return loaded


def place_shape_2d(
    loaded: LoadedShape2D,
    *,
    size: float,
    center_x: float | None = None,
    center_y: float | None = None,
) -> PlacedShape2D:
    """Uniformly scale so ``max(width, height) == size`` and keep aspect ratio.

    Default placement puts the bounding-box minimum at the origin (same convention
    as the square plane). Optional ``center_*`` move the bbox center.
    """
    xmin, ymin, xmax, ymax = loaded.native_bbox
    native_w = max(xmax - xmin, 1e-12)
    native_h = max(ymax - ymin, 1e-12)
    longest = max(native_w, native_h)
    scale = float(size) / longest
    placed: list[np.ndarray] = []
    for ring in loaded.rings:
        xy = np.asarray(ring, dtype=float)[:, :2]
        local = (xy - np.array([xmin, ymin])) * scale
        placed.append(local)
    width = native_w * scale
    height = native_h * scale
    if center_x is None and center_y is None:
        origin = np.array([0.0, 0.0], dtype=float)
    else:
        cx = 0.5 * width if center_x is None else float(center_x)
        cy = 0.5 * height if center_y is None else float(center_y)
        origin = np.array([cx - 0.5 * width, cy - 0.5 * height], dtype=float)
        placed = [ring + origin for ring in placed]
    bbox = (
        float(origin[0]),
        float(origin[1]),
        float(origin[0] + width),
        float(origin[1] + height),
    )
    return PlacedShape2D(
        rings=tuple(placed),
        bbox=bbox,
        scale=scale,
        aspect_ratio=loaded.aspect_ratio,
        source_path=loaded.path,
    )


def load_and_place_shape(
    path: str | Path,
    *,
    size: float,
    center_x: float | None = None,
    center_y: float | None = None,
) -> PlacedShape2D:
    """Load a vector file and place it with uniform scale."""
    return place_shape_2d(
        load_shape_2d(path),
        size=size,
        center_x=center_x,
        center_y=center_y,
    )
