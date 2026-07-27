"""Pipeline configuration: dataclass, JSON I/O, and config management."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime
from pathlib import Path

# Keep in sync with cymatics_geometry.grid.SOURCE_LABELS
_SOURCE_LABELS: tuple[str, ...] = ("SW", "SE", "NE", "NW", "S", "E", "N", "W")


@dataclass(frozen=True)
class PipelineConfig:
    """Parameters for one cymatics plane → line pipeline run.

    Up to eight wave sources: four corners (SW/SE/NE/NW) and four mid-edge
    points (S/E/N/W). Only *active* sources contribute.

    Per-source: amplitude, wavelength (λ), release (0 locked … radial XY shockwave).
    Global: time, frequency, decay, cloth (core-stiff / edge-soft gradient).
    Numeric controls default to 0 (static / locked until you raise them).
    """

    # Global lattice resolution along X and Y (point counts)
    grid_size_x: int = 100
    grid_size_y: int = 100
    side_length: float = 100.0

    # Per-source amplitudes (corners then mid-edges) — 0 = no wave
    amplitude_sw: float = 0.0
    amplitude_se: float = 0.0
    amplitude_ne: float = 0.0
    amplitude_nw: float = 0.0
    amplitude_s: float = 0.0
    amplitude_e: float = 0.0
    amplitude_n: float = 0.0
    amplitude_w: float = 0.0

    # Which sources emit (default: corners only)
    # Sources are always eligible; amp/λ/release = 0 means no contribution.
    # ``active_*`` flags remain for config compat but default to True.
    active_sw: bool = True
    active_se: bool = True
    active_ne: bool = True
    active_nw: bool = True
    active_s: bool = True
    active_e: bool = True
    active_n: bool = True
    active_w: bool = True

    # Per-source XY shockwave strength in [0, 150]:
    # 0 = locked; value ≈ peak shove in world units near the source.
    # Cloth neighbour springs keep the plane one connected surface.
    release_sw: float = 0.0
    release_se: float = 0.0
    release_ne: float = 0.0
    release_nw: float = 0.0
    release_s: float = 0.0
    release_e: float = 0.0
    release_n: float = 0.0
    release_w: float = 0.0

    # Per-source wavelength λ (0 = that source emits no spatial wave)
    wavelength_sw: float = 0.0
    wavelength_se: float = 0.0
    wavelength_ne: float = 0.0
    wavelength_nw: float = 0.0
    wavelength_s: float = 0.0
    wavelength_e: float = 0.0
    wavelength_n: float = 0.0
    wavelength_w: float = 0.0

    # Legacy / bulk λ (used when loading old configs without per-source λ)
    wavelength: float = 0.0

    # Global wave parameters
    frequency: float = 1.0
    time: float = 0.0
    decay: float = 0.0

    # Optional per-source phase offsets in radians
    phase_sw: float = 0.0
    phase_se: float = 0.0
    phase_ne: float = 0.0
    phase_nw: float = 0.0
    phase_s: float = 0.0
    phase_e: float = 0.0
    phase_n: float = 0.0
    phase_w: float = 0.0

    # Cloth / membrane stiffness in [0, 100]:
    # 0 = no internal springs; 100 = strong. Always applied with a spatial
    # gradient — core stiffer, edges more flexible — whatever the value.
    cloth: float = 0.0
    # Optional legacy dampener (kept for config compat; cloth is preferred)
    boundary_tension: float = 0.0
    # Deprecated: ignored by the radial release model (kept for config compat)
    release_pace: float = 0.0

    # Line geometry: "grid" = parallel X/Y lines; serpentine|row_major = legacy
    line_pattern: str = "grid"
    # Which grid directions to draw (only used when line_pattern == "grid")
    lines_x: bool = True
    lines_y: bool = True

    @property
    def grid_size(self) -> int:
        """Legacy square size alias (``grid_size_x``). Prefer ``grid_size_x`` / ``grid_size_y``."""
        return int(self.grid_size_x)

    @property
    def amplitudes(self) -> tuple[float, ...]:
        return (
            self.amplitude_sw,
            self.amplitude_se,
            self.amplitude_ne,
            self.amplitude_nw,
            self.amplitude_s,
            self.amplitude_e,
            self.amplitude_n,
            self.amplitude_w,
        )

    @property
    def active_flags(self) -> tuple[bool, ...]:
        return (
            self.active_sw,
            self.active_se,
            self.active_ne,
            self.active_nw,
            self.active_s,
            self.active_e,
            self.active_n,
            self.active_w,
        )

    @property
    def releases(self) -> tuple[float, ...]:
        return (
            self.release_sw,
            self.release_se,
            self.release_ne,
            self.release_nw,
            self.release_s,
            self.release_e,
            self.release_n,
            self.release_w,
        )

    @property
    def phases(self) -> tuple[float, ...]:
        return (
            self.phase_sw,
            self.phase_se,
            self.phase_ne,
            self.phase_nw,
            self.phase_s,
            self.phase_e,
            self.phase_n,
            self.phase_w,
        )

    @property
    def wavelengths(self) -> tuple[float, ...]:
        return (
            self.wavelength_sw,
            self.wavelength_se,
            self.wavelength_ne,
            self.wavelength_nw,
            self.wavelength_s,
            self.wavelength_e,
            self.wavelength_n,
            self.wavelength_w,
        )

    @property
    def effective_amplitudes(self) -> tuple[float, ...]:
        """Amplitudes as set (sources are always on; 0 amp = no wave)."""
        return self.amplitudes

    def engaged_source_labels(self) -> list[str]:
        """Sources with any non-zero amp, λ, or release."""
        engaged: list[str] = []
        for label, amp, wl, rel in zip(
            _SOURCE_LABELS, self.amplitudes, self.wavelengths, self.releases
        ):
            if amp > 0.0 or wl > 0.0 or rel > 0.0:
                engaged.append(label)
        return engaged

    def active_source_labels(self) -> list[str]:
        """Labels that are contributing (non-zero controls)."""
        return self.engaged_source_labels()

    @property
    def wave_numbers(self) -> tuple[float, ...]:
        """Per-source spatial wave numbers k_i = 2π / λ_i (0 when λ_i ≤ 0)."""
        out: list[float] = []
        for wl in self.wavelengths:
            out.append(0.0 if wl <= 0.0 else 2.0 * math.pi / wl)
        return tuple(out)

    @property
    def wave_number(self) -> float:
        """Legacy single k from bulk ``wavelength`` (prefer ``wave_numbers``)."""
        if self.wavelength <= 0.0:
            return 0.0
        return 2.0 * math.pi / self.wavelength

    @property
    def angular_frequency(self) -> float:
        """Temporal angular frequency ω = 2π f (global)."""
        return 2.0 * math.pi * self.frequency

    def with_amplitudes(
        self,
        *,
        sw: float | None = None,
        se: float | None = None,
        ne: float | None = None,
        nw: float | None = None,
        s: float | None = None,
        e: float | None = None,
        n: float | None = None,
        w: float | None = None,
    ) -> PipelineConfig:
        return replace(
            self,
            amplitude_sw=self.amplitude_sw if sw is None else sw,
            amplitude_se=self.amplitude_se if se is None else se,
            amplitude_ne=self.amplitude_ne if ne is None else ne,
            amplitude_nw=self.amplitude_nw if nw is None else nw,
            amplitude_s=self.amplitude_s if s is None else s,
            amplitude_e=self.amplitude_e if e is None else e,
            amplitude_n=self.amplitude_n if n is None else n,
            amplitude_w=self.amplitude_w if w is None else w,
        )

    def with_time(self, time: float) -> PipelineConfig:
        return replace(self, time=time)

    def to_dict(self) -> dict:
        return asdict(self)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load a PipelineConfig from a JSON file (unknown keys ignored)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {f.name: f for f in fields(PipelineConfig)}
    kwargs: dict = {}
    for name, field_info in known.items():
        if name not in raw:
            continue
        value = raw[name]
        default = field_info.default
        if isinstance(default, bool):
            kwargs[name] = bool(value)
        elif isinstance(default, int) and not isinstance(default, bool):
            kwargs[name] = int(value)
        elif isinstance(default, float):
            kwargs[name] = float(value)
        else:
            kwargs[name] = value

    # Old configs only had bulk `wavelength` — copy onto any missing per-source λ
    if "wavelength" in raw:
        bulk = float(raw["wavelength"])
        for label in _SOURCE_LABELS:
            key = f"wavelength_{label.lower()}"
            if key not in raw:
                kwargs[key] = bulk

    # Old configs used square `grid_size` only
    if "grid_size" in raw:
        n = int(raw["grid_size"])
        if "grid_size_x" not in raw:
            kwargs["grid_size_x"] = n
        if "grid_size_y" not in raw:
            kwargs["grid_size_y"] = n

    return PipelineConfig(**kwargs)


def save_pipeline_config(
    config: PipelineConfig,
    configs_dir: str | Path,
    *,
    allow_duplicates: bool = False,
) -> Path | None:
    """Save config JSON to configs_dir with a timestamp name."""
    configs_dir = Path(configs_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)
    cfg_data = config.to_dict()

    if not allow_duplicates:
        for existing_path in sorted(configs_dir.glob("*.json"), reverse=True):
            existing_data = json.loads(existing_path.read_text(encoding="utf-8"))
            if existing_data == cfg_data:
                return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = configs_dir / f"{ts}.json"
    path.write_text(json.dumps(cfg_data, indent=2), encoding="utf-8")
    return path


def list_saved_configs(configs_dir: str | Path) -> list[str]:
    """Return sorted list of saved config stems (newest first)."""
    return sorted(
        [f.stem for f in Path(configs_dir).glob("*.json")],
        reverse=True,
    )
