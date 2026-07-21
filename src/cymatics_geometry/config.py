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

    Per-source: amplitude, wavelength (λ), release (0 locked … 1 free … 10×).
    Global: time, frequency, decay. Optional boundary_tension dampens XY motion.
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
    active_sw: bool = True
    active_se: bool = True
    active_ne: bool = True
    active_nw: bool = True
    active_s: bool = False
    active_e: bool = False
    active_n: bool = False
    active_w: bool = False

    # Per-source XY release mobility in [0, 10]:
    # 0 = locked in original XY, 1 = free under wave influence, 10 = 10× influence.
    # Spreads radially from the source across the grid.
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

    # Optional global dampener on XY release (0 = none)
    boundary_tension: float = 0.0
    # Deprecated: ignored by the radial release model (kept for config compat)
    release_pace: float = 0.0

    line_pattern: str = "serpentine"

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
        """Amplitudes with inactive sources zeroed."""
        return tuple(
            amp if on else 0.0 for amp, on in zip(self.amplitudes, self.active_flags)
        )

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

    def active_source_labels(self) -> list[str]:
        return [label for label, on in zip(_SOURCE_LABELS, self.active_flags) if on]

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
