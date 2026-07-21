"""Pipeline configuration: dataclass, JSON I/O, and config management."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Parameters for one cymatics plane → line pipeline run.

    Four corner wave sources sit at the corners of a square grid.
    Amplitudes control how strongly each corner contributes to interference.
    """

    grid_size: int = 100
    side_length: float = 100.0
    # Corner amplitudes: (SW, SE, NE, NW) — counter-clockwise from bottom-left
    amplitude_sw: float = 1.0
    amplitude_se: float = 1.0
    amplitude_ne: float = 1.0
    amplitude_nw: float = 1.0
    wavelength: float = 25.0
    frequency: float = 1.0
    time: float = 0.0
    # Optional per-corner phase offsets in radians
    phase_sw: float = 0.0
    phase_se: float = 0.0
    phase_ne: float = 0.0
    phase_nw: float = 0.0
    # Distance decay of wave amplitude (0 = no decay)
    decay: float = 0.0
    # How line reconnection walks the grid: "serpentine" or "row_major"
    line_pattern: str = "serpentine"

    @property
    def amplitudes(self) -> tuple[float, float, float, float]:
        return (
            self.amplitude_sw,
            self.amplitude_se,
            self.amplitude_ne,
            self.amplitude_nw,
        )

    @property
    def phases(self) -> tuple[float, float, float, float]:
        return (self.phase_sw, self.phase_se, self.phase_ne, self.phase_nw)

    @property
    def wave_number(self) -> float:
        """Spatial angular wave number k = 2π / λ."""
        if self.wavelength <= 0.0:
            raise ValueError("wavelength must be > 0")
        return 2.0 * math.pi / self.wavelength

    @property
    def angular_frequency(self) -> float:
        """Temporal angular frequency ω = 2π f."""
        return 2.0 * math.pi * self.frequency

    def with_amplitudes(
        self,
        *,
        sw: float | None = None,
        se: float | None = None,
        ne: float | None = None,
        nw: float | None = None,
    ) -> PipelineConfig:
        return replace(
            self,
            amplitude_sw=self.amplitude_sw if sw is None else sw,
            amplitude_se=self.amplitude_se if se is None else se,
            amplitude_ne=self.amplitude_ne if ne is None else ne,
            amplitude_nw=self.amplitude_nw if nw is None else nw,
        )

    def with_time(self, time: float) -> PipelineConfig:
        return replace(self, time=time)

    def to_dict(self) -> dict:
        return asdict(self)


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load a PipelineConfig from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return PipelineConfig(
        grid_size=int(raw.get("grid_size", 100)),
        side_length=float(raw.get("side_length", 100.0)),
        amplitude_sw=float(raw.get("amplitude_sw", 1.0)),
        amplitude_se=float(raw.get("amplitude_se", 1.0)),
        amplitude_ne=float(raw.get("amplitude_ne", 1.0)),
        amplitude_nw=float(raw.get("amplitude_nw", 1.0)),
        wavelength=float(raw.get("wavelength", 25.0)),
        frequency=float(raw.get("frequency", 1.0)),
        time=float(raw.get("time", 0.0)),
        phase_sw=float(raw.get("phase_sw", 0.0)),
        phase_se=float(raw.get("phase_se", 0.0)),
        phase_ne=float(raw.get("phase_ne", 0.0)),
        phase_nw=float(raw.get("phase_nw", 0.0)),
        decay=float(raw.get("decay", 0.0)),
        line_pattern=str(raw.get("line_pattern", "serpentine")),
    )


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
