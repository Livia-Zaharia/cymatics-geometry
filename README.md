# cymatics-geometry

Plane-of-points geometry displaced by **multi-source cymatics wave interference**, then reconnected into **parallel X/Y grid lines**.

Built in the same spirit as [`enhancement-geometry`](https://github.com/Livia-Zaharia/enhancement-geometry) (the Voronoi / lofted-cylinder library used by [Materialized Enhancements](https://github.com/Livia-Zaharia/materialized-enchancements)): typed `PipelineConfig`, staged `run_pipeline`, PyVista visualization, Typer CLI, and a Jupyter notebook for debugging.

Author: **Livia Zaharia**

---

## Idea

Instead of lofting stacked circles into a cylindrical shell, this library:

1. Deploys points on an **X×Y grid** (default **100×100**, X and Y set independently)
2. Puts wave sources at the **four corners** and optional **mid-edge** points (S/E/N/W)
3. Controls per-source **amplitude**, **λ**, and **release** (0–150 world units, 1:1); sources are always eligible — leave sliders at 0 to idle
4. Moves points in **Z** by wave interference; **release** applies a radial **XY** shockwave with optional **cloth** springs (core stiff / edge soft)
5. Reconnects the displaced points into **parallel X-row and Y-column lines** (either direction can be turned off)

Each source with amp/λ emits a circular travelling wave
(`time` / `decay` / `cloth` are global). Numeric controls default to **0**:

```text
z_i(r) = A_i · sin(k_i · r − ω · t + φ_i) / (1 + decay · r)   with  k_i = 2π / λ_i
z     = sum of sources
release ∈ [0, 150]:  0 = XY locked · value ≈ peak world-unit shove near the source
```

---

## Install

```bash
uv sync
```

Optional notebook extras are already in the project dependencies (`jupyterlab`, `ipywidgets`, `matplotlib`).

---

## Quick start (Python)

```python
from cymatics_geometry import PipelineConfig, run_pipeline, export_line_obj
from cymatics_geometry.visualization import show_all_stages_matplotlib

config = PipelineConfig(
    grid_size_x=100,
    grid_size_y=100,
    amplitude_sw=1.0,
    amplitude_se=0.85,
    amplitude_ne=0.55,
    amplitude_nw=0.9,
    wavelength_sw=25.0,
    wavelength_se=25.0,
    wavelength_ne=25.0,
    wavelength_nw=25.0,
)

result = run_pipeline(config)
show_all_stages_matplotlib(result)
export_line_obj(result, "exports")
```

---

## Jupyter notebook

```bash
uv run jupyter lab notebooks/cymatics_plane_line.ipynb
```

The notebook explains each control and includes an **orbitable Plotly 3D line** with per-source on/off, amplitude, release, plus global λ / time / decay / boundary tension.

---

## CLI

```bash
# Generate line geometry with unequal corner intensities
uv run cymatics generate \
  --amp-sw 1.0 --amp-se 0.8 --amp-ne 0.4 --amp-nw 0.9 \
  --wavelength 25 --no-viewer

# From a saved config
uv run cymatics generate -c configs/default.json --screenshot exports/preview.png
```

Exports land in `exports/` as `.obj` and `.ply` line meshes. Config snapshots land in `configs/`.

---

## Pipeline stages

| Stage | What you see |
|------|----------------|
| 1 | Flat square point grid |
| 2 | Grid + corner wave sources |
| 3 | Interference field heatmap |
| 4 | Points displaced in space |
| 5 | Points reconnected as one continuous line |

---

## Layout

```text
cymatics-geometry/
├── configs/default.json
├── notebooks/cymatics_plane_line.ipynb
├── src/cymatics_geometry/
│   ├── config.py          # PipelineConfig + JSON I/O
│   ├── grid.py            # square grid + corner positions
│   ├── waves.py           # four-corner interference
│   ├── lines.py           # grid X/Y parallel lines (+ legacy serpentine)
│   ├── pipeline.py        # run_pipeline + exports
│   ├── visualization.py   # stage viewers
│   └── cli.py             # typer CLI (`cymatics`)
└── tests/
```

---

## Relation to enhancement-geometry

| | enhancement-geometry | cymatics-geometry |
|--|--|--|
| Base form | stacked circles → lofted half-cylinder | flat square point plane |
| Field | Voronoi cells on lofted surface | circular waves from 4 corners |
| Output | watertight printable shell (STL) | continuous line geometry (OBJ/PLY) |
| Controls | radii, spacing, seed, extrusion | corner amplitudes, λ, time, decay |

Same engineering habits: frozen config dataclass, staged pipeline result, notebook + CLI parity, PyVista for 3D previews.
