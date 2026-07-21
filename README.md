# cymatics-geometry

Plane-of-points geometry displaced by **four-corner cymatics wave interference**, then reconnected into a continuous **line**.

Built in the same spirit as [`enhancement-geometry`](https://github.com/Livia-Zaharia/enhancement-geometry) (the Voronoi / lofted-cylinder library used by [Materialized Enhancements](https://github.com/Livia-Zaharia/materialized-enchancements)): typed `PipelineConfig`, staged `run_pipeline`, PyVista visualization, Typer CLI, and a Jupyter notebook for debugging.

Author: **Livia Zaharia**

---

## Idea

Instead of lofting stacked circles into a cylindrical shell, this library:

1. Deploys points on a **square grid** (default **100×100**)
2. Puts **wave-producing outputs at the four corners** (SW, SE, NE, NW)
3. Lets sliders / config control each corner’s **amplitude / intensity**
4. Moves every point in **Z** according to **wave interference**
5. Reconnects the displaced points into a **serpentine line** geometry

Each corner emits a circular travelling wave:

```text
z_i(r) = A_i · sin(k · r − ω · t + φ_i) / (1 + decay · r)
z     = z_SW + z_SE + z_NE + z_NW
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
    grid_size=100,
    amplitude_sw=1.0,
    amplitude_se=0.85,
    amplitude_ne=0.55,
    amplitude_nw=0.9,
    wavelength=25.0,
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

The notebook shows every stage and includes **ipywidgets sliders** for the four corner amplitudes, wavelength, time, and decay.

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
│   ├── lines.py           # serpentine / row-major reconnection
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
