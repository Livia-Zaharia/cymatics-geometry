# cymatics-geometry

Plane-of-points geometry displaced by **multi-source cymatics wave interference**, optionally **mapped onto cylinder / cone / frustum**, then reconnected into **parallel U/V grid lines**.

Built in the same spirit as [`enhancement-geometry`](https://github.com/Livia-Zaharia/enhancement-geometry) (the Voronoi / lofted-cylinder library used by [Materialized Enhancements](https://github.com/Livia-Zaharia/materialized-enchancements)): typed `PipelineConfig`, staged `run_pipeline`, PyVista visualization, Typer CLI, and a Jupyter notebook for debugging.

Author: **Livia Zaharia**

---

## Idea

1. Deploy points on an **X×Y grid** (default **100×100**, X and Y set independently)
2. Put wave sources at the **four corners** and optional **mid-edge** points (S/E/N/W)
3. Control per-source **amplitude**, **λ**, and **release** (0–150 world units, 1:1)
4. Move points in **Z** by wave interference; **release** applies a radial **XY** shockwave with optional **cloth** springs
5. **Map** the same UV lattice + local offsets onto a target shape (`plane` / `cylinder` / `cone` / `frustum`)
6. Reconnect as **parallel U-row and V-column lines** (either direction can be turned off)

Waves always compute on the flat plane. Mapping reuses the plane-frame offset `δ = (dx, dy, dz)` in the target surface’s local frame `(Tu, Tv, N)`:

```text
P'(u,v) = P(u,v) + dx·Tu + dy·Tv + dz·N
```

---

## Install

```bash
uv sync
```

---

## Quick start (Python)

```python
from cymatics_geometry import PipelineConfig, run_pipeline, export_line_obj
from cymatics_geometry.visualization import show_all_stages_matplotlib

config = PipelineConfig(
    grid_size_x=80,
    grid_size_y=80,
    amplitude_sw=1.0,
    amplitude_se=0.85,
    amplitude_ne=0.55,
    amplitude_nw=0.9,
    wavelength_sw=25.0,
    wavelength_se=25.0,
    wavelength_ne=25.0,
    wavelength_nw=25.0,
    shape="cylinder",
    cylinder_diameter=40.0,
    cylinder_length=100.0,
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

Interactive Plotly viewer (right panel, top → bottom):

- **Display** — `line color` (`difference` / `uniform white`) + X/Y line toggles
- **Shape map** — plane / cylinder / cone / frustum + shape params
- **Sources** — per-source amp / λ / release (+ sync)
- **Global** — time, decay, cloth, grid resolution

Re-run the import cell after pulling library changes so the notebook picks up `src/`.

---

## CLI

Entry point: `uv run cymatics` (Typer app in `src/cymatics_geometry/cli.py`).

### Help

```bash
uv run cymatics --help
uv run cymatics generate --help
```

### Plane (default) — corner waves → grid lines

Matches the notebook’s static demo (unequal corner amps, λ=25):

```bash
uv run cymatics generate \
  --grid-x 80 --grid-y 80 \
  --amp-sw 1.0 --amp-se 0.85 --amp-ne 0.55 --amp-nw 0.9 \
  --wavelength 25 \
  --shape plane \
  --line-color difference \
  --no-viewer \
  --screenshot exports/plane_preview.png
```

Solid white lines (same geometry):

```bash
uv run cymatics generate \
  --amp-sw 1.0 --amp-se 0.85 --amp-ne 0.55 --amp-nw 0.9 \
  --wavelength 25 \
  --line-color white \
  --no-viewer \
  --screenshot exports/plane_white.png
```

### Cylinder — wrap plane UV onto a tube

```bash
uv run cymatics generate \
  --grid-x 60 --grid-y 60 \
  --amp-sw 1.2 --amp-se 1.0 --amp-ne 0.8 --amp-nw 1.1 \
  --wavelength 22 \
  --shape cylinder \
  --cylinder-diameter 40 \
  --cylinder-length 100 \
  --line-color difference \
  --no-viewer \
  --screenshot exports/cylinder_preview.png
```

### Cone

```bash
uv run cymatics generate \
  --amp-sw 1.0 --amp-se 0.9 --amp-ne 0.7 --amp-nw 0.95 \
  --wavelength 25 \
  --shape cone \
  --cone-height 100 \
  --cone-base-radius 30 \
  --line-color white \
  --no-viewer \
  --screenshot exports/cone_white.png
```

### Frustum (truncated cone — two circle diameters + height)

```bash
uv run cymatics generate \
  --amp-sw 1.0 --amp-se 0.85 --amp-ne 0.55 --amp-nw 0.9 \
  --wavelength 25 \
  --shape frustum \
  --frustum-height 100 \
  --frustum-base-diameter 60 \
  --frustum-top-diameter 20 \
  --line-color difference \
  --no-viewer \
  --screenshot exports/frustum_preview.png
```

### Release + cloth (XY shockwave on the plane, then mapped)

```bash
uv run cymatics generate \
  --amp-sw 0 --wavelength 0 \
  --release-sw 40 \
  --cloth 60 \
  --shape cylinder \
  --cylinder-diameter 50 \
  --cylinder-length 120 \
  --line-color difference \
  --no-viewer
```

### Open the PyVista window (interactive orbit)

Omit `--no-viewer` (viewer is on by default):

```bash
uv run cymatics generate --shape cone --line-color white
```

### From a saved config (notebook / previous CLI run)

```bash
uv run cymatics list-configs
uv run cymatics show-config configs/<timestamp>.json
uv run cymatics generate -c configs/<timestamp>.json --line-color white --screenshot exports/from_config.png
```

Exports land in `exports/` as `.obj` / `.ply`. Config snapshots land in `configs/`.

### Useful flags

| Flag | Meaning |
|------|---------|
| `--shape` | `plane` \| `cylinder` \| `cone` \| `frustum` |
| `--cylinder-diameter` / `--cylinder-length` | cylinder params |
| `--cone-height` / `--cone-base-radius` | cone params |
| `--frustum-height` / `--frustum-base-diameter` / `--frustum-top-diameter` | frustum params |
| `--line-color` | `difference` (Z coolwarm) \| `white` |
| `--lines-x` / `--no-lines-x` | U-parallel lines |
| `--lines-y` / `--no-lines-y` | V-parallel lines |
| `--grid-x` / `--grid-y` | lattice resolution |
| `--cloth` | membrane springs 0–100 |
| `--release-sw` | SW radial XY release 0–150 |
| `--viewer` / `--no-viewer` | PyVista window |
| `--screenshot PATH` | PNG preview |
| `-c` / `--config` | load `PipelineConfig` JSON |

---

## Pipeline stages

| Stage | What you see |
|------|----------------|
| 1 | Flat square point grid |
| 2 | Grid + wave sources |
| 3 | Interference field heatmap |
| 4 | Points displaced on the plane |
| 4b | Same UV + offsets mapped onto the target shape |
| 5 | Points reconnected as grid lines on the shape |

---

## Layout

```text
cymatics-geometry/
├── configs/
├── notebooks/cymatics_plane_line.ipynb
├── src/cymatics_geometry/
│   ├── config.py          # PipelineConfig + JSON I/O
│   ├── grid.py            # square grid + source positions
│   ├── waves.py           # multi-source interference + release/cloth
│   ├── shapes.py          # plane → cylinder/cone/frustum mapping
│   ├── lines.py           # grid U/V parallel lines
│   ├── pipeline.py        # run_pipeline + exports
│   ├── visualization.py   # notebook + PyVista viewers
│   └── cli.py             # typer CLI (`cymatics`)
└── tests/
```

---

## Relation to enhancement-geometry

| | enhancement-geometry | cymatics-geometry |
|--|--|--|
| Base form | stacked circles → lofted half-cylinder | flat square point plane → optional cylinder/cone/frustum |
| Field | Voronoi cells on lofted surface | circular waves from up to 8 sources |
| Output | watertight printable shell (STL) | continuous line geometry (OBJ/PLY) |
| Controls | radii, spacing, seed, extrusion | amps, λ, release, cloth, shape params |

Same engineering habits: frozen config dataclass, staged pipeline result, notebook + CLI parity, PyVista for 3D previews.
