# cymatics-geometry

Plane-of-points geometry displaced by **multi-source cymatics wave interference**, optionally **mapped onto cylinder / cone / frustum / variable cylinder / teardrop / bead / a custom 2D vector silhouette**, then reconnected into **parallel U/V grid lines**, clipped by an optional **section box**, and optionally **piped into 3D-printable voxel solids (STL)**.

Built in the same spirit as [`enhancement-geometry`](https://github.com/Livia-Zaharia/enhancement-geometry) (the Voronoi / lofted-cylinder library used by [Materialized Enhancements](https://github.com/Livia-Zaharia/materialized-enchancements)): typed `PipelineConfig`, staged `run_pipeline`, PyVista visualization, Typer CLI, and a Jupyter notebook for debugging.

Voxel solids use **[PicoPie](https://github.com/inventhq/PicoPie)** — Python bindings for LEAP 71’s PicoGK OpenVDB kernel — so pipes can carry **surface modulations** (radius ripples / angular lobes) along each curve spine.

Author: **Livia Zaharia**

---

## Idea

1. Deploy points on an **X×Y grid** (default **100×100**, X and Y set independently; notebook sliders allow up to **500**)
2. Put wave sources at the **four corners** and optional **mid-edge** points (S/E/N/W)
3. Control per-source **amplitude**, **λ**, and **release** (0–150 world units, 1:1)
4. Move points in **Z** by wave interference; **release** applies a radial **XY** shockwave with optional **cloth** springs
5. **Map** the same UV lattice + local offsets onto a target shape (`plane` / `cylinder` / `cone` / `frustum` / `variable_cylinder` / `teardrop` / `bead` / `custom`)
6. **Custom 2D** — load SVG / DXF / DWG, scale uniformly (aspect ratio kept), stretch the square to the bbox, **crop** anything outside the silhouette
7. **Section box** — optional oriented cube (size / center / rotation on 3 axes) that **clips** the mapped geometry
8. Reconnect as **parallel U-row and V-column lines** (either direction can be turned off)
9. **Voxel pipe** (optional) — cubic-spline-smooth each line spine (`spine_smooth`), densify by `voxel_size`, sweep solid rods / hollow pipes with PicoPie voxels, modulate the radius, mesh, export **STL**

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

### Voxel pipe → STL

```python
from cymatics_geometry import (
    PipelineConfig,
    run_pipeline,
    VoxelPipeConfig,
    pipe_and_export_stl,
    save_model_params,
)

pipe = PipelineConfig(
    grid_size_x=40,
    grid_size_y=40,
    amplitude_sw=1.0,
    amplitude_se=0.85,
    wavelength_sw=25.0,
    wavelength_se=25.0,
    shape="cylinder",
    cylinder_diameter=40.0,
    cylinder_length=100.0,
    lines_x=True,
    lines_y=True,
)
result = run_pipeline(pipe)
voxel = VoxelPipeConfig(
    voxel_size=0.6,       # smaller → smoother / denser spines
    pipe_radius=1.2,      # outer tube radius
    inner_radius=0.0,     # 0 = solid rod; >0 = hollow pipe
    modulation_amp=0.25,  # surface ripple strength
    modulation_freq=3.0,  # ripples along each line
    modulation_lobes=6,   # angular flutes (0 = none)
    line_stride=2,        # every N-th grid line
    point_stride=2,       # every N-th sample on a line
    spine_samples=40,     # cubic-spline samples along each bend
    spine_smooth=1.5,     # round sharp grid kinks on the lines
)

solid, path = pipe_and_export_stl(result, "exports", voxel)
print(path, solid.volume, solid.is_watertight)

# Snapshot pipeline + nested voxel settings (same JSON as notebook / `cymatics stl --save-config`)
save_model_params(pipe, voxel, "configs")
```

Spines are **cubic-spline** resampled (with optional `spine_smooth`) after line/point stride; sample count also rises when `voxel_size` is small (linear fallback when a segment has fewer than 4 distinct points). That applies to both `preview_pipe_mesh` and `pipe_lines_to_voxels` / STL.

---

## Jupyter notebook

```bash
uv run jupyter lab notebooks/cymatics_plane_line.ipynb
```

Interactive Plotly viewer (right panel, top → bottom):

- **Display** — saved-config **Load** · `line color` · X/Y toggles · **boundary** (closed polyline tracking the original 2D outline) · **line step** · **keep X / keep Y**
- **Sources** — amp may be **negative** (flips wave / Z direction)
- **Shape map** — plane / cylinder / cone / frustum / **variable cylinder** / **teardrop** / **bead** / **custom 2D** (upload SVG/DXF/DWG; size keeps imported aspect ratio) + shape params
- **Section box** — enable + size XYZ / center XYZ / rotation XYZ + **Fit box**
- **Sources** — per-source amp / λ / release (+ sync checkboxes)
- **Mirroring** — `corners` / `mids` / `all` / `clear`: copies amp·λ·release from the first selected source (order SW→…→W). Sources **not** in the new selection reset to **0**; `clear` zeros everything
- **Global** — time, decay, cloth, grid X/Y (**20–500**)
- **Voxel print** — pipe / voxel / smooth / modulation / stride controls + **Preview solid** (lofted tubes; `voxel` changes facet density and refreshes the overlay) / **Export STL** / **Save params**

Every numeric slider has a **side number box** — click and type for precise values (linked both ways with the slider).

### Variable cylinder

Three circle radii along the axis (begin / middle / end), total length, and middle station `t ∈ [0.1, 0.9]` (0 = begin, 1 = end; extremes are clamped so the middle circle never sits flush with either end). Radius is piecewise-linear between the three stations.

### Teardrop

Five control circles along the axis (radii `R0…R4`) plus three interior stations `t1, t2, t3` (0 = base, 1 = tip) and total **height**. Radius is piecewise-linear between neighboring circles; stations stay ordered with a 0.05 minimum gap. Default is a cone-like drop that tapers to a point (`R4 = 0`). Move the circles to make fatter, longer, or blunter drops.

### Bead

Five control circles with the same move/adjust controls as the teardrop. Defaults sample a **sliced sphere** (diameter + independent top/bottom opening radii) at five equal-v stations. Changing the sphere sliders / `--bead-diameter` / `--bead-*-radius` restamps the five circles; then you can override individual radii, stations, or height. Old configs that only store the three sphere params still load — the five-circle profile is filled from that sphere.

### Custom 2D shape

Load an **SVG**, **DXF**, or **DWG**. DWG is read in-process (no extra CAD app). The drawing is scaled so its longest bounding-box side equals `custom_shape_size` — width/height ratio is never stretched. The square wave lattice is mapped onto that bbox, then **cropped** to the silhouette (`cymatics_geometry.crop`).

### Section box

An oriented cube around the mapped geometry. Sliders / CLI flags control **size** (X/Y/Z), **center**, and **rotation** (degrees about X/Y/Z). When enabled, polylines (and therefore voxel pipes) are clipped to the cube.

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

### Variable cylinder (three radii + middle station)

```bash
uv run cymatics generate \
  --grid-x 80 --grid-y 80 \
  --amp-sw 1.0 --amp-se 0.85 --amp-ne 0.55 --amp-nw 0.9 \
  --wavelength 25 \
  --shape variable_cylinder \
  --var-cyl-r-begin 20 \
  --var-cyl-r-middle 32 \
  --var-cyl-r-end 12 \
  --var-cyl-length 100 \
  --var-cyl-middle 0.4 \
  --line-color difference \
  --no-viewer \
  --screenshot exports/variable_cylinder_preview.png
```

### Teardrop (five movable circles)

```bash
uv run cymatics generate \
  --grid-x 80 --grid-y 80 \
  --amp-sw 1.0 --amp-se 0.85 --amp-ne 0.55 --amp-nw 0.9 \
  --wavelength 25 \
  --shape teardrop \
  --teardrop-height 100 \
  --teardrop-r0 22 --teardrop-r1 20 --teardrop-r2 16 --teardrop-r3 8 --teardrop-r4 0 \
  --teardrop-t1 0.20 --teardrop-t2 0.45 --teardrop-t3 0.70 \
  --line-color difference \
  --no-viewer \
  --screenshot exports/teardrop_preview.png
```

### Bead (five circles, sphere-seeded)

```bash
uv run cymatics generate \
  --grid-x 80 --grid-y 80 \
  --amp-sw 1.0 --amp-se 0.85 --amp-ne 0.55 --amp-nw 0.9 \
  --wavelength 25 \
  --shape bead \
  --bead-diameter 40 \
  --bead-bottom-radius 10 \
  --bead-top-radius 16 \
  --line-color difference \
  --no-viewer \
  --screenshot exports/bead_preview.png
```

Sphere flags fill the five circles. Override any of them (for example a fatter bulge):

```bash
uv run cymatics generate --shape bead --bead-diameter 40 --bead-r2 28 --bead-t2 0.55
```

### Custom 2D silhouette (SVG / DXF / DWG)

```bash
uv run cymatics generate \
  --grid-x 80 --grid-y 80 \
  --amp-sw 1.0 --amp-se 0.85 --amp-ne 0.55 --amp-nw 0.9 \
  --wavelength 25 \
  --shape custom \
  --custom-shape tests/fixtures/l_shape.svg \
  --custom-shape-size 100 \
  --section-box \
  --box-size-x 80 --box-size-y 80 --box-size-z 40 \
  --box-cx 50 --box-cy 40 --box-cz 0 \
  --box-rx 0 --box-ry 0 --box-rz 15 \
  --line-color difference \
  --no-viewer
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

Exports land in `exports/` as `.obj` / `.ply` (and `.stl` from the voxel command). Config snapshots land in `configs/`.

- `cymatics generate --save-config` writes **pipeline-only** JSON (`save_pipeline_config`)
- Notebook **Save params** / `cymatics stl --save-config` write **pipeline + nested `voxel`** JSON (`save_model_params`)
- `load_pipeline_config` ignores unknown keys (including `voxel`); use `load_voxel_params` / `VoxelPipeConfig.from_dict` for the nested block

### Voxel STL (printable pipes along the lines)

```bash
uv run cymatics stl \
  --grid-x 40 --grid-y 40 \
  --amp-sw 1.0 --amp-se 0.85 --amp-ne 0.55 --amp-nw 0.9 \
  --wavelength 25 \
  --shape cylinder \
  --cylinder-diameter 40 --cylinder-length 100 \
  --voxel-size 0.6 \
  --pipe-radius 1.2 \
  --mod-amp 0.25 --mod-freq 3 --mod-lobes 6 \
  --line-stride 2 \
  --spine-samples 40 \
  --spine-smooth 1.5 \
  --export-dir exports \
  --save-config
```

Or reuse a saved model-params file (pipeline + nested voxel from notebook / previous `stl` run). When the JSON has a `voxel` object, those settings are used; CLI voxel flags apply when there is no nested block:

```bash
uv run cymatics stl -c configs/<timestamp>.json --export-dir exports
# pipeline-only JSON, set voxel on the CLI:
uv run cymatics stl -c configs/<pipeline-only>.json --voxel-size 0.5 --pipe-radius 1.0
```

### Useful flags

| Flag | Meaning |
|------|---------|
| `--shape` | `plane` \| `cylinder` \| `cone` \| `frustum` \| `variable_cylinder` \| `teardrop` \| `bead` \| `custom` |
| `--custom-shape` | SVG / DXF / DWG path (required when `--shape custom`) |
| `--custom-shape-size` | longest bbox side; imported aspect ratio is kept |
| `--section-box` / `--no-section-box` | clip mapped geometry with an oriented cube |
| `--box-size-x/y/z` | section-box dimensions |
| `--box-cx/cy/cz` | section-box center |
| `--box-rx/ry/rz` | section-box rotation in degrees |
| `--cylinder-diameter` / `--cylinder-length` | cylinder params |
| `--cone-height` / `--cone-base-radius` | cone params |
| `--frustum-height` / `--frustum-base-diameter` / `--frustum-top-diameter` | frustum params |
| `--var-cyl-r-begin` / `--var-cyl-r-middle` / `--var-cyl-r-end` | variable-cylinder circle radii |
| `--var-cyl-length` / `--var-cyl-middle` | variable-cylinder length + middle station ∈ [0.1, 0.9] |
| `--teardrop-height` / `--teardrop-r0`…`--teardrop-r4` / `--teardrop-t1`…`--teardrop-t3` | teardrop height, five radii, three interior stations |
| `--bead-diameter` / `--bead-bottom-radius` / `--bead-top-radius` | bead sphere seed (fills the five circles) |
| `--bead-height` / `--bead-r0`…`--bead-r4` / `--bead-t1`…`--bead-t3` | bead five-circle overlays (optional) |
| `--line-color` | `difference` (Z coolwarm) \| `white` |
| `--lines-x` / `--no-lines-x` | U-parallel lines |
| `--lines-y` / `--no-lines-y` | V-parallel lines |
| `--grid-x` / `--grid-y` | lattice resolution |
| `--cloth` | membrane springs 0–100 |
| `--release-sw` | SW radial XY release 0–150 |
| `--viewer` / `--no-viewer` | PyVista window |
| `--screenshot PATH` | PNG preview |
| `-c` / `--config` | load pipeline JSON (and nested `voxel` for `stl` when present) |
| `--save-config` / `--no-save-config` | snapshot used settings (`generate` → pipeline; `stl` → pipeline+voxel) |
| `cymatics stl` | pipe lines → cubic-spline spines → PicoPie voxels → STL |
| `--voxel-size` | OpenVDB cell size (also densifies spines; smaller = smoother/slower) |
| `--pipe-radius` / `--inner-radius` | outer / hollow bore |
| `--mod-amp` / `--mod-freq` / `--mod-lobes` | surface modulation |
| `--line-stride` / `--point-stride` | thin the lattice for faster builds |
| `--boundary-lines-x` / `--boundary-lines-y` | signed end treatment per axis (`+N` keep, `−N` remove) |
| `--spine-samples` | cubic-spline samples along each bend |
| `--spine-smooth` | line-smoothing strength before piping (0 = interpolate through samples) |

---

## Voxel parameters (practical)

| Parameter | What it does in practice |
|-----------|--------------------------|
| `voxel_size` | Edge length of one voxel. Smaller → smoother surface, denser spines, and (in the notebook preview) more ring facets. Preview at ~0.6–1.2; print at ~0.2–0.5. |
| `pipe_radius` | Outer radius of the tube swept along each line. Keep thick enough for your nozzle / wall after unit scaling. |
| `inner_radius` | Hollow bore. `0` = solid rod (strongest / fastest). Must stay below `pipe_radius`; wall ≈ outer − inner should be ≥ ~2× `voxel_size`. |
| `modulation_amp` | How far the outer radius ripples in/out (world units). `0` = smooth constant tube. |
| `modulation_freq` | Number of full radius waves along each line’s length. |
| `modulation_lobes` | Angular flutes around the circumference (`0` = round; `6` ≈ hexagonal ripple). |
| `line_stride` | Keep every N-th grid line. `1` = dense lattice; `2+` = lighter / faster. |
| `boundary_curve` | Naked boundary of the grid network surface (continuous; follows a custom 2D contour). On by default. |
| `boundary_lines_x` / `boundary_lines_y` | Signed end treatment per axis. `+N` keep only first/last N (drops the middle); `−N` remove first/last \|N\|; `0` = stride only. |
| `point_stride` | Keep every N-th sample along a polyline before building the spine. |
| `spine_samples` | Cubic-spline arc-length samples for each spine (preview + PicoPie). Higher → smoother bends; cost grows with samples × lines. |
| `spine_smooth` | Relative smoothing before piping. `0` interpolates through grid samples (can keep kinks); higher values round sharp corners on the **lines** themselves. |

Preview tubes are **lofted** along parallel-transport frames (not stacked short cylinders), so segment seams are much less visible. `voxel_size` also drives preview facet density; with a solid overlay open, changing voxel sliders refreshes the preview automatically.

Library entry points: `VoxelPipeConfig`, `VoxelPipeConfig.from_dict`, `pipe_lines_to_voxels`, `export_stl`, `pipe_and_export_stl` in `cymatics_geometry.voxels`; `save_model_params` / `load_voxel_params` in `cymatics_geometry.config` (also re-exported from the package root).

---

## Pipeline stages

| Stage | What you see |
|------|----------------|
| 1 | Flat square point grid |
| 2 | Grid + wave sources |
| 3 | Interference field heatmap |
| 4 | Points displaced on the plane |
| 4b | Same UV + offsets mapped onto the target shape |
| 5 | Points reconnected as grid lines on the shape, then cropped / section-boxed |
| 6 | (optional) Lines cubic-spline smoothed → voxel pipes → STL |

---

## Layout

```text
cymatics-geometry/
├── configs/
├── notebooks/cymatics_plane_line.ipynb
├── src/cymatics_geometry/
│   ├── config.py          # PipelineConfig + save/load model params JSON
│   ├── grid.py            # square grid + source positions
│   ├── waves.py           # multi-source interference + release/cloth
│   ├── shapes.py          # plane → cylinder/cone/frustum/variable_cylinder/teardrop/bead/custom mapping
│   ├── custom_shape.py    # SVG / DXF / DWG load + uniform scale
│   ├── crop.py            # 2D silhouette crop + oriented section box
│   ├── lines.py           # grid U/V parallel lines
│   ├── voxels.py          # spline spines + PicoPie pipe / STL
│   ├── pipeline.py        # run_pipeline + exports
│   ├── visualization.py   # notebook + PyVista viewers
│   └── cli.py             # typer CLI (`cymatics`)
└── tests/
```

---

## Relation to enhancement-geometry

| | enhancement-geometry | cymatics-geometry |
|--|--|--|
| Base form | stacked circles → lofted half-cylinder | flat square point plane → optional cylinder/cone/frustum/variable cylinder/teardrop/bead |
| Field | Voronoi cells on lofted surface | circular waves from up to 8 sources |
| Output | watertight printable shell (STL) | lines (OBJ/PLY) **and** voxel-piped solids (STL) |
| Controls | radii, spacing, seed, extrusion | amps, λ, release, cloth, shape params, voxel pipe |

Same engineering habits: frozen config dataclass, staged pipeline result, notebook + CLI parity, timestamped STL export via trimesh, PyVista / Plotly for 3D previews.
