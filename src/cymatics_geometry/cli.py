"""CLI for the cymatics plane → shape → line geometry generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="cymatics",
    help=(
        "Generate line geometry from a point plane displaced by multi-source "
        "cymatics waves, optionally mapped onto cylinder / cone / frustum / "
        "variable cylinder / bead / a custom 2D vector silhouette, then clipped "
        "by an optional section box."
    ),
    add_completion=False,
)


@app.command()
def generate(
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to a saved pipeline config JSON file."),
    ] = None,
    grid_size: Annotated[
        int,
        typer.Option(
            "--grid-size",
            "-n",
            help="Square grid shortcut (sets both X and Y point counts).",
        ),
    ] = 100,
    grid_size_x: Annotated[
        Optional[int],
        typer.Option("--grid-x", help="Point count along X (overrides --grid-size for X)."),
    ] = None,
    grid_size_y: Annotated[
        Optional[int],
        typer.Option("--grid-y", help="Point count along Y (overrides --grid-size for Y)."),
    ] = None,
    side_length: Annotated[
        float,
        typer.Option("--side", help="Square side length (plane UV domain)."),
    ] = 100.0,
    amp_sw: Annotated[float, typer.Option("--amp-sw", help="SW corner amplitude.")] = 1.0,
    amp_se: Annotated[float, typer.Option("--amp-se", help="SE corner amplitude.")] = 0.85,
    amp_ne: Annotated[float, typer.Option("--amp-ne", help="NE corner amplitude.")] = 0.55,
    amp_nw: Annotated[float, typer.Option("--amp-nw", help="NW corner amplitude.")] = 0.9,
    amp_s: Annotated[float, typer.Option("--amp-s", help="South mid-edge amplitude.")] = 0.0,
    amp_e: Annotated[float, typer.Option("--amp-e", help="East mid-edge amplitude.")] = 0.0,
    amp_n: Annotated[float, typer.Option("--amp-n", help="North mid-edge amplitude.")] = 0.0,
    amp_w: Annotated[float, typer.Option("--amp-w", help="West mid-edge amplitude.")] = 0.0,
    wavelength: Annotated[
        float,
        typer.Option("--wavelength", "-l", help="Wave length (lambda) for active corner sources."),
    ] = 25.0,
    frequency: Annotated[float, typer.Option("--frequency", "-f", help="Wave frequency.")] = 1.0,
    time: Annotated[float, typer.Option("--time", "-t", help="Simulation time.")] = 0.0,
    decay: Annotated[float, typer.Option("--decay", help="Distance amplitude decay.")] = 0.0,
    cloth: Annotated[
        float,
        typer.Option("--cloth", help="Cloth spring strength 0–100 (core stiff / edge soft)."),
    ] = 0.0,
    release_sw: Annotated[
        float,
        typer.Option("--release-sw", help="SW radial XY release 0–150."),
    ] = 0.0,
    pattern: Annotated[
        str,
        typer.Option(
            "--pattern",
            help="Line reconnection: grid|serpentine|row_major.",
        ),
    ] = "grid",
    lines_x: Annotated[
        bool,
        typer.Option("--lines-x/--no-lines-x", help="Draw X/U-parallel grid lines."),
    ] = True,
    lines_y: Annotated[
        bool,
        typer.Option("--lines-y/--no-lines-y", help="Draw Y/V-parallel grid lines."),
    ] = True,
    boundary_curve: Annotated[
        bool,
        typer.Option(
            "--boundary-curve/--no-boundary-curve",
            help="Closed polyline tracking the original 2D outline through the waves.",
        ),
    ] = True,
    line_stride: Annotated[
        int,
        typer.Option(
            "--line-stride",
            help="Keep every N-th grid line in the drawn geometry (1 = all).",
        ),
    ] = 1,
    boundary_lines_x: Annotated[
        int,
        typer.Option(
            "--boundary-lines-x",
            help="X-rows: +N keep only first/last N (drops the middle); −N shave ends.",
        ),
    ] = 0,
    boundary_lines_y: Annotated[
        int,
        typer.Option(
            "--boundary-lines-y",
            help="Y-cols: +N keep only first/last N (drops the middle); −N shave ends.",
        ),
    ] = 0,
    shape: Annotated[
        str,
        typer.Option(
            "--shape",
            help=(
                "Target surface: plane|cylinder|cone|frustum|variable_cylinder|"
                "bead|custom."
            ),
        ),
    ] = "plane",
    cylinder_diameter: Annotated[
        float,
        typer.Option("--cylinder-diameter", help="Cylinder diameter (shape=cylinder)."),
    ] = 40.0,
    cylinder_length: Annotated[
        float,
        typer.Option("--cylinder-length", help="Cylinder length along V (shape=cylinder)."),
    ] = 100.0,
    cone_height: Annotated[
        float,
        typer.Option("--cone-height", help="Cone height (shape=cone)."),
    ] = 100.0,
    cone_base_radius: Annotated[
        float,
        typer.Option("--cone-base-radius", help="Cone base radius at v=0 (shape=cone)."),
    ] = 30.0,
    frustum_height: Annotated[
        float,
        typer.Option("--frustum-height", help="Frustum height (shape=frustum)."),
    ] = 100.0,
    frustum_base_diameter: Annotated[
        float,
        typer.Option(
            "--frustum-base-diameter",
            help="Frustum base circle diameter at v=0.",
        ),
    ] = 60.0,
    frustum_top_diameter: Annotated[
        float,
        typer.Option(
            "--frustum-top-diameter",
            help="Frustum top circle diameter at v=1.",
        ),
    ] = 20.0,
    var_cyl_radius_begin: Annotated[
        float,
        typer.Option(
            "--var-cyl-r-begin",
            help="Variable-cylinder begin circle radius (v=0).",
        ),
    ] = 20.0,
    var_cyl_radius_middle: Annotated[
        float,
        typer.Option(
            "--var-cyl-r-middle",
            help="Variable-cylinder middle circle radius.",
        ),
    ] = 30.0,
    var_cyl_radius_end: Annotated[
        float,
        typer.Option(
            "--var-cyl-r-end",
            help="Variable-cylinder end circle radius (v=1).",
        ),
    ] = 15.0,
    var_cyl_length: Annotated[
        float,
        typer.Option(
            "--var-cyl-length",
            help="Variable-cylinder total length along V.",
        ),
    ] = 100.0,
    var_cyl_middle: Annotated[
        float,
        typer.Option(
            "--var-cyl-middle",
            help="Middle-circle station along V in [0.1, 0.9] (0=begin, 1=end).",
        ),
    ] = 0.5,
    bead_diameter: Annotated[
        float,
        typer.Option("--bead-diameter", help="Bead sphere diameter (shape=bead)."),
    ] = 40.0,
    bead_bottom_radius: Annotated[
        float,
        typer.Option(
            "--bead-bottom-radius",
            help="Bead bottom slice circle radius (independent of top).",
        ),
    ] = 12.0,
    bead_top_radius: Annotated[
        float,
        typer.Option(
            "--bead-top-radius",
            help="Bead top slice circle radius (independent of bottom).",
        ),
    ] = 12.0,
    custom_shape: Annotated[
        Optional[Path],
        typer.Option(
            "--custom-shape",
            help="2D vector file (SVG/DXF/DWG) when --shape custom. Aspect ratio kept.",
        ),
    ] = None,
    custom_shape_size: Annotated[
        float,
        typer.Option(
            "--custom-shape-size",
            help="Longest bbox side of the imported 2D shape (uniform scale).",
        ),
    ] = 100.0,
    section_box: Annotated[
        bool,
        typer.Option("--section-box/--no-section-box", help="Clip geometry with an oriented cube."),
    ] = False,
    box_size_x: Annotated[float, typer.Option("--box-size-x", help="Section box size X.")] = 120.0,
    box_size_y: Annotated[float, typer.Option("--box-size-y", help="Section box size Y.")] = 120.0,
    box_size_z: Annotated[float, typer.Option("--box-size-z", help="Section box size Z.")] = 120.0,
    box_cx: Annotated[float, typer.Option("--box-cx", help="Section box center X.")] = 50.0,
    box_cy: Annotated[float, typer.Option("--box-cy", help="Section box center Y.")] = 50.0,
    box_cz: Annotated[float, typer.Option("--box-cz", help="Section box center Z.")] = 0.0,
    box_rx: Annotated[float, typer.Option("--box-rx", help="Section box rotation X (deg).")] = 0.0,
    box_ry: Annotated[float, typer.Option("--box-ry", help="Section box rotation Y (deg).")] = 0.0,
    box_rz: Annotated[float, typer.Option("--box-rz", help="Section box rotation Z (deg).")] = 0.0,
    line_color: Annotated[
        str,
        typer.Option(
            "--line-color",
            help="Viewer/screenshot line colour: difference|white.",
        ),
    ] = "difference",
    export_dir: Annotated[
        Path,
        typer.Option("--export-dir", "-o", help="Directory for exported OBJ/PLY files."),
    ] = Path("exports"),
    configs_dir: Annotated[
        Path,
        typer.Option("--configs-dir", help="Directory for saving config snapshots."),
    ] = Path("configs"),
    save_config: Annotated[
        bool,
        typer.Option("--save-config/--no-save-config", help="Save the config used for this run."),
    ] = True,
    viewer: Annotated[
        bool,
        typer.Option("--viewer/--no-viewer", help="Open interactive 3D viewer after generation."),
    ] = True,
    screenshot: Annotated[
        Optional[Path],
        typer.Option("--screenshot", help="Save a PNG screenshot to this path."),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress progress output."),
    ] = False,
) -> None:
    """Run the full cymatics pipeline and export line geometry."""
    from cymatics_geometry.config import (
        PipelineConfig,
        load_pipeline_config,
        save_pipeline_config,
    )
    from cymatics_geometry.pipeline import export_line_obj, export_line_ply, run_pipeline
    from cymatics_geometry.shapes import SHAPE_KINDS

    color_mode = str(line_color).lower().strip()
    if color_mode not in {"difference", "white"}:
        typer.echo("--line-color must be 'difference' or 'white'", err=True)
        raise typer.Exit(1)

    shape_kind = str(shape).lower().strip()
    if shape_kind not in SHAPE_KINDS:
        typer.echo(
            f"--shape must be one of: {', '.join(SHAPE_KINDS)}",
            err=True,
        )
        raise typer.Exit(1)

    if shape_kind == "custom" and config is None:
        if custom_shape is None or not custom_shape.exists():
            typer.echo(
                "shape=custom requires --custom-shape PATH to an SVG/DXF/DWG file",
                err=True,
            )
            raise typer.Exit(1)

    if config is not None:
        if not config.exists():
            typer.echo(f"Config file not found: {config}", err=True)
            raise typer.Exit(1)
        pipeline_config = load_pipeline_config(config)
        if not quiet:
            typer.echo(f"Loaded config from {config}")
    else:
        nx = grid_size if grid_size_x is None else grid_size_x
        ny = grid_size if grid_size_y is None else grid_size_y
        pipeline_config = PipelineConfig(
            grid_size_x=nx,
            grid_size_y=ny,
            side_length=side_length,
            amplitude_sw=amp_sw,
            amplitude_se=amp_se,
            amplitude_ne=amp_ne,
            amplitude_nw=amp_nw,
            amplitude_s=amp_s,
            amplitude_e=amp_e,
            amplitude_n=amp_n,
            amplitude_w=amp_w,
            wavelength=wavelength,
            wavelength_sw=wavelength,
            wavelength_se=wavelength,
            wavelength_ne=wavelength,
            wavelength_nw=wavelength,
            frequency=frequency,
            time=time,
            decay=decay,
            cloth=cloth,
            release_sw=release_sw,
            line_pattern=pattern,
            lines_x=lines_x,
            lines_y=lines_y,
            boundary_curve=bool(boundary_curve),
            line_stride=max(1, int(line_stride)),
            boundary_lines_x=int(boundary_lines_x),
            boundary_lines_y=int(boundary_lines_y),
            shape=shape_kind,
            cylinder_diameter=cylinder_diameter,
            cylinder_length=cylinder_length,
            cone_height=cone_height,
            cone_base_radius=cone_base_radius,
            frustum_height=frustum_height,
            frustum_base_diameter=frustum_base_diameter,
            frustum_top_diameter=frustum_top_diameter,
            variable_cylinder_radius_begin=var_cyl_radius_begin,
            variable_cylinder_radius_middle=var_cyl_radius_middle,
            variable_cylinder_radius_end=var_cyl_radius_end,
            variable_cylinder_length=var_cyl_length,
            variable_cylinder_middle=max(0.1, min(0.9, float(var_cyl_middle))),
            bead_diameter=bead_diameter,
            bead_bottom_radius=bead_bottom_radius,
            bead_top_radius=bead_top_radius,
            custom_shape_path="" if custom_shape is None else str(custom_shape),
            custom_shape_size=float(custom_shape_size),
            section_box_enabled=bool(section_box),
            section_box_size_x=float(box_size_x),
            section_box_size_y=float(box_size_y),
            section_box_size_z=float(box_size_z),
            section_box_center_x=float(box_cx),
            section_box_center_y=float(box_cy),
            section_box_center_z=float(box_cz),
            section_box_rot_x=float(box_rx),
            section_box_rot_y=float(box_ry),
            section_box_rot_z=float(box_rz),
        )

    if not quiet:
        typer.echo("Running cymatics plane → shape → line pipeline...")

    result = run_pipeline(pipeline_config, verbose=not quiet)

    obj_path = export_line_obj(result, export_dir)
    ply_path = export_line_ply(result, export_dir)
    typer.echo(f"OBJ exported: {obj_path}")
    typer.echo(f"PLY exported: {ply_path}")

    if save_config:
        saved = save_pipeline_config(pipeline_config, configs_dir)
        if saved is not None:
            typer.echo(f"Config saved: {saved}")
        elif not quiet:
            typer.echo("Config identical to existing, not saved.")

    if screenshot is not None:
        from cymatics_geometry.visualization import save_line_screenshot

        save_line_screenshot(
            result,
            screenshot,
            line_color_mode=color_mode,
        )
        typer.echo(f"Screenshot saved: {screenshot}")

    if viewer:
        from cymatics_geometry.visualization import show_stage_line

        typer.echo("Opening interactive viewer (close window to continue)...")
        show_stage_line(result, line_color_mode=color_mode)


@app.command()
def stl(
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to a saved pipeline config JSON file."),
    ] = None,
    grid_size: Annotated[
        int,
        typer.Option("--grid-size", "-n", help="Square grid shortcut (X and Y)."),
    ] = 40,
    grid_size_x: Annotated[
        Optional[int],
        typer.Option("--grid-x", help="Point count along X."),
    ] = None,
    grid_size_y: Annotated[
        Optional[int],
        typer.Option("--grid-y", help="Point count along Y."),
    ] = None,
    amp_sw: Annotated[float, typer.Option("--amp-sw", help="SW corner amplitude.")] = 1.0,
    amp_se: Annotated[float, typer.Option("--amp-se", help="SE corner amplitude.")] = 0.85,
    amp_ne: Annotated[float, typer.Option("--amp-ne", help="NE corner amplitude.")] = 0.55,
    amp_nw: Annotated[float, typer.Option("--amp-nw", help="NW corner amplitude.")] = 0.9,
    wavelength: Annotated[
        float,
        typer.Option("--wavelength", "-l", help="Wave length for active corner sources."),
    ] = 25.0,
    shape: Annotated[
        str,
        typer.Option(
            "--shape",
            help=(
                "Target surface: plane|cylinder|cone|frustum|variable_cylinder|"
                "bead|custom."
            ),
        ),
    ] = "plane",
    cylinder_diameter: Annotated[
        float,
        typer.Option("--cylinder-diameter", help="Cylinder diameter (shape=cylinder)."),
    ] = 40.0,
    cylinder_length: Annotated[
        float,
        typer.Option("--cylinder-length", help="Cylinder length (shape=cylinder)."),
    ] = 100.0,
    cone_height: Annotated[
        float,
        typer.Option("--cone-height", help="Cone height (shape=cone)."),
    ] = 100.0,
    cone_base_radius: Annotated[
        float,
        typer.Option("--cone-base-radius", help="Cone base radius (shape=cone)."),
    ] = 30.0,
    frustum_height: Annotated[
        float,
        typer.Option("--frustum-height", help="Frustum height (shape=frustum)."),
    ] = 100.0,
    frustum_base_diameter: Annotated[
        float,
        typer.Option("--frustum-base-diameter", help="Frustum base diameter."),
    ] = 60.0,
    frustum_top_diameter: Annotated[
        float,
        typer.Option("--frustum-top-diameter", help="Frustum top diameter."),
    ] = 20.0,
    var_cyl_radius_begin: Annotated[
        float,
        typer.Option("--var-cyl-r-begin", help="Variable-cylinder begin radius."),
    ] = 20.0,
    var_cyl_radius_middle: Annotated[
        float,
        typer.Option("--var-cyl-r-middle", help="Variable-cylinder middle radius."),
    ] = 30.0,
    var_cyl_radius_end: Annotated[
        float,
        typer.Option("--var-cyl-r-end", help="Variable-cylinder end radius."),
    ] = 15.0,
    var_cyl_length: Annotated[
        float,
        typer.Option("--var-cyl-length", help="Variable-cylinder length."),
    ] = 100.0,
    var_cyl_middle: Annotated[
        float,
        typer.Option(
            "--var-cyl-middle",
            help="Middle-circle station along V in [0.1, 0.9].",
        ),
    ] = 0.5,
    bead_diameter: Annotated[
        float,
        typer.Option("--bead-diameter", help="Bead sphere diameter (shape=bead)."),
    ] = 40.0,
    bead_bottom_radius: Annotated[
        float,
        typer.Option("--bead-bottom-radius", help="Bead bottom slice radius."),
    ] = 12.0,
    bead_top_radius: Annotated[
        float,
        typer.Option("--bead-top-radius", help="Bead top slice radius."),
    ] = 12.0,
    custom_shape: Annotated[
        Optional[Path],
        typer.Option(
            "--custom-shape",
            help="2D vector file (SVG/DXF/DWG) when --shape custom.",
        ),
    ] = None,
    custom_shape_size: Annotated[
        float,
        typer.Option("--custom-shape-size", help="Longest bbox side (uniform scale)."),
    ] = 100.0,
    section_box: Annotated[
        bool,
        typer.Option("--section-box/--no-section-box", help="Clip geometry with an oriented cube."),
    ] = False,
    box_size_x: Annotated[float, typer.Option("--box-size-x", help="Section box size X.")] = 120.0,
    box_size_y: Annotated[float, typer.Option("--box-size-y", help="Section box size Y.")] = 120.0,
    box_size_z: Annotated[float, typer.Option("--box-size-z", help="Section box size Z.")] = 120.0,
    box_cx: Annotated[float, typer.Option("--box-cx", help="Section box center X.")] = 50.0,
    box_cy: Annotated[float, typer.Option("--box-cy", help="Section box center Y.")] = 50.0,
    box_cz: Annotated[float, typer.Option("--box-cz", help="Section box center Z.")] = 0.0,
    box_rx: Annotated[float, typer.Option("--box-rx", help="Section box rotation X (deg).")] = 0.0,
    box_ry: Annotated[float, typer.Option("--box-ry", help="Section box rotation Y (deg).")] = 0.0,
    box_rz: Annotated[float, typer.Option("--box-rz", help="Section box rotation Z (deg).")] = 0.0,
    lines_x: Annotated[
        bool,
        typer.Option("--lines-x/--no-lines-x", help="Include X/U-parallel grid lines."),
    ] = True,
    lines_y: Annotated[
        bool,
        typer.Option("--lines-y/--no-lines-y", help="Include Y/V-parallel grid lines."),
    ] = True,
    boundary_curve: Annotated[
        bool,
        typer.Option(
            "--boundary-curve/--no-boundary-curve",
            help="Closed polyline tracking the original 2D outline through the waves.",
        ),
    ] = True,
    voxel_size: Annotated[
        float,
        typer.Option(
            "--voxel-size",
            help="Voxel edge length (smaller = smoother/slower; denser spines).",
        ),
    ] = 0.8,
    pipe_radius: Annotated[
        float,
        typer.Option("--pipe-radius", help="Outer tube radius along each line."),
    ] = 1.2,
    inner_radius: Annotated[
        float,
        typer.Option("--inner-radius", help="Hollow bore (0 = solid rod)."),
    ] = 0.0,
    modulation_amp: Annotated[
        float,
        typer.Option("--mod-amp", help="Surface ripple amplitude (voxel modulation)."),
    ] = 0.0,
    modulation_freq: Annotated[
        float,
        typer.Option("--mod-freq", help="Ripples along each pipe length."),
    ] = 2.0,
    modulation_lobes: Annotated[
        int,
        typer.Option("--mod-lobes", help="Angular flutes around the tube (0 = none)."),
    ] = 0,
    line_stride: Annotated[
        int,
        typer.Option("--line-stride", help="Keep every N-th grid line."),
    ] = 2,
    boundary_lines_x: Annotated[
        int,
        typer.Option(
            "--boundary-lines-x",
            help="X-rows: +N keep only first/last N (drops the middle); −N shave ends.",
        ),
    ] = 0,
    boundary_lines_y: Annotated[
        int,
        typer.Option(
            "--boundary-lines-y",
            help="Y-cols: +N keep only first/last N (drops the middle); −N shave ends.",
        ),
    ] = 0,
    point_stride: Annotated[
        int,
        typer.Option("--point-stride", help="Keep every N-th sample along each line."),
    ] = 2,
    spine_samples: Annotated[
        int,
        typer.Option(
            "--spine-samples",
            help="Cubic-spline arc-length samples per spine.",
        ),
    ] = 40,
    spine_smooth: Annotated[
        float,
        typer.Option(
            "--spine-smooth",
            help="Line-smoothing strength before piping (0 = interpolate).",
        ),
    ] = 1.0,
    export_dir: Annotated[
        Path,
        typer.Option("--export-dir", "-o", help="Directory for the STL file."),
    ] = Path("exports"),
    configs_dir: Annotated[
        Path,
        typer.Option("--configs-dir", help="Directory for saving model-param snapshots."),
    ] = Path("configs"),
    save_config: Annotated[
        bool,
        typer.Option(
            "--save-config/--no-save-config",
            help="Save pipeline + voxel params JSON (same format as notebook Save params).",
        ),
    ] = True,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress progress output."),
    ] = False,
) -> None:
    """Pipe grid lines into a PicoPie voxel solid and export STL."""
    from cymatics_geometry.config import (
        PipelineConfig,
        load_pipeline_config,
        load_voxel_params,
        save_model_params,
    )
    from cymatics_geometry.pipeline import run_pipeline
    from cymatics_geometry.shapes import SHAPE_KINDS
    from cymatics_geometry.voxels import VoxelPipeConfig, pipe_and_export_stl

    shape_kind = str(shape).lower().strip()
    if shape_kind not in SHAPE_KINDS:
        typer.echo(f"--shape must be one of: {', '.join(SHAPE_KINDS)}", err=True)
        raise typer.Exit(1)

    voxel_from_file: dict[str, object] | None = None
    if config is not None:
        if not config.exists():
            typer.echo(f"Config file not found: {config}", err=True)
            raise typer.Exit(1)
        pipeline_config = load_pipeline_config(config)
        voxel_from_file = load_voxel_params(config)
        if not quiet:
            typer.echo(f"Loaded config from {config}")
            if voxel_from_file is not None:
                typer.echo("Using nested voxel params from config JSON")
    else:
        nx = grid_size if grid_size_x is None else grid_size_x
        ny = grid_size if grid_size_y is None else grid_size_y
        pipeline_config = PipelineConfig(
            grid_size_x=nx,
            grid_size_y=ny,
            amplitude_sw=amp_sw,
            amplitude_se=amp_se,
            amplitude_ne=amp_ne,
            amplitude_nw=amp_nw,
            wavelength=wavelength,
            wavelength_sw=wavelength,
            wavelength_se=wavelength,
            wavelength_ne=wavelength,
            wavelength_nw=wavelength,
            line_pattern="grid",
            lines_x=lines_x,
            lines_y=lines_y,
            boundary_curve=bool(boundary_curve),
            shape=shape_kind,
            cylinder_diameter=cylinder_diameter,
            cylinder_length=cylinder_length,
            cone_height=cone_height,
            cone_base_radius=cone_base_radius,
            frustum_height=frustum_height,
            frustum_base_diameter=frustum_base_diameter,
            frustum_top_diameter=frustum_top_diameter,
            variable_cylinder_radius_begin=var_cyl_radius_begin,
            variable_cylinder_radius_middle=var_cyl_radius_middle,
            variable_cylinder_radius_end=var_cyl_radius_end,
            variable_cylinder_length=var_cyl_length,
            variable_cylinder_middle=max(0.1, min(0.9, float(var_cyl_middle))),
            bead_diameter=bead_diameter,
            bead_bottom_radius=bead_bottom_radius,
            bead_top_radius=bead_top_radius,
            custom_shape_path="" if custom_shape is None else str(custom_shape),
            custom_shape_size=float(custom_shape_size),
            section_box_enabled=bool(section_box),
            section_box_size_x=float(box_size_x),
            section_box_size_y=float(box_size_y),
            section_box_size_z=float(box_size_z),
            section_box_center_x=float(box_cx),
            section_box_center_y=float(box_cy),
            section_box_center_z=float(box_cz),
            section_box_rot_x=float(box_rx),
            section_box_rot_y=float(box_ry),
            section_box_rot_z=float(box_rz),
        )

    if not quiet:
        typer.echo("Running cymatics pipeline, then voxel-piping lines…")

    result = run_pipeline(pipeline_config, verbose=not quiet)
    if voxel_from_file is not None:
        vcfg = VoxelPipeConfig.from_dict(voxel_from_file)
    else:
        vcfg = VoxelPipeConfig(
            voxel_size=voxel_size,
            pipe_radius=pipe_radius,
            inner_radius=inner_radius,
            modulation_amp=modulation_amp,
            modulation_freq=modulation_freq,
            modulation_lobes=modulation_lobes,
            line_stride=line_stride,
            boundary_lines_x=boundary_lines_x,
            boundary_lines_y=boundary_lines_y,
            point_stride=point_stride,
            spine_samples=spine_samples,
            spine_smooth=spine_smooth,
        )
    solid, path = pipe_and_export_stl(result, export_dir, vcfg, verbose=not quiet)
    typer.echo(f"STL exported: {path}")
    typer.echo(
        f"faces={solid.stats['faces']}  volume~={solid.volume:.2f}  "
        f"watertight={solid.is_watertight}"
    )

    if save_config:
        saved = save_model_params(pipeline_config, vcfg, configs_dir)
        if saved is not None:
            typer.echo(f"Model params saved: {saved}")
        elif not quiet:
            typer.echo("Model params identical to existing, not saved.")


@app.command(name="show-config")
def show_config(
    path: Annotated[Path, typer.Argument(help="Path to a config JSON file to display.")],
) -> None:
    """Display the contents of a saved config file."""
    if not path.exists():
        typer.echo(f"File not found: {path}", err=True)
        raise typer.Exit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    typer.echo(json.dumps(data, indent=2))


@app.command(name="list-configs")
def list_configs(
    configs_dir: Annotated[
        Path,
        typer.Option("--configs-dir", help="Directory containing config files."),
    ] = Path("configs"),
) -> None:
    """List all saved config snapshots."""
    from cymatics_geometry.config import list_saved_configs

    if not configs_dir.exists():
        typer.echo(f"Configs directory not found: {configs_dir}", err=True)
        raise typer.Exit(1)

    names = list_saved_configs(configs_dir)
    if not names:
        typer.echo("No saved configs found.")
        return

    typer.echo(f"Found {len(names)} saved config(s) in {configs_dir}/:")
    for name in names:
        typer.echo(f"  {name}.json")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
