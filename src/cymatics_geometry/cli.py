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
        "cymatics waves, optionally mapped onto cylinder / cone / frustum."
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
    shape: Annotated[
        str,
        typer.Option(
            "--shape",
            help="Target surface: plane|cylinder|cone|frustum.",
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
            shape=shape_kind,
            cylinder_diameter=cylinder_diameter,
            cylinder_length=cylinder_length,
            cone_height=cone_height,
            cone_base_radius=cone_base_radius,
            frustum_height=frustum_height,
            frustum_base_diameter=frustum_base_diameter,
            frustum_top_diameter=frustum_top_diameter,
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
