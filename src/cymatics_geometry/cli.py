"""CLI for the cymatics plane → line geometry generator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer(
    name="cymatics",
    help="Generate line geometry from a 100×100 point plane displaced by four-corner cymatics waves.",
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
        typer.Option("--grid-size", "-n", help="Grid resolution (N×N points)."),
    ] = 100,
    side_length: Annotated[
        float,
        typer.Option("--side", help="Square side length."),
    ] = 100.0,
    amp_sw: Annotated[float, typer.Option("--amp-sw", help="SW corner amplitude.")] = 1.0,
    amp_se: Annotated[float, typer.Option("--amp-se", help="SE corner amplitude.")] = 1.0,
    amp_ne: Annotated[float, typer.Option("--amp-ne", help="NE corner amplitude.")] = 0.6,
    amp_nw: Annotated[float, typer.Option("--amp-nw", help="NW corner amplitude.")] = 0.8,
    wavelength: Annotated[float, typer.Option("--wavelength", "-l", help="Wave length λ.")] = 25.0,
    frequency: Annotated[float, typer.Option("--frequency", "-f", help="Wave frequency.")] = 1.0,
    time: Annotated[float, typer.Option("--time", "-t", help="Simulation time.")] = 0.0,
    decay: Annotated[float, typer.Option("--decay", help="Distance amplitude decay.")] = 0.0,
    pattern: Annotated[
        str,
        typer.Option("--pattern", help="Line reconnection pattern: serpentine|row_major."),
    ] = "serpentine",
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

    if config is not None:
        if not config.exists():
            typer.echo(f"Config file not found: {config}", err=True)
            raise typer.Exit(1)
        pipeline_config = load_pipeline_config(config)
        if not quiet:
            typer.echo(f"Loaded config from {config}")
    else:
        pipeline_config = PipelineConfig(
            grid_size=grid_size,
            side_length=side_length,
            amplitude_sw=amp_sw,
            amplitude_se=amp_se,
            amplitude_ne=amp_ne,
            amplitude_nw=amp_nw,
            wavelength=wavelength,
            frequency=frequency,
            time=time,
            decay=decay,
            line_pattern=pattern,
        )

    if not quiet:
        typer.echo("Running cymatics plane → line pipeline...")

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

        save_line_screenshot(result, screenshot)
        typer.echo(f"Screenshot saved: {screenshot}")

    if viewer:
        from cymatics_geometry.visualization import show_stage_line

        typer.echo("Opening interactive viewer (close window to continue)...")
        show_stage_line(result)


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
