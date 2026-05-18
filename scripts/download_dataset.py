"""Download a YOLO-formatted dataset from Roboflow Universe.

Reads ROBOFLOW_API_KEY from .env. Workspace / project / version come from
configs/data/football.yaml or from CLI args (CLI overrides config).

Usage:
    poetry run python scripts/download_dataset.py
    poetry run python scripts/download_dataset.py --workspace foo --project bar --version 3
    poetry run python scripts/download_dataset.py --output-dir data/processed/football-players
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

app = typer.Typer(add_completion=False)


def _load_config_defaults() -> dict:
    """Read the data config YAML for default workspace/project/version."""
    cfg_path = Path(__file__).resolve().parent.parent / "configs" / "data" / "football.yaml"
    if not cfg_path.exists():
        return {}
    cfg = yaml.safe_load(cfg_path.read_text())
    rb = cfg.get("roboflow", {})
    return {
        "workspace": rb.get("workspace"),
        "project": rb.get("project"),
        "version": rb.get("version"),
        "format": rb.get("format", "yolov11"),
        "output_dir": cfg.get("root", "data/processed/football-players"),
    }


@app.command()
def main(
    workspace: str = typer.Option(None, help="Roboflow workspace slug"),
    project: str = typer.Option(None, help="Roboflow project slug"),
    version: int = typer.Option(None, help="Dataset version number"),
    format: str = typer.Option(None, "--format", help="Export format (yolov11, yolov8, coco)"),
    output_dir: Path = typer.Option(None, help="Where to put the dataset"),
    api_key: str = typer.Option(None, help="Roboflow API key (else read from env)"),
    overwrite: bool = typer.Option(False, help="Delete output_dir if it exists before downloading"),
) -> None:
    """Download a Roboflow dataset to disk in YOLO format."""
    load_dotenv()
    defaults = _load_config_defaults()

    workspace = workspace or defaults.get("workspace")
    project = project or defaults.get("project")
    version = version or defaults.get("version")
    fmt = format or defaults.get("format", "yolov11")
    output_dir = Path(output_dir or defaults.get("output_dir", "data/processed/football-players"))
    api_key = api_key or os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        typer.secho(
            "ERROR: ROBOFLOW_API_KEY not set. Add it to your .env file or pass --api-key.",
            fg=typer.colors.RED,
        )
        typer.echo("Get your key from: https://app.roboflow.com/settings/api")
        sys.exit(1)

    missing = [
        k
        for k, v in {"workspace": workspace, "project": project, "version": version}.items()
        if not v
    ]
    if missing:
        typer.secho(
            f"ERROR: missing required fields: {', '.join(missing)}. "
            "Set them in configs/data/football.yaml or pass as CLI args.",
            fg=typer.colors.RED,
        )
        sys.exit(1)

    if output_dir.exists():
        if overwrite:
            typer.secho(f"Removing existing {output_dir} ...", fg=typer.colors.YELLOW)
            shutil.rmtree(output_dir)
        else:
            typer.secho(
                f"{output_dir} already exists. Use --overwrite to replace it.",
                fg=typer.colors.YELLOW,
            )
            sys.exit(0)

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Import here so the script works even before dependencies are installed
    # (you'll get a clean error message).
    try:
        from roboflow import Roboflow
    except ImportError:
        typer.secho(
            "Roboflow package not installed. Run: poetry install --with dev",
            fg=typer.colors.RED,
        )
        sys.exit(1)

    typer.echo(f"Downloading {workspace}/{project} v{version} ({fmt}) -> {output_dir}")

    rf = Roboflow(api_key=api_key)
    rb_project = rf.workspace(workspace).project(project)
    rb_dataset = rb_project.version(version).download(fmt, location=str(output_dir))

    typer.secho(f"\n✓ Dataset downloaded to: {rb_dataset.location}", fg=typer.colors.GREEN)
    typer.echo("\nNext steps:")
    typer.echo(f"  1. Inspect: ls {output_dir}")
    typer.echo("  2. Run EDA notebook: poetry run jupyter lab notebooks/")
    typer.echo("  3. Verify YOLO data.yaml points to correct splits")


if __name__ == "__main__":
    app()
