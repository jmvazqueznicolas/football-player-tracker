"""Top-level CLI entry point. Run with: `football-tracker --help`."""

from __future__ import annotations

import typer

app = typer.Typer(
    name="football-tracker",
    help="Football Player Tracker — CV + MLOps CLI.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the package version."""
    from football_tracker import __version__

    typer.echo(__version__)


@app.command()
def info() -> None:
    """Print quick environment info (Python, Torch, CUDA, Ultralytics)."""
    import platform
    import sys

    typer.echo(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    try:
        import torch

        typer.echo(f"Torch: {torch.__version__}  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            typer.echo(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        typer.echo("Torch not installed.")
    try:
        import ultralytics

        typer.echo(f"Ultralytics: {ultralytics.__version__}")
    except ImportError:
        typer.echo("Ultralytics not installed.")


if __name__ == "__main__":
    app()
