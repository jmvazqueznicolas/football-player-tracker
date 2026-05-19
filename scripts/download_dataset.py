"""Download a YOLO-formatted dataset from Roboflow Universe.

Reads ROBOFLOW_API_KEY from .env. Workspace / project / version come from
configs/data/football.yaml unless overridden via CLI args.

Usage:
    poetry run python scripts/download_dataset.py
    poetry run python scripts/download_dataset.py --workspace foo --project bar --version 3
    poetry run python scripts/download_dataset.py --output-dir data/processed/football-players
    poetry run python scripts/download_dataset.py --overwrite
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv


def _color(text: str, color: str) -> str:
    """Tiny ANSI color helper for terminal output."""
    codes = {"red": "31", "green": "32", "yellow": "33", "reset": "0"}
    if not sys.stdout.isatty():
        return text
    return f"\033[{codes.get(color, '0')}m{text}\033[0m"


def _load_config_defaults() -> dict:
    """Read configs/data/football.yaml for default workspace/project/version."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Roboflow dataset to disk in YOLO format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workspace", default=None, help="Roboflow workspace slug")
    parser.add_argument("--project", default=None, help="Roboflow project slug")
    parser.add_argument("--version", type=int, default=None, help="Dataset version number")
    parser.add_argument(
        "--format",
        dest="fmt",
        default=None,
        help="Export format (yolov11, yolov8, coco, etc.)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to put the dataset",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Roboflow API key (defaults to ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_dir if it exists before downloading",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    defaults = _load_config_defaults()

    workspace = args.workspace or defaults.get("workspace")
    project = args.project or defaults.get("project")
    version = args.version or defaults.get("version")
    fmt = args.fmt or defaults.get("format", "yolov11")
    output_dir = Path(
        args.output_dir or defaults.get("output_dir", "data/processed/football-players")
    )
    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        print(
            _color(
                "ERROR: ROBOFLOW_API_KEY not set. Add it to your .env file or pass --api-key.",
                "red",
            )
        )
        print("Get your key from: https://app.roboflow.com/settings/api")
        sys.exit(1)

    missing = [
        k
        for k, v in {"workspace": workspace, "project": project, "version": version}.items()
        if not v
    ]
    if missing:
        print(
            _color(
                f"ERROR: missing required fields: {', '.join(missing)}. "
                "Set them in configs/data/football.yaml or pass as CLI args.",
                "red",
            )
        )
        sys.exit(1)

    if output_dir.exists():
        if args.overwrite:
            print(_color(f"Removing existing {output_dir} ...", "yellow"))
            shutil.rmtree(output_dir)
        else:
            print(
                _color(
                    f"{output_dir} already exists. Use --overwrite to replace it.",
                    "yellow",
                )
            )
            sys.exit(0)

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        from roboflow import Roboflow
    except ImportError:
        print(
            _color(
                "Roboflow package not installed. Run: poetry install --with dev",
                "red",
            )
        )
        sys.exit(1)

    print(f"Downloading {workspace}/{project} v{version} ({fmt}) -> {output_dir}")

    rf = Roboflow(api_key=api_key)
    rb_project = rf.workspace(workspace).project(project)
    rb_dataset = rb_project.version(version).download(fmt, location=str(output_dir))

    print(_color(f"\n✓ Dataset downloaded to: {rb_dataset.location}", "green"))
    print("\nNext steps:")
    print(f"  1. Inspect: ls {output_dir}")
    print("  2. Run EDA notebook: poetry run jupyter lab notebooks/")
    print("  3. Verify YOLO data.yaml points to correct splits")


if __name__ == "__main__":
    main()
