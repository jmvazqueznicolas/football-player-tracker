"""Helpers for training: config flattening, MLflow URI resolution, etc."""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dict into a single level with dotted keys.

    Example:
        {"a": {"b": 1, "c": 2}, "d": 3}
        becomes
        {"a.b": 1, "a.c": 2, "d": 3}

    Used to log Hydra configs to MLflow, which only accepts flat key→value params.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            items.append((new_key, ",".join(str(x) for x in v)))
        else:
            items.append((new_key, v))
    return dict(items)


def cfg_to_flat_params(cfg: DictConfig) -> dict[str, Any]:
    """Convert an OmegaConf DictConfig into a flat dict ready for MLflow.

    Truncates values that exceed MLflow's 500-char param-value limit.
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        return {}
    flat = flatten_dict(container)
    # MLflow truncates param values at ~500 chars. Be safe.
    return {k: (str(v)[:480] + "..." if len(str(v)) > 500 else v) for k, v in flat.items()}


def sanitize_metric_key(key: str) -> str:
    """MLflow allows: letters, numbers, _, -, ., space, /.

    Ultralytics emits keys like 'metrics/mAP50(B)' which contain parentheses.
    We strip them.
    """
    return key.replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace(",", "_")
