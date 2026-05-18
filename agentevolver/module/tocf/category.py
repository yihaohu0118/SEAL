from __future__ import annotations

from collections.abc import Mapping
from typing import Any


BFCL_CATEGORY_PREFIXES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)


def _plain_scalar(value: Any) -> Any:
    """Return a JSON-friendly scalar when values come from pandas/numpy records."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return str(value)
    return str(value)


def infer_task_category(task_id: str | None, env_type: str | None = None, metadata: Mapping[str, Any] | None = None) -> str:
    """Infer a stable task category without consulting the environment service."""
    metadata = metadata or {}

    for key in ("category", "dataset_type", "data_source"):
        value = metadata.get(key)
        if value:
            return str(_plain_scalar(value))

    task_id = str(task_id or "")
    if env_type == "bfcl" or task_id.startswith("multi_turn_"):
        for prefix in BFCL_CATEGORY_PREFIXES:
            if task_id.startswith(prefix + "_") or task_id == prefix:
                return prefix
        return task_id.rsplit("_", 1)[0] if "_" in task_id else task_id

    return metadata.get("env_type", env_type or "unknown")


def patch_task_metadata(metadata: Mapping[str, Any] | None, *, task_id: str | None, env_type: str | None) -> dict:
    """Create lightweight metadata used by TOCF, preserving existing metadata when safe."""
    patched = dict(metadata or {})
    patched.setdefault("category", infer_task_category(task_id, env_type, patched))
    patched.setdefault("env_type", env_type)
    return patched


def extract_bfcl_task_id(extras: Mapping[str, Any]) -> str:
    """Extract BFCL ids from either AgentEvolver extras or EnvTuning extra_info."""
    candidates = [
        extras.get("task_id"),
        extras.get("index"),
        extras.get("original_id"),
    ]
    interaction_kwargs = extras.get("interaction_kwargs")
    if isinstance(interaction_kwargs, Mapping):
        candidates.append(interaction_kwargs.get("id"))

    for candidate in candidates:
        if candidate:
            return str(_plain_scalar(candidate))

    return ""


def extract_lightweight_metadata(extras: Mapping[str, Any], *, env_type: str, task_id: str) -> dict:
    """Avoid putting large numpy arrays/function schemas into Task.metadata."""
    raw_metadata = extras.get("metadata") if isinstance(extras.get("metadata"), Mapping) else {}
    metadata = dict(raw_metadata)

    for key in ("dataset_type", "data_source", "split", "original_id"):
        value = extras.get(key)
        if value is not None:
            metadata[key] = _plain_scalar(value)

    metadata.setdefault("category", infer_task_category(task_id, env_type, metadata))
    metadata.setdefault("env_type", env_type)
    return metadata
