"""Shared rules for when a model record change invalidates loaded instances of that model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from invokeai.backend.model_manager.configs.factory import AnyModelConfig

_LOAD_AFFECTING_SETTINGS: tuple[str, ...] = ("fp8_storage", "cpu_only")
_MODEL_METADATA_FIELDS: tuple[str, ...] = (
    "name",
    "description",
    "cover_image",
    "source",
    "source_type",
    "source_api_response",
    "source_url",
    "trigger_phrases",
)


def _model_load_fingerprint(config: AnyModelConfig, models_path: Optional[Path]) -> dict[str, Any]:
    """Return record values whose change can make a cached model instance stale."""
    if model_dump := getattr(config, "model_dump", None):
        fingerprint = model_dump(mode="python")
    else:
        fingerprint = vars(config).copy()
    for field in _MODEL_METADATA_FIELDS:
        fingerprint.pop(field, None)

    # Keep hash and file_size: re-identification recomputes them from disk, and a changed value can mean
    # the bytes at an unchanged path no longer match the loaded module.

    if models_path is not None and fingerprint.get("path") is not None:
        # The loader rewrites `config.path` to an absolute path in place; a config object it has already
        # loaded from must still match its stored, models-relative record.
        fingerprint["path"] = (models_path / fingerprint["path"]).resolve()

    default_settings = fingerprint.pop("default_settings", None)
    fingerprint["default_settings"] = {
        field: default_settings.get(field)
        if isinstance(default_settings, dict)
        else getattr(default_settings, field, None)
        for field in _LOAD_AFFECTING_SETTINGS
    }
    return fingerprint


def load_settings_changed(
    previous: AnyModelConfig, updated: AnyModelConfig, models_path: Optional[Path] = None
) -> bool:
    """Return True if anything that influences how the model is loaded changed.

    Such values are read by the loader during `_load_model` and baked into the resulting nn.Module, so a
    cached entry built under the old value must be evicted for the change to take effect, and a load that
    started from the old value must not be admitted after that eviction. Pass `models_path` to compare
    paths after resolving them against it.
    """
    return _model_load_fingerprint(previous, models_path) != _model_load_fingerprint(updated, models_path)


@dataclass
class RecordEdit:
    """One bracketed write of a model record (see `ModelLoadServiceBase.record_edit`)."""

    # Whether the write changed how the model loads. A writer that has compared the record before and after
    # clears it; left set, as when the write fails partway, loads that overlapped it are re-checked.
    load_affecting: bool = True
