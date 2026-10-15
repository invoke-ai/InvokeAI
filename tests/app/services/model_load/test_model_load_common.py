"""Tests for `load_settings_changed` — the predicate that decides both whether a model record change evicts
cached model entries and whether a load that read the record before that change may still be admitted to the
cache. Load-affecting values (e.g. `fp8_storage`, `cpu_only`, loader identity) are baked into the loaded
nn.Module at load time, so changing them silently has no effect until the cached entry is evicted. The
predicate must catch those changes while ignoring ones that don't affect how the model loads (e.g. name,
description).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from invokeai.app.services.model_load.model_load_common import load_settings_changed


def _config(*, fp8: bool | None = None, cpu_only: bool | None = None, **fields):
    return SimpleNamespace(
        cpu_only=cpu_only,
        default_settings=SimpleNamespace(fp8_storage=fp8),
        **fields,
    )


def test_no_change_returns_false():
    assert load_settings_changed(_config(fp8=True), _config(fp8=True)) is False
    assert load_settings_changed(_config(fp8=None), _config(fp8=None)) is False


def test_fp8_storage_toggle_returns_true():
    """The primary motivating case: a user toggling FP8 storage in the Model Manager must
    drop the cached entry, otherwise inference keeps using the old (non-FP8) module."""
    assert load_settings_changed(_config(fp8=False), _config(fp8=True)) is True
    assert load_settings_changed(_config(fp8=True), _config(fp8=False)) is True
    assert load_settings_changed(_config(fp8=None), _config(fp8=True)) is True
    assert load_settings_changed(_config(fp8=True), _config(fp8=None)) is True


def test_cpu_only_toggle_returns_true():
    """`cpu_only` is also read by the loader (in `_get_execution_device`) and baked into the
    cache entry's execution device — toggling it after load is a silent no-op without eviction."""
    assert load_settings_changed(_config(cpu_only=False), _config(cpu_only=True)) is True
    assert load_settings_changed(_config(cpu_only=True), _config(cpu_only=None)) is True


def test_missing_default_settings_is_handled():
    """default_settings can legitimately be None (e.g. a freshly identified config)."""
    no_settings = SimpleNamespace(cpu_only=None, default_settings=None)
    assert load_settings_changed(no_settings, no_settings) is False
    assert load_settings_changed(no_settings, _config(fp8=True)) is True


def test_default_generation_steps_change_does_not_invalidate_model_cache():
    """Generation defaults are read per invocation, not baked into the loaded model module."""
    previous = SimpleNamespace(cpu_only=None, default_settings=SimpleNamespace(fp8_storage=None, steps=20))
    updated = SimpleNamespace(cpu_only=None, default_settings=SimpleNamespace(fp8_storage=None, steps=40))

    assert load_settings_changed(previous, updated) is False


@pytest.mark.parametrize("field", ["fp8_storage", "cpu_only"])
def test_pydantic_nested_load_setting_changes_trigger_invalidation(field: str) -> None:
    from pydantic import BaseModel

    from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings

    class PydanticConfig(BaseModel):
        default_settings: MainModelDefaultSettings | None = None

    previous = PydanticConfig(default_settings=MainModelDefaultSettings(**{field: False}))
    updated = PydanticConfig(default_settings=MainModelDefaultSettings(**{field: True}))

    assert load_settings_changed(previous, updated) is True


def test_unrelated_field_does_not_trigger_invalidation():
    """A config missing the fp8/cpu_only attributes entirely (e.g. a model type with no such
    fields) must not falsely report a change."""
    bare_a = SimpleNamespace()
    bare_b = SimpleNamespace()
    assert load_settings_changed(bare_a, bare_b) is False


@pytest.mark.parametrize("field", ["path", "base", "type", "format", "variant", "repo_variant", "hash", "file_size"])
def test_model_identity_changes_trigger_invalidation(field: str):
    """Loader identity or changed on-disk content must evict a module loaded from the previous record."""
    previous = _config(**{field: "old"})
    updated = _config(**{field: "new"})
    assert load_settings_changed(previous, updated) is True


def test_path_made_absolute_by_the_loader_matches_its_stored_record(tmp_path: Path):
    """The loader rewrites `config.path` in place; that must not read as a change against the record."""
    stored = _config(path="sd-1/main/model.safetensors")
    loaded_from = _config(path=str(tmp_path / "sd-1/main/model.safetensors"))

    assert load_settings_changed(loaded_from, stored, models_path=tmp_path) is False
    assert load_settings_changed(loaded_from, _config(path="sd-1/main/other.safetensors"), models_path=tmp_path)
