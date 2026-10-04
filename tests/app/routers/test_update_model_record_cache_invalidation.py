"""Tests for `_load_settings_changed` — the predicate that decides whether to evict cached
model entries after an `update_model_record` call. Settings like `fp8_storage` and `cpu_only`
are baked into the loaded nn.Module at load time, so toggling them silently has no effect
until the cached entry is evicted. The predicate must catch changes to those settings while
ignoring changes that don't affect how the model loads (e.g. name, description).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.api.routers import model_manager as model_manager_router
from invokeai.app.api.routers.model_manager import _load_settings_changed
from invokeai.app.services.model_records.model_records_base import ModelRecordChanges


def _config(*, fp8: bool | None = None, cpu_only: bool | None = None, **fields):
    return SimpleNamespace(
        cpu_only=cpu_only,
        default_settings=SimpleNamespace(fp8_storage=fp8),
        **fields,
    )


def test_no_change_returns_false():
    assert _load_settings_changed(_config(fp8=True), _config(fp8=True)) is False
    assert _load_settings_changed(_config(fp8=None), _config(fp8=None)) is False


def test_fp8_storage_toggle_returns_true():
    """The primary motivating case: a user toggling FP8 storage in the Model Manager must
    drop the cached entry, otherwise inference keeps using the old (non-FP8) module."""
    assert _load_settings_changed(_config(fp8=False), _config(fp8=True)) is True
    assert _load_settings_changed(_config(fp8=True), _config(fp8=False)) is True
    assert _load_settings_changed(_config(fp8=None), _config(fp8=True)) is True
    assert _load_settings_changed(_config(fp8=True), _config(fp8=None)) is True


def test_cpu_only_toggle_returns_true():
    """`cpu_only` is also read by the loader (in `_get_execution_device`) and baked into the
    cache entry's execution device — toggling it after load is a silent no-op without eviction."""
    assert _load_settings_changed(_config(cpu_only=False), _config(cpu_only=True)) is True
    assert _load_settings_changed(_config(cpu_only=True), _config(cpu_only=None)) is True


def test_missing_default_settings_is_handled():
    """default_settings can legitimately be None (e.g. a freshly identified config)."""
    no_settings = SimpleNamespace(cpu_only=None, default_settings=None)
    assert _load_settings_changed(no_settings, no_settings) is False
    assert _load_settings_changed(no_settings, _config(fp8=True)) is True


def test_unrelated_field_does_not_trigger_invalidation():
    """A config missing the fp8/cpu_only attributes entirely (e.g. a model type with no such
    fields) must not falsely report a change."""
    bare_a = SimpleNamespace()
    bare_b = SimpleNamespace()
    assert _load_settings_changed(bare_a, bare_b) is False


@pytest.mark.parametrize("field", ["path", "base", "type", "format", "variant"])
def test_model_identity_changes_trigger_invalidation(field: str):
    """Fields exposed by ModelRecordChanges can change the module selected or its source path."""
    previous = _config(**{field: "old"})
    updated = _config(**{field: "new"})
    assert _load_settings_changed(previous, updated) is True


@pytest.mark.parametrize("field", ["path", "base", "type", "format", "variant", "name", "description"])
def test_update_only_evicts_caches_for_load_affecting_changes(field: str, monkeypatch: pytest.MonkeyPatch):
    previous = _config(**{field: "old"})
    updated = _config(**{field: "new"})
    record_store = SimpleNamespace(
        get_model=MagicMock(return_value=previous),
        update_model=MagicMock(return_value=updated),
    )
    cache_a = SimpleNamespace(drop_model=MagicMock(return_value=1))
    cache_b = SimpleNamespace(drop_model=MagicMock(return_value=1))
    services = SimpleNamespace(
        logger=MagicMock(),
        model_manager=SimpleNamespace(
            store=record_store,
            load=SimpleNamespace(ram_caches={"cuda:0": cache_a, "cuda:1": cache_b}),
        ),
    )
    monkeypatch.setattr(
        model_manager_router.ApiDependencies,
        "invoker",
        SimpleNamespace(services=services),
        raising=False,
    )
    monkeypatch.setattr(model_manager_router, "prepare_model_config_for_response", lambda config, _deps: config)

    result = model_manager_router._update_model_record(key="model-key", changes=ModelRecordChanges())
    assert result is updated

    for cache in (cache_a, cache_b):
        if field in {"name", "description"}:
            cache.drop_model.assert_not_called()
        else:
            cache.drop_model.assert_called_once_with("model-key")
