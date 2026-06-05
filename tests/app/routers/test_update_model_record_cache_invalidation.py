"""Tests for `_load_settings_changed` — the predicate that decides whether to evict cached
model entries after an `update_model_record` call. Settings like `fp8_storage` and `cpu_only`
are baked into the loaded nn.Module at load time, so toggling them silently has no effect
until the cached entry is evicted. The predicate must catch changes to those settings while
ignoring changes that don't affect how the model loads (e.g. name, description).
"""

from types import SimpleNamespace

from invokeai.app.api.routers.model_manager import _load_settings_changed


def _config(*, fp8: bool | None = None, cpu_only: bool | None = None):
    return SimpleNamespace(
        cpu_only=cpu_only,
        default_settings=SimpleNamespace(fp8_storage=fp8),
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
