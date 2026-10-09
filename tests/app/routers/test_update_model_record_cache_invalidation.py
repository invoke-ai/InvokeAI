"""Tests for `_load_settings_changed` — the predicate that decides whether to evict cached
model entries after an `update_model_record` call. Settings like `fp8_storage` and `cpu_only`
are baked into the loaded nn.Module at load time, so toggling them silently has no effect
until the cached entry is evicted. The predicate must catch changes to those settings while
ignoring changes that don't affect how the model loads (e.g. name, description).
"""

from pathlib import Path
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


def test_default_generation_steps_change_does_not_invalidate_model_cache():
    """Generation defaults are read per invocation, not baked into the loaded model module."""
    previous = SimpleNamespace(cpu_only=None, default_settings=SimpleNamespace(fp8_storage=None, steps=20))
    updated = SimpleNamespace(cpu_only=None, default_settings=SimpleNamespace(fp8_storage=None, steps=40))

    assert _load_settings_changed(previous, updated) is False


@pytest.mark.parametrize("field", ["fp8_storage", "cpu_only"])
def test_pydantic_nested_load_setting_changes_trigger_invalidation(field: str) -> None:
    from pydantic import BaseModel

    from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings

    class PydanticConfig(BaseModel):
        default_settings: MainModelDefaultSettings | None = None

    previous = PydanticConfig(default_settings=MainModelDefaultSettings(**{field: False}))
    updated = PydanticConfig(default_settings=MainModelDefaultSettings(**{field: True}))

    assert _load_settings_changed(previous, updated) is True


def test_unrelated_field_does_not_trigger_invalidation():
    """A config missing the fp8/cpu_only attributes entirely (e.g. a model type with no such
    fields) must not falsely report a change."""
    bare_a = SimpleNamespace()
    bare_b = SimpleNamespace()
    assert _load_settings_changed(bare_a, bare_b) is False


@pytest.mark.parametrize("field", ["path", "base", "type", "format", "variant", "repo_variant", "hash", "file_size"])
def test_model_identity_changes_trigger_invalidation(field: str):
    """Loader identity or changed on-disk content must evict a module loaded from the previous record."""
    previous = _config(**{field: "old"})
    updated = _config(**{field: "new"})
    assert _load_settings_changed(previous, updated) is True


@pytest.mark.parametrize(
    "field",
    [
        "path",
        "base",
        "type",
        "format",
        "variant",
        "repo_variant",
        "name",
        "description",
        "cover_image",
        "source",
        "source_type",
        "source_api_response",
        "source_url",
        "trigger_phrases",
    ],
)
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
        if field in {
            "name",
            "description",
            "cover_image",
            "source",
            "source_type",
            "source_api_response",
            "source_url",
            "trigger_phrases",
        }:
            cache.drop_model.assert_not_called()
        else:
            cache.drop_model.assert_called_once_with("model-key")


def test_cache_invalidation_holds_model_load_write_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager

    previous = _config(path="old")
    updated = _config(path="new")
    inside_write_lock = False
    observed_drop_under_lock: list[bool] = []

    class TrackingModelLoadLock:
        @contextmanager
        def write_lock(self):
            nonlocal inside_write_lock
            inside_write_lock = True
            try:
                yield
            finally:
                inside_write_lock = False

    def drop_model(_key: str) -> int:
        observed_drop_under_lock.append(inside_write_lock)
        return 1

    cache = SimpleNamespace(drop_model=drop_model)
    services = SimpleNamespace(
        logger=MagicMock(),
        model_manager=SimpleNamespace(load=SimpleNamespace(ram_caches={"cpu": cache})),
    )
    monkeypatch.setattr(model_manager_router, "MODEL_LOAD_LOCK", TrackingModelLoadLock())
    monkeypatch.setattr(
        model_manager_router.ApiDependencies, "invoker", SimpleNamespace(services=services), raising=False
    )

    model_manager_router._invalidate_model_load_caches("model-key", previous, updated)

    assert observed_drop_under_lock == [True]


@pytest.mark.parametrize(("field", "expected_eviction"), [("path", True), ("name", False), ("description", False)])
def test_record_update_handles_real_cpu_cache_entries(
    field: str, expected_eviction: bool, monkeypatch: pytest.MonkeyPatch
):
    """Load-affecting edits drop all model entries; metadata edits retain cached entries."""
    import logging

    import torch

    from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache

    previous = _config(**{field: "old"})
    updated = _config(**{field: "new"})
    record_store = SimpleNamespace(
        get_model=MagicMock(return_value=previous),
        update_model=MagicMock(return_value=updated),
    )
    caches = [
        ModelCache(
            execution_device_working_mem_gb=1,
            enable_partial_loading=False,
            keep_ram_copy_of_weights=True,
            execution_device="cpu",
            storage_device="cpu",
            logger=logging.getLogger("test.model_manager_cache_invalidation"),
        )
        for _ in range(2)
    ]
    try:
        for cache in caches:
            cache.put("model-key", torch.ones(2))
            cache.put("model-key:unet", torch.ones(2))
            cache.put("other-model", torch.ones(2))

        services = SimpleNamespace(
            logger=MagicMock(),
            model_manager=SimpleNamespace(
                store=record_store,
                load=SimpleNamespace(ram_caches={"cpu": caches[0], "cpu:1": caches[1]}),
            ),
        )
        monkeypatch.setattr(
            model_manager_router.ApiDependencies,
            "invoker",
            SimpleNamespace(services=services),
            raising=False,
        )
        monkeypatch.setattr(model_manager_router, "prepare_model_config_for_response", lambda config, _deps: config)

        assert model_manager_router._update_model_record(key="model-key", changes=ModelRecordChanges()) is updated

        for cache in caches:
            assert ("model-key" not in cache._cached_models) is expected_eviction
            assert ("model-key:unet" not in cache._cached_models) is expected_eviction
            assert "other-model" in cache._cached_models
    finally:
        for cache in caches:
            cache.shutdown()


def test_reidentify_invalidates_caches_when_identification_changes_model_class(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Re-probing can change loader identity just like an explicit model-record edit."""
    from types import SimpleNamespace

    from invokeai.app.api.routers import model_manager as model_manager_router
    from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelSourceType, ModelType

    previous = SimpleNamespace(
        key="model-key",
        path="model.safetensors",
        name="Model",
        description="",
        cover_image=None,
        source="local-source",
        source_type=ModelSourceType.Path,
        image_encoder_model_id="custom/image-encoder",
        base=BaseModelType.StableDiffusion1,
        type=ModelType.Main,
        format=ModelFormat.Checkpoint,
        variant="old",
        cpu_only=False,
        default_settings=None,
    )
    identified = SimpleNamespace(
        key="model-key",
        path="model.safetensors",
        name="identified name",
        description="identified description",
        cover_image=None,
        source="new-source",
        source_type=ModelSourceType.Path,
        base=BaseModelType.StableDiffusionXL,
        type=ModelType.Main,
        format=ModelFormat.Checkpoint,
        variant="new",
        cpu_only=False,
        default_settings=None,
    )
    stored = SimpleNamespace(
        replace_model=MagicMock(return_value=identified), get_model=MagicMock(return_value=previous)
    )
    caches = [SimpleNamespace(drop_model=MagicMock(return_value=1)) for _ in range(2)]
    services = SimpleNamespace(
        model_manager=SimpleNamespace(
            store=stored, load=SimpleNamespace(ram_caches={str(i): cache for i, cache in enumerate(caches)})
        ),
        configuration=SimpleNamespace(models_path=tmp_path / "models"),
        logger=MagicMock(),
    )
    monkeypatch.setattr(
        model_manager_router.ApiDependencies, "invoker", SimpleNamespace(services=services), raising=False
    )
    monkeypatch.setattr(model_manager_router, "ModelOnDisk", lambda _path: object())
    factory_overrides: dict[str, object] = {}

    def identify(_model_on_disk: object, override_fields: dict[str, object]) -> object:
        factory_overrides.update(override_fields)
        return SimpleNamespace(config=identified)

    monkeypatch.setattr(model_manager_router.ModelConfigFactory, "from_model_on_disk", identify)

    assert model_manager_router._reidentify_model("model-key") is identified
    assert factory_overrides["image_encoder_model_id"] == "custom/image-encoder"

    for cache in caches:
        cache.drop_model.assert_called_once_with("model-key")


def test_reidentify_preserves_ip_adapter_encoder_override_with_blank_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    from invokeai.app.api.routers import model_manager as model_manager_router
    from invokeai.backend.model_manager.configs.factory import ModelConfigFactory
    from invokeai.backend.model_manager.configs.ip_adapter import IPAdapter_InvokeAI_SD1_Config
    from invokeai.backend.model_manager.taxonomy import ModelSourceType

    models_path = tmp_path / "models"
    model_path = models_path / "ip-adapter"
    model_path.mkdir(parents=True)
    torch.save({"ip_adapter": {"1.to_k_ip.weight": torch.empty(1, 768)}}, model_path / "ip_adapter.bin")
    (model_path / "image_encoder.txt").write_text("  \n", encoding="utf-8")
    result = ModelConfigFactory.from_model_on_disk(
        model_path, override_fields={"image_encoder_model_id": "custom/image-encoder"}, allow_unknown=False
    )
    assert isinstance(result.config, IPAdapter_InvokeAI_SD1_Config)
    previous = result.config
    previous.key = "ip-adapter-key"
    previous.path = "ip-adapter"
    previous.name = "IP-Adapter"
    previous.source = "local-source"
    previous.source_type = ModelSourceType.Path

    store = SimpleNamespace(get_model=lambda _key: previous, replace_model=lambda _key, updated: updated)
    services = SimpleNamespace(
        model_manager=SimpleNamespace(store=store, load=SimpleNamespace(ram_caches={})),
        configuration=SimpleNamespace(models_path=models_path),
        logger=MagicMock(),
    )
    monkeypatch.setattr(
        model_manager_router.ApiDependencies, "invoker", SimpleNamespace(services=services), raising=False
    )

    updated = model_manager_router._reidentify_model("ip-adapter-key")

    assert isinstance(updated, IPAdapter_InvokeAI_SD1_Config)
    assert updated.image_encoder_model_id == "custom/image-encoder"
