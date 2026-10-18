"""Model record edits and re-identification evict cached instances of the model exactly when the change
affects how it loads (see `test_model_load_common.py` for the predicate itself)."""

import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.api.routers import model_manager as model_manager_router
from invokeai.app.services.model_load.model_load_common import RecordEdit
from invokeai.app.services.model_records.model_records_base import ModelRecordChanges


def _no_record_edit(_key: str) -> contextlib.AbstractContextManager[RecordEdit]:
    return contextlib.nullcontext(RecordEdit())


def _config(*, fp8: bool | None = None, cpu_only: bool | None = None, **fields):
    return SimpleNamespace(
        cpu_only=cpu_only,
        default_settings=SimpleNamespace(fp8_storage=fp8),
        **fields,
    )


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
    edit = RecordEdit()
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
            load=SimpleNamespace(
                record_edit=lambda _key: contextlib.nullcontext(edit), ram_caches={"cuda:0": cache_a, "cuda:1": cache_b}
            ),
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
    # Loads that overlapped the edit are re-checked only when it changed how the model loads.
    assert edit.load_affecting is (field in {"path", "base", "type", "format", "variant", "repo_variant"})


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
        model_manager=SimpleNamespace(load=SimpleNamespace(record_edit=_no_record_edit, ram_caches={"cpu": cache})),
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
                load=SimpleNamespace(record_edit=_no_record_edit, ram_caches={"cpu": caches[0], "cpu:1": caches[1]}),
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
        cpu_only=True,
        default_settings=SimpleNamespace(fp8_storage=True),
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
        default_settings=SimpleNamespace(fp8_storage=False),
    )
    stored = SimpleNamespace(
        replace_model=MagicMock(return_value=identified), get_model=MagicMock(return_value=previous)
    )
    caches = [SimpleNamespace(drop_model=MagicMock(return_value=1)) for _ in range(2)]
    services = SimpleNamespace(
        model_manager=SimpleNamespace(
            store=stored,
            load=SimpleNamespace(
                record_edit=_no_record_edit, ram_caches={str(i): cache for i, cache in enumerate(caches)}
            ),
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
    assert identified.cpu_only is True
    assert identified.default_settings.fp8_storage is True

    for cache in caches:
        cache.drop_model.assert_called_once_with("model-key")


@pytest.mark.parametrize("cpu_only", [False, None])
@pytest.mark.parametrize("fp8_storage", [False, None])
def test_reidentify_preserves_explicit_unset_preferences(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cpu_only: bool | None, fp8_storage: bool | None
):
    """False and None are user choices too; probe defaults must not replace them."""
    from invokeai.app.api.routers import model_manager as model_manager_router
    from invokeai.backend.model_manager.taxonomy import ModelSourceType

    previous = SimpleNamespace(
        key="model-key",
        path="model.safetensors",
        name="Model",
        description="",
        cover_image=None,
        source="local-source",
        source_type=ModelSourceType.Path,
        cpu_only=cpu_only,
        default_settings=SimpleNamespace(fp8_storage=fp8_storage),
    )
    identified = SimpleNamespace(
        key="model-key",
        path="model.safetensors",
        name="identified name",
        description="identified description",
        cover_image=None,
        source="new-source",
        source_type=ModelSourceType.Path,
        cpu_only=not cpu_only if cpu_only is not None else True,
        default_settings=SimpleNamespace(fp8_storage=not fp8_storage if fp8_storage is not None else True),
    )
    store = SimpleNamespace(get_model=lambda _key: previous, replace_model=lambda _key, updated: updated)
    services = SimpleNamespace(
        model_manager=SimpleNamespace(
            store=store,
            load=SimpleNamespace(record_edit=_no_record_edit, ram_caches={}),
        ),
        configuration=SimpleNamespace(models_path=tmp_path / "models"),
        logger=MagicMock(),
    )
    monkeypatch.setattr(
        model_manager_router.ApiDependencies, "invoker", SimpleNamespace(services=services), raising=False
    )
    monkeypatch.setattr(model_manager_router, "ModelOnDisk", lambda _path: object())
    monkeypatch.setattr(
        model_manager_router.ModelConfigFactory,
        "from_model_on_disk",
        lambda *_args, **_kwargs: SimpleNamespace(config=identified),
    )

    updated = model_manager_router._reidentify_model("model-key")

    assert updated.cpu_only is cpu_only
    assert updated.default_settings.fp8_storage is fp8_storage


@pytest.mark.parametrize("cpu_only,fp8_storage", [(True, True), (False, False), (None, None)])
def test_reidentify_preserves_main_model_default_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cpu_only: bool | None,
    fp8_storage: bool | None,
):
    """Main model CPU and FP8 preferences live in the typed default_settings object."""
    from invokeai.app.api.routers import model_manager as model_manager_router
    from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
    from invokeai.backend.model_manager.configs.main import Main_Checkpoint_SD1_Config
    from invokeai.backend.model_manager.taxonomy import ModelSourceType, ModelVariantType, SchedulerPredictionType

    def main_config(*, name: str, settings: MainModelDefaultSettings) -> Main_Checkpoint_SD1_Config:
        return Main_Checkpoint_SD1_Config(
            hash=f"hash-{name}",
            path="model.safetensors",
            file_size=1,
            name=name,
            source="local-source",
            source_type=ModelSourceType.Path,
            prediction_type=SchedulerPredictionType.Epsilon,
            variant=ModelVariantType.Normal,
            default_settings=settings,
        )

    previous = main_config(
        name="Model",
        settings=MainModelDefaultSettings(cpu_only=cpu_only, fp8_storage=fp8_storage),
    )
    previous.key = "model-key"
    identified = main_config(
        name="Probed model",
        settings=MainModelDefaultSettings(
            cpu_only=not cpu_only if cpu_only is not None else True,
            fp8_storage=not fp8_storage if fp8_storage is not None else True,
        ),
    )
    store = SimpleNamespace(get_model=lambda _key: previous, replace_model=lambda _key, updated: updated)
    services = SimpleNamespace(
        model_manager=SimpleNamespace(
            store=store,
            load=SimpleNamespace(record_edit=_no_record_edit, ram_caches={}),
        ),
        configuration=SimpleNamespace(models_path=tmp_path / "models"),
        logger=MagicMock(),
    )
    monkeypatch.setattr(
        model_manager_router.ApiDependencies, "invoker", SimpleNamespace(services=services), raising=False
    )
    monkeypatch.setattr(model_manager_router, "ModelOnDisk", lambda _path: object())
    monkeypatch.setattr(
        model_manager_router.ModelConfigFactory,
        "from_model_on_disk",
        lambda *_args, **_kwargs: SimpleNamespace(config=identified),
    )

    updated = model_manager_router._reidentify_model("model-key")

    assert isinstance(updated, Main_Checkpoint_SD1_Config)
    assert updated.default_settings is not None
    assert updated.default_settings.cpu_only is cpu_only
    assert updated.default_settings.fp8_storage is fp8_storage


def test_reidentify_does_not_mutate_shared_architecture_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from invokeai.app.api.routers import model_manager as model_manager_router
    from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
    from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
    from invokeai.backend.model_manager.configs.main import Main_Checkpoint_SD1_Config
    from invokeai.backend.model_manager.taxonomy import ModelSourceType, ModelVariantType, SchedulerPredictionType

    shared_defaults = MainModelDefaultSettings(cpu_only=None, fp8_storage=True)
    facet = DefaultSettingsFacet(by_variant={None: shared_defaults})
    assert facet.resolve() is shared_defaults

    def main_config(*, name: str, settings: MainModelDefaultSettings) -> Main_Checkpoint_SD1_Config:
        return Main_Checkpoint_SD1_Config(
            hash=f"hash-{name}",
            path="model.safetensors",
            file_size=1,
            name=name,
            source="local-source",
            source_type=ModelSourceType.Path,
            prediction_type=SchedulerPredictionType.Epsilon,
            variant=ModelVariantType.Normal,
            default_settings=settings,
        )

    previous = main_config(
        name="Model",
        settings=MainModelDefaultSettings(cpu_only=True, fp8_storage=False),
    )
    previous.key = "model-key"
    identified = main_config(name="Probed model", settings=facet.resolve())
    store = SimpleNamespace(get_model=lambda _key: previous, replace_model=lambda _key, updated: updated)
    services = SimpleNamespace(
        model_manager=SimpleNamespace(
            store=store,
            load=SimpleNamespace(record_edit=_no_record_edit, ram_caches={}),
        ),
        configuration=SimpleNamespace(models_path=tmp_path / "models"),
        logger=MagicMock(),
    )
    monkeypatch.setattr(
        model_manager_router.ApiDependencies, "invoker", SimpleNamespace(services=services), raising=False
    )
    monkeypatch.setattr(model_manager_router, "ModelOnDisk", lambda _path: object())
    monkeypatch.setattr(
        model_manager_router.ModelConfigFactory,
        "from_model_on_disk",
        lambda *_args, **_kwargs: SimpleNamespace(config=identified),
    )

    updated = model_manager_router._reidentify_model("model-key")
    later_model_defaults = facet.resolve()

    assert updated.default_settings is not shared_defaults
    assert updated.default_settings is not None
    assert updated.default_settings.cpu_only is True
    assert updated.default_settings.fp8_storage is False
    assert later_model_defaults is shared_defaults
    assert later_model_defaults.cpu_only is None
    assert later_model_defaults.fp8_storage is True


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
        model_manager=SimpleNamespace(store=store, load=SimpleNamespace(record_edit=_no_record_edit, ram_caches={})),
        configuration=SimpleNamespace(models_path=models_path),
        logger=MagicMock(),
    )
    monkeypatch.setattr(
        model_manager_router.ApiDependencies, "invoker", SimpleNamespace(services=services), raising=False
    )

    updated = model_manager_router._reidentify_model("ip-adapter-key")

    assert isinstance(updated, IPAdapter_InvokeAI_SD1_Config)
    assert updated.image_encoder_model_id == "custom/image-encoder"
