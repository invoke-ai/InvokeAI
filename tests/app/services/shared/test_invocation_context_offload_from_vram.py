"""The invocation offload interface preserves model keys, freed bytes, failures and real weights."""

from types import SimpleNamespace

import pytest

from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.shared.invocation_context import ModelsInterface
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType


def _models(cache):
    services = SimpleNamespace(model_manager=SimpleNamespace(load=SimpleNamespace(ram_cache=cache)))
    return ModelsInterface(services=services, data=None, util=None)


def test_invocation_offload_retains_weights_during_concurrent_construction(offload_race) -> None:
    def offload(cache):
        return _models(cache).offload_from_vram("old")

    assert offload_race(offload, construction_first=True) == 16
    assert offload_race(offload, construction_first=False) == 16


def test_offload_accepts_model_identifier_and_returns_cache_answer() -> None:
    keys = []

    def offload(key):
        keys.append(key)
        return 123

    identifier = ModelIdentifierField(
        key="encoder", hash="blake3:encoder", name="encoder", base=BaseModelType.StableDiffusion1, type=ModelType.Main
    )
    models = _models(SimpleNamespace(offload_model_from_vram=offload))
    assert models.offload_from_vram(identifier) == 123
    assert keys == ["encoder"]


def test_failed_offload_preserves_exception() -> None:
    error = RuntimeError("offload failed")

    def fail(key):
        raise error

    models = _models(SimpleNamespace(offload_model_from_vram=fail))
    with pytest.raises(RuntimeError) as raised:
        models.offload_from_vram("encoder")
    assert raised.value is error
