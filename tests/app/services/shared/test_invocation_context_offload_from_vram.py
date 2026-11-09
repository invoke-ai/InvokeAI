"""The invocation offload interface preserves model keys, freed bytes, failures and real weights."""

import threading
from types import SimpleNamespace

import pytest
import torch

from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.shared.invocation_context import ModelsInterface
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from tests.fixtures import model_offload
from tests.fixtures.model_offload import assert_load_lock_idle

offload_race = model_offload.offload_race


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
    assert_load_lock_idle()
    assert models.offload_from_vram(identifier) == 123
    assert keys == ["encoder"]
    assert_load_lock_idle()


def test_failed_offload_releases_load_lock_and_preserves_exception() -> None:
    error = RuntimeError("offload failed")

    def fail(key):
        raise error

    models = _models(SimpleNamespace(offload_model_from_vram=fail))
    assert_load_lock_idle()
    with pytest.raises(RuntimeError) as raised:
        models.offload_from_vram("encoder")
    assert raised.value is error
    assert_load_lock_idle()


def test_offloads_on_independent_caches_can_assign_weights_concurrently() -> None:
    """Two device caches may move together. Only cache I/O is replaced; registration and the shared lock are real."""
    assert_load_lock_idle()
    ready = [threading.Event(), threading.Event()]
    models = [torch.nn.Linear(2, 2, bias=False) for _ in range(2)]
    expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    errors, results = [], []

    def move(index):
        def assign(key):
            ready[index].set()
            assert ready[1 - index].wait(20), "independent offloads were serialized"
            models[index].load_state_dict({"weight": expected.clone()}, assign=True)
            return 16

        try:
            results.append(_models(SimpleNamespace(offload_model_from_vram=assign)).offload_from_vram("encoder"))
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=move, args=(index,), daemon=True) for index in range(2)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(20)
    finally:
        for event in ready:
            event.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(20)
        assert all(not thread.is_alive() for thread in threads)
        assert_load_lock_idle()

    assert not errors
    assert results == [16, 16]
    for model in models:
        assert model.weight.device.type == "cpu"
        torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
