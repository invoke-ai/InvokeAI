"""`context.models.make_room_in_vram` must reach the calling thread's device cache and hand back its answer: the
invocation context is the only route an invocation has to the cache, and the post-offload availability is what the
caller compares its request against."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from invokeai.app.services.shared.invocation_context import ModelsInterface


def test_make_room_in_vram_delegates_to_the_threads_ram_cache():
    services = MagicMock()
    services.model_manager.load.ram_cache.make_room_in_vram.return_value = 123
    models = ModelsInterface(services=services, data=MagicMock(), util=MagicMock())

    assert models.make_room_in_vram(10) == 123
    services.model_manager.load.ram_cache.make_room_in_vram.assert_called_once_with(10)


def test_make_room_retains_weights_during_concurrent_construction(offload_race, monkeypatch):
    def offload(cache):
        monkeypatch.setattr(cache, "_get_physical_vram_available", lambda *_: 0)
        services = SimpleNamespace(model_manager=SimpleNamespace(load=SimpleNamespace(ram_cache=cache)))
        return ModelsInterface(services=services, data=None, util=None).make_room_in_vram(16)

    assert offload_race(offload, construction_first=True) == 0
    assert offload_race(offload, construction_first=False) == 0
