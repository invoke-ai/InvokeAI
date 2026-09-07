"""`context.models.make_room_in_vram` must reach the calling thread's device cache with both arguments intact:
the invocation context is the only route an invocation has to the cache, and a dropped `working_mem_bytes` would
silently fall back to the configured default."""

from unittest.mock import MagicMock

from invokeai.app.services.shared.invocation_context import ModelsInterface


def test_make_room_in_vram_delegates_to_the_threads_ram_cache():
    services = MagicMock()
    services.model_manager.load.ram_cache.make_room_in_vram.return_value = 123
    models = ModelsInterface(services=services, data=MagicMock(), util=MagicMock())

    assert models.make_room_in_vram(10, working_mem_bytes=7) == 123
    services.model_manager.load.ram_cache.make_room_in_vram.assert_called_once_with(10, 7)
