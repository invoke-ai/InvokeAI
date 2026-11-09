"""Native offload/construction races shared by the three application entrypoints."""

import threading
from time import monotonic

import pytest
import torch
from accelerate import init_empty_weights

from invokeai.backend.model_manager.load.model_cache.model_cache import MODEL_LOAD_LOCK, ModelCache
from invokeai.backend.util.devices import TorchDevice

WAIT_TIMEOUT = 20


def assert_load_lock_idle() -> None:
    with MODEL_LOAD_LOCK._cond:
        assert MODEL_LOAD_LOCK._readers == MODEL_LOAD_LOCK._writers_waiting == 0
        assert not MODEL_LOAD_LOCK._writer_active


@pytest.fixture
def offload_race(monkeypatch):
    """Use real cache locks and CPU-backed full offloads; only device residency and allocator I/O are simulated.

    The wait observer forwards to the real condition. It acknowledges contention without timing a negative event,
    and bounds deadlocks so real context managers can unwind without resetting process-global lock state.
    """

    def run(offload, construction_first: bool):
        assert_load_lock_idle()
        with monkeypatch.context() as patch:
            patch.setattr(TorchDevice, "empty_cache", classmethod(lambda cls, **kwargs: None))
            cache = ModelCache(
                execution_device_working_mem_gb=0,
                execution_device="cpu",
                storage_device="cpu",
                enable_partial_loading=False,
                keep_ram_copy_of_weights=True,
                shared_cpu_weights=None,
            )
            model = torch.nn.Linear(2, 2, bias=False)
            expected = torch.arange(4, dtype=torch.float32).reshape(2, 2)
            model.load_state_dict({"weight": expected.clone()}, assign=True)
            cache.put("old:encoder", model)
            cached = cache._cached_models["old:encoder"].cached_model
            cached.full_load_to_vram()  # CPU copies, but the real wrapper records residency and retains its RAM copy.
            patch.setattr(cache, "_execution_device", torch.device("cuda:0"))
            ready, proceed, attempted, registered = (threading.Event() for _ in range(4))
            errors, results, constructed = [], [], []
            deadline = monotonic() + WAIT_TIMEOUT
            real_wait, real_assign = MODEL_LOAD_LOCK._cond.wait, model.load_state_dict

            def wait(timeout=None):
                # Waiting for construction must happen before the calling thread owns the cache lock.
                assert not cache._lock._is_owned(), "cache lock acquired before model load lock"
                attempted.set()
                remaining = deadline - monotonic()
                assert remaining > 0, "model load lock did not make progress"
                return real_wait(remaining if timeout is None else min(timeout, remaining))

            def assign(*args, **kwargs):
                try:
                    if not construction_first:
                        ready.set()
                        assert proceed.wait(WAIT_TIMEOUT), "offload was not released"
                    return real_assign(*args, **kwargs)
                finally:
                    registered.set()
                    if construction_first:
                        attempted.set()

            def move():
                try:
                    results.append(offload(cache))
                except BaseException as error:
                    errors.append(error)
                    attempted.set()

            def construct():
                try:
                    with MODEL_LOAD_LOCK.write_lock(), init_empty_weights():
                        constructed.append(torch.nn.Linear(2, 2).weight.device.type)
                        attempted.set()
                        assert registered.wait(WAIT_TIMEOUT), "offload never registered its weights"
                except BaseException as error:
                    errors.append(error)

            patch.setattr(MODEL_LOAD_LOCK._cond, "wait", wait)
            patch.setattr(model, "load_state_dict", assign)
            mover = threading.Thread(target=move, daemon=True)
            builder = threading.Thread(target=construct, daemon=True)
            try:
                if construction_first:
                    with MODEL_LOAD_LOCK.write_lock(), init_empty_weights():
                        constructed.append(torch.nn.Linear(2, 2).weight.device.type)
                        mover.start()
                        assert attempted.wait(WAIT_TIMEOUT), "offload neither waited nor assigned"
                else:
                    mover.start()
                    assert ready.wait(WAIT_TIMEOUT), f"offload never reached assignment: {errors!r}"
                    builder.start()
                    assert attempted.wait(WAIT_TIMEOUT), "construction neither waited nor started"
                    proceed.set()
            finally:
                proceed.set()
                if mover.ident is not None:
                    mover.join(WAIT_TIMEOUT)
                registered.set()
                if builder.ident is not None:
                    builder.join(WAIT_TIMEOUT)
                assert not mover.is_alive() and not builder.is_alive(), "offload/construction deadlocked"
                cache.shutdown()
                assert_load_lock_idle()

            assert constructed == ["meta"]  # The real accelerate patch was active, not a stand-in constructor.
            assert model.weight.device.type == "cpu", f"offload stranded weights on meta: {errors!r}"
            torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
            assert not errors, f"offload/construction failed: {errors!r}"
            assert cached.cur_vram_bytes() == 0
            return results[0]

    return run
