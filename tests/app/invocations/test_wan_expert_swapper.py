"""Tests for ``_ExpertSwapper``'s LoRA-context lifecycle.

The swapper is responsible for entering and exiting both the
``model_on_device`` context and the ``LayerPatcher.apply_smart_model_patches``
context in the right order across an expert swap:

  enter HIGH:  enter device(HIGH)  ->  enter lora(HIGH)
  swap:        exit lora(HIGH)     ->  exit device(HIGH)
               enter device(LOW)   ->  enter lora(LOW)
  close:       exit lora(LOW)      ->  exit device(LOW)

These tests use a tiny ``nn.Linear`` standing in for each transformer expert
so we can verify the swapper hands back the right model and routes the right
LoRA factory at each step.
"""

from typing import Iterable, Tuple
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from invokeai.app.invocations.wan_denoise import _ExpertSwapper
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw


class _FakeModelOnDevice:
    """Minimal stand-in for the model-cache record's ``model_on_device`` context.

    Tracks enter/exit to verify the swapper's lifecycle invariants."""

    def __init__(self, label: str, model: nn.Module, log: list[str]) -> None:
        self._label = label
        self._model = model
        self._log = log

    def __enter__(self):
        self._log.append(f"device-enter:{self._label}")
        # Return shape mirrors the real model cache: (cached_weights, model).
        return (None, self._model)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._log.append(f"device-exit:{self._label}")
        return False


class _FakeCachedModel:
    """Stand-in for ``CachedModelWithPartialLoad``: records full_unload_from_vram calls."""

    def __init__(self, label: str, log: list[str]) -> None:
        self._label = label
        self._log = log
        self.unload_calls = 0

    def full_unload_from_vram(self) -> int:
        self._log.append(f"full-unload:{self._label}")
        self.unload_calls += 1
        return 0


class _FakeCacheRecord:
    def __init__(self, cached_model: _FakeCachedModel) -> None:
        self.cached_model = cached_model


class _FakeInfo:
    """Mirrors the runtime ``LoadedModel`` enough for the swapper to reach
    ``info._cache_record.cached_model.full_unload_from_vram()`` on swap."""

    def __init__(self, label: str, model: nn.Module, log: list[str]) -> None:
        self._label = label
        self._model = model
        self._log = log
        self._cache_record = _FakeCacheRecord(_FakeCachedModel(label, log))

    def model_on_device(self):
        return _FakeModelOnDevice(self._label, self._model, self._log)


class _FakeContext:
    """Mocks ``InvocationContext.models.load`` returning a fresh ``_FakeInfo``
    for each call — mirrors the real behaviour where the swapper expects a
    fresh handle per ``get()``."""

    def __init__(self, infos_by_model_id: dict[str, _FakeInfo], log: list[str]) -> None:
        self._infos = infos_by_model_id
        self._log = log
        # Track how many times each model id was loaded — the lazy-load fix
        # depends on this count being 1 per swap, not 1 upfront.

        class _Models:
            def __init__(self, outer):
                self._outer = outer
                self.load_calls: list[str] = []

            def load(self, model_id):
                self.load_calls.append(model_id)
                self._outer._log.append(f"models.load:{model_id}")
                return self._outer._infos[model_id]

        self.models = _Models(self)


def _make_factory(log: list[str], label: str) -> "callable":
    """Build a LoRAIteratorFactory that records each invocation in ``log``."""

    def factory() -> Iterable[Tuple[ModelPatchRaw, float]]:
        log.append(f"lora-factory-call:{label}")
        return iter([])

    return factory


def _stub_lora_context_manager(log: list[str]):
    """Patch ``LayerPatcher.apply_smart_model_patches`` to a stub that records
    enter/exit in ``log`` and returns a no-op context manager.

    The stub introspects its arguments so we can verify the swapper passes
    the correct ``model``, ``patches`` factory output, and prefix.
    """
    calls: list[dict] = []

    class _Stub:
        def __init__(self, model, patches, prefix, dtype, cached_weights, force_sidecar_patching):
            self.model = model
            self.patches = patches
            self.prefix = prefix
            self.dtype = dtype
            self.cached_weights = cached_weights
            self.force_sidecar_patching = force_sidecar_patching
            calls.append(
                {
                    "model": model,
                    "prefix": prefix,
                    "dtype": dtype,
                    "force_sidecar_patching": force_sidecar_patching,
                }
            )

        def __enter__(self):
            log.append("lora-enter")
            # Force the factory's iterator to evaluate so we can assert it was
            # consumed (mirrors the real LayerPatcher behavior).
            list(self.patches)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            log.append("lora-exit")
            return False

    def factory(model, patches, prefix, dtype, cached_weights, force_sidecar_patching=False):
        return _Stub(model, patches, prefix, dtype, cached_weights, force_sidecar_patching)

    return factory, calls


def test_lifecycle_high_only():
    """Single-expert (TI2V-5B / A14B with only high loaded): enter HIGH, close."""
    log: list[str] = []
    high_nn = nn.Linear(1, 1)
    ctx = _FakeContext({"high": _FakeInfo("HIGH", high_nn, log)}, log)

    stub, calls = _stub_lora_context_manager(log)
    with patch(
        "invokeai.app.invocations.wan_denoise.LayerPatcher.apply_smart_model_patches",
        side_effect=stub,
    ):
        swapper = _ExpertSwapper(
            context=ctx,
            high_model="high",
            low_model=None,
            inference_dtype=torch.bfloat16,
            high_lora_factory=_make_factory(log, "HIGH"),
            low_lora_factory=None,
        )
        model = swapper.get(_ExpertSwapper.HIGH)
        assert model is high_nn
        swapper.close()

    assert log == [
        "models.load:high",
        "device-enter:HIGH",
        "lora-factory-call:HIGH",
        "lora-enter",
        "lora-exit",
        "device-exit:HIGH",
    ]
    assert len(calls) == 1
    assert calls[0]["model"] is high_nn
    assert calls[0]["prefix"] == "lora_transformer-"


def test_lifecycle_dual_expert_swap():
    """A14B: HIGH first, then LOW. Each LoRA context opens/closes with its expert."""
    log: list[str] = []
    high_nn = nn.Linear(1, 1)
    low_nn = nn.Linear(1, 1)
    ctx = _FakeContext(
        {"high": _FakeInfo("HIGH", high_nn, log), "low": _FakeInfo("LOW", low_nn, log)},
        log,
    )

    stub, calls = _stub_lora_context_manager(log)
    with patch(
        "invokeai.app.invocations.wan_denoise.LayerPatcher.apply_smart_model_patches",
        side_effect=stub,
    ):
        swapper = _ExpertSwapper(
            context=ctx,
            high_model="high",
            low_model="low",
            inference_dtype=torch.bfloat16,
            high_lora_factory=_make_factory(log, "HIGH"),
            low_lora_factory=_make_factory(log, "LOW"),
        )
        first = swapper.get(_ExpertSwapper.HIGH)
        assert first is high_nn

        second = swapper.get(_ExpertSwapper.LOW)
        assert second is low_nn

        swapper.close()

    expected = [
        # enter HIGH (models.load first, then device, then lora)
        "models.load:high",
        "device-enter:HIGH",
        "lora-factory-call:HIGH",
        "lora-enter",
        # swap to LOW: LoRA out -> device out -> force-unload HIGH -> models.load LOW
        # -> device in -> LoRA in. The full-unload step shoves HIGH's weights off GPU
        # before the cache decides how much room LOW gets.
        "lora-exit",
        "device-exit:HIGH",
        "full-unload:HIGH",
        "models.load:low",
        "device-enter:LOW",
        "lora-factory-call:LOW",
        "lora-enter",
        # close
        "lora-exit",
        "device-exit:LOW",
    ]
    assert log == expected
    # Two patcher invocations, each bound to the expected model.
    assert len(calls) == 2
    assert calls[0]["model"] is high_nn
    assert calls[1]["model"] is low_nn


def test_quantized_flag_forwards_to_sidecar():
    """GGUF (quantized) experts must request sidecar patching."""
    log: list[str] = []
    high_nn = nn.Linear(1, 1)
    ctx = _FakeContext({"high": _FakeInfo("HIGH", high_nn, log)}, log)

    stub, calls = _stub_lora_context_manager(log)
    with patch(
        "invokeai.app.invocations.wan_denoise.LayerPatcher.apply_smart_model_patches",
        side_effect=stub,
    ):
        swapper = _ExpertSwapper(
            context=ctx,
            high_model="high",
            low_model=None,
            inference_dtype=torch.bfloat16,
            high_lora_factory=_make_factory(log, "HIGH"),
            high_is_quantized=True,
        )
        swapper.get(_ExpertSwapper.HIGH)
        swapper.close()

    assert calls[0]["force_sidecar_patching"] is True


def test_no_lora_factory_skips_lora_context():
    """When no LoRAs are wired, the swapper doesn't enter the LoRA context."""
    log: list[str] = []
    high_nn = nn.Linear(1, 1)
    ctx = _FakeContext({"high": _FakeInfo("HIGH", high_nn, log)}, log)

    stub, calls = _stub_lora_context_manager(log)
    with patch(
        "invokeai.app.invocations.wan_denoise.LayerPatcher.apply_smart_model_patches",
        side_effect=stub,
    ):
        swapper = _ExpertSwapper(
            context=ctx,
            high_model="high",
            low_model=None,
            inference_dtype=torch.bfloat16,
            high_lora_factory=None,  # no LoRAs
            low_lora_factory=None,
        )
        swapper.get(_ExpertSwapper.HIGH)
        swapper.close()

    # No "lora-enter" / "lora-exit" entries — LayerPatcher was never invoked.
    assert "lora-enter" not in log
    assert "lora-exit" not in log
    assert len(calls) == 0


def test_repeat_get_same_label_is_a_no_op():
    """Calling get(HIGH) twice in a row must not re-enter the contexts.

    Critically, ``models.load`` must only be called once per actual swap —
    not on every ``get()``. Caching the loaded model on first entry, and
    short-circuiting re-entry, prevents per-step cache thrash."""
    log: list[str] = []
    high_nn = nn.Linear(1, 1)
    ctx = _FakeContext({"high": _FakeInfo("HIGH", high_nn, log)}, log)

    stub, calls = _stub_lora_context_manager(log)
    with patch(
        "invokeai.app.invocations.wan_denoise.LayerPatcher.apply_smart_model_patches",
        side_effect=stub,
    ):
        swapper = _ExpertSwapper(
            context=ctx,
            high_model="high",
            low_model=None,
            inference_dtype=torch.bfloat16,
            high_lora_factory=_make_factory(log, "HIGH"),
        )
        swapper.get(_ExpertSwapper.HIGH)
        swapper.get(_ExpertSwapper.HIGH)  # should be a no-op
        swapper.close()

    # device-enter + lora-enter happen exactly once, and crucially
    # models.load is called only once — repeat get() must short-circuit
    # so the cache isn't re-touched every step of the denoise loop.
    assert log.count("models.load:high") == 1
    assert log.count("device-enter:HIGH") == 1
    assert log.count("lora-enter") == 1
    assert log.count("lora-exit") == 1
    assert log.count("device-exit:HIGH") == 1


def test_lazy_load_per_swap_not_upfront():
    """Regression for the cache-eviction warning that triggered this fix.

    ``models.load`` must NOT be called at swapper construction. It is called
    only on the first ``get()`` for each expert. This keeps the per-handle
    cache window small enough that the LRU policy doesn't drop one expert
    while the other is being used."""
    log: list[str] = []
    high_nn = nn.Linear(1, 1)
    low_nn = nn.Linear(1, 1)
    ctx = _FakeContext(
        {"high": _FakeInfo("HIGH", high_nn, log), "low": _FakeInfo("LOW", low_nn, log)},
        log,
    )

    stub, _ = _stub_lora_context_manager(log)
    with patch(
        "invokeai.app.invocations.wan_denoise.LayerPatcher.apply_smart_model_patches",
        side_effect=stub,
    ):
        # Construction alone must not trigger any models.load call.
        swapper = _ExpertSwapper(
            context=ctx,
            high_model="high",
            low_model="low",
            inference_dtype=torch.bfloat16,
            high_lora_factory=_make_factory(log, "HIGH"),
            low_lora_factory=_make_factory(log, "LOW"),
        )
        assert ctx.models.load_calls == [], (
            "Swapper must not call models.load until get() is invoked — see issue #7513 for cache-eviction rationale."
        )

        # First get(HIGH): loads HIGH only.
        swapper.get(_ExpertSwapper.HIGH)
        assert ctx.models.load_calls == ["high"]

        # Swap to LOW: loads LOW only. HIGH is NOT re-loaded — its handle
        # was used and released, the next call to it (if any) will re-load.
        swapper.get(_ExpertSwapper.LOW)
        assert ctx.models.load_calls == ["high", "low"]

        # Back to HIGH: a fresh load (the previous handle is gone). This is
        # the right behaviour — each swap gets a guaranteed-fresh handle
        # rather than a stale reference into the cache.
        swapper.get(_ExpertSwapper.HIGH)
        assert ctx.models.load_calls == ["high", "low", "high"]

        swapper.close()


def test_empty_cache_called_on_swap():
    """Regression: each expert swap must trigger ``TorchDevice.empty_cache()`` so
    the next ``partial_load_to_vram`` sees an un-fragmented allocator.

    A14B users reported the low-noise expert ending up far more CPU-resident than
    the high-noise one — the previous expert's freed blocks stayed pinned in the
    PyTorch caching allocator across the swap, and partial_load decided there
    wasn't room for as much of the incoming expert as there actually was."""
    log: list[str] = []
    high_nn = nn.Linear(1, 1)
    low_nn = nn.Linear(1, 1)
    ctx = _FakeContext(
        {"high": _FakeInfo("HIGH", high_nn, log), "low": _FakeInfo("LOW", low_nn, log)},
        log,
    )

    stub, _ = _stub_lora_context_manager(log)
    with (
        patch(
            "invokeai.app.invocations.wan_denoise.LayerPatcher.apply_smart_model_patches",
            side_effect=stub,
        ),
        patch("invokeai.app.invocations.wan_denoise.TorchDevice.empty_cache") as empty_cache_mock,
    ):
        swapper = _ExpertSwapper(
            context=ctx,
            high_model="high",
            low_model="low",
            inference_dtype=torch.bfloat16,
            high_lora_factory=_make_factory(log, "HIGH"),
            low_lora_factory=_make_factory(log, "LOW"),
        )
        swapper.get(_ExpertSwapper.HIGH)
        first_call_count = empty_cache_mock.call_count
        assert first_call_count >= 1, "empty_cache should run on the initial expert load too"

        swapper.get(_ExpertSwapper.LOW)
        assert empty_cache_mock.call_count > first_call_count, (
            "empty_cache must be called on each HIGH→LOW (or LOW→HIGH) swap"
        )

        # Same-label re-get is a no-op; empty_cache must NOT be re-invoked.
        before_no_op = empty_cache_mock.call_count
        swapper.get(_ExpertSwapper.LOW)
        assert empty_cache_mock.call_count == before_no_op, (
            "Re-getting the active expert must short-circuit before empty_cache."
        )

        swapper.close()


def test_outgoing_expert_force_unloaded_from_vram():
    """Regression: on swap, the previous expert's weights must be explicitly forced
    off VRAM via ``cached_model.full_unload_from_vram()``.

    A14B users observed the high-noise transformer continuing to occupy ~9 GB of
    VRAM during the low-noise step, because the cache's automatic offload heuristic
    underestimated how much room the new expert needed when workspace memory from
    the previous denoise step was still allocated. The swapper sidesteps that by
    invoking full_unload_from_vram on the outgoing expert directly."""
    log: list[str] = []
    high_info = _FakeInfo("HIGH", nn.Linear(1, 1), log)
    low_info = _FakeInfo("LOW", nn.Linear(1, 1), log)
    ctx = _FakeContext({"high": high_info, "low": low_info}, log)

    stub, _ = _stub_lora_context_manager(log)
    with patch(
        "invokeai.app.invocations.wan_denoise.LayerPatcher.apply_smart_model_patches",
        side_effect=stub,
    ):
        swapper = _ExpertSwapper(
            context=ctx,
            high_model="high",
            low_model="low",
            inference_dtype=torch.bfloat16,
            high_lora_factory=_make_factory(log, "HIGH"),
            low_lora_factory=_make_factory(log, "LOW"),
        )
        # Initial load: nothing to unload yet.
        swapper.get(_ExpertSwapper.HIGH)
        assert high_info._cache_record.cached_model.unload_calls == 0
        assert low_info._cache_record.cached_model.unload_calls == 0

        # Swap to LOW: HIGH must be force-unloaded; LOW is the incoming expert and
        # must not be unloaded.
        swapper.get(_ExpertSwapper.LOW)
        assert high_info._cache_record.cached_model.unload_calls == 1
        assert low_info._cache_record.cached_model.unload_calls == 0

        # Swap back to HIGH: LOW must now be force-unloaded.
        swapper.get(_ExpertSwapper.HIGH)
        assert low_info._cache_record.cached_model.unload_calls == 1

        swapper.close()


def test_device_context_released_when_lora_enter_raises():
    """Regression: if the LoRA patcher's ``__enter__`` raises, the device context
    must still be released on the next swap or close.

    Earlier shape stashed ``self._active_device_ctx`` only after the LoRA enter
    succeeded, so an exception there left the device context entered but
    unreachable — ``_release`` saw ``None`` and walked away, leaving 8–9 GB of
    GGUF expert weights pinned to GPU until the model cache LRU evicted them."""
    log: list[str] = []
    high_nn = nn.Linear(1, 1)
    ctx = _FakeContext({"high": _FakeInfo("HIGH", high_nn, log)}, log)

    class _RaisingLoraStub:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            raise RuntimeError("LoRA patcher blew up")

        def __exit__(self, *_args):
            return False

    with patch(
        "invokeai.app.invocations.wan_denoise.LayerPatcher.apply_smart_model_patches",
        side_effect=lambda **_kwargs: _RaisingLoraStub(),
    ):
        swapper = _ExpertSwapper(
            context=ctx,
            high_model="high",
            low_model=None,
            inference_dtype=torch.bfloat16,
            high_lora_factory=_make_factory(log, "HIGH"),
            low_lora_factory=None,
        )
        with pytest.raises(RuntimeError, match="LoRA patcher blew up"):
            swapper.get(_ExpertSwapper.HIGH)
        # close() must succeed and must call the device context's __exit__ so
        # the model leaves GPU. If the device context were unreachable,
        # device-exit:HIGH would be missing from the log.
        swapper.close()


def test_device_context_released_when_lora_exit_raises():
    """Counterpart to the enter-raises regression: if LoRA weight-restore (``__exit__``)
    raises — e.g. OOM restoring original weights on a nearly full card — the device
    context must still exit so the cache record unlocks and the expert leaves GPU."""
    log: list[str] = []
    high_nn = nn.Linear(1, 1)
    ctx = _FakeContext({"high": _FakeInfo("HIGH", high_nn, log)}, log)

    class _ExitRaisingLoraStub:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            raise RuntimeError("LoRA weight restore blew up")

    with patch(
        "invokeai.app.invocations.wan_denoise.LayerPatcher.apply_smart_model_patches",
        side_effect=lambda **_kwargs: _ExitRaisingLoraStub(),
    ):
        swapper = _ExpertSwapper(
            context=ctx,
            high_model="high",
            low_model=None,
            inference_dtype=torch.bfloat16,
            high_lora_factory=_make_factory(log, "HIGH"),
            low_lora_factory=None,
        )
        swapper.get(_ExpertSwapper.HIGH)
        with pytest.raises(RuntimeError, match="LoRA weight restore blew up"):
            swapper.close()
        assert "device-exit:HIGH" in log
        # The swapper must also have cleared its slots so a later close() is a no-op.
        assert swapper._active_device_ctx is None
        assert swapper._active_lora_ctx is None

    assert "device-exit:HIGH" in log, "device context must be exited even if LoRA enter raised"


def test_force_unload_failure_does_not_break_swap():
    """If full_unload_from_vram raises (e.g. cache evicted the entry between unlock
    and now), the swap must still succeed. Reaching into a private attribute is the
    pragmatic choice today; this test pins the defensive try/except so a future
    refactor of LoadedModel doesn't break swap reliability."""
    log: list[str] = []

    class _RaisingCachedModel:
        def full_unload_from_vram(self):
            raise RuntimeError("cache evicted me between unlock and unload")

    raising_high = _FakeInfo("HIGH", nn.Linear(1, 1), log)
    raising_high._cache_record = _FakeCacheRecord(_RaisingCachedModel())
    low_info = _FakeInfo("LOW", nn.Linear(1, 1), log)
    ctx = _FakeContext({"high": raising_high, "low": low_info}, log)

    stub, _ = _stub_lora_context_manager(log)
    with patch(
        "invokeai.app.invocations.wan_denoise.LayerPatcher.apply_smart_model_patches",
        side_effect=stub,
    ):
        swapper = _ExpertSwapper(
            context=ctx,
            high_model="high",
            low_model="low",
            inference_dtype=torch.bfloat16,
            high_lora_factory=_make_factory(log, "HIGH"),
            low_lora_factory=_make_factory(log, "LOW"),
        )
        swapper.get(_ExpertSwapper.HIGH)
        # Should not raise even though the outgoing expert's full_unload throws.
        model = swapper.get(_ExpertSwapper.LOW)
        assert model is low_info._model
        swapper.close()


def test_slots_cleared_when_device_exit_raises():
    """Counterpart to the LoRA-exit regression (JPPhoto review 2026-07-21): if the
    *device* context's ``__exit__`` raises, the swapper must still clear every active
    slot — otherwise a later ``close()`` double-exits the already-exited LoRA context
    and ``get()`` can hand back a stale model."""
    log: list[str] = []
    high_nn = nn.Linear(1, 1)

    class _ExitRaisingDeviceCtx(_FakeModelOnDevice):
        def __exit__(self, exc_type, exc_val, exc_tb):
            self._log.append(f"device-exit:{self._label}")
            raise RuntimeError("device teardown blew up")

    class _ExitRaisingInfo(_FakeInfo):
        def model_on_device(self):
            return _ExitRaisingDeviceCtx(self._label, self._model, self._log)

    ctx = _FakeContext({"high": _ExitRaisingInfo("HIGH", high_nn, log)}, log)
    stub, _calls = _stub_lora_context_manager(log)
    with patch(
        "invokeai.app.invocations.wan_denoise.LayerPatcher.apply_smart_model_patches",
        side_effect=stub,
    ):
        swapper = _ExpertSwapper(
            context=ctx,
            high_model="high",
            low_model=None,
            inference_dtype=torch.bfloat16,
            high_lora_factory=_make_factory(log, "HIGH"),
            low_lora_factory=None,
        )
        swapper.get(_ExpertSwapper.HIGH)

        with pytest.raises(RuntimeError, match="device teardown blew up"):
            swapper.close()

        # Every slot must be cleared despite the device-exit failure...
        assert swapper._active_device_ctx is None
        assert swapper._active_lora_ctx is None
        assert swapper._active_model is None
        assert swapper._active_label is None

        # ...so a second close is a true no-op: no double LoRA exit, no second
        # device exit attempt.
        swapper.close()

    assert log.count("lora-exit") == 1
    assert log.count("device-exit:HIGH") == 1
