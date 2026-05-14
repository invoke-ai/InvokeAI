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


class _FakeInfo:
    def __init__(self, label: str, model: nn.Module, log: list[str]) -> None:
        self._label = label
        self._model = model
        self._log = log

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
        # swap to LOW: LoRA out -> device out -> models.load -> device in -> LoRA in
        "lora-exit",
        "device-exit:HIGH",
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
