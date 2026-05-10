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

from typing import Iterable, Iterator, Tuple
from unittest.mock import MagicMock, patch

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
    high_model = nn.Linear(1, 1)
    high_info = _FakeInfo("HIGH", high_model, log)

    stub, calls = _stub_lora_context_manager(log)
    with patch(
        "invokeai.app.invocations.wan_denoise.LayerPatcher.apply_smart_model_patches",
        side_effect=stub,
    ):
        swapper = _ExpertSwapper(
            high_info=high_info,
            low_info=None,
            inference_dtype=torch.bfloat16,
            high_lora_factory=_make_factory(log, "HIGH"),
            low_lora_factory=None,
        )
        model = swapper.get(_ExpertSwapper.HIGH)
        assert model is high_model
        swapper.close()

    assert log == [
        "device-enter:HIGH",
        "lora-factory-call:HIGH",
        "lora-enter",
        "lora-exit",
        "device-exit:HIGH",
    ]
    assert len(calls) == 1
    assert calls[0]["model"] is high_model
    assert calls[0]["prefix"] == "lora_transformer-"


def test_lifecycle_dual_expert_swap():
    """A14B: HIGH first, then LOW. Each LoRA context opens/closes with its expert."""
    log: list[str] = []
    high_model = nn.Linear(1, 1)
    low_model = nn.Linear(1, 1)
    high_info = _FakeInfo("HIGH", high_model, log)
    low_info = _FakeInfo("LOW", low_model, log)

    stub, calls = _stub_lora_context_manager(log)
    with patch(
        "invokeai.app.invocations.wan_denoise.LayerPatcher.apply_smart_model_patches",
        side_effect=stub,
    ):
        swapper = _ExpertSwapper(
            high_info=high_info,
            low_info=low_info,
            inference_dtype=torch.bfloat16,
            high_lora_factory=_make_factory(log, "HIGH"),
            low_lora_factory=_make_factory(log, "LOW"),
        )
        first = swapper.get(_ExpertSwapper.HIGH)
        assert first is high_model

        second = swapper.get(_ExpertSwapper.LOW)
        assert second is low_model

        swapper.close()

    expected = [
        # enter HIGH (device, then lora)
        "device-enter:HIGH",
        "lora-factory-call:HIGH",
        "lora-enter",
        # swap to LOW: LoRA out -> device out -> device in -> LoRA in
        "lora-exit",
        "device-exit:HIGH",
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
    assert calls[0]["model"] is high_model
    assert calls[1]["model"] is low_model


def test_quantized_flag_forwards_to_sidecar():
    """GGUF (quantized) experts must request sidecar patching."""
    log: list[str] = []
    high_model = nn.Linear(1, 1)
    high_info = _FakeInfo("HIGH", high_model, log)

    stub, calls = _stub_lora_context_manager(log)
    with patch(
        "invokeai.app.invocations.wan_denoise.LayerPatcher.apply_smart_model_patches",
        side_effect=stub,
    ):
        swapper = _ExpertSwapper(
            high_info=high_info,
            low_info=None,
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
    high_model = nn.Linear(1, 1)
    high_info = _FakeInfo("HIGH", high_model, log)

    stub, calls = _stub_lora_context_manager(log)
    with patch(
        "invokeai.app.invocations.wan_denoise.LayerPatcher.apply_smart_model_patches",
        side_effect=stub,
    ):
        swapper = _ExpertSwapper(
            high_info=high_info,
            low_info=None,
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
    """Calling get(HIGH) twice in a row must not re-enter the contexts."""
    log: list[str] = []
    high_model = nn.Linear(1, 1)
    high_info = _FakeInfo("HIGH", high_model, log)

    stub, calls = _stub_lora_context_manager(log)
    with patch(
        "invokeai.app.invocations.wan_denoise.LayerPatcher.apply_smart_model_patches",
        side_effect=stub,
    ):
        swapper = _ExpertSwapper(
            high_info=high_info,
            low_info=None,
            inference_dtype=torch.bfloat16,
            high_lora_factory=_make_factory(log, "HIGH"),
        )
        swapper.get(_ExpertSwapper.HIGH)
        swapper.get(_ExpertSwapper.HIGH)  # should be a no-op
        swapper.close()

    # device-enter + lora-enter happen exactly once.
    assert log.count("device-enter:HIGH") == 1
    assert log.count("lora-enter") == 1
    assert log.count("lora-exit") == 1
    assert log.count("device-exit:HIGH") == 1
