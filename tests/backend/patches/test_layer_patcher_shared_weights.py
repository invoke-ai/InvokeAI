"""Regression tests: LoRA direct patching must not mutate the model's canonical CPU weights.

In multi-GPU mode the per-device caches share one canonical CPU state_dict (SharedCpuWeightsStore),
and that same dict is the keep_ram_copy used to restore a model after unpatching. Direct patching
must therefore never mutate those tensors in place — otherwise a LoRA applied on one GPU would
corrupt the weights seen by the other GPU (and taint the cached "clean" copy even with one GPU).

These run on CPU and force direct patching, which is the path that touches CPU-resident weights.
"""

import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw
from tests.backend.patches.test_layer_patcher import DummyModuleWithOneLayer


def _make_loras(num_loras: int, in_features: int, out_features: int, rank: int):
    lora_models: list[tuple[ModelPatchRaw, float]] = []
    for _ in range(num_loras):
        layers = {
            "linear_layer_1": LoRALayer.from_state_dict_values(
                values={
                    "lora_down.weight": torch.ones((rank, in_features), device="cpu", dtype=torch.float32),
                    "lora_up.weight": torch.ones((out_features, rank), device="cpu", dtype=torch.float32),
                },
            )
        }
        lora_models.append((ModelPatchRaw(layers), 0.5))
    return lora_models


@torch.no_grad()
def test_force_direct_patch_does_not_mutate_canonical_cpu_weights():
    in_features, out_features, rank = 4, 8, 2
    model = DummyModuleWithOneLayer(in_features, out_features, device="cpu", dtype=torch.float32)
    apply_custom_layers_to_model(model)

    # `canonical` holds references to the model's actual parameter tensors — exactly what the shared
    # store would hand out as the canonical CPU copy and what model_on_device() passes as
    # cached_weights. We snapshot their values to detect any in-place mutation.
    canonical = dict(model.state_dict())
    snapshot = {k: v.detach().clone() for k, v in canonical.items()}

    lora_models = _make_loras(num_loras=2, in_features=in_features, out_features=out_features, rank=rank)
    x = torch.randn(1, in_features, dtype=torch.float32)
    out_before = model(x)

    with LayerPatcher.apply_smart_model_patches(
        model=model,
        patches=lora_models,
        prefix="",
        dtype=torch.float32,
        cached_weights=canonical,
        force_direct_patching=True,
    ):
        # Sanity: this really is the direct path (no sidecar wrappers), so weights were applied
        # directly — and the patch actually changed the output.
        assert model.linear_layer_1.get_num_patches() == 0
        out_during = model(x)
        assert not torch.allclose(out_before, out_during)

        # The canonical tensors must be untouched even while the patch is active.
        for k in canonical:
            torch.testing.assert_close(canonical[k], snapshot[k])

    # ...and after unpatching.
    for k in canonical:
        torch.testing.assert_close(canonical[k], snapshot[k])
    assert torch.allclose(out_before, model(x))


@torch.no_grad()
def test_two_models_sharing_canonical_are_isolated_under_direct_patch():
    """Patch one model built from the shared canonical weights; a second model built from the same
    canonical tensors must be unaffected (no cross-device bleed)."""
    in_features, out_features, rank = 4, 8, 2
    model_a = DummyModuleWithOneLayer(in_features, out_features, device="cpu", dtype=torch.float32)
    apply_custom_layers_to_model(model_a)
    canonical = dict(model_a.state_dict())

    # model_b shares the canonical tensors (as a second device's cache would via load_state_dict).
    model_b = DummyModuleWithOneLayer(in_features, out_features, device="cpu", dtype=torch.float32)
    apply_custom_layers_to_model(model_b)
    model_b.load_state_dict(canonical, assign=True)

    x = torch.randn(1, in_features, dtype=torch.float32)
    out_b_before = model_b(x)

    lora_models = _make_loras(num_loras=1, in_features=in_features, out_features=out_features, rank=rank)
    with LayerPatcher.apply_smart_model_patches(
        model=model_a,
        patches=lora_models,
        prefix="",
        dtype=torch.float32,
        cached_weights=canonical,
        force_direct_patching=True,
    ):
        # model_a is patched; model_b (sharing the canonical weights) must be unchanged.
        assert torch.allclose(model_b(x), out_b_before)

    assert torch.allclose(model_b(x), out_b_before)
