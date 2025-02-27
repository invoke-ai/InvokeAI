import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.cached_model.cached_model_with_partial_load import (
    CachedModelWithPartialLoad,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    apply_custom_layers_to_model,
)
from invokeai.backend.patches.layer_patcher import LayerPatcher
from invokeai.backend.patches.layers.lora_layer import LoRALayer
from invokeai.backend.patches.model_patch_raw import ModelPatchRaw


class DummyModuleWithOneLayer(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: str, dtype: torch.dtype):
        super().__init__()
        self.linear_layer_1 = torch.nn.Linear(in_features, out_features, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer_1(x)


class DummyModuleWithTwoLayers(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: str, dtype: torch.dtype):
        super().__init__()
        self.linear_layer_1 = torch.nn.Linear(in_features, out_features, device=device, dtype=dtype)
        self.linear_layer_2 = torch.nn.Linear(out_features, out_features, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer_2(self.linear_layer_1(x))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")),
    ],
)
@pytest.mark.parametrize("num_loras", [1, 2])
@pytest.mark.parametrize(
    ["force_sidecar_patching", "force_direct_patching"], [(True, False), (False, True), (False, False)]
)
@torch.no_grad()
def test_apply_smart_model_patches(
    device: str, num_loras: int, force_sidecar_patching: bool, force_direct_patching: bool
):
    """Test the basic behavior of ModelPatcher.apply_smart_model_patches(...). Check that unpatching works correctly."""
    dtype = torch.float16
    linear_in_features = 4
    linear_out_features = 8
    lora_rank = 2
    model = DummyModuleWithOneLayer(linear_in_features, linear_out_features, device=device, dtype=dtype)
    apply_custom_layers_to_model(model)

    # Initialize num_loras LoRA models with weights of 0.5.
    lora_weight = 0.5
    lora_models: list[tuple[ModelPatchRaw, float]] = []
    for _ in range(num_loras):
        lora_layers = {
            "linear_layer_1": LoRALayer.from_state_dict_values(
                values={
                    "lora_down.weight": torch.ones((lora_rank, linear_in_features), device="cpu", dtype=torch.float16),
                    "lora_up.weight": torch.ones((linear_out_features, lora_rank), device="cpu", dtype=torch.float16),
                },
            )
        }
        lora = ModelPatchRaw(lora_layers)
        lora_models.append((lora, lora_weight))

    orig_linear_weight = model.linear_layer_1.weight.data.detach().clone()
    expected_patched_linear_weight = orig_linear_weight + (lora_rank * lora_weight * num_loras)

    # Run inference before patching the model.
    input = torch.randn(1, linear_in_features, device=device, dtype=dtype)
    output_before_patch = model(input)

    expect_sidecar_wrappers = device == "cpu"
    if force_sidecar_patching:
        expect_sidecar_wrappers = True
    elif force_direct_patching:
        expect_sidecar_wrappers = False

    # Patch the model and run inference during the patch.
    with LayerPatcher.apply_smart_model_patches(
        model=model,
        patches=lora_models,
        prefix="",
        dtype=dtype,
        force_direct_patching=force_direct_patching,
        force_sidecar_patching=force_sidecar_patching,
    ):
        if expect_sidecar_wrappers:
            # There should be sidecar patches in the model.
            assert model.linear_layer_1.get_num_patches() == num_loras
        else:
            # There should be no sidecar patches in the model.
            assert model.linear_layer_1.get_num_patches() == 0
            torch.testing.assert_close(model.linear_layer_1.weight.data, expected_patched_linear_weight)

            # After patching, the patched model should still be on its original device.
            assert model.linear_layer_1.weight.data.device.type == device

            # After patching, all LoRA layer weights should have been moved back to the cpu.
            for lora, _ in lora_models:
                assert lora.layers["linear_layer_1"].up.device.type == "cpu"
                assert lora.layers["linear_layer_1"].down.device.type == "cpu"

        output_during_patch = model(input)

    # Run inference after unpatching.
    output_after_patch = model(input)

    # Check that the output before patching is different from the output during patching.
    assert not torch.allclose(output_before_patch, output_during_patch)

    # Check that the output before patching is the same as the output after patching.
    assert torch.allclose(output_before_patch, output_after_patch)


@pytest.mark.parametrize(["num_loras"], [(1,), (2,)])
@torch.no_grad()
def test_apply_smart_lora_patches_to_partially_loaded_model(num_loras: int):
    """Test the behavior of ModelPatcher.apply_smart_lora_patches(...) when it is applied to a
    CachedModelWithPartialLoad that is partially loaded into VRAM.
    """

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA device")

    # Initialize the model on the CPU.
    dtype = torch.float16
    linear_in_features = 4
    linear_out_features = 8
    lora_rank = 2
    model = DummyModuleWithTwoLayers(linear_in_features, linear_out_features, device="cpu", dtype=dtype)
    apply_custom_layers_to_model(model)
    cached_model = CachedModelWithPartialLoad(model=model, compute_device=torch.device("cuda"))
    model_total_bytes = cached_model.total_bytes()
    assert cached_model.cur_vram_bytes() == 0

    # Partially load the model into VRAM.
    target_vram_bytes = int(model_total_bytes * 0.6)
    _ = cached_model.partial_load_to_vram(target_vram_bytes)
    assert cached_model.model.linear_layer_1.weight.device.type == "cuda"
    assert cached_model.model.linear_layer_2.weight.device.type == "cpu"

    # Initialize num_loras LoRA models with weights of 0.5.
    lora_weight = 0.5
    lora_models: list[tuple[ModelPatchRaw, float]] = []
    for _ in range(num_loras):
        lora_layers = {
            "linear_layer_1": LoRALayer.from_state_dict_values(
                values={
                    "lora_down.weight": torch.ones((lora_rank, linear_in_features), device="cpu", dtype=torch.float16),
                    "lora_up.weight": torch.ones((linear_out_features, lora_rank), device="cpu", dtype=torch.float16),
                },
            ),
            "linear_layer_2": LoRALayer.from_state_dict_values(
                values={
                    "lora_down.weight": torch.ones((lora_rank, linear_out_features), device="cpu", dtype=torch.float16),
                    "lora_up.weight": torch.ones((linear_out_features, lora_rank), device="cpu", dtype=torch.float16),
                },
            ),
        }
        lora = ModelPatchRaw(lora_layers)
        lora_models.append((lora, lora_weight))

    # Run inference before patching the model.
    input = torch.randn(1, linear_in_features, device="cuda", dtype=dtype)
    output_before_patch = cached_model.model(input)

    # Patch the model and run inference during the patch.
    with LayerPatcher.apply_smart_model_patches(model=cached_model.model, patches=lora_models, prefix="", dtype=dtype):
        # Check that the second layer has sidecar patches, but the first layer does not.
        assert cached_model.model.linear_layer_1.get_num_patches() == 0
        assert cached_model.model.linear_layer_2.get_num_patches() == num_loras

        output_during_patch = cached_model.model(input)

    # Run inference after unpatching.
    output_after_patch = cached_model.model(input)

    # Check that the output before patching is different from the output during patching.
    assert not torch.allclose(output_before_patch, output_during_patch)

    # Check that the output before patching is the same as the output after patching.
    assert torch.allclose(output_before_patch, output_after_patch)


@torch.no_grad()
@pytest.mark.parametrize(["num_loras"], [(1,), (2,)])
def test_all_patching_methods_produce_same_output(num_loras: int):
    """Test that apply_lora_wrapper_patches(...) produces the same model outputs as apply_lora_patches(...)."""
    dtype = torch.float32
    linear_in_features = 4
    linear_out_features = 8
    lora_rank = 2
    model = DummyModuleWithOneLayer(linear_in_features, linear_out_features, device="cpu", dtype=dtype)
    apply_custom_layers_to_model(model)

    # Initialize num_loras LoRA models with weights of 0.5.
    lora_weight = 0.5
    lora_models: list[tuple[ModelPatchRaw, float]] = []
    for _ in range(num_loras):
        lora_layers = {
            "linear_layer_1": LoRALayer.from_state_dict_values(
                values={
                    "lora_down.weight": torch.ones((lora_rank, linear_in_features), device="cpu", dtype=torch.float16),
                    "lora_up.weight": torch.ones((linear_out_features, lora_rank), device="cpu", dtype=torch.float16),
                },
            )
        }
        lora = ModelPatchRaw(lora_layers)
        lora_models.append((lora, lora_weight))

    input = torch.randn(1, linear_in_features, device="cpu", dtype=dtype)

    with LayerPatcher.apply_smart_model_patches(
        model=model, patches=lora_models, prefix="", dtype=dtype, force_direct_patching=True
    ):
        output_force_direct = model(input)

    with LayerPatcher.apply_smart_model_patches(
        model=model, patches=lora_models, prefix="", dtype=dtype, force_sidecar_patching=True
    ):
        output_force_sidecar = model(input)

    with LayerPatcher.apply_smart_model_patches(model=model, patches=lora_models, prefix="", dtype=dtype):
        output_smart = model(input)

    # Note: We set atol=1e-5 because the test failed occasionally with the default atol=1e-8. Slight numerical
    # differences are tolerable and expected due to the difference between sidecar vs. patching.
    assert torch.allclose(output_force_direct, output_force_sidecar, atol=1e-5)
    assert torch.allclose(output_force_direct, output_smart, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")
@torch.no_grad()
def test_apply_smart_model_patches_change_device():
    """Test that if LoRA patching is applied on the CPU, and then the patched model is moved to the GPU, unpatching
    still behaves correctly.
    """
    linear_in_features = 4
    linear_out_features = 8
    lora_dim = 2
    # Initialize the model on the CPU.
    model = DummyModuleWithOneLayer(linear_in_features, linear_out_features, device="cpu", dtype=torch.float16)
    apply_custom_layers_to_model(model)

    lora_layers = {
        "linear_layer_1": LoRALayer.from_state_dict_values(
            values={
                "lora_down.weight": torch.ones((lora_dim, linear_in_features), device="cpu", dtype=torch.float16),
                "lora_up.weight": torch.ones((linear_out_features, lora_dim), device="cpu", dtype=torch.float16),
            },
        )
    }
    lora = ModelPatchRaw(lora_layers)

    orig_linear_weight = model.linear_layer_1.weight.data.detach().clone()

    with LayerPatcher.apply_smart_model_patches(
        model=model, patches=[(lora, 0.5)], prefix="", dtype=torch.float16, force_direct_patching=True
    ):
        # After patching, all LoRA layer weights should have been moved back to the cpu.
        assert lora_layers["linear_layer_1"].up.device.type == "cpu"
        assert lora_layers["linear_layer_1"].down.device.type == "cpu"

        # After patching, the patched model should still be on the CPU.
        assert model.linear_layer_1.weight.data.device.type == "cpu"

        # There should be no sidecar patches in the model.
        assert model.linear_layer_1.get_num_patches() == 0

        # Move the model to the GPU.
        assert model.to("cuda")

    # After unpatching, the original model weights should have been restored on the GPU.
    assert model.linear_layer_1.weight.data.device.type == "cuda"
    torch.testing.assert_close(model.linear_layer_1.weight.data, orig_linear_weight, check_device=False)


def test_apply_smart_model_patches_force_sidecar_and_direct_patching():
    """Test that ModelPatcher.apply_smart_model_patches(..., force_direct_patching=True, force_sidecar_patching=True)
    raises an error.
    """
    linear_in_features = 4
    linear_out_features = 8
    lora_rank = 2
    model = DummyModuleWithOneLayer(linear_in_features, linear_out_features, device="cpu", dtype=torch.float16)
    apply_custom_layers_to_model(model)

    lora_layers = {
        "linear_layer_1": LoRALayer.from_state_dict_values(
            values={
                "lora_down.weight": torch.ones((lora_rank, linear_in_features), device="cpu", dtype=torch.float16),
                "lora_up.weight": torch.ones((linear_out_features, lora_rank), device="cpu", dtype=torch.float16),
            },
        )
    }
    lora = ModelPatchRaw(lora_layers)
    with pytest.raises(ValueError, match="Cannot force both direct and sidecar patching."):
        with LayerPatcher.apply_smart_model_patches(
            model=model,
            patches=[(lora, 0.5)],
            prefix="",
            dtype=torch.float16,
            force_direct_patching=True,
            force_sidecar_patching=True,
        ):
            pass
