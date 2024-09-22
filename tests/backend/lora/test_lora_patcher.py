import pytest
import torch

from invokeai.backend.lora.layers.lora_layer import LoRALayer
from invokeai.backend.lora.lora_model_raw import LoRAModelRaw
from invokeai.backend.lora.lora_patcher import LoRAPatcher


class DummyModule(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: str, dtype: torch.dtype):
        super().__init__()
        self.linear_layer_1 = torch.nn.Linear(in_features, out_features, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer_1(x)


@pytest.mark.parametrize(
    ["device", "num_layers"],
    [
        ("cpu", 1),
        pytest.param("cuda", 1, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")),
        ("cpu", 2),
        pytest.param("cuda", 2, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")),
    ],
)
@torch.no_grad()
def test_apply_lora_patches(device: str, num_layers: int):
    """Test the basic behavior of ModelPatcher.apply_lora_patches(...). Check that patching and unpatching produce the
    correct result, and that model/LoRA tensors are moved between devices as expected.
    """

    linear_in_features = 4
    linear_out_features = 8
    lora_rank = 2
    model = DummyModule(linear_in_features, linear_out_features, device=device, dtype=torch.float16)

    # Initialize num_layers LoRA models with weights of 0.5.
    lora_weight = 0.5
    lora_models: list[tuple[LoRAModelRaw, float]] = []
    for _ in range(num_layers):
        lora_layers = {
            "linear_layer_1": LoRALayer.from_state_dict_values(
                values={
                    "lora_down.weight": torch.ones((lora_rank, linear_in_features), device="cpu", dtype=torch.float16),
                    "lora_up.weight": torch.ones((linear_out_features, lora_rank), device="cpu", dtype=torch.float16),
                },
            )
        }
        lora = LoRAModelRaw(lora_layers)
        lora_models.append((lora, lora_weight))

    orig_linear_weight = model.linear_layer_1.weight.data.detach().clone()
    expected_patched_linear_weight = orig_linear_weight + (lora_rank * lora_weight * num_layers)

    with LoRAPatcher.apply_lora_patches(model=model, patches=lora_models, prefix=""):
        # After patching, all LoRA layer weights should have been moved back to the cpu.
        for lora, _ in lora_models:
            assert lora.layers["linear_layer_1"].up.device.type == "cpu"
            assert lora.layers["linear_layer_1"].down.device.type == "cpu"

        # After patching, the patched model should still be on its original device.
        assert model.linear_layer_1.weight.data.device.type == device

        torch.testing.assert_close(model.linear_layer_1.weight.data, expected_patched_linear_weight)

    # After unpatching, the original model weights should have been restored on the original device.
    assert model.linear_layer_1.weight.data.device.type == device
    torch.testing.assert_close(model.linear_layer_1.weight.data, orig_linear_weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")
@torch.no_grad()
def test_apply_lora_patches_change_device():
    """Test that if LoRA patching is applied on the CPU, and then the patched model is moved to the GPU, unpatching
    still behaves correctly.
    """
    linear_in_features = 4
    linear_out_features = 8
    lora_dim = 2
    # Initialize the model on the CPU.
    model = DummyModule(linear_in_features, linear_out_features, device="cpu", dtype=torch.float16)

    lora_layers = {
        "linear_layer_1": LoRALayer.from_state_dict_values(
            values={
                "lora_down.weight": torch.ones((lora_dim, linear_in_features), device="cpu", dtype=torch.float16),
                "lora_up.weight": torch.ones((linear_out_features, lora_dim), device="cpu", dtype=torch.float16),
            },
        )
    }
    lora = LoRAModelRaw(lora_layers)

    orig_linear_weight = model.linear_layer_1.weight.data.detach().clone()

    with LoRAPatcher.apply_lora_patches(model=model, patches=[(lora, 0.5)], prefix=""):
        # After patching, all LoRA layer weights should have been moved back to the cpu.
        assert lora_layers["linear_layer_1"].up.device.type == "cpu"
        assert lora_layers["linear_layer_1"].down.device.type == "cpu"

        # After patching, the patched model should still be on the CPU.
        assert model.linear_layer_1.weight.data.device.type == "cpu"

        # Move the model to the GPU.
        assert model.to("cuda")

    # After unpatching, the original model weights should have been restored on the GPU.
    assert model.linear_layer_1.weight.data.device.type == "cuda"
    torch.testing.assert_close(model.linear_layer_1.weight.data, orig_linear_weight, check_device=False)


@pytest.mark.parametrize(
    ["device", "num_layers"],
    [
        ("cpu", 1),
        pytest.param("cuda", 1, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")),
        ("cpu", 2),
        pytest.param("cuda", 2, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")),
    ],
)
def test_apply_lora_sidecar_patches(device: str, num_layers: int):
    """Test the basic behavior of ModelPatcher.apply_lora_sidecar_patches(...). Check that unpatching works correctly."""
    dtype = torch.float16
    linear_in_features = 4
    linear_out_features = 8
    lora_rank = 2
    model = DummyModule(linear_in_features, linear_out_features, device=device, dtype=dtype)

    # Initialize num_layers LoRA models with weights of 0.5.
    lora_weight = 0.5
    lora_models: list[tuple[LoRAModelRaw, float]] = []
    for _ in range(num_layers):
        lora_layers = {
            "linear_layer_1": LoRALayer.from_state_dict_values(
                values={
                    "lora_down.weight": torch.ones((lora_rank, linear_in_features), device="cpu", dtype=torch.float16),
                    "lora_up.weight": torch.ones((linear_out_features, lora_rank), device="cpu", dtype=torch.float16),
                },
            )
        }
        lora = LoRAModelRaw(lora_layers)
        lora_models.append((lora, lora_weight))

    # Run inference before patching the model.
    input = torch.randn(1, linear_in_features, device=device, dtype=dtype)
    output_before_patch = model(input)

    # Patch the model and run inference during the patch.
    with LoRAPatcher.apply_lora_sidecar_patches(model=model, patches=lora_models, prefix="", dtype=dtype):
        output_during_patch = model(input)

    # Run inference after unpatching.
    output_after_patch = model(input)

    # Check that the output before patching is different from the output during patching.
    assert not torch.allclose(output_before_patch, output_during_patch)

    # Check that the output before patching is the same as the output after patching.
    assert torch.allclose(output_before_patch, output_after_patch)


@torch.no_grad()
@pytest.mark.parametrize(["num_layers"], [(1,), (2,)])
def test_apply_lora_sidecar_patches_matches_apply_lora_patches(num_layers: int):
    """Test that apply_lora_sidecar_patches(...) produces the same model outputs as apply_lora_patches(...)."""
    dtype = torch.float32
    linear_in_features = 4
    linear_out_features = 8
    lora_rank = 2
    model = DummyModule(linear_in_features, linear_out_features, device="cpu", dtype=dtype)

    # Initialize num_layers LoRA models with weights of 0.5.
    lora_weight = 0.5
    lora_models: list[tuple[LoRAModelRaw, float]] = []
    for _ in range(num_layers):
        lora_layers = {
            "linear_layer_1": LoRALayer.from_state_dict_values(
                values={
                    "lora_down.weight": torch.ones((lora_rank, linear_in_features), device="cpu", dtype=torch.float16),
                    "lora_up.weight": torch.ones((linear_out_features, lora_rank), device="cpu", dtype=torch.float16),
                },
            )
        }
        lora = LoRAModelRaw(lora_layers)
        lora_models.append((lora, lora_weight))

    input = torch.randn(1, linear_in_features, device="cpu", dtype=dtype)

    with LoRAPatcher.apply_lora_patches(model=model, patches=lora_models, prefix=""):
        output_lora_patches = model(input)

    with LoRAPatcher.apply_lora_sidecar_patches(model=model, patches=lora_models, prefix="", dtype=dtype):
        output_lora_sidecar_patches = model(input)

    # Note: We set atol=1e-5 because the test failed occasionally with the default atol=1e-8. Slight numerical
    # differences are tolerable and expected due to the difference between sidecar vs. patching.
    assert torch.allclose(output_lora_patches, output_lora_sidecar_patches, atol=1e-5)
