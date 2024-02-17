# test that if the model's device changes while the lora is applied, the weights can still be restored

# test that LoRA patching works on both CPU and CUDA

import pytest
import torch

from invokeai.backend.model_patcher import ModelPatcher
from invokeai.backend.lora import LoRALayer, LoRAModelRaw


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")),
    ],
)
@torch.no_grad()
def test_apply_lora(device):
    """Test the basic behavior of ModelPatcher.apply_lora(...). Check that patching and unpatching produce the correct
    result, and that model/LoRA tensors are moved between devices as expected.
    """

    linear_in_features = 4
    linear_out_features = 8
    lora_dim = 2
    model = torch.nn.ModuleDict(
        {"linear_layer_1": torch.nn.Linear(linear_in_features, linear_out_features, device=device, dtype=torch.float16)}
    )

    lora_layers = {
        "linear_layer_1": LoRALayer(
            layer_key="linear_layer_1",
            values={
                "lora_down.weight": torch.ones((lora_dim, linear_in_features), device="cpu", dtype=torch.float16),
                "lora_up.weight": torch.ones((linear_out_features, lora_dim), device="cpu", dtype=torch.float16),
            },
        )
    }
    lora = LoRAModelRaw("lora_name", lora_layers)

    lora_weight = 0.5
    orig_linear_weight = model["linear_layer_1"].weight.data.detach().clone()
    expected_patched_linear_weight = orig_linear_weight + (lora_dim * lora_weight)

    with ModelPatcher.apply_lora(model, [(lora, lora_weight)], prefix=""):
        # After patching, all LoRA layer weights should have been moved back to the cpu.
        assert lora_layers["linear_layer_1"].up.device.type == "cpu"
        assert lora_layers["linear_layer_1"].down.device.type == "cpu"

        # After patching, the patched model should still be on its original device.
        assert model["linear_layer_1"].weight.data.device.type == device

        torch.testing.assert_close(model["linear_layer_1"].weight.data, expected_patched_linear_weight)

    # After unpatching, the original model weights should have been restored on the original device.
    assert model["linear_layer_1"].weight.data.device.type == device
    torch.testing.assert_close(model["linear_layer_1"].weight.data, orig_linear_weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")
@torch.no_grad()
def test_apply_lora_change_device():
    """Test that if LoRA patching is applied on the CPU, and then the patched model is moved to the GPU, unpatching
    still behaves correctly.
    """
    linear_in_features = 4
    linear_out_features = 8
    lora_dim = 2
    # Initialize the model on the CPU.
    model = torch.nn.ModuleDict(
        {"linear_layer_1": torch.nn.Linear(linear_in_features, linear_out_features, device="cpu", dtype=torch.float16)}
    )

    lora_layers = {
        "linear_layer_1": LoRALayer(
            layer_key="linear_layer_1",
            values={
                "lora_down.weight": torch.ones((lora_dim, linear_in_features), device="cpu", dtype=torch.float16),
                "lora_up.weight": torch.ones((linear_out_features, lora_dim), device="cpu", dtype=torch.float16),
            },
        )
    }
    lora = LoRAModelRaw("lora_name", lora_layers)

    orig_linear_weight = model["linear_layer_1"].weight.data.detach().clone()

    with ModelPatcher.apply_lora(model, [(lora, 0.5)], prefix=""):
        # After patching, all LoRA layer weights should have been moved back to the cpu.
        assert lora_layers["linear_layer_1"].up.device.type == "cpu"
        assert lora_layers["linear_layer_1"].down.device.type == "cpu"

        # After patching, the patched model should still be on the CPU.
        assert model["linear_layer_1"].weight.data.device.type == "cpu"

        # Move the model to the GPU.
        assert model.to("cuda")

    # After unpatching, the original model weights should have been restored on the GPU.
    assert model["linear_layer_1"].weight.data.device.type == "cuda"
    torch.testing.assert_close(model["linear_layer_1"].weight.data, orig_linear_weight, check_device=False)
