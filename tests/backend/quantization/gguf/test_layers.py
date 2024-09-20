import torch
import pytest
import torch.nn as nn

from invokeai.backend.quantization.gguf.torch_patcher import TorchPatcher
from invokeai.backend.quantization.gguf.layers import GGUFLayer

quantized_sd = {
    "linear.weight": torch.load("tests/assets/gguf_qweight.pt"),
    "linear.bias": torch.load("tests/assets/gguf_qbias.pt"),
}

class TestGGUFPatcher(TorchPatcher):
    class Linear(GGUFLayer, nn.Linear):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            weight, bias = self.cast_bias_weight(input)
            return nn.functional.linear(input, weight, bias)

class Test2GGUFPatcher(TorchPatcher):
    class Linear(GGUFLayer, nn.Linear):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            weight, bias = self.cast_bias_weight(input)
            return nn.functional.linear(input, weight, bias)

# Define a dummy module for testing
class DummyModule(nn.Module):
    def __init__(self, device: str='cpu', dtype: torch.dtype=torch.float32):
        super().__init__()
        self.linear = nn.Linear(3072, 18432, device=device, dtype=dtype)

    def forward(self, x):
        x = self.linear(x)
        return x

# Test that TorchPatcher patches and unpatches nn.Linear correctly
def test_torch_patcher_patches_nn_linear():
    original_linear = nn.Linear

    with TorchPatcher.wrap():
        # nn.Linear should not be replaced
        assert nn.Linear is original_linear
    assert nn.Linear is original_linear

# Test that GGUFPatcher patches and unpatches nn.Linear correctly
def test_gguf_patcher_patches_nn_linear():
    original_linear = nn.Linear

    with TestGGUFPatcher.wrap():
        # nn.Linear should be replaced
        assert nn.Linear is not original_linear
        # Create a linear layer and check its type
        linear_layer = nn.Linear(3072, 18432)
        assert isinstance(linear_layer, TestGGUFPatcher.Linear)
    # nn.Linear should be restored
    assert nn.Linear is original_linear

# Test that unpatching restores the original behavior
def test_gguf_patcher_unpatch_restores_behavior():
    device = 'cpu'
    dtype = torch.float32

    input_tensor = torch.randn(1, 3072, device=device, dtype=dtype)
    model = DummyModule(device=device, dtype=dtype)
    with pytest.raises(Exception):
        model.load_state_dict(quantized_sd)

    with TestGGUFPatcher.wrap():
        patched_model = DummyModule(device=device, dtype=dtype)
        patched_model.load_state_dict(quantized_sd)
        # Will raise if patch is not applied
        patched_model(input_tensor)

    # Ensure nn.Linear is restored
    assert nn.Linear is not TestGGUFPatcher.Linear
    assert isinstance(nn.Linear(4, 8), nn.Linear)

# Test that the patched Linear layer behaves as expected
def test_gguf_patcher_linear_layer_behavior():
    device = 'cpu'
    dtype = torch.float32

    input_tensor = torch.randn(1, 3072, device=device, dtype=dtype)
    model = DummyModule(device=device, dtype=dtype)
    with pytest.raises(Exception):
        model.load_state_dict(quantized_sd)

    with TestGGUFPatcher.wrap():
        patched_model = DummyModule(device=device, dtype=dtype)
        patched_model.load_state_dict(quantized_sd)

        patched_tensor = patched_model(input_tensor)

    # After unpatching, run forward and ensure patched classes are still applied
    assert torch.equal(patched_tensor, patched_model(input_tensor))


# Test that the TorchPatcher works correctly when nesting contexts
def test_torch_patcher_nested_contexts():
    original_linear = nn.Linear

    with TestGGUFPatcher.wrap():
        # First level patching
        first_level_linear = nn.Linear
        assert first_level_linear is not original_linear

        with Test2GGUFPatcher.wrap():
            # Second level patching
            second_level_linear = nn.Linear
            assert second_level_linear is not first_level_linear

        # After exiting inner context, nn.Linear should be restored to first level patch
        assert nn.Linear is first_level_linear

    # After exiting outer context, nn.Linear should be restored to original
    assert nn.Linear is original_linear

