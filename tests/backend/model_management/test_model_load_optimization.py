import pytest
import torch

from invokeai.backend.model_management.model_load_optimizations import _no_op, skip_torch_weight_init


@pytest.mark.parametrize(
    ["torch_module", "layer_args"],
    [
        (torch.nn.Linear, {"in_features": 10, "out_features": 20}),
        (torch.nn.Conv1d, {"in_channels": 10, "out_channels": 20, "kernel_size": 3}),
        (torch.nn.Conv2d, {"in_channels": 10, "out_channels": 20, "kernel_size": 3}),
        (torch.nn.Conv3d, {"in_channels": 10, "out_channels": 20, "kernel_size": 3}),
    ],
)
def test_skip_torch_weight_init_linear(torch_module, layer_args):
    """Test the interactions between `skip_torch_weight_init()` and various torch modules."""
    seed = 123

    # Initialize a torch layer *before* applying `skip_torch_weight_init()`.
    reset_params_fn_before = torch_module.reset_parameters
    torch.manual_seed(seed)
    layer_before = torch_module(**layer_args)

    # Initialize a torch layer while `skip_torch_weight_init()` is applied.
    with skip_torch_weight_init():
        reset_params_fn_during = torch_module.reset_parameters
        torch.manual_seed(123)
        layer_during = torch_module(**layer_args)

    # Initialize a torch layer *after* applying `skip_torch_weight_init()`.
    reset_params_fn_after = torch_module.reset_parameters
    torch.manual_seed(123)
    layer_after = torch_module(**layer_args)

    # Check that reset_parameters is skipped while `skip_torch_weight_init()` is active.
    assert reset_params_fn_during == _no_op
    assert not torch.allclose(layer_before.weight, layer_during.weight)
    assert not torch.allclose(layer_before.bias, layer_during.bias)

    # Check that the original behavior is restored after `skip_torch_weight_init()` ends.
    assert reset_params_fn_before is reset_params_fn_after
    assert torch.allclose(layer_before.weight, layer_after.weight)
    assert torch.allclose(layer_before.bias, layer_after.bias)
