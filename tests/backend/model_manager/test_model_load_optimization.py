import pytest
import torch

from invokeai.backend.model_manager.load.optimizations import _no_op, skip_torch_weight_init


@pytest.mark.parametrize(
    ["torch_module", "layer_args"],
    [
        (torch.nn.Linear, {"in_features": 10, "out_features": 20}),
        (torch.nn.Conv1d, {"in_channels": 10, "out_channels": 20, "kernel_size": 3}),
        (torch.nn.Conv2d, {"in_channels": 10, "out_channels": 20, "kernel_size": 3}),
        (torch.nn.Conv3d, {"in_channels": 10, "out_channels": 20, "kernel_size": 3}),
        (torch.nn.Embedding, {"num_embeddings": 10, "embedding_dim": 10}),
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
    if hasattr(layer_before, "bias"):
        assert not torch.allclose(layer_before.bias, layer_during.bias)

    # Check that the original behavior is restored after `skip_torch_weight_init()` ends.
    assert reset_params_fn_before is reset_params_fn_after
    assert torch.allclose(layer_before.weight, layer_after.weight)
    if hasattr(layer_before, "bias"):
        assert torch.allclose(layer_before.bias, layer_after.bias)


def test_skip_torch_weight_init_restores_base_class_behavior():
    """Test that `skip_torch_weight_init()` correctly restores the original behavior of torch.nn.Conv*d modules. This
    test was created to catch a previous bug where `reset_parameters` was being copied from the base `_ConvNd` class to
    its child classes (like `Conv1d`).
    """
    with skip_torch_weight_init():
        # There is no need to do anything while the context manager is applied, we're just testing that the original
        # behavior is restored correctly.
        pass

    # Mock the behavior of another library that monkey patches `torch.nn.modules.conv._ConvNd.reset_parameters` and
    # expects it to affect all of the sub-classes (e.g. `torch.nn.Conv1D`, `torch.nn.Conv2D`, etc.).
    called_monkey_patched_fn = False

    def monkey_patched_fn(*args, **kwargs):
        nonlocal called_monkey_patched_fn
        called_monkey_patched_fn = True

    saved_fn = torch.nn.modules.conv._ConvNd.reset_parameters
    torch.nn.modules.conv._ConvNd.reset_parameters = monkey_patched_fn
    _ = torch.nn.Conv1d(10, 20, 3)
    torch.nn.modules.conv._ConvNd.reset_parameters = saved_fn

    assert called_monkey_patched_fn
