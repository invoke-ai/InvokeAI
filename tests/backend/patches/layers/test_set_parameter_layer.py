import pytest
import torch

from invokeai.backend.patches.layers.set_parameter_layer import SetParameterLayer


def test_set_parameter_layer_get_parameters():
    orig_module = torch.nn.Linear(4, 8)

    target_weight = torch.randn(8, 4)
    layer = SetParameterLayer(param_name="weight", weight=target_weight)

    params = layer.get_parameters(dict(orig_module.named_parameters(recurse=False)), weight=1.0)
    assert len(params) == 1
    new_weight = orig_module.weight + params["weight"]
    assert torch.allclose(new_weight, target_weight)


@pytest.mark.parametrize(
    ["device"],
    [
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA device")),
        pytest.param(
            "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS device")
        ),
    ],
)
def test_set_parameter_layer_to(device: str):
    """Test moving SetParameterLayer to different device/dtype."""

    target_weight = torch.randn(8, 4)
    layer = SetParameterLayer(param_name="weight", weight=target_weight)

    # SetParameterLayer should be initialized on CPU.
    assert layer.weight.device.type == "cpu"  # type: ignore

    # Move to device.
    layer.to(device=torch.device(device))
    assert layer.weight.device.type == device  # type: ignore


def test_set_parameter_layer_calc_size():
    """Test calculating parameter size of SetParameterLayer"""
    param = torch.randn(4, 8)
    layer = SetParameterLayer(param_name="weight", weight=param)

    # Size should be number of elements * bytes per element
    expected_size = param.nelement() * param.element_size()
    assert layer.calc_size() == expected_size
