import torch

from invokeai.backend.flux.modules.layers import RMSNorm
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_flux_rms_norm import (
    CustomFluxRMSNorm,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.torch_module_autocast import (
    wrap_custom_layer,
)
from invokeai.backend.patches.layers.set_parameter_layer import SetParameterLayer


def test_custom_flux_rms_norm_patch():
    """Test a SetParameterLayer patch on a CustomFluxRMSNorm layer."""
    # Create a RMSNorm layer.
    dim = 8
    rms_norm = RMSNorm(dim)

    # Create a SetParameterLayer.
    new_scale = torch.randn(dim)
    set_parameter_layer = SetParameterLayer("scale", new_scale)

    # Wrap the RMSNorm layer in a CustomFluxRMSNorm layer.
    custom_flux_rms_norm = wrap_custom_layer(rms_norm, CustomFluxRMSNorm)
    custom_flux_rms_norm.add_patch(set_parameter_layer, 1.0)

    # Run the CustomFluxRMSNorm layer.
    input = torch.randn(1, dim)
    expected_output = torch.nn.functional.rms_norm(input, new_scale.shape, new_scale, eps=1e-6)
    output_custom = custom_flux_rms_norm(input)
    assert torch.allclose(output_custom, expected_output, atol=1e-6)
