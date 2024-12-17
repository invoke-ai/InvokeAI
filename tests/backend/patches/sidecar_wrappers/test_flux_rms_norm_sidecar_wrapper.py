import torch

from invokeai.backend.patches.layers.set_parameter_layer import SetParameterLayer
from invokeai.backend.patches.sidecar_wrappers.flux_rms_norm_sidecar_wrapper import FluxRMSNormSidecarWrapper


def test_flux_rms_norm_sidecar_wrapper():
    # Create a RMSNorm layer.
    dim = 10
    rms_norm = torch.nn.RMSNorm(dim)

    # Create a SetParameterLayer.
    new_scale = torch.randn(dim)
    set_parameter_layer = SetParameterLayer("scale", new_scale)

    # Create a FluxRMSNormSidecarWrapper.
    rms_norm_wrapped = FluxRMSNormSidecarWrapper(rms_norm, [(set_parameter_layer, 1.0)])

    # Run the FluxRMSNormSidecarWrapper.
    input = torch.randn(1, dim)
    expected_output = torch.nn.functional.rms_norm(input, new_scale.shape, new_scale, eps=1e-6)
    output_wrapped = rms_norm_wrapped(input)
    assert torch.allclose(output_wrapped, expected_output, atol=1e-6)
