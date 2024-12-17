import torch

from invokeai.backend.patches.layers.set_parameter_layer import SetParameterLayer
from invokeai.backend.patches.sidecar_wrappers.base_sidecar_wrapper import BaseSidecarWrapper


class FluxRMSNormSidecarWrapper(BaseSidecarWrapper):
    """A sidecar wrapper for a FLUX RMSNorm layer.

    This wrapper is a special case. It is added specifically to enable FLUX structural control LoRAs, which overwrite
    the RMSNorm scale parameters.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Given the narrow focus of this wrapper, we only support a very particular patch configuration:
        assert len(self._patches_and_weights) == 1
        patch, _patch_weight = self._patches_and_weights[0]
        assert isinstance(patch, SetParameterLayer)
        assert patch.param_name == "scale"

        # Apply the patch.
        # NOTE(ryand): Currently, we ignore the patch weight when running as a sidecar. It's not clear how this should
        # be handled.
        return torch.nn.functional.rms_norm(input, patch.weight.shape, patch.weight, eps=1e-6)
