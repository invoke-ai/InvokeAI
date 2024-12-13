import torch

from invokeai.backend.patches.sidecar_wrappers.base_sidecar_wrapper import BaseSidecarWrapper


class Conv2dSidecarWrapper(BaseSidecarWrapper):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        aggregated_param_residuals = self._aggregate_patch_parameters(self._patches_and_weights)
        return self.orig_module(input) + torch.nn.functional.conv1d(
            input, aggregated_param_residuals["weight"], aggregated_param_residuals.get("bias", None)
        )
