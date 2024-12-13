import torch

from invokeai.backend.patches.sidecar_wrappers.base_sidecar_wrapper import BaseSidecarWrapper


class LinearSidecarWrapper(BaseSidecarWrapper):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        aggregated_param_residuals = self._aggregate_patch_parameters(self._patches_and_weights)
        added_residual = torch.nn.functional.linear(
            input, aggregated_param_residuals["weight"], aggregated_param_residuals.get("bias", None)
        )

        return self.orig_module(input) + added_residual
