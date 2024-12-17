import torch

from invokeai.backend.patches.layers.base_layer_patch import BaseLayerPatch


class BaseSidecarWrapper(torch.nn.Module):
    """A base class for sidecar wrappers.

    A sidecar wrapper is a wrapper for an existing torch.nn.Module that applies a
    list of patches as 'sidecar' patches. I.e. it applies the sidecar patches during forward inference without modifying
    the original module.

    Sidecar wrappers are typically used over regular patches when:
    - The original module is quantized and so the weights can't be patched in the usual way.
    - The original module is on the CPU and modifying the weights would require backing up the original weights and
      doubling the CPU memory usage.
    """

    def __init__(
        self, orig_module: torch.nn.Module, patches_and_weights: list[tuple[BaseLayerPatch, float]] | None = None
    ):
        super().__init__()
        self._orig_module = orig_module
        self._patches_and_weights = [] if patches_and_weights is None else patches_and_weights

    @property
    def orig_module(self) -> torch.nn.Module:
        return self._orig_module

    def add_patch(self, patch: BaseLayerPatch, patch_weight: float):
        """Add a patch to the sidecar wrapper."""
        self._patches_and_weights.append((patch, patch_weight))

    def _aggregate_patch_parameters(
        self, patches_and_weights: list[tuple[BaseLayerPatch, float]]
    ) -> dict[str, torch.Tensor]:
        """Helper function that aggregates the parameters from all patches into a single dict."""
        params: dict[str, torch.Tensor] = {}

        for patch, patch_weight in patches_and_weights:
            # TODO(ryand): self._orig_module could be quantized. Depending on what the patch is doing with the original
            # module, this might fail or return incorrect results.
            layer_params = patch.get_parameters(self._orig_module, weight=patch_weight)

            for param_name, param_weight in layer_params.items():
                if param_name not in params:
                    params[param_name] = param_weight
                else:
                    params[param_name] += param_weight

        return params

    def forward(self, *args, **kwargs):  # type: ignore
        raise NotImplementedError()
