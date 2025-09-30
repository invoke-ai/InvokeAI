import torch

from invokeai.backend.flux.modules.layers import MLPEmbedder
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.cast_to_device import cast_to_device
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_module_mixin import (
    CustomModuleMixin,
)


class CustomFluxMLPEmbedder(MLPEmbedder, CustomModuleMixin):
    def _autocast_forward_with_patches(self, x: torch.Tensor) -> torch.Tensor:
        # Example patch logic: apply LoRA weights to in_layer and out_layer
        for patch, patch_weight in self._patches_and_weights:
            if hasattr(patch, "lora_up"):
                if hasattr(self.in_layer, "weight"):
                    self.in_layer.weight.data += patch.lora_up.weight.data * patch_weight
                if hasattr(self.out_layer, "weight"):
                    self.out_layer.weight.data += patch.lora_up.weight.data * patch_weight
        # Move weights to input device
        device = x.device
        if hasattr(self.in_layer, "weight"):
            self.in_layer.weight.data = cast_to_device(self.in_layer.weight, device)
        if hasattr(self.out_layer, "weight"):
            self.out_layer.weight.data = cast_to_device(self.out_layer.weight, device)
        return super().forward(x)

    def _autocast_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move weights to input device
        device = x.device
        if hasattr(self.in_layer, "weight"):
            self.in_layer.weight.data = cast_to_device(self.in_layer.weight, device)
        if hasattr(self.out_layer, "weight"):
            self.out_layer.weight.data = cast_to_device(self.out_layer.weight, device)
        return super().forward(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.get_num_patches() > 0:
            return self._autocast_forward_with_patches(x)
        elif self._device_autocasting_enabled:
            return self._autocast_forward(x)
        else:
            return super().forward(x)
