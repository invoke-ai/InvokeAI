from contextlib import contextmanager
from typing import Dict, Iterator, Optional, Tuple

import torch

from invokeai.backend.lora.lora_model_raw import LoRAModelRaw
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.original_weights_storage import OriginalWeightsStorage


class PeftPatcher:
    @classmethod
    @torch.no_grad()
    @contextmanager
    def apply_peft_patches(
        cls,
        model: torch.nn.Module,
        patches: Iterator[Tuple[LoRAModelRaw, float]],
        prefix: str,
        cached_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Apply one or more PEFT patches to a model.

        :param model: The model to patch.
        :param loras: An iterator that returns tuples of PEFT patches and associated weights. An iterator is used so
            that the PEFT patches do not need to be loaded into memory all at once.
        :param prefix: The keys in the patches will be filtered to only include weights with this prefix.
        :cached_weights: Read-only copy of the model's state dict in CPU, for efficient unpatching purposes.
        """
        original_weights = OriginalWeightsStorage(cached_weights)
        try:
            for patch, patch_weight in patches:
                cls._apply_peft_patch(
                    model=model,
                    prefix=prefix,
                    patch=patch,
                    patch_weight=patch_weight,
                    original_weights=original_weights,
                )

            yield
        finally:
            for param_key, weight in original_weights.get_changed_weights():
                model.get_parameter(param_key).copy_(weight)

    @classmethod
    @torch.no_grad()
    def _apply_peft_patch(
        cls,
        model: torch.nn.Module,
        prefix: str,
        patch: LoRAModelRaw,
        patch_weight: float,
        original_weights: OriginalWeightsStorage,
    ):
        """
        Apply one a LoRA to a model.
        :param model: The model to patch.
        :param patch: LoRA model to patch in.
        :param patch_weight: LoRA patch weight.
        :param prefix: A string prefix that precedes keys used in the LoRAs weight layers.
        :param original_weights: Storage with original weights, filled by weights which lora patches, used for unpatching.
        """

        if patch_weight == 0:
            return

        for layer_key, layer in patch.layers.items():
            if not layer_key.startswith(prefix):
                continue

            module = model.get_submodule(layer_key)

            # All of the LoRA weight calculations will be done on the same device as the module weight.
            # (Performance will be best if this is a CUDA device.)
            device = module.weight.device
            dtype = module.weight.dtype

            layer_scale = layer.alpha / layer.rank if (layer.alpha and layer.rank) else 1.0

            # We intentionally move to the target device first, then cast. Experimentally, this was found to
            # be significantly faster for 16-bit CPU tensors being moved to a CUDA device than doing the
            # same thing in a single call to '.to(...)'.
            layer.to(device=device)
            layer.to(dtype=torch.float32)

            # TODO(ryand): Using torch.autocast(...) over explicit casting may offer a speed benefit on CUDA
            # devices here. Experimentally, it was found to be very slow on CPU. More investigation needed.
            for param_name, lora_param_weight in layer.get_parameters(module).items():
                param_key = layer_key + "." + param_name
                module_param = module.get_parameter(param_name)

                # Save original weight
                original_weights.save(param_key, module_param)

                if module_param.shape != lora_param_weight.shape:
                    lora_param_weight = lora_param_weight.reshape(module_param.shape)

                lora_param_weight *= patch_weight * layer_scale
                module_param += lora_param_weight.to(dtype=dtype)

            layer.to(device=TorchDevice.CPU_DEVICE)
