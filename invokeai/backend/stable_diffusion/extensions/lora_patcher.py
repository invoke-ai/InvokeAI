from __future__ import annotations

import torch
from contextlib import contextmanager
from typing import List, Tuple, Dict
from diffusers import UNet2DConditionModel
from .base import ExtensionBase
from invokeai.backend.util.devices import TorchDevice
#from invokeai.backend.lora import LoRAModelRaw


class LoRAPatcherExt(ExtensionBase):
    def __init__(
        self,
        node_context: "InvocationContext",
        loras: List["LoRAField"], #ModelIdentifierField
        prefix: str,
        priority: int,
    ):
        super().__init__(priority=priority)
        self.loras = loras
        self.prefix = prefix
        self.node_context = node_context

    @contextmanager
    def patch_unet(self, model_state_dict: Dict[str, torch.Tensor], model: UNet2DConditionModel):
        """
        Apply one or more LoRAs to a model.

        :param model: The model to patch.
        :param loras: An iterator that returns the LoRA to patch in and its patch weight.
        :param prefix: A string prefix that precedes keys used in the LoRAs weight layers.
        :model_state_dict: Read-only copy of the model's state dict in CPU, for unpatching purposes.
        """


        changed_keys = set()
        changed_unknown_keys = dict()
        try:
            with torch.no_grad():
                # for lora, lora_weight in loras:
                for lora_field in self.loras:
                    lora_model_info = self.node_context.models.load(lora_field.lora)
                    from invokeai.backend.lora import LoRAModelRaw # TODO: circular import
                    assert isinstance(lora_model_info.model, LoRAModelRaw)

                    lora = lora_model_info.model
                    lora_weight = lora_field.weight

                
                    # assert lora.device.type == "cpu"
                    for layer_key, layer in lora.layers.items():
                        if not layer_key.startswith(self.prefix):
                            continue

                        # TODO(ryand): A non-negligible amount of time is currently spent resolving LoRA keys. This
                        # should be improved in the following ways:
                        # 1. The key mapping could be more-efficiently pre-computed. This would save time every time a
                        #    LoRA model is applied.
                        # 2. From an API perspective, there's no reason that the `ModelPatcher` should be aware of the
                        #    intricacies of Stable Diffusion key resolution. It should just expect the input LoRA
                        #    weights to have valid keys.
                        assert isinstance(model, torch.nn.Module)
                        module_key, module = self._resolve_lora_key(model, layer_key, self.prefix)

                        # All of the LoRA weight calculations will be done on the same device as the module weight.
                        # (Performance will be best if this is a CUDA device.)
                        device = module.weight.device
                        dtype = module.weight.dtype

                        if module_key not in changed_keys and module_key not in changed_unknown_keys:
                            if module_key is model_state_dict:
                                changed_keys.add(module_key)
                            else:
                                changed_unknown_keys[module_key] = module.weight.detach().to(device="cpu", copy=True)

                        layer_scale = layer.alpha / layer.rank if (layer.alpha and layer.rank) else 1.0

                        # We intentionally move to the target device first, then cast. Experimentally, this was found to
                        # be significantly faster for 16-bit CPU tensors being moved to a CUDA device than doing the
                        # same thing in a single call to '.to(...)'.
                        layer.to(device=device, non_blocking=TorchDevice.get_non_blocking(device))
                        layer.to(dtype=torch.float32, non_blocking=TorchDevice.get_non_blocking(device))
                        # TODO(ryand): Using torch.autocast(...) over explicit casting may offer a speed benefit on CUDA
                        # devices here. Experimentally, it was found to be very slow on CPU. More investigation needed.
                        layer_weight = layer.get_weight(module.weight) * (lora_weight * layer_scale)
                        layer.to(
                            device=TorchDevice.CPU_DEVICE,
                            non_blocking=TorchDevice.get_non_blocking(TorchDevice.CPU_DEVICE),
                        )

                        assert isinstance(layer_weight, torch.Tensor)  # mypy thinks layer_weight is a float|Any ??!
                        if module.weight.shape != layer_weight.shape:
                            # TODO: debug on lycoris
                            assert hasattr(layer_weight, "reshape")
                            layer_weight = layer_weight.reshape(module.weight.shape)

                        assert isinstance(layer_weight, torch.Tensor)  # mypy thinks layer_weight is a float|Any ??!
                        module.weight += layer_weight.to(dtype=dtype, non_blocking=TorchDevice.get_non_blocking(device))

                    del lora_model_info

            yield changed_keys, changed_unknown_keys # wait for context manager exit

        finally:
            # nothing to do as weights restored by extensions manager
            pass

    @staticmethod
    def _resolve_lora_key(model: torch.nn.Module, lora_key: str, prefix: str) -> Tuple[str, torch.nn.Module]:
        assert "." not in lora_key

        if not lora_key.startswith(prefix):
            raise Exception(f"lora_key with invalid prefix: {lora_key}, {prefix}")

        module = model
        module_key = ""
        key_parts = lora_key[len(prefix) :].split("_")

        submodule_name = key_parts.pop(0)

        while len(key_parts) > 0:
            try:
                module = module.get_submodule(submodule_name)
                module_key += "." + submodule_name
                submodule_name = key_parts.pop(0)
            except Exception:
                submodule_name += "_" + key_parts.pop(0)

        module = module.get_submodule(submodule_name)
        module_key = (module_key + "." + submodule_name).lstrip(".")

        return (module_key, module)
