from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple

import torch
from diffusers import UNet2DConditionModel

from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase
from invokeai.backend.util.devices import TorchDevice

if TYPE_CHECKING:
    from invokeai.app.invocations.model import LoRAField
    from invokeai.app.services.shared.invocation_context import InvocationContext
    from invokeai.backend.lora import LoRAModelRaw  # TODO: circular import


class LoRAPatcherExt(ExtensionBase):
    def __init__(
        self,
        node_context: InvocationContext,
        loras: List[LoRAField],
        prefix: str,
        priority: int,
    ):
        super().__init__(priority=priority)
        self.loras = loras
        self.prefix = prefix
        self.node_context = node_context

    @contextmanager
    def patch_unet(self, model_state_dict: Dict[str, torch.Tensor], model: UNet2DConditionModel):
        def _lora_loader() -> Iterator[Tuple[LoRAModelRaw, float]]:
            for lora in self.loras:
                lora_info = self.node_context.models.load(lora.lora)
                lora_model = lora_info.model
                from invokeai.backend.lora import LoRAModelRaw

                assert isinstance(lora_model, LoRAModelRaw)
                yield (lora_model, lora.weight)
                del lora_info
            return

        yield self._patch_model(
            model=model,
            prefix=self.prefix,
            loras=_lora_loader(),
            model_state_dict=model_state_dict,
        )

    @classmethod
    @contextmanager
    def static_patch_model(
        cls,
        model: torch.nn.Module,
        prefix: str,
        loras: Iterator[Tuple[LoRAModelRaw, float]],
        model_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        changed_keys = None
        changed_unknown_keys = None
        try:
            changed_keys, changed_unknown_keys = cls._patch_model(
                model=model,
                prefix=prefix,
                loras=loras,
                model_state_dict=model_state_dict,
            )

            yield

        finally:
            assert hasattr(model, "get_submodule")  # mypy not picking up fact that torch.nn.Module has get_submodule()
            with torch.no_grad():
                if changed_keys:
                    for module_key in changed_keys:
                        weight = model_state_dict[module_key]
                        model.get_submodule(module_key).weight.copy_(
                            weight, non_blocking=TorchDevice.get_non_blocking(weight.device)
                        )
                if changed_unknown_keys:
                    for module_key, weight in changed_unknown_keys.items():
                        model.get_submodule(module_key).weight.copy_(
                            weight, non_blocking=TorchDevice.get_non_blocking(weight.device)
                        )

    @classmethod
    def _patch_model(
        cls,
        model: UNet2DConditionModel,
        prefix: str,
        loras: Iterator[Tuple[LoRAModelRaw, float]],
        model_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Apply one or more LoRAs to a model.

        :param model: The model to patch.
        :param loras: An iterator that returns the LoRA to patch in and its patch weight.
        :param prefix: A string prefix that precedes keys used in the LoRAs weight layers.
        :model_state_dict: Read-only copy of the model's state dict in CPU, for unpatching purposes.
        """
        if model_state_dict is None:
            model_state_dict = {}

        changed_keys = set()
        changed_unknown_keys = {}
        with torch.no_grad():
            for lora, lora_weight in loras:
                # assert lora.device.type == "cpu"
                for layer_key, layer in lora.layers.items():
                    if not layer_key.startswith(prefix):
                        continue

                    # TODO(ryand): A non-negligible amount of time is currently spent resolving LoRA keys. This
                    # should be improved in the following ways:
                    # 1. The key mapping could be more-efficiently pre-computed. This would save time every time a
                    #    LoRA model is applied.
                    # 2. From an API perspective, there's no reason that the `ModelPatcher` should be aware of the
                    #    intricacies of Stable Diffusion key resolution. It should just expect the input LoRA
                    #    weights to have valid keys.
                    assert isinstance(model, torch.nn.Module)
                    module_key, module = cls._resolve_lora_key(model, layer_key, prefix)

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

        return changed_keys, changed_unknown_keys

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
