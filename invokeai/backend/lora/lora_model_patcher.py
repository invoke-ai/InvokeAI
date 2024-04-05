from contextlib import contextmanager
from typing import Iterator, Tuple

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from transformers import CLIPTextModel

from invokeai.backend.lora.lora_model import LoRAModelRaw
from invokeai.backend.model_manager.any_model_type import AnyModel


class LoraModelPatcher:
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

    @classmethod
    @contextmanager
    def apply_lora_unet(
        cls,
        unet: UNet2DConditionModel,
        loras: Iterator[Tuple[LoRAModelRaw, float]],
    ):
        with cls.apply_lora(unet, loras, "lora_unet_"):
            yield

    @classmethod
    @contextmanager
    def apply_lora_text_encoder(
        cls,
        text_encoder: CLIPTextModel,
        loras: Iterator[Tuple[LoRAModelRaw, float]],
    ):
        with cls.apply_lora(text_encoder, loras, "lora_te_"):
            yield

    @classmethod
    @contextmanager
    def apply_sdxl_lora_text_encoder(
        cls,
        text_encoder: CLIPTextModel,
        loras: Iterator[Tuple[LoRAModelRaw, float]],
    ):
        with cls.apply_lora(text_encoder, loras, "lora_te1_"):
            yield

    @classmethod
    @contextmanager
    def apply_sdxl_lora_text_encoder2(
        cls,
        text_encoder: CLIPTextModel,
        loras: Iterator[Tuple[LoRAModelRaw, float]],
    ):
        with cls.apply_lora(text_encoder, loras, "lora_te2_"):
            yield

    @classmethod
    @contextmanager
    def apply_lora(
        cls,
        model: AnyModel,
        loras: Iterator[Tuple[LoRAModelRaw, float]],
        prefix: str,
    ):
        original_weights = {}
        try:
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
                        # 2. From an API perspective, there's no reason that the `LoraModelPatcher` should be aware of
                        #    the intricacies of Stable Diffusion key resolution. It should just expect the input LoRA
                        #    weights to have valid keys.
                        assert isinstance(model, torch.nn.Module)
                        module_key, module = cls._resolve_lora_key(model, layer_key, prefix)

                        # All of the LoRA weight calculations will be done on the same device as the module weight.
                        # (Performance will be best if this is a CUDA device.)
                        device = module.weight.device
                        dtype = module.weight.dtype

                        if module_key not in original_weights:
                            original_weights[module_key] = module.weight.detach().to(device="cpu", copy=True)

                        layer_scale = layer.alpha / layer.rank if (layer.alpha and layer.rank) else 1.0

                        # We intentionally move to the target device first, then cast. Experimentally, this was found to
                        # be significantly faster for 16-bit CPU tensors being moved to a CUDA device than doing the
                        # same thing in a single call to '.to(...)'.
                        layer.to(device=device)
                        layer.to(dtype=torch.float32)
                        # TODO(ryand): Using torch.autocast(...) over explicit casting may offer a speed benefit on CUDA
                        # devices here. Experimentally, it was found to be very slow on CPU. More investigation needed.
                        layer_weight = layer.get_weight(module.weight) * (lora_weight * layer_scale)
                        layer.to(device=torch.device("cpu"))

                        if module.weight.shape != layer_weight.shape:
                            layer_weight = layer_weight.reshape(module.weight.shape)

                        module.weight += layer_weight.to(dtype=dtype)

            yield  # wait for context manager exit

        finally:
            assert hasattr(model, "get_submodule")  # mypy not picking up fact that torch.nn.Module has get_submodule()
            with torch.no_grad():
                for module_key, weight in original_weights.items():
                    model.get_submodule(module_key).weight.copy_(weight)
