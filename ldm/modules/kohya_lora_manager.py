import re
from pathlib import Path

import torch
from compel import Compel
from diffusers.models import UNet2DConditionModel
from safetensors.torch import load_file
from torch.utils.hooks import RemovableHandle
from transformers import CLIPTextModel

from ldm.invoke.devices import choose_torch_device

"""
This module supports loading LoRA weights trained with https://github.com/kohya-ss/sd-scripts
To be removed once support for diffusers LoRA weights is well supported
"""


class LoRALayer:
    lora_name: str
    name: str
    scale: float
    up: torch.nn.Module
    down: torch.nn.Module

    def __init__(self, lora_name: str, name: str, rank=4, alpha=1.0):
        self.lora_name = lora_name
        self.name = name
        self.scale = alpha / rank if (alpha and rank) else 1.0


class LoRAModuleWrapper:
    unet: UNet2DConditionModel
    text_encoder: CLIPTextModel
    hooks: list[RemovableHandle]

    def __init__(self, unet, text_encoder):
        self.unet = unet
        self.text_encoder = text_encoder
        self.hooks = []
        self.text_modules = None
        self.unet_modules = None

        self.applied_loras = {}
        self.loaded_loras = {}

        self.UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
        self.TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
        self.LORA_PREFIX_UNET = "lora_unet"
        self.LORA_PREFIX_TEXT_ENCODER = "lora_te"


        def find_modules(
            prefix, root_module: torch.nn.Module, target_replace_modules
        ) -> dict[str, torch.nn.Module]:
            mapping = {}
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        layer_type = child_module.__class__.__name__
                        if layer_type == "Linear" or (
                            layer_type == "Conv2d"
                            and child_module.kernel_size == (1, 1)
                        ):
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")
                            mapping[lora_name] = child_module
                            self.apply_module_forward(child_module, lora_name)
            return mapping

        if self.text_modules is None:
            self.text_modules = find_modules(
                self.LORA_PREFIX_TEXT_ENCODER,
                text_encoder,
                self.TEXT_ENCODER_TARGET_REPLACE_MODULE,
            )

        if self.unet_modules is None:
            self.unet_modules = find_modules(
                self.LORA_PREFIX_UNET, unet, self.UNET_TARGET_REPLACE_MODULE
            )


    def lora_forward_hook(self, name):
        wrapper = self

        def lora_forward(module, input_h, output):
            if len(wrapper.loaded_loras) == 0:
                return output

            for lora in wrapper.applied_loras.values():
                layer = lora.layers.get(name, None)
                if layer is None:
                    continue
                output = (
                    output
                    + layer.up(layer.down(*input_h)) * lora.multiplier * layer.scale
                )
            return output

        return lora_forward

    def apply_module_forward(self, module, name):
        handle = module.register_forward_hook(self.lora_forward_hook(name))
        self.hooks.append(handle)

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()

        self.hooks.clear()

    def clear_applied_loras(self):
        self.applied_loras.clear()

    def clear_loaded_loras(self):
        self.loaded_loras.clear()

class LoRA:
    name: str
    layers: dict[str, LoRALayer]
    device: torch.device
    dtype: torch.dtype
    wrapper: LoRAModuleWrapper
    multiplier: float

    def __init__(self, name: str, device, dtype, wrapper, multiplier=1.0):
        self.name = name
        self.layers = {}
        self.multiplier = multiplier
        self.device = device
        self.dtype = dtype
        self.wrapper = wrapper
        self.rank = None
        self.alpha = None

    def load_from_dict(self, state_dict):
        for key, value in state_dict.items():
            stem, leaf = key.split(".", 1)

            if leaf.endswith("alpha"):
                if self.alpha is None:
                    self.alpha = value.item()
                continue

            if stem.startswith(self.wrapper.LORA_PREFIX_TEXT_ENCODER):
                wrapped = self.wrapper.text_modules.get(stem, None)
                if wrapped is None:
                    print(f">> Missing layer: {stem}")
                    continue

                if (
                    self.rank is None
                    and leaf == "lora_down.weight"
                    and len(value.size()) == 2
                ):
                    self.rank = value.shape[0]
                self.load_lora_layer(stem, leaf, value, wrapped)
                continue
            elif stem.startswith(self.wrapper.LORA_PREFIX_UNET):
                wrapped = self.wrapper.unet_modules.get(stem, None)
                if wrapped is None:
                    print(f">> Missing layer: {stem}")
                    continue

                if (
                    self.rank is None
                    and leaf == "lora_down.weight"
                    and len(value.size()) == 2
                ):
                    self.rank = value.shape[0]
                self.load_lora_layer(stem, leaf, value, wrapped)
                continue
            else:
                continue

    def load_lora_layer(self, stem: str, leaf: str, value, wrapped: torch.nn.Module):
        layer = self.layers.get(stem, None)
        if layer is None:
            layer = LoRALayer(self.name, stem, self.rank, self.alpha)
            self.layers[stem] = layer

        if type(wrapped) == torch.nn.Linear:
            module = torch.nn.Linear(value.shape[1], value.shape[0], bias=False)
        elif type(wrapped) == torch.nn.Conv2d:
            module = torch.nn.Conv2d(value.shape[1], value.shape[0], (1, 1), bias=False)
        else:
            print(
                f">> Encountered unknown lora layer module in {self.name}: {type(value).__name__}"
            )
            return

        with torch.no_grad():
            module.weight.copy_(value)

        module.to(device=self.device, dtype=self.dtype)

        if leaf == "lora_up.weight":
            layer.up = module
        elif leaf == "lora_down.weight":
            layer.down = module
        else:
            print(f">> Encountered unknown layer in lora {self.name}: {leaf}")
            return


class KohyaLoraManager:
    def __init__(self, pipe, lora_path):
        self.unet = pipe.unet
        self.lora_path = lora_path
        self.wrapper = LoRAModuleWrapper(pipe.unet, pipe.text_encoder)
        self.text_encoder = pipe.text_encoder
        self.device = torch.device(choose_torch_device())
        self.dtype = pipe.unet.dtype
        self.loras_to_load = {}

    def load_lora_module(self, name, path_file, multiplier: float = 1.0):
        print(f"   | Found lora {name} at {path_file}")
        if path_file.suffix == ".safetensors":
            checkpoint = load_file(path_file.absolute().as_posix(), device="cpu")
        else:
            checkpoint = torch.load(path_file, map_location="cpu")

        lora = LoRA(name, self.device, self.dtype, self.wrapper, multiplier)
        lora.load_from_dict(checkpoint)
        self.wrapper.loaded_loras[name] = lora

        return lora

    def apply_lora_model(self, name, mult: float = 1.0):
        for suffix in ["ckpt", "safetensors", "pt"]:
            path_file = Path(self.lora_path, f"{name}.{suffix}")
            if path_file.exists():
                print(f"   | Loading lora {path_file.name} with weight {mult}")
                break
        if not path_file.exists():
            print(f"   ** Unable to find lora: {name}")
            return

        lora = self.wrapper.loaded_loras.get(name, None)
        if lora is None:
            lora = self.load_lora_module(name, path_file, mult)

        lora.multiplier = mult
        self.wrapper.applied_loras[name] = lora

    def unload_applied_loras(self, loras_to_load):
        # unload any lora's not defined by loras_to_load
        for name in list(self.wrapper.applied_loras.keys()):
            if name not in loras_to_load:
                self.unload_applied_lora(name)

    def unload_applied_lora(self, lora_name: str):
        if lora_name in self.wrapper.applied_loras:
            del self.wrapper.applied_loras[lora_name]

    def unload_lora(self, lora_name: str):
        if lora_name in self.wrapper.loaded_loras:
            del self.wrapper.loaded_loras[lora_name]

    def clear_loras(self):
        self.loras_to_load = {}
        self.wrapper.clear_applied_loras()
