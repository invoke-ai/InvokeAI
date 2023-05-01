import json
from pathlib import Path
from typing import Optional

import torch
from diffusers.models import UNet2DConditionModel
from filelock import FileLock, Timeout
from safetensors.torch import load_file
from torch.utils.hooks import RemovableHandle
from transformers import CLIPTextModel

from ..invoke.globals import global_lora_models_dir, Globals
from ..invoke.devices import choose_torch_device

"""
This module supports loading LoRA weights trained with https://github.com/kohya-ss/sd-scripts
To be removed once support for diffusers LoRA weights is well supported
"""


class IncompatibleModelException(Exception):
    "Raised when there is an attempt to load a LoRA into a model that is incompatible with it"
    pass


class LoRALayer:
    lora_name: str
    name: str
    scale: float

    up: torch.nn.Module
    mid: Optional[torch.nn.Module] = None
    down: torch.nn.Module

    def __init__(self, lora_name: str, name: str, rank=4, alpha=1.0):
        self.lora_name = lora_name
        self.name = name
        self.scale = alpha / rank if (alpha and rank) else 1.0

    def forward(self, lora, input_h):
        if self.mid is None:
            weight = self.up(self.down(*input_h))
        else:
            weight = self.up(self.mid(self.down(*input_h)))

        return weight * lora.multiplier * self.scale


class LoHALayer:
    lora_name: str
    name: str
    scale: float

    w1_a: torch.Tensor
    w1_b: torch.Tensor
    w2_a: torch.Tensor
    w2_b: torch.Tensor
    t1: Optional[torch.Tensor] = None
    t2: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None

    org_module: torch.nn.Module

    def __init__(self, lora_name: str, name: str, rank=4, alpha=1.0):
        self.lora_name = lora_name
        self.name = name
        self.scale = alpha / rank if (alpha and rank) else 1.0

    def forward(self, lora, input_h):
        if type(self.org_module) == torch.nn.Conv2d:
            op = torch.nn.functional.conv2d
            extra_args = dict(
                stride=self.org_module.stride,
                padding=self.org_module.padding,
                dilation=self.org_module.dilation,
                groups=self.org_module.groups,
            )

        else:
            op = torch.nn.functional.linear
            extra_args = {}

        if self.t1 is None:
            weight = (self.w1_a @ self.w1_b) * (self.w2_a @ self.w2_b)

        else:
            rebuild1 = torch.einsum(
                "i j k l, j r, i p -> p r k l", self.t1, self.w1_b, self.w1_a
            )
            rebuild2 = torch.einsum(
                "i j k l, j r, i p -> p r k l", self.t2, self.w2_b, self.w2_a
            )
            weight = rebuild1 * rebuild2

        bias = self.bias if self.bias is not None else 0
        return op(
            *input_h,
            (weight + bias).view(self.org_module.weight.shape),
            None,
            **extra_args,
        ) * lora.multiplier * self.scale

class LoKRLayer:
    lora_name: str
    name: str
    scale: float

    w1: Optional[torch.Tensor] = None
    w1_a: Optional[torch.Tensor] = None
    w1_b: Optional[torch.Tensor] = None
    w2: Optional[torch.Tensor] = None
    w2_a: Optional[torch.Tensor] = None
    w2_b: Optional[torch.Tensor] = None
    t2: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None

    org_module: torch.nn.Module

    def __init__(self, lora_name: str, name: str, rank=4, alpha=1.0):
        self.lora_name = lora_name
        self.name = name
        self.scale = alpha / rank if (alpha and rank) else 1.0

    def forward(self, lora, input_h):

        if type(self.org_module) == torch.nn.Conv2d:
            op = torch.nn.functional.conv2d
            extra_args = dict(
                stride=self.org_module.stride,
                padding=self.org_module.padding,
                dilation=self.org_module.dilation,
                groups=self.org_module.groups,
            )

        else:
            op = torch.nn.functional.linear
            extra_args = {}

        w1 = self.w1
        if w1 is None:
            w1 = self.w1_a @ self.w1_b

        w2 = self.w2
        if w2 is None:
            if self.t2 is None:
                w2 = self.w2_a @ self.w2_b
            else:
                w2 = torch.einsum('i j k l, i p, j r -> p r k l', self.t2, self.w2_a, self.w2_b)


        if len(w2.shape) == 4:
            w1 = w1.unsqueeze(2).unsqueeze(2)
        w2 = w2.contiguous()
        weight = torch.kron(w1, w2).reshape(self.org_module.weight.shape)


        bias = self.bias if self.bias is not None else 0
        return op(
            *input_h, 
            (weight + bias).view(self.org_module.weight.shape),
            None,
            **extra_args
        ) * lora.multiplier * self.scale


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

        self.UNET_TARGET_REPLACE_MODULE = [
            "Transformer2DModel",
            "Attention",
            "ResnetBlock2D",
            "Downsample2D",
            "Upsample2D",
            "SpatialTransformer",
        ]
        self.TEXT_ENCODER_TARGET_REPLACE_MODULE = [
            "ResidualAttentionBlock",
            "CLIPAttention",
            "CLIPMLP",
        ]
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
                            and child_module.kernel_size in [(1, 1), (3, 3)]
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
                output += layer.forward(lora, input_h)
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

    def load_from_dict(self, state_dict):
        state_dict_groupped = dict()

        for key, value in state_dict.items():
            stem, leaf = key.split(".", 1)
            if stem not in state_dict_groupped:
                state_dict_groupped[stem] = dict()
            state_dict_groupped[stem][leaf] = value

        for stem, values in state_dict_groupped.items():
            if stem.startswith(self.wrapper.LORA_PREFIX_TEXT_ENCODER):
                wrapped = self.wrapper.text_modules.get(stem, None)
            elif stem.startswith(self.wrapper.LORA_PREFIX_UNET):
                wrapped = self.wrapper.unet_modules.get(stem, None)
            else:
                continue

            if wrapped is None:
                print(f">> Missing layer: {stem}")
                continue

            # TODO: diff key

            bias = None
            alpha = None

            if "alpha" in values:
                alpha = values["alpha"].item()

            if (
                "bias_indices" in values
                and "bias_values" in values
                and "bias_size" in values
            ):
                bias = torch.sparse_coo_tensor(
                    values["bias_indices"],
                    values["bias_values"],
                    tuple(values["bias_size"]),
                ).to(device=self.device, dtype=self.dtype)

            # lora and locon
            if "lora_down.weight" in values:
                value_down = values["lora_down.weight"]
                value_mid = values.get("lora_mid.weight", None)
                value_up = values["lora_up.weight"]

                if type(wrapped) == torch.nn.Conv2d:
                    if value_mid is not None:
                        layer_down = torch.nn.Conv2d(
                            value_down.shape[1], value_down.shape[0], (1, 1), bias=False
                        )
                        layer_mid = torch.nn.Conv2d(
                            value_mid.shape[1],
                            value_mid.shape[0],
                            wrapped.kernel_size,
                            wrapped.stride,
                            wrapped.padding,
                            bias=False,
                        )
                    else:
                        layer_down = torch.nn.Conv2d(
                            value_down.shape[1],
                            value_down.shape[0],
                            wrapped.kernel_size,
                            wrapped.stride,
                            wrapped.padding,
                            bias=False,
                        )
                        layer_mid = None

                    layer_up = torch.nn.Conv2d(
                        value_up.shape[1], value_up.shape[0], (1, 1), bias=False
                    )

                elif type(wrapped) == torch.nn.Linear:
                    layer_down = torch.nn.Linear(
                        value_down.shape[1], value_down.shape[0], bias=False
                    )
                    layer_mid = None
                    layer_up = torch.nn.Linear(
                        value_up.shape[1], value_up.shape[0], bias=False
                    )

                else:
                    print(
                        f">> Encountered unknown lora layer module in {self.name}: {stem} - {type(wrapped).__name__}"
                    )
                    return

                with torch.no_grad():
                    layer_down.weight.copy_(value_down)
                    if layer_mid is not None:
                        layer_mid.weight.copy_(value_mid)
                    layer_up.weight.copy_(value_up)

                layer_down.to(device=self.device, dtype=self.dtype)
                if layer_mid is not None:
                    layer_mid.to(device=self.device, dtype=self.dtype)
                layer_up.to(device=self.device, dtype=self.dtype)

                rank = value_down.shape[0]

                layer = LoRALayer(self.name, stem, rank, alpha)
                # layer.bias = bias # TODO: find and debug lora/locon with bias
                layer.down = layer_down
                layer.mid = layer_mid
                layer.up = layer_up

            # loha
            elif "hada_w1_b" in values:
                rank = values["hada_w1_b"].shape[0]

                layer = LoHALayer(self.name, stem, rank, alpha)
                layer.org_module = wrapped
                layer.bias = bias

                layer.w1_a = values["hada_w1_a"].to(
                    device=self.device, dtype=self.dtype
                )
                layer.w1_b = values["hada_w1_b"].to(
                    device=self.device, dtype=self.dtype
                )
                layer.w2_a = values["hada_w2_a"].to(
                    device=self.device, dtype=self.dtype
                )
                layer.w2_b = values["hada_w2_b"].to(
                    device=self.device, dtype=self.dtype
                )

                if "hada_t1" in values:
                    layer.t1 = values["hada_t1"].to(
                        device=self.device, dtype=self.dtype
                    )
                else:
                    layer.t1 = None

                if "hada_t2" in values:
                    layer.t2 = values["hada_t2"].to(
                        device=self.device, dtype=self.dtype
                    )
                else:
                    layer.t2 = None

            # lokr
            elif "lokr_w1_b" in values or "lokr_w1" in values:

                if "lokr_w1_b" in values:
                    rank = values["lokr_w1_b"].shape[0]
                elif "lokr_w2_b" in values:
                    rank = values["lokr_w2_b"].shape[0]
                else:
                    rank = None # unscaled

                layer = LoKRLayer(self.name, stem, rank, alpha)
                layer.org_module = wrapped
                layer.bias = bias

                if "lokr_w1" in values:
                    layer.w1 = values["lokr_w1"].to(device=self.device, dtype=self.dtype)
                else:
                    layer.w1_a = values["lokr_w1_a"].to(device=self.device, dtype=self.dtype)
                    layer.w1_b = values["lokr_w1_b"].to(device=self.device, dtype=self.dtype)

                if "lokr_w2" in values:
                    layer.w2 = values["lokr_w2"].to(device=self.device, dtype=self.dtype)
                else:
                    layer.w2_a = values["lokr_w2_a"].to(device=self.device, dtype=self.dtype)
                    layer.w2_b = values["lokr_w2_b"].to(device=self.device, dtype=self.dtype)

                if "lokr_t2" in values:
                    layer.t2 = values["lokr_t2"].to(device=self.device, dtype=self.dtype)


            else:
                print(
                    f">> Encountered unknown lora layer module in {self.name}: {stem} - {type(wrapped).__name__}"
                )
                return

            self.layers[stem] = layer


class KohyaLoraManager:
    
    def __init__(self, pipe):
        self.vector_length_cache_path = self.lora_path / '.vectorlength.cache'
        self.unet = pipe.unet
        self.wrapper = LoRAModuleWrapper(pipe.unet, pipe.text_encoder)
        self.text_encoder = pipe.text_encoder
        self.device = torch.device(choose_torch_device())
        self.dtype = pipe.unet.dtype

    @classmethod
    @property
    def lora_path(cls)->Path:
        return Path(global_lora_models_dir())

    @classmethod
    @property
    def vector_length_cache_path(cls)->Path:
        return cls.lora_path / '.vectorlength.cache'        

    def load_lora_module(self, name, path_file, multiplier: float = 1.0):
        print(f"   | Found lora {name} at {path_file}")
        if path_file.suffix == ".safetensors":
            checkpoint = load_file(path_file.absolute().as_posix(), device="cpu")
        else:
            checkpoint = torch.load(path_file, map_location="cpu")

        if not self.check_model_compatibility(checkpoint):
            raise IncompatibleModelException

        lora = LoRA(name, self.device, self.dtype, self.wrapper, multiplier)
        lora.load_from_dict(checkpoint)
        self.wrapper.loaded_loras[name] = lora

        return lora

    def apply_lora_model(self, name, mult: float = 1.0):
        path_file = None
        for suffix in ["ckpt", "safetensors", "pt"]:
            path_files = [x for x in Path(self.lora_path).glob(f"**/{name}.{suffix}")]
            if len(path_files):
                path_file = path_files[0]
                print(f"   | Loading lora {path_file.name} with weight {mult}")
                break
        if not path_file:
            print(f"   ** Unable to find lora: {name}")
            return

        lora = self.wrapper.loaded_loras.get(name, None)
        if lora is None:
            lora = self.load_lora_module(name, path_file, mult)

        lora.multiplier = mult
        self.wrapper.applied_loras[name] = lora

    def unload_applied_lora(self, lora_name: str) -> bool:
        """If the indicated LoRA has previously been applied then
        unload it and return True. Return False if the LoRA was
        not previously applied (for status reporting)
        """
        if lora_name in self.wrapper.applied_loras:
            del self.wrapper.applied_loras[lora_name]
            return True
        return False

    def unload_lora(self, lora_name: str) -> bool:
        if lora_name in self.wrapper.loaded_loras:
            del self.wrapper.loaded_loras[lora_name]
            return True
        return False

    def clear_loras(self):
        self.wrapper.clear_applied_loras()

    def check_model_compatibility(self, checkpoint) -> bool:
        """Checks whether the LoRA checkpoint is compatible with the token vector
        length of the model that this manager is associated with.
        """
        model_token_vector_length = (
            self.text_encoder.get_input_embeddings().weight.data[0].shape[0]
        )
        lora_token_vector_length = self.vector_length_from_checkpoint(checkpoint)
        return model_token_vector_length == lora_token_vector_length

    @staticmethod
    def vector_length_from_checkpoint(checkpoint: dict) -> int:
        """Return the vector token length for the passed LoRA checkpoint object.
        This is used to determine which SD model version the LoRA was based on.
        768 -> SDv1
        1024-> SDv2
        """
        key1 = "lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight"
        key2 = "lora_te_text_model_encoder_layers_0_self_attn_k_proj.hada_w1_a"
        lora_token_vector_length = (
            checkpoint[key1].shape[1]
            if key1 in checkpoint
            else checkpoint[key2].shape[0]
            if key2 in checkpoint
            else 768
        )
        return lora_token_vector_length

    @classmethod
    def vector_length_from_checkpoint_file(self, checkpoint_path: Path) -> int:
        with LoraVectorLengthCache(self.vector_length_cache_path) as cache:
            if str(checkpoint_path) not in cache:
                if checkpoint_path.suffix == ".safetensors":
                    checkpoint = load_file(
                        checkpoint_path.absolute().as_posix(), device="cpu"
                    )
                else:
                    checkpoint = torch.load(checkpoint_path, map_location="cpu")
                cache[str(checkpoint_path)] = KohyaLoraManager.vector_length_from_checkpoint(
                    checkpoint
                )
            return cache[str(checkpoint_path)]

class LoraVectorLengthCache(object):
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.lock = FileLock(Path(cache_path.parent, ".cachelock"))
        self.cache = {}

    def __enter__(self):
        self.lock.acquire(timeout=10)
        try:
            if self.cache_path.exists():
                with open(self.cache_path, "r") as json_file:
                    self.cache = json.load(json_file)
        except Timeout:
            print(
                "** Can't acquire lock on lora vector length cache. Operations will be slower"
            )
        except (json.JSONDecodeError, OSError):
            self.cache_path.unlink()
        return self.cache

    def __exit__(self, type, value, traceback):
        with open(self.cache_path, "w") as json_file:
            json.dump(self.cache, json_file)
        self.lock.release()
