import re
from pathlib import Path
from ldm.invoke.globals import global_models_dir
from ldm.invoke.devices import choose_torch_device
from safetensors.torch import load_file
import torch
from torch.utils.hooks import RemovableHandle


class LoRALayer:
    lora_name: str
    name: str
    scale: float
    up: torch.nn.Module
    down: torch.nn.Module

    def __init__(self, lora_name: str, name: str, rank=4, alpha=1.0):
        super().__init__()
        self.lora_name = lora_name
        self.name = name
        self.scale = alpha / rank


class LoRA:
    name: str
    layers: dict[str, LoRALayer]
    multiplier: float

    def __init__(self, name: str, multiplier=1.0):
        self.name = name
        self.layers = {}
        self.multiplier = multiplier


UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel", "Attention"]
TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
LORA_PREFIX_UNET = 'lora_unet'
LORA_PREFIX_TEXT_ENCODER = 'lora_te'


def lora_forward_hook(name):
    def lora_forward(module, input_h, output):
        if len(loaded_loras) == 0:
            return output

        for lora in applied_loras.values():
            layer = lora.layers.get(name, None)
            if layer is None:
                continue
            output = output + layer.up(layer.down(*input_h)) * lora.multiplier * layer.scale
        return output

    return lora_forward


def load_lora(
    name: str,
    path_file: Path,
    device: torch.device,
    dtype: torch.dtype,
    text_modules: dict[str, torch.nn.Module],
    unet_modules: dict[str, torch.nn.Module],
    multiplier=1.0
):
    print(f">> Loading lora {name} from {path_file}")
    if path_file.suffix == '.safetensors':
        checkpoint = load_file(path_file.absolute().as_posix(), device='cpu')
    else:
        checkpoint = torch.load(path_file, map_location='cpu')

    lora = LoRA(name, multiplier)

    alpha = None
    rank = None
    for key, value in checkpoint.items():
        stem, leaf = key.split(".", 1)

        if leaf.endswith("alpha"):
            if alpha is None:
                alpha = value.item()
            continue

        if stem.startswith(LORA_PREFIX_TEXT_ENCODER):
            # text encoder layer
            wrapped = text_modules.get(stem, None)
            if wrapped is None:
                print(f">> Missing layer: {stem}")
                continue
        elif stem.startswith(LORA_PREFIX_UNET):
            # unet layer
            wrapped = unet_modules.get(stem, None)
            if wrapped is None:
                print(f">> Missing layer: {stem}")
                continue
        else:
            continue

        if rank is None and leaf == 'lora_down.weight' and len(value.size()) == 2:
            rank = value.shape[0]

        if wrapped is None:
            continue

        layer = lora.layers.get(stem, None)
        if layer is None:
            layer = LoRALayer(name, stem, rank, alpha)
            lora.layers[stem] = layer

        if type(wrapped) == torch.nn.Linear:
            module = torch.nn.Linear(
                value.shape[1], value.shape[0], bias=False)
        elif type(wrapped) == torch.nn.Conv2d:
            module = torch.nn.Conv2d(
                value.shape[1], value.shape[0], (1, 1), bias=False)
        else:
            print(
                f">> Encountered unknown lora layer module in {name}: {type(value).__name__}")
            continue

        with torch.no_grad():
            module.weight.copy_(value)

        module.to(device=device, dtype=dtype)

        if leaf == "lora_up.weight":
            layer.up = module
        elif leaf == "lora_down.weight":
            layer.down = module
        else:
            print(f">> Encountered unknown layer in lora {name}: {key}")
            continue

    return lora


class LoraManager:
    loras_to_load: dict[str, float]
    hooks: list[RemovableHandle]

    def __init__(self, pipe):
        self.lora_path = Path(global_models_dir(), 'lora')
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        self.device = torch.device(choose_torch_device())
        self.dtype = pipe.unet.dtype
        self.loras_to_load = {}
        self.hooks = []

        def find_modules(prefix, root_module: torch.nn.Module, target_replace_modules) -> dict[str, torch.nn.Module]:
            mapping = {}
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        layer_type = child_module.__class__.__name__
                        if layer_type == "Linear" or (layer_type == "Conv2d" and child_module.kernel_size == (1, 1)):
                            lora_name = prefix + '.' + name + '.' + child_name
                            lora_name = lora_name.replace('.', '_')
                            mapping[lora_name] = child_module
                            self.apply_module_forward(child_module, lora_name)
            return mapping

        self.text_modules = find_modules(
            LORA_PREFIX_TEXT_ENCODER, self.text_encoder, TEXT_ENCODER_TARGET_REPLACE_MODULE)
        self.unet_modules = find_modules(
            LORA_PREFIX_UNET, self.unet, UNET_TARGET_REPLACE_MODULE)

    def _load_lora(self, name, path_file, multiplier: float = 1.0):
        lora = load_lora(name, path_file, self.device, self.dtype,
                         self.text_modules, self.unet_modules, multiplier)
        loaded_loras[name] = lora
        return lora

    def apply_module_forward(self, module, lora_name):
        handle = module.register_module_forward_hook(lora_forward_hook(lora_name))
        self.hooks.append(handle)

    def apply_lora_model(self, name, mult: float = 1.0):
        path = Path(self.lora_path, name)
        file = Path(path, "pytorch_lora_weights.bin")

        if path.is_dir() and file.is_file():
            print(f"Diffusers lora is currently disabled: {path}")
            # print(f"loading lora: {path}")
            # self.unet.load_attn_procs(path.absolute().as_posix())
        else:
            path_file = Path(self.lora_path, f'{name}.ckpt')
            if Path(self.lora_path, f'{name}.safetensors').exists():
                path_file = Path(self.lora_path, f'{name}.safetensors')

            if not path_file.exists():
                print(f">> Unable to find lora: {name}")
                return

            lora = loaded_loras.get(name, None)
            if lora is None:
                lora = self._load_lora(name, path_file, mult)

            lora.multiplier = mult
            applied_loras[name] = lora

    def load_lora(self):
        for name, multiplier in self.loras_to_load.items():
            self.apply_lora_model(name, multiplier)

        # unload any lora's not defined by loras_to_load
        for name in list(applied_loras.keys()):
            if name not in self.loras_to_load:
                self.unload_applied_lora(name)

    @staticmethod
    def unload_applied_lora(lora_name: str):
        if lora_name in applied_loras:
            del applied_loras[lora_name]

    @staticmethod
    def unload_lora(lora_name: str):
        if lora_name in loaded_loras:
            del loaded_loras[lora_name]

    # Define a lora to be loaded
    # Can be used to define a lora to be loaded outside of prompts
    def set_lora(self, name, multiplier: float = 1.0):
        self.loras_to_load[name] = multiplier

        # update the multiplier if the lora was already loaded
        if name in loaded_loras:
            loaded_loras[name].multiplier = multiplier

    # Load the lora from a prompt, syntax is <lora:lora_name:multiplier>
    # Multiplier should be a value between 0.0 and 1.0
    def configure_prompt(self, prompt: str) -> str:
        self.clear_loras()

        lora_match = re.compile(r"<lora:([^>]+)>")

        for match in re.findall(lora_match, prompt):
            match = match.split(':')
            name = match[0]

            mult = 1.0
            if len(match) == 2:
                mult = float(match[1])

            self.set_lora(name, mult)

        # remove lora and return prompt to avoid the lora prompt causing issues in inference
        return re.sub(lora_match, "", prompt)

    def clear_loras(self):
        clear_applied_loras()
        self.loras_to_load = {}

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()

        self.hooks.clear()

    def __del__(self):
        self.clear_hooks()
        clear_applied_loras()
        clear_loaded_loras()
        del self.text_modules
        del self.unet_modules
        del self.loras_to_load


applied_loras = {}
loaded_loras = {}


def clear_applied_loras():
    applied_loras.clear()


def clear_loaded_loras():
    loaded_loras.clear()
