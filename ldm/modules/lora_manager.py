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
                f">> Encoundered unknown lora layer module in {name}: {type(value).__name__}")
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
    loras: dict[str, LoRA]
    applied_loras: dict[str, LoRA]
    hooks: list[RemovableHandle]

    def __init__(self, pipe):
        self.lora_path = Path(global_models_dir(), 'lora')
        self.lora_match = re.compile(r"<lora:([^>]+)>")
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        self.device = torch.device(choose_torch_device())
        self.dtype = pipe.unet.dtype
        self.loras = {}
        self.applied_loras = {}
        self.hooks = []
        self.prompt = ""

        def find_modules(prefix, root_module: torch.nn.Module, target_replace_modules) -> dict[str, torch.nn.Module]:
            mapping = {}
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        if child_module.__class__.__name__ == "Linear" or (child_module.__class__.__name__ == "Conv2d" and child_module.kernel_size == (1, 1)):
                            # It's easier to just convert into "lora" naming instead of trying to inverse map from "lora" -> diffuser
                            lora_name = prefix + '.' + name + '.' + child_name
                            lora_name = lora_name.replace('.', '_')
                            mapping[lora_name] = child_module
                            self.hooks.append(child_module.register_forward_hook(self._make_hook(lora_name)))
            return mapping

        self.text_modules = find_modules(
            LORA_PREFIX_TEXT_ENCODER, self.text_encoder, TEXT_ENCODER_TARGET_REPLACE_MODULE)
        self.unet_modules = find_modules(
            LORA_PREFIX_UNET, self.unet, UNET_TARGET_REPLACE_MODULE)

    def _make_hook(self, layer: str):
        def hook(module, input, output):
            for lora in self.applied_loras.values():
                lora_layer = lora.layers.get(layer, None)
                if lora_layer is None:
                    continue
                output = output + \
                    lora_layer.up(lora_layer.down(*input)) * \
                    lora.multiplier * lora_layer.scale
            return output
        return hook

    def _load_lora(self, name, path_file, multiplier=1.0):
        lora = load_lora(name, path_file, self.device, self.dtype,
                         self.text_modules, self.unet_modules, multiplier)
        self.loras[name] = lora
        self.applied_loras[name] = lora
        return lora

    def apply_lora_model(self, args):
        args = args.split(':')
        name = args[0]
        path = Path(self.lora_path, name)
        file = Path(path, "pytorch_lora_weights.bin")

        if path.is_dir() and file.is_file():
            print(f"loading lora: {path}")
            self.unet.load_attn_procs(path.absolute().as_posix())
        else:
            # converting and saving in diffusers format
            path_file = Path(self.lora_path, f'{name}.ckpt')
            if Path(self.lora_path, f'{name}.safetensors').exists():
                path_file = Path(self.lora_path, f'{name}.safetensors')

            if not path_file.exists():
                print(f">> Unable to find lora: {name}")
                return

            mult = 1.0
            if len(args) == 2:
                mult = float(args[1])

            lora = self.loras.get(name, None)
            if lora is None:
                lora = self._load_lora(name, path_file, mult)

            lora.multiplier = mult
            self.applied_loras[name] = lora

    def load_lora_from_prompt(self, prompt: str):
        self.applied_loras = {}
        for m in re.findall(self.lora_match, prompt):
            self.apply_lora_model(m)

    def load_lora(self):
        self.load_lora_from_prompt(self.prompt)

    def unload_lora(self, lora_name: str):
        if lora_name in self.loras:
            del self.loras[lora_name]

    def configure_prompt(self, prompt: str) -> str:
        self.prompt = prompt

        def found(m):
            return ""

        return re.sub(self.lora_match, found, prompt)

    def __del__(self):
        del self.loras
        del self.applied_loras
        del self.text_modules
        del self.unet_modules
        for cb in self.hooks:
            cb.remove()
        del self.hooks
