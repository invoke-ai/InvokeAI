import re
from pathlib import Path
from ldm.invoke.globals import global_models_dir
import torch
from safetensors.torch import load_file
from typing import List, Optional, Set, Type


class LoraLinear(torch.nn.Module):
    def __init__(
        self, in_features, out_features, rank=4
    ):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )
        self.rank = rank
        self.linear = torch.nn.Linear(in_features, out_features, bias=False)
        self.lora = torch.nn.Linear(in_features, out_features, bias=False)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.lora.weight.dtype
        return self.lora(hidden_states.to(dtype)).to(orig_dtype)


class LoraManager:

    def __init__(self, pipe):
        self.pipe = pipe
        self.lora_path = Path(global_models_dir(), 'lora')
        self.lora_match = re.compile(r"<lora:([^>]+)>")
        self.prompt = None

    def _process_lora(self, lora):
        processed_lora = {
            "unet": [],
            "text_encoder": []
        }
        visited = []
        for key in lora:
            if ".alpha" in key or key in visited:
                continue
            if "text" in key:
                lora_type, pair_keys = self._find_layer(
                    "text_encoder",
                    key.split(".")[0].split("lora_te" + "_")[-1].split("_"),
                    key
                )
            else:
                lora_type, pair_keys = self._find_layer(
                    "unet",
                    key.split(".")[0].split("lora_unet" + "_")[-1].split("_"),
                    key
                )

            if len(lora[pair_keys[0]].shape) == 4:
                weight_up = lora[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
                weight_down = lora[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
                weight = torch.mm(weight_up, weight_down)
            else:
                weight_up = lora[pair_keys[0]].to(torch.float32)
                weight_down = lora[pair_keys[1]].to(torch.float32)
                weight = torch.mm(weight_up, weight_down)

            processed_lora[lora_type].append({
                "weight": weight,
                "rank": lora[pair_keys[1]].shape[0]
            })

            for item in pair_keys:
                visited.append(item)

        return processed_lora

    def _find_layer(self, lora_type, layer_key, key):
        temp_name = layer_key.pop(0)
        if lora_type == "unet":
            curr_layer = self.pipe.unet
        elif lora_type == "text_encoder":
            curr_layer = self.pipe.text_encoder
        else:
            raise ValueError("Invalid Lora Type")

        while len(layer_key) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_key) > 0:
                    temp_name = layer_key.pop(0)
                elif len(layer_key) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_key.pop(0)
                else:
                    temp_name = layer_key.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        return lora_type, pair_keys

    @staticmethod
    def _find_modules(
        model,
        ancestor_class: Optional[Set[str]] = None,
        search_class: List[Type[torch.nn.Module]] = [torch.nn.Linear],
        exclude_children_of: Optional[List[Type[torch.nn.Module]]] = [LoraLinear],
    ):
        """
        Find all modules of a certain class (or union of classes) that are direct or
        indirect descendants of other modules of a certain class (or union of classes).
        Returns all matching modules, along with the parent of those modules and the
        names they are referenced by.
        """

        # Get the targets we should replace all linears under
        if ancestor_class is not None:
            ancestors = (
                module
                for module in model.modules()
                if module.__class__.__name__ in ancestor_class
            )
        else:
            # this, incase you want to naively iterate over all modules.
            ancestors = [module for module in model.modules()]

        # For each target find every linear_class module that isn't a child of a LoraInjectedLinear
        for ancestor in ancestors:
            for fullname, module in ancestor.named_modules():
                if any([isinstance(module, _class) for _class in search_class]):
                    # Find the direct parent if this is a descendant, not a child, of target
                    *path, name = fullname.split(".")
                    parent = ancestor
                    while path:
                        parent = parent.get_submodule(path.pop(0))
                    # Skip this linear if it's a child of a LoraInjectedLinear
                    if exclude_children_of and any(
                        [isinstance(parent, _class) for _class in exclude_children_of]
                    ):
                        continue
                    # Otherwise, yield it
                    yield parent, name, module

    @staticmethod
    def patch_module(lora_type, processed_lora, module, name, child_module, scale: float = 1.0):
        _source = (
            child_module.linear
            if isinstance(child_module, LoraLinear)
            else child_module
        )

        lora = processed_lora[lora_type].pop(0)

        weight = _source.weight
        _tmp = LoraLinear(
            in_features=_source.in_features,
            out_features=_source.out_features,
            rank=lora["rank"]
        )
        _tmp.linear.weight = weight

        # switch the module
        module._modules[name] = _tmp
        module._modules[name].lora.weight.data = lora["weight"]
        module._modules[name].to(weight.device)

    def patch_lora(self, lora_path, scale: float = 1.0):
        lora = load_file(lora_path)
        processed_lora = self._process_lora(lora)
        for module, name, child_module in self._find_modules(
            self.pipe.unet,
            {"CrossAttention", "Attention", "GEGLU"},
            search_class=[torch.nn.Linear, LoraLinear]
        ):
            self.patch_module("unet", processed_lora, module, name, child_module, scale)

        for module, name, child_module in self._find_modules(
            self.pipe.text_encoder,
            {"CLIPAttention"},
            search_class=[torch.nn.Linear, LoraLinear]
        ):
            self.patch_module("text_encoder", processed_lora, module, name, child_module, scale)

    def apply_lora_model(self, args):
        args = args.split(':')
        name = args[0]

        path = Path(self.lora_path, name)
        file = Path(path, "pytorch_lora_weights.bin")

        if path.is_dir() and file.is_file():
            print(f"loading diffusers lora: {path}")
            self.pipe.unet.load_attn_procs(path.absolute().as_posix())
        else:
            file = Path(self.lora_path, f"{name}.safetensors")
            print(f"loading lora: {file}")
            scale = 1.0
            if len(args) == 2:
                scale = float(args[1])

            self.patch_lora(file.absolute().as_posix(), scale)

    @staticmethod
    def remove_lora(child_module):
        _source = child_module.linear
        weight = _source.weight

        _tmp = torch.nn.Linear(_source.in_features, _source.out_features)
        _tmp.weight = weight

    def reset_lora(self):
        for module, name, child_module in self._find_modules(
            self.pipe.unet,
            search_class=[LoraLinear]
        ):
            self.remove_lora(child_module)

        for module, name, child_module in self._find_modules(
            self.pipe.text_encoder,
            search_class=[LoraLinear]
        ):
            self.remove_lora(child_module)

    def load_lora_from_prompt(self, prompt: str):
        for m in re.findall(self.lora_match, prompt):
            self.apply_lora_model(m)

    def load_lora(self):
        self.load_lora_from_prompt(self.prompt)

    def configure_prompt(self, prompt: str) -> str:
        self.prompt = prompt

        def found(m):
            return ""

        return re.sub(self.lora_match, found, prompt)
