import re
from pathlib import Path
from ldm.invoke.globals import global_models_dir
import torch
from safetensors.torch import load_file

# modified from script at https://github.com/huggingface/diffusers/pull/2403
def merge_lora_into_pipe(pipeline, checkpoint_path, alpha):
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device=torch.cuda.current_device())

    visited = []

    # directly update weight in diffusers model
    for key in state_dict:

        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue
        if "text" in key:
            layer_infos = key.split(".")[0].split("lora_te" + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = key.split(".")[0].split("lora_unet" + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            curr_layer.weight.data += float(alpha) * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            curr_layer.weight.data += float(alpha) * torch.mm(weight_up, weight_down)

        # update visited list
        for item in pair_keys:
            visited.append(item)


class LoraManager:

    def __init__(self, pipe):
        self.weights = {}
        self.pipe = pipe
        self.lora_path = Path(global_models_dir(), 'lora')
        self.lora_match = re.compile(r"<lora:([^>]+)>")
        self.prompt = None

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
            alpha = 1
            if len(args) == 2:
                alpha = args[1]

            merge_lora_into_pipe(self.pipe, file.absolute().as_posix(), alpha)

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

