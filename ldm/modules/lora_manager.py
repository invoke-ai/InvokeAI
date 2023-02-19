import re
from pathlib import Path
from ldm.invoke.globals import global_models_dir
from lora_diffusion import tune_lora_scale, patch_pipe


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
            print(f"loading lora: {path}")
            self.pipe.unet.load_attn_procs(path.absolute().as_posix())
            if len(args) == 2:
                self.weights[name] = args[1]
        else:
            # converting and saving in diffusers format
            path_file = Path(self.lora_path, f'{name}.ckpt')
            if Path(self.lora_path, f'{name}.safetensors').exists():
                path_file = Path(self.lora_path, f'{name}.safetensors')

            if path_file.is_file():
                print(f"loading lora: {path}")
                patch_pipe(
                    self.pipe,
                    path_file.absolute().as_posix(),
                    patch_text=True,
                    patch_ti=True,
                    patch_unet=True,
                )
                if len(args) == 2:
                    tune_lora_scale(self.pipe.unet, args[1])
                    tune_lora_scale(self.pipe.text_encoder, args[1])

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

