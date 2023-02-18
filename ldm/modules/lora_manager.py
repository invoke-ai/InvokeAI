import re
from pathlib import Path

from ldm.invoke.globals import global_models_dir
from diffusers.models import UNet2DConditionModel

class LoraManager:

    def __init__(self, model: UNet2DConditionModel):
        self.weights = {}
        self.model = model
        self.lora_path = Path(global_models_dir(), 'lora')
        self.lora_match = re.compile(r"<lora:([^>]+)>")
        self.prompt = None

    def apply_lora_model(self, args):
        args = args.split(':')
        name = args[0]
        path = Path(self.lora_path, name)

        if path.is_dir():
            print(f"loading lora: {path}")
            self.model.load_attn_procs(path.absolute().as_posix())

            if len(args) == 2:
                self.weights[name] = args[1]

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

