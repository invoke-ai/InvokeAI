from pathlib import Path
from ldm.invoke.globals import global_models_dir
from .legacy_lora_manager import LegacyLoraManager


class LoraManager:
    models: list[str]

    def __init__(self, pipe):
        self.lora_path = Path(global_models_dir(), 'lora')
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        # Legacy class handles lora not generated through diffusers
        self.legacy = LegacyLoraManager(pipe, self.lora_path)
        self.models = []

    def apply_lora_model(self, name):
        path = Path(self.lora_path, name)
        file = Path(path, "pytorch_lora_weights.bin")

        if path.is_dir() and file.is_file():
            print(f">> Loading LoRA: {path}")
            self.unet.load_attn_procs(path.absolute().as_posix())
        else:
            print(f">> Unable to find valid LoRA at: {path}")

    def set_lora_model(self, name):
        self.models.append(name)

    def set_loras_compel(self, lora_weights: list):
        if len(lora_weights) > 0:
            for lora in lora_weights:
                self.set_lora_model(lora.model)

    def load_loras(self):
        for name in self.models:
            self.apply_lora_model(name)

    # Legacy functions, to pipe to LoraLegacyManager
    # To be removed once support for diffusers LoRA weights is high enough
    def configure_prompt_legacy(self, prompt: str) -> str:
        return self.legacy.configure_prompt(prompt)

    def load_lora_legacy(self):
        self.legacy.load_lora()
