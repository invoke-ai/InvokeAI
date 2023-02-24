from pathlib import Path
from ldm.invoke.globals import global_lora_models_dir
from .legacy_lora_manager import LegacyLoraManager


class LoraCondition:
    name: str
    weight: float

    def __init__(self, name, weight: float = 1.0):
        self.name = name
        self.weight = weight

    def __call__(self, model):
        path = Path(global_lora_models_dir(), self.name)
        file = Path(path, "pytorch_lora_weights.bin")

        if path.is_dir() and file.is_file():
            if model.load_attn_procs:
                print(f">> Loading LoRA: {path}")
                model.load_attn_procs(path.absolute().as_posix())
            else:
                print(f">> Invalid Model to load LoRA")
        else:
            print(f">> Unable to find valid LoRA at: {path}")


class LoraManager:
    conditions: list[LoraCondition]

    def __init__(self, pipe):
        self.unet = pipe.unet
        self.text_encoder = pipe.text_encoder
        # Legacy class handles lora not generated through diffusers
        self.legacy = LegacyLoraManager(pipe, global_lora_models_dir())
        self.conditions = []

    def set_lora_model(self, name, weight: float = 1.0):
        self.conditions.append(LoraCondition(name, weight))

    def set_loras_conditions(self, lora_weights: list):
        if len(lora_weights) > 0:
            for lora in lora_weights:
                self.set_lora_model(lora.model, lora.weight)

        if len(self.conditions) > 0:
            return self.conditions

        return None

    # Legacy functions, to pipe to LoraLegacyManager
    # To be removed once support for diffusers LoRA weights is high enough
    def configure_prompt_legacy(self, prompt: str) -> str:
        return self.legacy.configure_prompt(prompt)

    def load_lora_legacy(self):
        self.legacy.load_lora()
