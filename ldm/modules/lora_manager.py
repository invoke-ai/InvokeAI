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
        print(f'DEBUG: path = {path}')

        # TODO: make model able to load from huggingface, rather then just local files
        if path.is_dir():
            if model.load_attn_procs:
                file = Path(path, "pytorch_lora_weights.bin")
                if file.is_file():
                    print(f">> Loading LoRA: {path}")
                    model.load_attn_procs(path.absolute().as_posix())
                else:
                    print(f">> Unable to find valid LoRA at: {path}")
            else:
                print(f">> Invalid Model to load LoRA")
        else:
            print(f">> Unable to find valid LoRA at: {path}")


class LoraManager:
    def __init__(self, pipe):
        # Legacy class handles lora not generated through diffusers
        self.legacy = LegacyLoraManager(pipe, global_lora_models_dir())

    @staticmethod
    def set_loras_conditions(lora_weights: list):
        conditions = []
        if len(lora_weights) > 0:
            for lora in lora_weights:
                conditions.append(LoraCondition(lora.model, lora.weight))

        if len(conditions) > 0:
            return conditions

        return None

    # Legacy functions, to pipe to LoraLegacyManager
    # To be removed once support for diffusers LoRA weights is high enough
    def configure_prompt_legacy(self, prompt: str) -> str:
        return self.legacy.configure_prompt(prompt)

    def load_lora_legacy(self):
        self.legacy.load_lora()
