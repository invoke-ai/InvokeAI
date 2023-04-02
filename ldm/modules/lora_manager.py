import traceback

from pathlib import Path
from ldm.invoke.globals import global_lora_models_dir
from .kohya_lora_manager import KohyaLoraManager


class KohyaLoraContext:
    def __init__(self, kohya_manager: KohyaLoraManager):
        self.kohya=kohya_manager

    def __enter__(self):
        self.kohya.clear_loras()

    def __exit__(self,*exc):
        self.kohya.clear_loras()

class LoraCondition:
    name: str
    weight: float

    def __init__(self, name, weight: float = 1.0, kohya_manager: Optional[KohyaLoraManager]=None):
        self.name = name
        self.weight = weight
        self.kohya_manager = kohya_manager

    def __call__(self, model):
        # TODO: make model able to load from huggingface, rather then just local files
        path = Path(global_lora_models_dir(), self.name)
        if path.is_dir():
            if model.load_attn_procs:
                file = Path(path, "pytorch_lora_weights.bin")
                if file.is_file():
                    print(f">> Loading LoRA: {path}")
                    model.load_attn_procs(path.absolute().as_posix())
                else:
                    print(f"   ** Unable to find valid LoRA at: {path}")
            else:
                print(f"   ** Invalid Model to load LoRA")
        elif self.kohya_manager:
            self.kohya_manager.apply_lora_model(self.name,self.weight)
        else:
            print(f"   ** Unable to load LoRA")

class LoraManager:
    def __init__(self, pipe):
        # Kohya class handles lora not generated through diffusers
        self.kohya = KohyaLoraManager(pipe, global_lora_models_dir())

    def set_loras_conditions(self, lora_weights: list):
        conditions = []
        if len(lora_weights) > 0:
            for lora in lora_weights:
                conditions.append(LoraCondition(lora.model, lora.weight, self.kohya))

        if len(conditions) > 0:
            return conditions

        return None

    def kohya_context(self)->KohyaLoraContext:
        return KohyaLoraContext(self.kohya)
