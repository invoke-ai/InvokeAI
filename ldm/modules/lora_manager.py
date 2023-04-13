import os
from diffusers import StableDiffusionPipeline
from pathlib import Path
from ldm.invoke.globals import global_lora_models_dir
from .kohya_lora_manager import KohyaLoraManager, IncompatibleModelException
from typing import Optional, Dict

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
                print("   ** Invalid Model to load LoRA")
        elif self.kohya_manager:
            try:
                self.kohya_manager.apply_lora_model(self.name,self.weight)
            except IncompatibleModelException:
                print(f"   ** LoRA {self.name} is incompatible with this model; will generate without the LoRA applied.")
        else:
            print("   ** Unable to load LoRA")

    def unload(self):
        if self.kohya_manager and self.kohya_manager.unload_applied_lora(self.name):
            print(f'>> unloading LoRA {self.name}')
            

class LoraManager:
    def __init__(self, pipe):
        # Kohya class handles lora not generated through diffusers
        self.kohya = KohyaLoraManager(pipe)

    def set_loras_conditions(self, lora_weights: list):
        conditions = []
        if len(lora_weights) > 0:
            for lora in lora_weights:
                conditions.append(LoraCondition(lora.model, lora.weight, self.kohya))

        if len(conditions) > 0:
            return conditions

        return None

    @classmethod
    def list_loras(self, token_vector_length:int=None)->Dict[str, Path]:
        '''List the LoRAS in the global lora directory.
        If token_vector_length is provided, then only return
        LoRAS that have the indicated length:
        768: v1 models
        1024: v2 models
        '''
        path = Path(global_lora_models_dir())
        models_found = dict()
        for root,_,files in os.walk(path):
            for x in files:
                name = Path(x).stem
                suffix = Path(x).suffix
                if suffix not in [".ckpt", ".pt", ".safetensors"]:
                    continue
                path = Path(root,x)
                if token_vector_length is None:
                    models_found[name]=Path(root,x)  # unconditional addition
                elif token_vector_length == KohyaLoraManager.vector_length_from_checkpoint_file(path):
                    models_found[name]=Path(root,x)  # conditional on the base model matching
        return models_found
            
