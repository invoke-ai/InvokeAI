from peft import LoraModel, LoraConfig, set_peft_model_state_dict
import torch
import json
from pathlib import Path
from ldm.invoke.globals import global_lora_models_dir

class LoraPeftModule:
    def __init__(self, lora_dir, multiplier: float = 1.0):
        self.lora_dir = lora_dir
        self.multiplier = multiplier
        self.config = self.load_config()
        self.checkpoint = self.load_checkpoint()

    def load_config(self):
        lora_config_file = Path(self.lora_dir, f'lora_config.json')
        with open(lora_config_file, "r") as f:
            return json.load(f)

    def load_checkpoint(self):
        return torch.load(Path(self.lora_dir, f'lora.pt'))

    def unet(self, text_encoder):
        lora_ds = {
            k.replace("text_encoder_", ""): v for k, v in self.checkpoint.items() if "text_encoder_" in k
        }
        config = LoraConfig(**self.config["peft_config"])
        model = LoraModel(config, text_encoder)
        set_peft_model_state_dict(model, lora_ds)
        return model

    def text_encoder(self, unet):
        lora_ds = {
            k: v for k, v in self.checkpoint.items() if "text_encoder_" not in k
        }
        config = LoraConfig(**self.config["text_encoder_peft_config"])
        model = LoraModel(config, unet)
        set_peft_model_state_dict(model, lora_ds)
        return model

    def apply(self, pipe, dtype):
        pipe.unet = self.unet(pipe.unet)
        if "text_encoder_peft_config" in self.config:
            pipe.text_encoder = self.text_encoder(pipe.text_encoder)

        if dtype in (torch.float16, torch.bfloat16):
            pipe.unet.half()
            pipe.text_encoder.half()

        return pipe


class PeftManager:
    modules: list[LoraPeftModule]

    def __init__(self):
        self.lora_path = global_lora_models_dir()
        self.modules = []

    def set_loras(self, lora_weights: list):
        if len(lora_weights) > 0:
            for lora in lora_weights:
                self.add(lora.model, lora.weight)

    def add(self, name, multiplier: float = 1.0):
        lora_dir = Path(self.lora_path, name)

        if lora_dir.exists():
            lora_config_file = Path(lora_dir, f'lora_config.json')
            lora_checkpoint = Path(lora_dir, f'lora.pt')

            if lora_config_file.exists() and lora_checkpoint.exists():
                self.modules.append(LoraPeftModule(lora_dir, multiplier))
                return

        print(f">> Failed to load lora {name}")

    def load(self, pipe, dtype):
        if len(self.modules) > 0:
            for module in self.modules:
                pipe = module.apply(pipe, dtype)

        return pipe

    # Simple check to allow previous functionality
    def should_use(self, lora_weights: list):
        if len(lora_weights) > 0:
            for lora in lora_weights:
                lora_dir = Path(self.lora_path, lora.model)
                if lora_dir.exists():
                    lora_config_file = Path(lora_dir, f'lora_config.json')
                    lora_checkpoint = Path(lora_dir, f'lora.pt')
                    if lora_config_file.exists() and lora_checkpoint.exists():
                        return False

        return True
