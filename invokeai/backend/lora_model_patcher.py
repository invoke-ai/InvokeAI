
from diffusers.loaders.lora import LoraLoaderMixin
from contextlib import contextmanager
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.utils.peft_utils import recurse_remove_peft_layers
from typing import Iterator, Tuple, Union
from transformers import CLIPTextModel
from invokeai.backend.lora_model_raw import LoRAModelRaw


class LoraModelPatcher:

    @classmethod
    def unload_lora_from_model(cls, m: Union[UNet2DConditionModel, CLIPTextModel]):
        """Unload all LoRA models from a UNet or Text Encoder.
        This implementation is base on LoraLoaderMixin.unload_lora_weights().
        """
        recurse_remove_peft_layers(m)
        if hasattr(m, "peft_config"):
            del m.peft_config
        if hasattr(m, "_hf_peft_config_loaded"):
            m._hf_peft_config_loaded = None


    @classmethod
    @contextmanager
    def apply_lora_to_unet(cls, unet: UNet2DConditionModel, loras: Iterator[Tuple[LoRAModelRaw, float]])
        try:
            for lora, lora_weight in loras:
                LoraLoaderMixin.load_lora_into_unet(
                    state_dict=lora.state_dict,
                    network_alphas=lora.network_alphas,
                    unet=unet,
                    low_cpu_mem_usage=False,
                    adapter_name=lora.name,
                    _pipeline=None,
                )
            yield
        finally:
            pass
