from contextlib import contextmanager
from typing import Iterator, Tuple, Union

from diffusers.loaders.lora import LoraLoaderMixin
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.utils.peft_utils import recurse_remove_peft_layers
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
            del m.peft_config  # type: ignore
        if hasattr(m, "_hf_peft_config_loaded"):
            m._hf_peft_config_loaded = None  # type: ignore

    @classmethod
    @contextmanager
    def apply_lora_to_unet(cls, unet: UNet2DConditionModel, loras: Iterator[Tuple[LoRAModelRaw, float]]):
        try:
            # TODO(ryand): Test speed of low_cpu_mem_usage=True.
            for lora, lora_weight in loras:
                LoraLoaderMixin.load_lora_into_unet(
                    state_dict=lora.state_dict,
                    network_alphas=lora.network_alphas,
                    unet=unet,
                    low_cpu_mem_usage=True,
                    adapter_name=lora.name,
                    _pipeline=None,
                )
            yield
        finally:
            cls.unload_lora_from_model(unet)

    @classmethod
    @contextmanager
    def apply_lora_to_text_encoder(
        cls, text_encoder: CLIPTextModel, loras: Iterator[Tuple[LoRAModelRaw, float]], prefix: str
    ):
        assert prefix in ["text_encoder", "text_encoder_2"]
        try:
            for lora, lora_weight in loras:
                # Filter the state_dict to only include the keys that start with the prefix.
                text_encoder_state_dict = {
                    key: value for key, value in lora.state_dict.items() if key.startswith(prefix + ".")
                }
                if len(text_encoder_state_dict) > 0:
                    LoraLoaderMixin.load_lora_into_text_encoder(
                        state_dict=text_encoder_state_dict,
                        network_alphas=lora.network_alphas,
                        text_encoder=text_encoder,
                        low_cpu_mem_usage=True,
                        adapter_name=lora.name,
                        _pipeline=None,
                    )
            yield
        finally:
            cls.unload_lora_from_model(text_encoder)
