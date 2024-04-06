import pathlib
from typing import Optional

import safetensors.torch
from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

from invokeai.backend.model_manager.config import AnyModelConfig, BaseModelType, ModelFormat, ModelType, SubModelType
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.util.devices import choose_torch_device

CLIP_VISION_STANDARD_CONFIG = {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "projection_dim": 512,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "num_channels": 3,
    "image_size": 224,
    "patch_size": 32,
    "hidden_act": "quick_gelu",
    "layer_norm_eps": 1e-05,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "initializer_factor": 1.0,
    "torch_dtype": "float16",
}


CLIP_VISION_VIT_H_CONFIG = {
    **CLIP_VISION_STANDARD_CONFIG,
    **{
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "projection_dim": 1024,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "patch_size": 14,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-05,
    },
}

CLIP_VISION_VIT_G_CONFIG = {
    **CLIP_VISION_STANDARD_CONFIG,
    **{
        "hidden_size": 1664,
        "intermediate_size": 8192,
        "projection_dim": 1280,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "patch_size": 14,
        "hidden_act": "gelu",
    },
}


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.CLIPVision, format=ModelFormat.Checkpoint)
class CLIPVisionModelLoader(ModelLoader):
    """Class to load CLIP Vision Checkpoint Models"""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> CLIPVisionModelWithProjection:
        model_path = pathlib.Path(config.path)
        clip_vision_state_dict = safetensors.torch.load_file(model_path, device=choose_torch_device().type)
        clip_vision_keys = clip_vision_state_dict.keys()

        if not any(key.startswith("vision_model.") for key in clip_vision_keys):
            raise RuntimeError("Not a recognized CLIP Vision model.")

        if "vision_model.encoder.layers.30.layer_norm1.weight" in clip_vision_keys:
            clip_config = CLIPVisionConfig(**CLIP_VISION_VIT_H_CONFIG)
        elif "vision_model.encoder.layers.47.layer_norm1.weight" in clip_vision_keys:
            clip_config = CLIPVisionConfig(**CLIP_VISION_VIT_G_CONFIG)
        else:
            raise RuntimeError("Unrecognized CLIP Vision Model. Failed to load.")

        clip_vision_model = CLIPVisionModelWithProjection(clip_config)
        clip_vision_model.load_state_dict(clip_vision_state_dict, strict=False)

        return clip_vision_model
