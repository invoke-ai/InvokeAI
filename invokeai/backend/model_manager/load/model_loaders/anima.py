# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for Anima model loading in InvokeAI."""

from pathlib import Path
from typing import Optional

import accelerate

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Anima_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger

logger = InvokeAILogger.get_logger(__name__)


@ModelLoaderRegistry.register(base=BaseModelType.Anima, type=ModelType.Main, format=ModelFormat.Checkpoint)
class AnimaCheckpointModel(ModelLoader):
    """Class to load Anima transformer models from single-file checkpoints.

    The Anima checkpoint contains both the MiniTrainDIT backbone and the LLM Adapter
    under a shared `net.` prefix. The loader strips this prefix and instantiates
    the AnimaTransformer model with the correct architecture parameters.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Checkpoint_Config_Base):
            raise ValueError("Only CheckpointConfigBase models are currently supported here.")

        match submodel_type:
            case SubModelType.Transformer:
                return self._load_from_singlefile(config)

        raise ValueError(
            f"Only Transformer submodels are currently supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )

    def _load_from_singlefile(
        self,
        config: AnyModelConfig,
    ) -> AnyModel:
        from safetensors.torch import load_file

        from invokeai.backend.anima.anima_transformer import AnimaTransformer

        if not isinstance(config, Main_Checkpoint_Anima_Config):
            raise TypeError(
                f"Expected Main_Checkpoint_Anima_Config, got {type(config).__name__}. "
                "Model configuration type mismatch."
            )
        model_path = Path(config.path)

        # Load the state dict from safetensors
        sd = load_file(model_path)

        # Strip the `net.` prefix that all Anima checkpoint keys have
        # e.g., "net.blocks.0.self_attn.q_proj.weight" -> "blocks.0.self_attn.q_proj.weight"
        prefix_to_strip = None
        for prefix in ["net."]:
            if any(k.startswith(prefix) for k in sd.keys() if isinstance(k, str)):
                prefix_to_strip = prefix
                break

        if prefix_to_strip:
            stripped_sd = {}
            for key, value in sd.items():
                if isinstance(key, str) and key.startswith(prefix_to_strip):
                    stripped_sd[key[len(prefix_to_strip) :]] = value
                else:
                    stripped_sd[key] = value
            sd = stripped_sd

        # Create an empty AnimaTransformer with Anima's default architecture parameters
        with accelerate.init_empty_weights():
            model = AnimaTransformer(
                max_img_h=240,
                max_img_w=240,
                max_frames=1,
                in_channels=16,
                out_channels=16,
                patch_spatial=2,
                patch_temporal=1,
                concat_padding_mask=True,
                model_channels=2048,
                num_blocks=28,
                num_heads=16,
                mlp_ratio=4.0,
                crossattn_emb_channels=1024,
                pos_emb_cls="rope3d",
                use_adaln_lora=True,
                adaln_lora_dim=256,
                extra_per_block_abs_pos_emb=False,
                image_model="anima",
            )

        # Determine safe dtype
        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        # Handle memory management
        new_sd_size = sum(ten.nelement() * model_dtype.itemsize for ten in sd.values())
        self._ram_cache.make_room(new_sd_size)

        # Convert to target dtype (skip non-float tensors like embedding indices)
        for k in sd.keys():
            if sd[k].is_floating_point():
                sd[k] = sd[k].to(model_dtype)

        # Filter out rotary embedding inv_freq buffers that are regenerated at runtime
        keys_to_remove = [k for k in sd.keys() if k.endswith(".inv_freq")]
        for k in keys_to_remove:
            del sd[k]

        load_result = model.load_state_dict(sd, assign=True, strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint contains {len(load_result.unexpected_keys)} unexpected keys. "
                f"This may indicate a corrupted or incompatible checkpoint. "
                f"First 5 unexpected keys: {load_result.unexpected_keys[:5]}"
            )
        if load_result.missing_keys:
            logger.warning(
                f"Checkpoint is missing {len(load_result.missing_keys)} keys "
                f"(expected for inv_freq buffers). First 5: {load_result.missing_keys[:5]}"
            )
        return model
