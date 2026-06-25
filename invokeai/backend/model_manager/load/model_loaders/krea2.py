# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for Krea-2 model loading in InvokeAI."""

from pathlib import Path
from typing import Any, Optional

import accelerate
from transformers import AutoTokenizer

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Krea2_Config
from invokeai.backend.model_manager.configs.qwen3_vl_encoder import Qwen3VLEncoder_Qwen3VLEncoder_Config
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.util.devices import TorchDevice

# Default Krea2Transformer2DModel config (from the Krea-2-Turbo transformer/config.json). Used when
# loading a bare single-file checkpoint that has no accompanying config.json.
KREA2_TRANSFORMER_CONFIG = {
    "attention_head_dim": 128,
    "axes_dims_rope": [32, 48, 48],
    "in_channels": 64,
    "intermediate_size": 16384,
    "norm_eps": 1e-05,
    "num_attention_heads": 48,
    "num_key_value_heads": 12,
    "num_layers": 28,
    "num_layerwise_text_blocks": 2,
    "num_refiner_text_blocks": 2,
    "num_text_layers": 12,
    "rope_theta": 1000.0,
    "text_hidden_dim": 2560,
    "text_intermediate_size": 6912,
    "text_num_attention_heads": 20,
    "text_num_key_value_heads": 20,
    "timestep_embed_dim": 256,
}


@ModelLoaderRegistry.register(base=BaseModelType.Krea2, type=ModelType.Main, format=ModelFormat.Diffusers)
class Krea2DiffusersModel(GenericDiffusersLoader):
    """Class to load Krea-2 main models (Krea-2-Turbo) in diffusers format.

    Loads every submodel (transformer, vae, text_encoder, tokenizer, scheduler) from the diffusers
    pipeline folder via the class names declared in model_index.json. The transformer resolves to
    diffusers' ``Krea2Transformer2DModel`` (only available in diffusers main / >=0.39); the VAE to
    ``AutoencoderKLQwenImage`` and the text encoder to ``Qwen3VLModel``.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, Checkpoint_Config_Base):
            raise NotImplementedError("CheckpointConfigBase is not implemented for the Krea-2 diffusers loader.")

        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading main pipelines.")

        model_path = Path(config.path)

        # model_index.json declares the tokenizer as the slow `Qwen2Tokenizer`, which requires
        # vocab.json/merges.txt. Krea-2 ships only a fast tokenizer.json, so load via AutoTokenizer
        # (which resolves to Qwen2TokenizerFast from tokenizer.json).
        #
        # Krea-2's tokenizer_config.json stores `extra_special_tokens` as a list (the special tokens
        # are already baked into tokenizer.json as added tokens). Newer transformers expects a dict and
        # crashes on the list, so override it with an empty dict — the special tokens are still
        # recognized from tokenizer.json.
        if submodel_type is SubModelType.Tokenizer:
            return AutoTokenizer.from_pretrained(
                model_path / submodel_type.value, local_files_only=True, extra_special_tokens={}
            )

        load_class = self.get_hf_load_class(model_path, submodel_type)
        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None
        model_path = model_path / submodel_type.value

        # Krea-2 prefers bfloat16; use a safe dtype based on target device capabilities.
        target_device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        extra_kwargs: dict[str, Any] = {}
        if submodel_type is SubModelType.TextEncoder:
            # Krea-2's Qwen3-VL text_encoder config stores rope settings under `rope_parameters`, but the
            # installed transformers' Qwen3VL rotary embedding reads `rope_scaling` (None here) → crash.
            # Patch the config so rope_scaling mirrors rope_parameters before instantiating the model.
            from transformers import AutoConfig

            te_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
            text_config = getattr(te_config, "text_config", None)
            if text_config is not None:
                rope_params = getattr(text_config, "rope_parameters", None)
                if getattr(text_config, "rope_scaling", None) is None and rope_params is not None:
                    text_config.rope_scaling = rope_params
            extra_kwargs["config"] = te_config

        try:
            result: AnyModel = load_class.from_pretrained(
                model_path,
                torch_dtype=dtype,
                variant=variant,
                **extra_kwargs,
            )
        except OSError as e:
            if variant and "no file named" in str(e):
                # try without the variant, just in case the user's preferences changed
                result = load_class.from_pretrained(model_path, torch_dtype=dtype, **extra_kwargs)
            else:
                raise e

        result = self._apply_fp8_layerwise_casting(result, config, submodel_type)
        return result


@ModelLoaderRegistry.register(base=BaseModelType.Krea2, type=ModelType.Main, format=ModelFormat.Checkpoint)
class Krea2CheckpointModel(ModelLoader):
    """Class to load Krea-2 transformer models from single-file checkpoints (safetensors).

    NOTE: the official Krea-2-Turbo release ships only in (sharded) diffusers format. This single-file
    path mirrors the Z-Image checkpoint loader and uses the known transformer config; it has not been
    validated against a real single-file Krea-2 checkpoint.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Checkpoint_Config_Base):
            raise ValueError("Only CheckpointConfigBase models are supported here.")

        if submodel_type is not SubModelType.Transformer:
            raise ValueError(
                f"Only Transformer submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
            )
        return self._load_from_singlefile(config)

    def _load_from_singlefile(self, config: AnyModelConfig) -> AnyModel:
        from diffusers import Krea2Transformer2DModel
        from safetensors.torch import load_file

        if not isinstance(config, Main_Checkpoint_Krea2_Config):
            raise TypeError(f"Expected Main_Checkpoint_Krea2_Config, got {type(config).__name__}.")
        model_path = Path(config.path)

        sd = load_file(model_path)

        # Strip ComfyUI-style key prefixes if present.
        prefix_to_strip = None
        for prefix in ("model.diffusion_model.", "diffusion_model."):
            if any(k.startswith(prefix) for k in sd.keys() if isinstance(k, str)):
                prefix_to_strip = prefix
                break
        if prefix_to_strip:
            sd = {
                (k[len(prefix_to_strip) :] if isinstance(k, str) and k.startswith(prefix_to_strip) else k): v
                for k, v in sd.items()
            }

        target_device = TorchDevice.choose_torch_device()
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)

        with accelerate.init_empty_weights():
            model = Krea2Transformer2DModel(**KREA2_TRANSFORMER_CONFIG)

        new_sd_size = sum(ten.nelement() * model_dtype.itemsize for ten in sd.values())
        self._ram_cache.make_room(new_sd_size)
        for k in sd.keys():
            sd[k] = sd[k].to(model_dtype)

        model.load_state_dict(sd, assign=True, strict=False)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen3VLEncoder, format=ModelFormat.Qwen3VLEncoder)
class Qwen3VLEncoderLoader(ModelLoader):
    """Class to load standalone Qwen3-VL text encoder models for Krea-2 (directory format)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        from transformers import Qwen3VLModel

        if not isinstance(config, Qwen3VLEncoder_Qwen3VLEncoder_Config):
            raise ValueError("Only Qwen3VLEncoder_Qwen3VLEncoder_Config models are supported here.")

        model_path = Path(config.path)

        # Support both a full pipeline-style layout (text_encoder/ + tokenizer/) and a standalone
        # download where the encoder files live directly at the root.
        text_encoder_path = model_path / "text_encoder"
        tokenizer_path = model_path / "tokenizer"
        is_standalone = not text_encoder_path.exists() and (model_path / "config.json").exists()
        if is_standalone:
            text_encoder_path = model_path
            tokenizer_path = model_path

        match submodel_type:
            case SubModelType.Tokenizer:
                # extra_special_tokens={} works around Krea-2's list-format tokenizer_config (see
                # Krea2DiffusersModel); harmless for well-formed configs.
                return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, extra_special_tokens={})
            case SubModelType.TextEncoder:
                target_device = TorchDevice.choose_torch_device()
                model_dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)
                return Qwen3VLModel.from_pretrained(
                    text_encoder_path,
                    torch_dtype=model_dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are supported. "
            f"Received: {submodel_type.value if submodel_type else 'None'}"
        )
