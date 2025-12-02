# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for Z-Image model loading in InvokeAI."""

from pathlib import Path
from typing import Any, Optional

import accelerate
import torch

from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_GGUF_ZImage_Config
from invokeai.backend.model_manager.configs.qwen3_encoder import Qwen3Encoder_Qwen3Encoder_Config
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
from invokeai.backend.quantization.gguf.loaders import gguf_sd_loader


def _convert_z_image_gguf_to_diffusers(sd: dict[str, Any]) -> dict[str, Any]:
    """Convert Z-Image GGUF state dict keys to diffusers format.

    The GGUF format uses original model keys that differ from diffusers:
    - qkv.weight (fused) -> to_q.weight, to_k.weight, to_v.weight (split)
    - out.weight -> to_out.0.weight
    - q_norm.weight -> norm_q.weight
    - k_norm.weight -> norm_k.weight
    - x_embedder.* -> all_x_embedder.2-1.*
    - final_layer.* -> all_final_layer.2-1.*
    """
    new_sd: dict[str, Any] = {}

    for key, value in sd.items():
        if not isinstance(key, str):
            new_sd[key] = value
            continue

        # Handle x_embedder -> all_x_embedder.2-1
        if key.startswith("x_embedder."):
            suffix = key[len("x_embedder.") :]
            new_key = f"all_x_embedder.2-1.{suffix}"
            new_sd[new_key] = value
            continue

        # Handle final_layer -> all_final_layer.2-1
        if key.startswith("final_layer."):
            suffix = key[len("final_layer.") :]
            new_key = f"all_final_layer.2-1.{suffix}"
            new_sd[new_key] = value
            continue

        # Handle fused QKV weights - need to split
        if ".attention.qkv." in key:
            # Get the layer prefix and suffix
            prefix = key.rsplit(".attention.qkv.", 1)[0]
            suffix = key.rsplit(".attention.qkv.", 1)[1]  # "weight" or "bias"

            # Split the fused QKV tensor into Q, K, V
            tensor = value
            if hasattr(tensor, "shape"):
                dim = tensor.shape[0] // 3
                q = tensor[:dim]
                k = tensor[dim : 2 * dim]
                v = tensor[2 * dim :]

                new_sd[f"{prefix}.attention.to_q.{suffix}"] = q
                new_sd[f"{prefix}.attention.to_k.{suffix}"] = k
                new_sd[f"{prefix}.attention.to_v.{suffix}"] = v
            continue

        # Handle attention key renaming
        if ".attention." in key:
            new_key = key.replace(".q_norm.", ".norm_q.")
            new_key = new_key.replace(".k_norm.", ".norm_k.")
            new_key = new_key.replace(".attention.out.", ".attention.to_out.0.")
            new_sd[new_key] = value
            continue

        # For all other keys, just copy as-is
        new_sd[key] = value

    return new_sd


@ModelLoaderRegistry.register(base=BaseModelType.ZImage, type=ModelType.Main, format=ModelFormat.Diffusers)
class ZImageDiffusersModel(GenericDiffusersLoader):
    """Class to load Z-Image main models (Z-Image-Turbo, Z-Image-Base, Z-Image-Edit)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, Checkpoint_Config_Base):
            raise NotImplementedError("CheckpointConfigBase is not implemented for Z-Image models.")

        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading main pipelines.")

        model_path = Path(config.path)
        load_class = self.get_hf_load_class(model_path, submodel_type)
        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None
        model_path = model_path / submodel_type.value

        # Z-Image requires bfloat16 for correct inference.
        dtype = torch.bfloat16
        try:
            result: AnyModel = load_class.from_pretrained(
                model_path,
                torch_dtype=dtype,
                variant=variant,
            )
        except OSError as e:
            if variant and "no file named" in str(
                e
            ):  # try without the variant, just in case user's preferences changed
                result = load_class.from_pretrained(model_path, torch_dtype=dtype)
            else:
                raise e

        return result


@ModelLoaderRegistry.register(base=BaseModelType.ZImage, type=ModelType.Main, format=ModelFormat.GGUFQuantized)
class ZImageGGUFCheckpointModel(ModelLoader):
    """Class to load GGUF-quantized Z-Image transformer models."""

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
        from diffusers import ZImageTransformer2DModel

        assert isinstance(config, Main_GGUF_ZImage_Config)
        model_path = Path(config.path)

        # Load the GGUF state dict
        sd = gguf_sd_loader(model_path, compute_dtype=torch.bfloat16)

        # Some Z-Image GGUF models have keys prefixed with "diffusion_model."
        # Check if we need to strip this prefix
        has_prefix = any(k.startswith("diffusion_model.") for k in sd.keys() if isinstance(k, str))

        if has_prefix:
            stripped_sd = {}
            prefix = "diffusion_model."
            for key, value in sd.items():
                if isinstance(key, str) and key.startswith(prefix):
                    stripped_sd[key[len(prefix) :]] = value
                else:
                    stripped_sd[key] = value
            sd = stripped_sd

        # Convert GGUF format keys to diffusers format
        sd = _convert_z_image_gguf_to_diffusers(sd)

        # Create an empty model with the default Z-Image config
        # Z-Image-Turbo uses these default parameters from diffusers
        with accelerate.init_empty_weights():
            model = ZImageTransformer2DModel(
                all_patch_size=(2,),
                all_f_patch_size=(1,),
                in_channels=16,
                dim=3840,
                n_layers=30,
                n_refiner_layers=2,
                n_heads=30,
                n_kv_heads=30,
                norm_eps=1e-05,
                qk_norm=True,
                cap_feat_dim=2560,
                rope_theta=256.0,
                t_scale=1000.0,
                axes_dims=[32, 48, 48],
                axes_lens=[1024, 512, 512],
            )

        model.load_state_dict(sd, assign=True)
        return model


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.Qwen3Encoder, format=ModelFormat.Qwen3Encoder)
class Qwen3EncoderLoader(ModelLoader):
    """Class to load standalone Qwen3 Encoder models for Z-Image."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Qwen3Encoder_Qwen3Encoder_Config):
            raise ValueError("Only Qwen3Encoder_Qwen3Encoder_Config models are supported here.")

        match submodel_type:
            case SubModelType.Tokenizer:
                return AutoTokenizer.from_pretrained(Path(config.path) / "tokenizer")
            case SubModelType.TextEncoder:
                return Qwen2VLForConditionalGeneration.from_pretrained(
                    Path(config.path) / "text_encoder",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )

        raise ValueError(
            f"Only Tokenizer and TextEncoder submodels are supported. Received: {submodel_type.value if submodel_type else 'None'}"
        )
