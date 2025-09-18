# Copyright (c) 2024, Brandon W. Rising and the InvokeAI Development Team
"""Class for Qwen-Image model loading in InvokeAI."""

from pathlib import Path
from typing import Optional

from diffusers import DiffusionPipeline

from invokeai.backend.model_manager.config import AnyModelConfig, MainDiffusersConfig
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_util import calc_model_size_by_fs
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)


@ModelLoaderRegistry.register(base=BaseModelType.QwenImage, type=ModelType.Main, format=ModelFormat.Diffusers)
class QwenImageLoader(ModelLoader):
    """Class to load Qwen-Image models."""

    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        """Calculate the size of the Qwen-Image model on disk."""
        if not isinstance(config, MainDiffusersConfig):
            raise ValueError("Only MainDiffusersConfig models are currently supported here.")

        # For Qwen-Image, we need to calculate the size of the entire model or specific submodels
        return calc_model_size_by_fs(
            model_path=model_path,
            subfolder=submodel_type.value if submodel_type else None,
            variant=config.repo_variant.value if config.repo_variant else None,
        )

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, MainDiffusersConfig):
            raise ValueError("Only MainDiffusersConfig models are currently supported here.")

        if config.base != BaseModelType.QwenImage:
            raise ValueError("This loader only supports Qwen-Image models.")

        model_path = Path(config.path)

        if submodel_type is not None:
            # Load individual submodel components with memory optimizations
            import torch
            from diffusers import QwenImageTransformer2DModel
            from diffusers.models import AutoencoderKLQwenImage

            # Force bfloat16 for memory efficiency if not already set
            torch_dtype = self._torch_dtype if self._torch_dtype is not None else torch.bfloat16

            # Load only the specific submodel, not the entire pipeline
            if submodel_type == SubModelType.VAE:
                # Load VAE directly from subfolder
                vae_path = model_path / "vae"
                if vae_path.exists():
                    return AutoencoderKLQwenImage.from_pretrained(
                        vae_path,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                    )
            elif submodel_type == SubModelType.Transformer:
                # Load transformer directly from subfolder
                transformer_path = model_path / "transformer"
                if transformer_path.exists():
                    return QwenImageTransformer2DModel.from_pretrained(
                        transformer_path,
                        torch_dtype=torch_dtype,
                        low_cpu_mem_usage=True,
                    )

            # Fallback to loading full pipeline if direct loading fails
            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                variant=config.repo_variant.value if config.repo_variant else None,
                low_cpu_mem_usage=True,
            )

            # Return the specific submodel
            if hasattr(pipeline, submodel_type.value):
                return getattr(pipeline, submodel_type.value)
            else:
                raise ValueError(f"Submodel {submodel_type} not found in Qwen-Image pipeline.")
        else:
            # Load the full pipeline with memory optimizations
            import torch

            # Force bfloat16 for memory efficiency if not already set
            torch_dtype = self._torch_dtype if self._torch_dtype is not None else torch.bfloat16

            pipeline = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                variant=config.repo_variant.value if config.repo_variant else None,
                low_cpu_mem_usage=True,  # Important for reducing memory during loading
            )
            return pipeline
