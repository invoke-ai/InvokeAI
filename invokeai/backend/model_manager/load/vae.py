# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for VAE model loading in InvokeAI."""

from pathlib import Path
from typing import Dict, Optional

import torch

from invokeai.backend.model_manager import BaseModelType, ModelFormat, ModelRepoVariant, ModelType, SubModelType
from invokeai.backend.model_manager.load.load_base import AnyModelLoader
from invokeai.backend.model_manager.load.load_default import ModelLoader


@AnyModelLoader.register(base=BaseModelType.Any, type=ModelType.Vae, format=ModelFormat.Diffusers)
class VaeDiffusersModel(ModelLoader):
    """Class to load VAE models."""

    def _load_model(
        self,
        model_path: Path,
        model_variant: Optional[ModelRepoVariant] = None,
        submodel_type: Optional[SubModelType] = None,
    ) -> Dict[str, torch.Tensor]:
        if submodel_type is not None:
            raise Exception("There are no submodels in VAEs")
        vae_class = self._get_hf_load_class(model_path)
        variant = model_variant.value if model_variant else ""
        result: Dict[str, torch.Tensor] = vae_class.from_pretrained(
            model_path, torch_dtype=self._torch_dtype, variant=variant
        )  # type: ignore
        return result
