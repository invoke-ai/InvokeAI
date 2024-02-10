# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for simple diffusers model loading in InvokeAI."""

from pathlib import Path
from typing import Optional

from invokeai.backend.model_manager import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelRepoVariant,
    ModelType,
    SubModelType,
)

from ..load_base import AnyModelLoader
from ..load_default import ModelLoader


@AnyModelLoader.register(base=BaseModelType.Any, type=ModelType.CLIPVision, format=ModelFormat.Diffusers)
@AnyModelLoader.register(base=BaseModelType.Any, type=ModelType.T2IAdapter, format=ModelFormat.Diffusers)
class GenericDiffusersLoader(ModelLoader):
    """Class to load simple diffusers models."""

    def _load_model(
        self,
        model_path: Path,
        model_variant: Optional[ModelRepoVariant] = None,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        model_class = self._get_hf_load_class(model_path)
        if submodel_type is not None:
            raise Exception(f"There are no submodels in models of type {model_class}")
        variant = model_variant.value if model_variant else None
        result: AnyModel = model_class.from_pretrained(model_path, torch_dtype=self._torch_dtype, variant=variant)  # type: ignore
        return result
