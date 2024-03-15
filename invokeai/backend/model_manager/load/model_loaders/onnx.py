# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for Onnx model loading in InvokeAI."""

# This should work the same as Stable Diffusion pipelines
from pathlib import Path
from typing import Optional

from invokeai.backend.model_manager import (
    AnyModelConfig,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.model_manager.any_model_type import AnyModel

from .. import ModelLoaderRegistry
from .generic_diffusers import GenericDiffusersLoader


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.ONNX, format=ModelFormat.ONNX)
@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.ONNX, format=ModelFormat.Olive)
class OnnyxDiffusersModel(GenericDiffusersLoader):
    """Class to load onnx models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not submodel_type is not None:
            raise Exception("A submodel type must be provided when loading onnx pipelines.")
        model_path = Path(config.path)
        load_class = self.get_hf_load_class(model_path, submodel_type)
        repo_variant = getattr(config, "repo_variant", None)
        variant = repo_variant.value if repo_variant else None
        model_path = model_path / submodel_type.value
        result: AnyModel = load_class.from_pretrained(
            model_path,
            torch_dtype=self._torch_dtype,
            variant=variant,
        )
        return result
