from pathlib import Path
from typing import Optional

from invokeai.backend.llava_onevision_model import LlavaOnevisionModel
from invokeai.backend.model_manager.config import (
    AnyModel,
    AnyModelConfig,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.LlavaOnevision, format=ModelFormat.Diffusers)
class SpandrelImageToImageModelLoader(ModelLoader):
    """Class for loading LLaVA Onevision VLLM models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if submodel_type is not None:
            raise ValueError("Unexpected submodel requested for Spandrel model.")

        model_path = Path(config.path)
        model = LlavaOnevisionModel.load_from_path(model_path)
        model.to(dtype=self._torch_dtype)
        return model
