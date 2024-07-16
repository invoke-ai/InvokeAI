from pathlib import Path
from typing import Optional

import torch

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
from invokeai.backend.spandrel_image_to_image_model import SpandrelImageToImageModel


@ModelLoaderRegistry.register(
    base=BaseModelType.Any, type=ModelType.SpandrelImageToImage, format=ModelFormat.Checkpoint
)
class SpandrelImageToImageModelLoader(ModelLoader):
    """Class for loading Spandrel Image-to-Image models (i.e. models wrapped by spandrel.ImageModelDescriptor)."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if submodel_type is not None:
            raise ValueError("Unexpected submodel requested for Spandrel model.")

        model_path = Path(config.path)
        model = SpandrelImageToImageModel.load_from_file(model_path)

        torch_dtype = self._torch_dtype
        if not model.supports_dtype(torch_dtype):
            self._logger.warning(
                f"The configured dtype ('{self._torch_dtype}') is not supported by the {model.get_model_type_name()} "
                "model. Falling back to 'float32'."
            )
            torch_dtype = torch.float32
        model.to(dtype=torch_dtype)

        return model
