from pathlib import Path
from typing import Optional

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
from invokeai.backend.sig_lip.sig_lip_pipeline import SigLipPipeline


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.SigLIP, format=ModelFormat.Diffusers)
class SigLIPModelLoader(ModelLoader):
    """Class for loading SigLIP models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if submodel_type is not None:
            raise ValueError("Unexpected submodel requested for LLaVA OneVision model.")

        model_path = Path(config.path)
        model = SigLipPipeline.load_from_path(model_path)
        model.to(dtype=self._torch_dtype)
        return model
