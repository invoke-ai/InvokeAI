from pathlib import Path
from typing import Optional

from transformers import CLIPVisionModelWithProjection

from invokeai.backend.model_manager.configs.base import Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import AnyModel, BaseModelType, ModelFormat, ModelType, SubModelType


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.CLIPVision, format=ModelFormat.Diffusers)
class ClipVisionLoader(ModelLoader):
    """Class to load CLIPVision models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Diffusers_Config_Base):
            raise ValueError("Only DiffusersConfigBase models are currently supported here.")

        if submodel_type is not None:
            raise Exception("There are no submodels in CLIP Vision models.")

        model_path = Path(config.path)

        model = CLIPVisionModelWithProjection.from_pretrained(
            model_path, torch_dtype=self._torch_dtype, local_files_only=True
        )
        assert isinstance(model, CLIPVisionModelWithProjection)

        return model
