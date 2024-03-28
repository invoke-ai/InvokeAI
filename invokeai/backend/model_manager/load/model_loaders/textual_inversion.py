# Copyright (c) 2024, Lincoln D. Stein and the InvokeAI Development Team
"""Class for TI model loading in InvokeAI."""

from pathlib import Path
from typing import Optional

from invokeai.backend.model_manager import (
    AnyModel,
    AnyModelConfig,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.textual_inversion import TextualInversionModelRaw

from .. import ModelLoader, ModelLoaderRegistry


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.TextualInversion, format=ModelFormat.EmbeddingFile)
@ModelLoaderRegistry.register(
    base=BaseModelType.Any, type=ModelType.TextualInversion, format=ModelFormat.EmbeddingFolder
)
class TextualInversionLoader(ModelLoader):
    """Class to load TI models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if submodel_type is not None:
            raise ValueError("There are no submodels in a TI model.")
        model = TextualInversionModelRaw.from_checkpoint(
            file_path=self._get_model_path(config),
            dtype=self._torch_dtype,
        )
        return model

    # override
    def _get_model_path(self, config: AnyModelConfig) -> Path:
        model_path = self._app_config.models_path / config.path

        if config.format == ModelFormat.EmbeddingFolder:
            path = model_path / "learned_embeds.bin"
        else:
            path = model_path

        if not path.exists():
            raise OSError(f"The embedding file at {path} was not found")

        return path
