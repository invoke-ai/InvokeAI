from pathlib import Path
from typing import Optional

from invokeai.backend.model_manager.config import (
    AnyModelConfig,
    CheckpointConfigBase,
    DiffusersConfigBase,
    ControlNetDiffusersConfig,
    ControlNetCheckpointConfig,
)
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)


@ModelLoaderRegistry.register(base=BaseModelType.Bria, type=ModelType.ControlNet, format=ModelFormat.Diffusers)
class BriaControlNetDiffusersModel(GenericDiffusersLoader):
    """Class to load Bria control net models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, ControlNetCheckpointConfig):
            raise NotImplementedError("CheckpointConfigBase is not implemented for Bria models.")

        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading control net pipelines.")

        model_path = Path(config.path)
        load_class = self.get_hf_load_class(model_path, submodel_type)
        repo_variant = config.repo_variant if isinstance(config, ControlNetDiffusersConfig) else None
        variant = repo_variant.value if repo_variant else None
        model_path = model_path / submodel_type.value

        dtype = self._torch_dtype

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

@ModelLoaderRegistry.register(base=BaseModelType.Bria, type=ModelType.Main, format=ModelFormat.Diffusers)
class BriaDiffusersModel(GenericDiffusersLoader):
    """Class to load Bria main models."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, CheckpointConfigBase):
            raise NotImplementedError("CheckpointConfigBase is not implemented for Bria models.")

        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading main pipelines.")

        model_path = Path(config.path)
        load_class = self.get_hf_load_class(model_path, submodel_type)
        repo_variant = config.repo_variant if isinstance(config, DiffusersConfigBase) else None
        variant = repo_variant.value if repo_variant else None
        model_path = model_path / submodel_type.value

        dtype = self._torch_dtype
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
