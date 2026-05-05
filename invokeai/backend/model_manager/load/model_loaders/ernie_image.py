# Copyright (c) 2026, Lincoln D. Stein and the InvokeAI Development Team
"""Loader for Baidu ERNIE-Image diffusers pipelines."""

from pathlib import Path
from typing import Optional

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.util.devices import TorchDevice


@ModelLoaderRegistry.register(base=BaseModelType.ErnieImage, type=ModelType.Main, format=ModelFormat.Diffusers)
class ErnieImageDiffusersModel(GenericDiffusersLoader):
    """Loads ERNIE-Image submodels (transformer, vae, text_encoder, tokenizer, pe, pe_tokenizer)
    from a diffusers pipeline directory.
    """

    # Map our internal SubModelType values to the actual subdir names in the
    # ERNIE-Image diffusers layout. Most match 1:1, but the prompt-enhancer dirs
    # use short names ("pe" / "pe_tokenizer") in the upstream Baidu repos.
    _SUBDIR_OVERRIDES = {
        SubModelType.PromptEnhancer: "pe",
        SubModelType.PromptEnhancerTokenizer: "pe_tokenizer",
    }

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, Checkpoint_Config_Base):
            raise NotImplementedError("Single-file checkpoints are not yet supported for ERNIE-Image.")

        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading ERNIE-Image pipelines.")

        model_path = Path(config.path)
        load_class = self.get_hf_load_class(model_path, submodel_type)

        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None

        subdir = self._SUBDIR_OVERRIDES.get(submodel_type, submodel_type.value)
        model_path = model_path / subdir

        target_device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)
        try:
            return load_class.from_pretrained(model_path, torch_dtype=dtype, variant=variant)
        except OSError as e:
            if variant and "no file named" in str(e):
                return load_class.from_pretrained(model_path, torch_dtype=dtype)
            raise
