# Copyright (c) 2026, Lincoln D. Stein and the InvokeAI Development Team
"""Loader for Baidu ERNIE-Image diffusers pipelines."""

from pathlib import Path
from typing import Optional

from invokeai.backend.model_manager.configs.base import Diffusers_Config_Base
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

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading ERNIE-Image pipelines.")

        model_path = Path(config.path)
        load_class = self.get_hf_load_class(model_path, submodel_type)

        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None

        # The SubModelType values match the ERNIE-Image pipeline's subdir names 1:1, including
        # the prompt enhancer ("pe" / "pe_tokenizer").
        model_path = model_path / submodel_type.value

        # Tokenizers take neither a dtype nor a repo variant — mirror the sibling loaders and load
        # them bare. `local_files_only=True` everywhere keeps loading offline-safe: the files are
        # already on disk, and without it transformers/diffusers may reach out to the Hub to
        # validate the repo.
        if submodel_type in (SubModelType.Tokenizer, SubModelType.PromptEnhancerTokenizer):
            result: AnyModel = load_class.from_pretrained(model_path, local_files_only=True)
            return result

        target_device = TorchDevice.choose_torch_device()
        dtype = TorchDevice.choose_bfloat16_safe_dtype(target_device)
        try:
            result = load_class.from_pretrained(model_path, torch_dtype=dtype, variant=variant, local_files_only=True)
        except OSError as e:
            if variant and "no file named" in str(e):
                result = load_class.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True)
            else:
                raise

        return self._apply_fp8_layerwise_casting(result, config, submodel_type)
