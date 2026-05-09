"""Loader registrations for Wan 2.2 image-generation models.

Currently covers:
- Main: Diffusers format (T2V-A14B with dual experts via Transformer +
  Transformer2 submodels, plus TI2V-5B). Phase 4 will add a GGUFQuantized loader.
- WanT5Encoder: standalone UMT5-XXL encoder folder (``text_encoder/`` +
  ``tokenizer/`` subdirs, or a flat ``text_encoder/`` folder).
- VAE: handled in ``vae.py`` (registered for type=VAE generically).
"""

from pathlib import Path
from typing import Optional

import torch

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)


@ModelLoaderRegistry.register(base=BaseModelType.Wan, type=ModelType.Main, format=ModelFormat.Diffusers)
class WanDiffusersModel(GenericDiffusersLoader):
    """Loader for Wan 2.2 diffusers-format models (T2V-A14B and TI2V-5B).

    Forces bfloat16 for the transformer and VAE — fp16 is unstable on Wan VAE
    (same issue affects the Flux VAE). Resolves the appropriate Hugging Face
    class for each submodel via the parent loader's ``get_hf_load_class``.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, Checkpoint_Config_Base):
            raise NotImplementedError("Single-file checkpoint format is not yet supported for Wan models.")

        if submodel_type is None:
            raise Exception("A submodel type must be provided when loading Wan main pipelines.")

        model_path = Path(config.path)
        load_class = self.get_hf_load_class(model_path, submodel_type)
        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None
        model_path = model_path / submodel_type.value

        # bfloat16 across the board: matches Diffusers WanPipeline reference and
        # avoids the fp16 instability seen in the Wan VAE.
        dtype_kwarg = {"dtype": torch.bfloat16}
        try:
            result: AnyModel = load_class.from_pretrained(
                model_path,
                **dtype_kwarg,
                variant=variant,
                local_files_only=True,
            )
        except TypeError:
            # Older diffusers releases use torch_dtype instead of dtype.
            dtype_kwarg = {"torch_dtype": torch.bfloat16}
            result = load_class.from_pretrained(
                model_path,
                **dtype_kwarg,
                variant=variant,
                local_files_only=True,
            )
        except OSError as e:
            # Some Wan repos ship without a fp16 variant suffix on every submodel.
            # If the requested variant isn't on disk, fall back to the default weights.
            if variant and "no file named" in str(e):
                result = load_class.from_pretrained(model_path, **dtype_kwarg, local_files_only=True)
            else:
                raise

        return result


@ModelLoaderRegistry.register(
    base=BaseModelType.Any, type=ModelType.WanT5Encoder, format=ModelFormat.WanT5Encoder
)
class WanT5EncoderLoader(ModelLoader):
    """Loader for the standalone Wan UMT5-XXL encoder.

    Accepts two on-disk layouts:
    1. Parent dir with ``text_encoder/`` (and typically ``tokenizer/``) subdirs —
       what ``Wan-AI/Wan2.2-T2V-A14B::text_encoder+tokenizer`` produces.
    2. A flat ``text_encoder/`` folder with ``config.json`` declaring
       ``model_type: umt5`` directly at the root. In this case the tokenizer
       is loaded from the same folder via ``AutoTokenizer.from_pretrained``.
    """

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if submodel_type is None:
            raise ValueError("A submodel type (Tokenizer or TextEncoder) must be provided.")

        root = Path(config.path)
        nested_text_encoder = root / "text_encoder"
        nested_tokenizer = root / "tokenizer"

        if submodel_type == SubModelType.TextEncoder:
            from transformers import UMT5EncoderModel

            target = nested_text_encoder if nested_text_encoder.exists() else root
            return UMT5EncoderModel.from_pretrained(
                str(target),
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
        if submodel_type == SubModelType.Tokenizer:
            from transformers import AutoTokenizer

            # Prefer a sibling tokenizer/ directory; fall back to the encoder dir
            # itself, which is normal for "flat" downloads.
            target = nested_tokenizer if nested_tokenizer.exists() else (
                nested_text_encoder if nested_text_encoder.exists() else root
            )
            return AutoTokenizer.from_pretrained(str(target), local_files_only=True)

        raise ValueError(
            f"Unsupported submodel type for WanT5Encoder: {submodel_type.value if submodel_type else 'None'}"
        )
