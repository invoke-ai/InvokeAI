from abc import ABC
from pathlib import Path
from typing import (
    Literal,
    Self,
)

import torch
from pydantic import BaseModel, Field
from typing_extensions import Any

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
)


class TI_Config_Base(ABC, BaseModel):
    type: Literal[ModelType.TextualInversion] = Field(default=ModelType.TextualInversion)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk, path: Path | None = None) -> None:
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod, path)
        if expected_base is not recognized_base:
            raise NotAMatchError(f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _file_looks_like_embedding(cls, mod: ModelOnDisk, path: Path | None = None) -> bool:
        try:
            p = path or mod.path

            if not p.exists():
                return False

            if p.is_dir():
                return False

            if p.name in [f"learned_embeds.{s}" for s in mod.weight_files()]:
                return True

            state_dict = mod.load_state_dict(p)

            # Heuristic: textual inversion embeddings have these keys
            if any(key in {"string_to_param", "emb_params", "clip_g"} for key in state_dict.keys()):
                return True

            # Heuristic: small state dict with all tensor values
            if (len(state_dict)) < 10 and all(isinstance(v, torch.Tensor) for v in state_dict.values()):
                return True

            return False
        except Exception:
            return False

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk, path: Path | None = None) -> BaseModelType:
        p = path or mod.path

        try:
            state_dict = mod.load_state_dict(p)
        except Exception as e:
            raise NotAMatchError(f"unable to load state dict from {p}: {e}") from e

        try:
            if "string_to_token" in state_dict:
                token_dim = list(state_dict["string_to_param"].values())[0].shape[-1]
            elif "emb_params" in state_dict:
                token_dim = state_dict["emb_params"].shape[-1]
            elif "clip_g" in state_dict:
                token_dim = state_dict["clip_g"].shape[-1]
            else:
                token_dim = list(state_dict.values())[0].shape[0]
        except Exception as e:
            raise NotAMatchError(f"unable to determine token dimension from state dict in {p}: {e}") from e

        match token_dim:
            case 768:
                return BaseModelType.StableDiffusion1
            case 1024:
                return BaseModelType.StableDiffusion2
            case 1280:
                return BaseModelType.StableDiffusionXL
            case _:
                raise NotAMatchError(f"unrecognized token dimension {token_dim}")


class TI_File_Config_Base(TI_Config_Base):
    """Model config for textual inversion embeddings."""

    format: Literal[ModelFormat.EmbeddingFile] = Field(default=ModelFormat.EmbeddingFile)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        if not cls._file_looks_like_embedding(mod):
            raise NotAMatchError("model does not look like a textual inversion embedding file")

        cls._validate_base(mod)

        return cls(**override_fields)


class TI_File_SD1_Config(TI_File_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class TI_File_SD2_Config(TI_File_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class TI_File_SDXL_Config(TI_File_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class TI_Folder_Config_Base(TI_Config_Base):
    """Model config for textual inversion embeddings."""

    format: Literal[ModelFormat.EmbeddingFolder] = Field(default=ModelFormat.EmbeddingFolder)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        for p in mod.weight_files():
            if cls._file_looks_like_embedding(mod, p):
                cls._validate_base(mod, p)
                return cls(**override_fields)

        raise NotAMatchError("model does not look like a textual inversion embedding folder")


class TI_Folder_SD1_Config(TI_Folder_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class TI_Folder_SD2_Config(TI_Folder_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class TI_Folder_SDXL_Config(TI_Folder_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)
