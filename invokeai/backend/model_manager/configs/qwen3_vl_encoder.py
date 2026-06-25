from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType


class Qwen3VLEncoder_Qwen3VLEncoder_Config(Config_Base):
    """Configuration for standalone Qwen3-VL text encoder models (diffusers-like directory format).

    Used by Krea-2, whose text conditioning comes from a Qwen3-VL model (``Qwen3VLModel``). The model
    weights are expected either in a ``text_encoder`` subfolder of the model directory or directly at the
    root (standalone download). This is distinct from the text-only ``Qwen3Encoder`` (Z-Image / FLUX.2
    Klein) and the Qwen2.5-VL ``QwenVLEncoder`` (Qwen Image).
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.Qwen3VLEncoder] = Field(default=ModelType.Qwen3VLEncoder)
    format: Literal[ModelFormat.Qwen3VLEncoder] = Field(default=ModelFormat.Qwen3VLEncoder)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # Exclude full pipeline models - these should be matched as main models, not just encoders.
        model_index_path = mod.path / "model_index.json"
        transformer_path = mod.path / "transformer"
        if model_index_path.exists() or transformer_path.exists():
            raise NotAMatchError(
                "directory looks like a full diffusers pipeline (has model_index.json or transformer folder), "
                "not a standalone Qwen3-VL encoder"
            )

        # Support both a nested text_encoder/config.json and a standalone config.json at the root.
        config_path_nested = mod.path / "text_encoder" / "config.json"
        config_path_direct = mod.path / "config.json"

        if config_path_nested.exists():
            expected_config_path = config_path_nested
        elif config_path_direct.exists():
            expected_config_path = config_path_direct
        else:
            raise NotAMatchError(f"unable to load config file: {config_path_nested} does not exist")

        # Qwen3-VL uses the Qwen3VLModel / Qwen3VLForConditionalGeneration architecture.
        raise_for_class_name(
            expected_config_path,
            {
                "Qwen3VLModel",
                "Qwen3VLForConditionalGeneration",
            },
        )

        return cls(**override_fields)
