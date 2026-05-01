import json
from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_override_fields,
    raise_if_not_dir,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType

_RECOGNIZED_TEXT_ENCODER_CLASSES = {
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
}


class QwenVLEncoder_Diffusers_Config(Config_Base):
    """Configuration for standalone Qwen2.5-VL encoder models in diffusers-style folder layout.

    Expected structure:
        <model_root>/
            text_encoder/
                config.json (with `_class_name` or `architectures` listing
                             `Qwen2_5_VLForConditionalGeneration`)
                model.safetensors
            tokenizer/
                tokenizer_config.json
                ...
            processor/                  (optional, for vision preprocessing)
                preprocessor_config.json

    This lets users avoid downloading the full ~40 GB Qwen Image diffusers pipeline
    when they only need the Qwen2.5-VL encoder for use with a GGUF transformer.
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.QwenVLEncoder] = Field(default=ModelType.QwenVLEncoder)
    format: Literal[ModelFormat.QwenVLEncoder] = Field(default=ModelFormat.QwenVLEncoder)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # Reject anything that looks like a full pipeline (those are matched as Main models).
        if (mod.path / "model_index.json").exists() or (mod.path / "transformer").exists():
            raise NotAMatchError(
                "directory looks like a full diffusers pipeline (has model_index.json or transformer folder), "
                "not a standalone Qwen VL encoder"
            )

        text_encoder_dir = mod.path / "text_encoder"
        tokenizer_dir = mod.path / "tokenizer"

        if not text_encoder_dir.is_dir():
            raise NotAMatchError("missing text_encoder/ subfolder")
        if not tokenizer_dir.is_dir():
            raise NotAMatchError("missing tokenizer/ subfolder")

        config_path = text_encoder_dir / "config.json"
        if not config_path.is_file():
            raise NotAMatchError(f"missing {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise NotAMatchError(f"could not read text_encoder/config.json: {e}") from e

        class_name = cfg.get("_class_name")
        architectures = cfg.get("architectures") or []
        candidates = {class_name, *architectures} - {None}

        if not candidates & _RECOGNIZED_TEXT_ENCODER_CLASSES:
            raise NotAMatchError(
                f"text_encoder class is {sorted(candidates) or 'unknown'}, "
                f"expected one of {sorted(_RECOGNIZED_TEXT_ENCODER_CLASSES)}"
            )

        return cls(**override_fields)
