import json
from pathlib import Path
from typing import Any, Iterable, Literal, Self

from pydantic import Field
from safetensors import safe_open

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType

_RECOGNIZED_TEXT_ENCODER_CLASSES = {
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
}


def _has_qwen_vl_keys(keys: Iterable[str]) -> bool:
    """A Qwen2.5-VL/Qwen2-VL checkpoint must have both LM weights and a visual
    tower — that's what distinguishes it from text-only Qwen3/Qwen2 encoders."""
    has_lm = False
    has_vision = False
    for k in keys:
        if not isinstance(k, str):
            continue
        if not has_lm and (k == "model.embed_tokens.weight" or k.startswith("model.layers.")):
            has_lm = True
        if not has_vision and (k.startswith("visual.patch_embed.") or k.startswith("visual.blocks.")):
            has_vision = True
        if has_lm and has_vision:
            return True
    return False


def _read_safetensors_keys(path: Path) -> list[str]:
    """Read only the key index from a safetensors file without loading tensor data.

    Avoids holding multi-GB encoder weights in RAM just to classify the file.
    """
    with safe_open(str(path), framework="pt", device="cpu") as f:
        return list(f.keys())


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


class QwenVLEncoder_Checkpoint_Config(Checkpoint_Config_Base, Config_Base):
    """Configuration for single-file Qwen2.5-VL encoder checkpoints (safetensors).

    This matches ComfyUI-style consolidated single-file encoders such as
    `qwen_2.5_vl_7b_fp8_scaled.safetensors`, which bundle the language model
    and the visual tower into one file (typically with FP8 + per-tensor
    `weight_scale` ComfyUI quantization).

    The matching tokenizer + processor are pulled from HuggingFace
    (`Qwen/Qwen2.5-VL-7B-Instruct`) on first use and cached for offline use.
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.QwenVLEncoder] = Field(default=ModelType.QwenVLEncoder)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        # Only safetensors checkpoints are supported as single-file Qwen VL encoders.
        # Reject other extensions cheaply before attempting to read keys.
        if mod.path.suffix != ".safetensors":
            raise NotAMatchError(f"expected a .safetensors file, got {mod.path.suffix or '(no suffix)'}")

        # Read only the key index — a 7GB fp8 encoder weighs ~7GB on disk, but we
        # only need the key names to classify it, not the tensor data.
        try:
            keys = _read_safetensors_keys(mod.path)
        except Exception as e:
            raise NotAMatchError(f"could not read safetensors header: {e}") from e

        if not _has_qwen_vl_keys(keys):
            raise NotAMatchError("state dict does not look like a Qwen2.5-VL/Qwen2-VL checkpoint")

        return cls(**override_fields)
