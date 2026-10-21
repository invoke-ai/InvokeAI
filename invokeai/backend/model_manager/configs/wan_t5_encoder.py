"""Configurations for the UMT5-XXL text encoder used by Wan 2.2.

Wan ships a UMT5-XXL encoder (not the more common T5-XXL). The two are not
weight-compatible — UMT5 has a different vocabulary and ``model_type``. We
register a dedicated config + ModelType so users can't accidentally wire a
FLUX/SD3-style T5-XXL into a Wan slot.

Two layouts are accepted: the diffusers folder, and a single llama.cpp GGUF
(``city96/umt5-xxl-encoder-gguf``) that carries its tokenizer in its metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    InvalidMatchError,
    NotAMatchError,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.model_manager.util.llamacpp_t5 import (
    BLOCK_COMPONENTS,
    TOKENIZER_SCALAR_FIELDS,
    has_per_layer_relative_bias,
    is_llamacpp_t5_encoder_state_dict,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor


def _read_text_encoder_model_type(mod: ModelOnDisk) -> str | None:
    """Return ``model_type`` from the encoder's ``config.json``.

    Diffusers encoder folders may live at the root (``config.json``) or under a
    ``text_encoder/`` subdirectory. UMT5-XXL sets ``model_type`` to ``"umt5"``;
    a regular T5-XXL would be ``"t5"``.
    """
    candidates: list[Path] = [mod.path / "text_encoder" / "config.json", mod.path / "config.json"]
    for path in candidates:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    config = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            mt = config.get("model_type")
            if isinstance(mt, str):
                return mt.lower()
    return None


class WanT5Encoder_WanT5Encoder_Config(Config_Base):
    """UMT5-XXL encoder in diffusers folder layout.

    Accepts either:
    - A directory containing ``text_encoder/`` (and typically ``tokenizer/``) ─ the
      shape produced by ``Wan-AI/Wan2.2-T2V-A14B::text_encoder+tokenizer``.
    - A bare ``text_encoder/`` directory whose own ``config.json`` declares
      ``model_type: umt5``.
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.WanT5Encoder] = Field(default=ModelType.WanT5Encoder)
    format: Literal[ModelFormat.WanT5Encoder] = Field(default=ModelFormat.WanT5Encoder)

    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)
        raise_for_override_fields(cls, override_fields)

        # Refuse to claim full Wan pipelines — they should match Main_Diffusers_Wan_Config.
        if (mod.path / "model_index.json").exists() or (mod.path / "transformer").exists():
            raise NotAMatchError(
                "directory looks like a full Wan pipeline (model_index.json or transformer/), "
                "not a standalone Wan T5 encoder"
            )

        model_type = _read_text_encoder_model_type(mod)
        if model_type is None:
            raise NotAMatchError("no encoder config.json found at root or text_encoder/")
        if model_type != "umt5":
            raise NotAMatchError(f"encoder model_type is {model_type!r}, not 'umt5'")

        return cls(**override_fields)


class WanT5Encoder_GGUF_Config(Checkpoint_Config_Base, Config_Base):
    """UMT5-XXL encoder in one llama.cpp ``t5encoder`` GGUF file, e.g. ``city96/umt5-xxl-encoder-gguf``.

    The naming is the one FLUX's T5 v1.1 GGUFs use (see ``util/llamacpp_t5.py``); a relative attention
    bias in blocks past the first is what makes the file UMT5. Once it is UMT5, a file the loader would
    fail on is refused at install rather than at the first prompt: an encoder-decoder conversion,
    missing tensors, or missing tokenizer metadata (a GGUF ships no tokenizer files beside it). Only the
    tokenizer's scalar fields are checked here; its vocabulary arrays, which ``ModelOnDisk.metadata``
    leaves out, are checked when the tokenizer is loaded.
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.WanT5Encoder] = Field(default=ModelType.WanT5Encoder)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)

    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)
        raise_for_override_fields(cls, override_fields)

        state_dict = mod.load_state_dict()
        if not is_llamacpp_t5_encoder_state_dict(state_dict):
            raise NotAMatchError("state dict does not look like a llama.cpp T5 encoder (no 'enc.blk.*' keys)")
        if not has_per_layer_relative_bias(state_dict):
            raise NotAMatchError("relative attention bias only in block 0: a T5 v1.1 encoder, not UMT5")
        if not any(isinstance(v, GGMLTensor) for v in state_dict.values()):
            raise NotAMatchError("state dict does not look like GGUF quantized")

        metadata = mod.metadata()
        architecture = metadata.get("general.architecture")
        if architecture != "t5encoder":
            raise InvalidMatchError(
                f"UMT5 GGUF has general.architecture {architecture!r}: only encoder-only 't5encoder' "
                "conversions, such as city96/umt5-xxl-encoder-gguf, are supported"
            )
        cls._raise_if_incomplete(state_dict, metadata)
        missing_tokenizer_fields = [name for name in TOKENIZER_SCALAR_FIELDS if name not in metadata]
        if missing_tokenizer_fields:
            raise InvalidMatchError(
                f"UMT5 GGUF carries no usable tokenizer: its metadata lacks {', '.join(missing_tokenizer_fields)}"
            )
        if metadata["tokenizer.ggml.model"] != "t5":
            raise InvalidMatchError(
                f"UMT5 GGUF declares a {metadata['tokenizer.ggml.model']!r} tokenizer, not the T5 Unigram one"
            )

        return cls(**override_fields)

    @staticmethod
    def _raise_if_incomplete(state_dict: dict[str | int, Any], metadata: dict[str, Any]) -> None:
        block_count = metadata.get("t5encoder.block_count")
        if not isinstance(block_count, int) or block_count < 1:
            raise InvalidMatchError("UMT5 GGUF is incomplete: its metadata has no t5encoder.block_count")
        expected = [
            "token_embd.weight",
            "enc.output_norm.weight",
            *(f"enc.blk.{block}.{component}.weight" for block in range(block_count) for component in BLOCK_COMPONENTS),
        ]
        missing = [key for key in expected if key not in state_dict]
        if missing:
            raise InvalidMatchError(
                f"UMT5 GGUF is incomplete: {len(missing)} of the tensors a {block_count}-block encoder needs are "
                f"missing, e.g. {missing[0]}"
            )
