"""Configurations for the UMT5-XXL text encoder used by Wan 2.2.

Wan ships a UMT5-XXL encoder (not the more common T5-XXL). The two are not
weight-compatible — UMT5 has a different vocabulary and ``model_type``. We
register a dedicated config + ModelType so users can't accidentally wire a
FLUX/SD3-style T5-XXL into a Wan slot.

For Phase 3 we accept the diffusers-folder layout only. Single-file UMT5
checkpoints are uncommon; if they show up later, a checkpoint config can be
added alongside this one.
"""

from __future__ import annotations

import json
from pathlib import Path
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
