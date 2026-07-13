from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
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

        if config_path_nested.exists():
            weights_path = mod.path / "text_encoder"
            tokenizer_path = mod.path / "tokenizer"
        else:
            weights_path = mod.path
            tokenizer_path = mod.path

        has_weights = any(weights_path.glob("*.safetensors")) or any(weights_path.glob("*.bin"))
        has_tokenizer = (tokenizer_path / "tokenizer.json").exists() or (
            (tokenizer_path / "vocab.json").exists() and (tokenizer_path / "merges.txt").exists()
        )
        if not has_weights:
            raise NotAMatchError("standalone Qwen3-VL encoder directory does not contain model weights")
        if not has_tokenizer:
            raise NotAMatchError("standalone Qwen3-VL encoder directory does not contain tokenizer files")

        return cls(**override_fields)


def _is_qwen3_vl_encoder_state_dict(state_dict: dict[str | int, Any]) -> bool:
    """True for a single-file Qwen3-VL encoder: a Qwen3 text decoder PLUS a visual tower.

    The visual tower (``visual.*`` / ``model.visual.*``) distinguishes Qwen3-VL from the text-only
    ``Qwen3Encoder`` (Z-Image / FLUX.2 Klein), which has ``model.layers.*`` but no visual tower.
    """
    str_keys = [k for k in state_dict if isinstance(k, str)]
    has_text_decoder = any(".layers." in k and ("model." in k or k.startswith("layers.")) for k in str_keys)
    has_visual_tower = any(k.startswith(("visual.", "model.visual.")) or ".visual." in k for k in str_keys)
    return has_text_decoder and has_visual_tower


class Qwen3VLEncoder_Checkpoint_Config(Checkpoint_Config_Base, Config_Base):
    """Configuration for a single-file Qwen3-VL text encoder checkpoint (e.g. ComfyUI ``qwen3vl_4b_*``).

    Distinguished from the text-only ``Qwen3Encoder`` checkpoint (Z-Image) by the presence of the
    Qwen3-VL visual tower. The tokenizer is not bundled in single-file checkpoints and is pulled from
    HuggingFace (``Qwen/Qwen3-VL-4B-Instruct``) by the loader.
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.Qwen3VLEncoder] = Field(default=ModelType.Qwen3VLEncoder)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        if not _is_qwen3_vl_encoder_state_dict(mod.load_state_dict()):
            raise NotAMatchError("state dict does not look like a single-file Qwen3-VL encoder")

        return cls(**override_fields)
