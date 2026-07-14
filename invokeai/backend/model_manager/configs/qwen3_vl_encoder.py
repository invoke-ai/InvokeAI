from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    get_config_dict_or_raise,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType

_KREA2_QWEN3_VL_HIDDEN_SIZE = 2560
_KREA2_QWEN3_VL_NUM_HIDDEN_LAYERS = 36


def _validate_krea2_qwen3_vl_config(config_path: Path) -> None:
    config = get_config_dict_or_raise(config_path)
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise NotAMatchError("Qwen3-VL text_config must be an object")
    hidden_size = text_config.get("hidden_size")
    num_hidden_layers = text_config.get("num_hidden_layers")
    if hidden_size != _KREA2_QWEN3_VL_HIDDEN_SIZE:
        raise NotAMatchError(
            f"Krea-2 requires the Qwen3-VL 4B hidden size {_KREA2_QWEN3_VL_HIDDEN_SIZE}, got {hidden_size}"
        )
    if num_hidden_layers != _KREA2_QWEN3_VL_NUM_HIDDEN_LAYERS:
        raise NotAMatchError(
            f"Krea-2 requires {_KREA2_QWEN3_VL_NUM_HIDDEN_LAYERS} Qwen3-VL layers, got {num_hidden_layers}"
        )


def _has_complete_pretrained_weights(weights_path: Path) -> bool:
    if (weights_path / "model.safetensors").is_file() or (weights_path / "pytorch_model.bin").is_file():
        return True

    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = weights_path / index_name
        if not index_path.is_file():
            continue
        index = get_config_dict_or_raise(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        filenames = list(weight_map.values())
        if not all(isinstance(filename, str) and filename for filename in filenames):
            return False
        root = weights_path.resolve()
        referenced_files: set[Path] = set()
        for filename in filenames:
            filename_path = Path(filename)
            if filename_path.is_absolute():
                return False
            candidate = (weights_path / filename_path).resolve()
            if not candidate.is_relative_to(root):
                return False
            referenced_files.add(candidate)
        return bool(referenced_files) and all(path.is_file() for path in referenced_files)
    return False


def _validate_krea2_qwen3_vl_checkpoint_shape(state_dict: dict[str | int, Any]) -> None:
    embed_keys = (
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
        "embed_tokens.weight",
    )
    embed = next((state_dict[key] for key in embed_keys if key in state_dict), None)
    shape = getattr(embed, "shape", ())
    if len(shape) < 2 or shape[1] != _KREA2_QWEN3_VL_HIDDEN_SIZE:
        hidden_size = shape[1] if len(shape) >= 2 else None
        raise NotAMatchError(
            f"Krea-2 requires a Qwen3-VL 4B checkpoint with hidden size "
            f"{_KREA2_QWEN3_VL_HIDDEN_SIZE}, got {hidden_size}"
        )
    if not any(isinstance(key, str) and ".layers.35." in key for key in state_dict):
        raise NotAMatchError("Krea-2 requires a Qwen3-VL 4B checkpoint containing language-model layer 35")


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
        _validate_krea2_qwen3_vl_config(expected_config_path)

        if config_path_nested.exists():
            weights_path = mod.path / "text_encoder"
            tokenizer_path = mod.path / "tokenizer"
        else:
            weights_path = mod.path
            tokenizer_path = mod.path

        has_weights = _has_complete_pretrained_weights(weights_path)
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

        if mod.path.suffix.lower() != ".safetensors":
            raise NotAMatchError(f"expected a .safetensors file, got {mod.path.suffix or '(no suffix)'}")

        state_dict = mod.load_state_dict()
        if not _is_qwen3_vl_encoder_state_dict(state_dict):
            raise NotAMatchError("state dict does not look like a single-file Qwen3-VL encoder")
        _validate_krea2_qwen3_vl_checkpoint_shape(state_dict)

        return cls(**override_fields)
