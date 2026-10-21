import json
from pathlib import Path
from typing import Any, Iterable, Literal, Self

from pydantic import Field
from safetensors import safe_open

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
from invokeai.backend.quantization.gguf.loaders import WrappedGGUFReader

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

    The tokenizer, architecture config and image preprocessor config all ship with InvokeAI
    (vendored from the Apache-2.0 `Qwen/Qwen2.5-VL-7B-Instruct` release), so this encoder loads
    without network access.
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


# llama.cpp names both Qwen2-VL and Qwen2.5-VL language models "qwen2vl".
_QWEN_VL_GGUF_ARCHITECTURE = "qwen2vl"
# llama.cpp writes the vision tower to a companion `mmproj-*.gguf` under the generic "clip" architecture; the
# projector names whose tower it is.
_QWEN_VL_MMPROJ_PROJECTORS = frozenset({"qwen2vl_merger", "qwen2.5vl_merger"})
# The geometry of the vendored Qwen2.5-VL-7B-Instruct config, which is what the loader builds. Qwen-Image's
# transformer projects a 3584-wide prompt, so no other size could condition it; a resized vocabulary would fail the
# embedding's load.
_QWEN_IMAGE_ENCODER_HIDDEN_SIZE = 3584
_QWEN_IMAGE_ENCODER_NUM_LAYERS = 28
_QWEN_IMAGE_ENCODER_VOCAB_SIZE = 152064
# Qwen2-VL-7B is the same architecture name, geometry and tensor set as Qwen2.5-VL-7B, so it would load and condition
# Qwen-Image with the wrong model. Its context length is what tells it apart: 32768 in every Qwen2-VL release,
# 128000 in Qwen2.5-VL. Only Qwen2-VL's value is refused, not everything but Qwen2.5-VL's: a finetune or conversion
# may change the context length for its own reasons, and refusing it would turn away a working Qwen2.5-VL encoder.
_QWEN2_VL_CONTEXT_LENGTH = 32768
_GGUF_EMBED_KEY = "token_embd.weight"
_GGUF_TOP_LEVEL_TENSORS = (_GGUF_EMBED_KEY, "output_norm.weight")
# Every tensor of one decoder block the loader fills. `output.weight` (the LM head) is deliberately not required:
# the loader never reads it.
_GGUF_BLOCK_TENSORS = (
    "attn_q.weight",
    "attn_q.bias",
    "attn_k.weight",
    "attn_k.bias",
    "attn_v.weight",
    "attn_v.bias",
    "attn_output.weight",
    "attn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
    "ffn_norm.weight",
)
_EDIT_NEEDS_VISION_TOWER = (
    "Qwen-Image Edit with reference images needs the vision tower, which only the safetensors or Diffusers "
    "encoder carries (e.g. the 'Qwen2.5-VL Encoder (fp8 scaled)' starter model)."
)


def _raise_if_qwen_vl_mmproj(mod: ModelOnDisk) -> None:
    """Refuse the companion vision tower with the reason, rather than letting it register as an unknown model.

    The GGUF repos ship it next to the language model, so installing one of them hands the user both files.
    """
    if mod.metadata().get("clip.projector_type") not in _QWEN_VL_MMPROJ_PROJECTORS:
        return
    raise InvalidMatchError(
        "this is the Qwen2-VL / Qwen2.5-VL vision tower ('mmproj'), not a text encoder. Only text-only GGUF encoders "
        "are supported: install the language model instead (e.g. 'Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf'). "
        + _EDIT_NEEDS_VISION_TOWER
    )


def _gguf_context_length(path: Path) -> int | None:
    """The file's `qwen2vl.context_length`, or None when it has none or cannot be read.

    Read with a reader of its own because `ModelOnDisk.metadata()` keeps only string fields. That costs one more
    header parse, paid only by files whose architecture already matched.
    """
    try:
        with WrappedGGUFReader(path) as reader:
            field = reader.fields.get(f"{_QWEN_VL_GGUF_ARCHITECTURE}.context_length")
            value = field.contents() if field is not None else None
    except Exception:
        return None
    return value if isinstance(value, int) else None


def _raise_if_qwen2_vl(mod: ModelOnDisk) -> None:
    if _gguf_context_length(mod.path) != _QWEN2_VL_CONTEXT_LENGTH:
        return
    raise InvalidMatchError(
        f"this GGUF is Qwen2-VL, not Qwen2.5-VL, which Qwen-Image uses: its context length is "
        f"{_QWEN2_VL_CONTEXT_LENGTH}, Qwen2-VL's. The two share an architecture and tensor layout, so it would load "
        "and condition Qwen-Image with the wrong model. Install a Qwen2.5-VL-7B-Instruct GGUF instead."
    )


def _raise_if_not_a_complete_qwen_image_encoder(state_dict: dict[str | int, Any]) -> None:
    """Refuse a qwen2vl GGUF that is not, in full, the 7B encoder Qwen-Image conditions on.

    Judged from the tensors rather than the `block_count`/`embedding_length` metadata: a later part of a
    multi-part quant and an interrupted download both carry the complete metadata and only some of the tensors.

    `InvalidMatchError`: the architecture metadata already identified the file, so "recognised but unusable" is
    the verdict, and it is the only one the user sees. A `NotAMatchError` would register the file as an Unknown
    model, since `allow_unknown_models` defaults to True.
    """
    embed = state_dict.get(_GGUF_EMBED_KEY)
    if embed is None:
        raise InvalidMatchError(
            f"a Qwen2.5-VL encoder GGUF must contain '{_GGUF_EMBED_KEY}'; this file does not. Multi-part quants "
            "('...-00002-of-00003.gguf') must be joined before installing."
        )
    shape = tuple(getattr(embed, "shape", ()))
    vocab_size, hidden_size = shape if len(shape) == 2 else (None, None)
    layers = {
        int(key.split(".")[1])
        for key in state_dict
        if isinstance(key, str) and key.startswith("blk.") and key.split(".")[1].isdigit()
    }
    num_layers = max(layers, default=-1) + 1
    if (
        hidden_size != _QWEN_IMAGE_ENCODER_HIDDEN_SIZE
        or vocab_size != _QWEN_IMAGE_ENCODER_VOCAB_SIZE
        or num_layers > _QWEN_IMAGE_ENCODER_NUM_LAYERS
    ):
        raise InvalidMatchError(
            f"this Qwen2-VL GGUF is {hidden_size} wide with {num_layers} layers and a {vocab_size}-token vocabulary; "
            f"Qwen-Image conditions only on the 7B encoder ({_QWEN_IMAGE_ENCODER_HIDDEN_SIZE} wide, "
            f"{_QWEN_IMAGE_ENCODER_NUM_LAYERS} layers, {_QWEN_IMAGE_ENCODER_VOCAB_SIZE} tokens)."
        )

    expected = {
        *_GGUF_TOP_LEVEL_TENSORS,
        *(f"blk.{layer}.{name}" for layer in range(_QWEN_IMAGE_ENCODER_NUM_LAYERS) for name in _GGUF_BLOCK_TENSORS),
    }
    missing = sorted(expected - state_dict.keys())
    if missing:
        raise InvalidMatchError(
            f"this Qwen2.5-VL encoder GGUF is incomplete: {len(missing)} tensor(s) are missing, e.g. "
            f"{missing[:3]}. Multi-part quants must be joined before installing, and an interrupted download "
            "must be re-fetched."
        )


class QwenVLEncoder_GGUF_Config(Checkpoint_Config_Base, Config_Base):
    """Configuration for a llama.cpp GGUF of the Qwen2.5-VL-7B language model (e.g.
    ``Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf``), used as a text-only Qwen-Image encoder.

    llama.cpp splits Qwen2.5-VL across two files: the language model, which is all text prompts condition on, and a
    companion ``mmproj-*.gguf`` holding the vision tower. Only the language model is accepted; the ``mmproj`` file is
    refused with its own reason. Without the vision tower the encoder cannot read Qwen-Image Edit's reference
    images, and the prompt node refuses that combination before loading anything.

    Identified by the ``general.architecture`` metadata: the tensors alone are a plain llama.cpp decoder that other
    families' GGUFs share. Qwen2-VL shares that architecture name and is refused by its context length.
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.QwenVLEncoder] = Field(default=ModelType.QwenVLEncoder)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        if mod.path.suffix.lower() != ".gguf":
            raise NotAMatchError(f"expected a .gguf file, got {mod.path.suffix or '(no suffix)'}")

        # `mod.metadata()` is cached across the identification sweep, so this costs one shared header read.
        architecture = mod.metadata().get("general.architecture")
        if architecture is None:
            raise NotAMatchError("not a readable GGUF file, or it carries no 'general.architecture' field")
        if architecture != _QWEN_VL_GGUF_ARCHITECTURE:
            _raise_if_qwen_vl_mmproj(mod)
            raise NotAMatchError(f"GGUF architecture is {architecture!r}, not {_QWEN_VL_GGUF_ARCHITECTURE!r}")

        _raise_if_qwen2_vl(mod)
        _raise_if_not_a_complete_qwen_image_encoder(mod.load_state_dict())

        return cls(**override_fields)
