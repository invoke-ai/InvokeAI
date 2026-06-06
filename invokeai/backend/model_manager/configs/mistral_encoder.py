import json
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
from invokeai.backend.model_manager.taxonomy import BaseModelType, MistralVariantType, ModelFormat, ModelType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor

# Mistral Small 3 family hidden_size — both the BFL canonical 40-layer encoder
# (``black-forest-labs/FLUX.2-dev/text_encoder``) and the 30-layer "cow" community
# distillation share this. Anything else is rejected as not a FLUX.2 encoder.
_MISTRAL_3_HIDDEN_SIZE = 5120

# Layer counts ComfyUI's reference implementation accepts:
# - 40 layers → BFL canonical (Mistral3_24B), keep final RMSNorm enabled.
# - 30 layers → BFL "cow" distillation, final RMSNorm dropped at load time.
# Anything else is rejected.
_MISTRAL_24B_NUM_LAYERS = 40
_COW_NUM_LAYERS = 30
_ACCEPTED_NUM_LAYERS = (_COW_NUM_LAYERS, _MISTRAL_24B_NUM_LAYERS)


def _has_mistral_keys(state_dict: dict[str | int, Any]) -> bool:
    """Check if a state dict looks like a Mistral causal-LM / multimodal model.

    Supports both:
    - PyTorch/diffusers/transformers format: model.layers.0., model.embed_tokens.weight
      (with optional language_model. prefix for multimodal Mistral3ForConditionalGeneration)
    - GGUF/llama.cpp format: blk.0., token_embd.weight
    """
    pytorch_indicators = (
        "model.layers.",
        "model.embed_tokens.weight",
        "language_model.model.layers.",
        "language_model.model.embed_tokens.weight",
    )
    gguf_indicators = ("blk.", "token_embd.weight")

    for key in state_dict.keys():
        if not isinstance(key, str):
            continue
        if key.startswith(pytorch_indicators):
            return True
        if key.startswith(gguf_indicators):
            return True
    return False


def _has_ggml_tensors(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict contains GGML tensors (GGUF quantized)."""
    return any(isinstance(v, GGMLTensor) for v in state_dict.values())


def _count_mistral_layers(state_dict: dict[str | int, Any]) -> int:
    """Count transformer layers in a Mistral state dict.

    Supports both transformers' ``model.layers.N.*`` layout and llama.cpp's
    ``blk.N.*`` layout. Returns 0 if no per-layer keys are present.
    """
    indices: set[int] = set()
    for key in state_dict.keys():
        if not isinstance(key, str):
            continue
        # transformers / diffusers: model.layers.N.* or language_model.model.layers.N.*
        if ".layers." in key:
            parts = key.split(".layers.", 1)[1].split(".", 1)
            if parts and parts[0].isdigit():
                indices.add(int(parts[0]))
                continue
        # llama.cpp GGUF: blk.N.*
        if key.startswith("blk."):
            parts = key.split(".", 2)
            if len(parts) >= 2 and parts[1].isdigit():
                indices.add(int(parts[1]))
    return (max(indices) + 1) if indices else 0


def _embed_hidden_size(state_dict: dict[str | int, Any]) -> int | None:
    """Read the embedding hidden size from a Mistral-like state dict.

    Returns None if no recognized embedding tensor is present.
    """
    candidate_keys = (
        "model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "token_embd.weight",
    )
    for key in candidate_keys:
        if key not in state_dict:
            continue
        tensor = state_dict[key]
        if isinstance(tensor, GGMLTensor):
            shape = getattr(tensor, "tensor_shape", None) or getattr(tensor, "shape", None)
        else:
            shape = getattr(tensor, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[1])
    return None


def _get_mistral_variant_from_state_dict(state_dict: dict[str | int, Any]) -> MistralVariantType | None:
    """Return the Mistral variant for a state dict, or ``None`` if unrecognized.

    Recognized variants:
    - 30-layer + hidden_size=5120 → ``MistralVariantType.Cow`` (BFL distillation)
    - 40-layer + hidden_size=5120 → ``MistralVariantType.Mistral24B`` (BFL canonical / upstream Mistral Small 3.x)
    """
    if _embed_hidden_size(state_dict) != _MISTRAL_3_HIDDEN_SIZE:
        return None
    num_layers = _count_mistral_layers(state_dict)
    if num_layers == _COW_NUM_LAYERS:
        return MistralVariantType.Cow
    if num_layers == _MISTRAL_24B_NUM_LAYERS:
        return MistralVariantType.Mistral24B
    return None


def _get_mistral_variant_from_config(config_path) -> MistralVariantType | None:
    """Return the Mistral variant for a HF ``config.json``, or ``None`` if unrecognized."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Mistral3ForConditionalGeneration nests the LM config under text_config.
    hidden_size = config.get("hidden_size")
    num_layers = config.get("num_hidden_layers")
    if hidden_size is None or num_layers is None:
        text_config = config.get("text_config") or {}
        if hidden_size is None:
            hidden_size = text_config.get("hidden_size")
        if num_layers is None:
            num_layers = text_config.get("num_hidden_layers")

    if hidden_size != _MISTRAL_3_HIDDEN_SIZE:
        return None
    if num_layers == _COW_NUM_LAYERS:
        return MistralVariantType.Cow
    if num_layers == _MISTRAL_24B_NUM_LAYERS:
        return MistralVariantType.Mistral24B
    return None


class MistralEncoder_Diffusers_Config(Config_Base):
    """Configuration for a Mistral text encoder in HuggingFace transformers/diffusers folder layout.

    Matches:
    - Full pipelines downloaded as just the `text_encoder/` subfolder
      (e.g. `black-forest-labs/FLUX.2-dev/text_encoder/`)
    - Quantized variants such as `diffusers/FLUX.2-dev-bnb-4bit/text_encoder/`

    Does NOT match a full FLUX.2 pipeline directory — those are picked up by the
    `Main_Diffusers_Flux2_Config` instead.

    Accepts both:
    - 30-layer "cow" distillation (recommended, produces the cleanest output)
    - 40-layer Mistral Small 3 (BFL canonical / upstream Mistral 3.x — also works,
      slightly weaker prompt adherence than cow in our tests)

    The variant field records which one was probed so the loader can decide
    whether to keep the final RMSNorm (40-layer) or strip it (30-layer cow).
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.MistralEncoder] = Field(default=ModelType.MistralEncoder)
    format: Literal[ModelFormat.MistralEncoder] = Field(default=ModelFormat.MistralEncoder)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")
    variant: MistralVariantType = Field(description="Mistral text encoder variant")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # Exclude full pipeline models; those should match Main_Diffusers_Flux2_Config.
        if (mod.path / "model_index.json").exists() or (mod.path / "transformer").exists():
            raise NotAMatchError(
                "directory looks like a full diffusers pipeline (has model_index.json or transformer/), "
                "not a standalone Mistral encoder"
            )

        # Find config.json: either nested under text_encoder/ or at the directory root.
        config_path_nested = mod.path / "text_encoder" / "config.json"
        config_path_direct = mod.path / "config.json"
        if config_path_nested.exists():
            expected_config_path = config_path_nested
        elif config_path_direct.exists():
            expected_config_path = config_path_direct
        else:
            raise NotAMatchError(f"no config.json found at {config_path_nested} or {config_path_direct}")

        raise_for_class_name(
            expected_config_path,
            {
                "Mistral3ForConditionalGeneration",
                "MistralModel",
                "MistralForCausalLM",
            },
        )

        variant = _get_mistral_variant_from_config(expected_config_path)
        if variant is None:
            raise NotAMatchError(
                f"config.json does not describe a recognized Mistral variant "
                f"(expected hidden_size={_MISTRAL_3_HIDDEN_SIZE} and num_hidden_layers in {_ACCEPTED_NUM_LAYERS})."
            )

        return cls(variant=variant, **override_fields)


class MistralEncoder_Checkpoint_Config(Checkpoint_Config_Base, Config_Base):
    """Configuration for a single-file Mistral text encoder (safetensors).

    Accepts both 30-layer cow (Comfy-Org bf16/fp8/fp4) and 40-layer Mistral Small 3
    (BFL canonical / upstream Mistral 3.x single-files). The loader uses the
    detected variant to decide whether to keep or strip the final RMSNorm.
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.MistralEncoder] = Field(default=ModelType.MistralEncoder)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")
    variant: MistralVariantType = Field(description="Mistral text encoder variant")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        state_dict = mod.load_state_dict()

        if not _has_mistral_keys(state_dict):
            raise NotAMatchError("state dict does not look like a Mistral encoder")

        if _has_ggml_tensors(state_dict):
            raise NotAMatchError("state dict looks like GGUF quantized")

        variant = _get_mistral_variant_from_state_dict(state_dict)
        if variant is None:
            raise NotAMatchError(
                f"unrecognized Mistral geometry (got hidden_size={_embed_hidden_size(state_dict)}, "
                f"layers={_count_mistral_layers(state_dict)}). Expected hidden_size={_MISTRAL_3_HIDDEN_SIZE} "
                f"and num_hidden_layers in {_ACCEPTED_NUM_LAYERS}."
            )

        return cls(variant=variant, **override_fields)


class MistralEncoder_GGUF_Config(Checkpoint_Config_Base, Config_Base):
    """Configuration for a GGUF-quantized Mistral text encoder.

    Accepts both 30-layer cow GGUFs and 40-layer Mistral Small 3 GGUFs — see
    ``MistralEncoder_Checkpoint_Config`` for variant handling.
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.MistralEncoder] = Field(default=ModelType.MistralEncoder)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")
    variant: MistralVariantType = Field(description="Mistral text encoder variant")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        state_dict = mod.load_state_dict()

        if not _has_mistral_keys(state_dict):
            raise NotAMatchError("state dict does not look like a Mistral encoder")

        if not _has_ggml_tensors(state_dict):
            raise NotAMatchError("state dict does not look like GGUF quantized")

        variant = _get_mistral_variant_from_state_dict(state_dict)
        if variant is None:
            raise NotAMatchError(
                f"unrecognized Mistral geometry (got hidden_size={_embed_hidden_size(state_dict)}, "
                f"layers={_count_mistral_layers(state_dict)}). Expected hidden_size={_MISTRAL_3_HIDDEN_SIZE} "
                f"and num_hidden_layers in {_ACCEPTED_NUM_LAYERS}."
            )

        return cls(variant=variant, **override_fields)
