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

# ERNIE-Image's encoder is Ministral 3B — a different member of the Mistral family, not a
# smaller Mistral Small 3. It is half as wide (3072) over 26 layers and uses YaRN RoPE, so the
# loader builds it as ``Ministral3Model``; see ``MistralVariantType.Ministral3B``. Geometry is
# what separates the families, which is why the width is part of the match rather than assumed.
_MINISTRAL_3B_HIDDEN_SIZE = 3072
_MINISTRAL_3B_NUM_LAYERS = 26

_ACCEPTED_GEOMETRIES = (
    f"hidden_size={_MISTRAL_3_HIDDEN_SIZE} with num_hidden_layers in {_ACCEPTED_NUM_LAYERS}, "
    f"or hidden_size={_MINISTRAL_3B_HIDDEN_SIZE} with num_hidden_layers={_MINISTRAL_3B_NUM_LAYERS}"
)

# Minimum vocab size to accept as a FLUX.2 Mistral encoder. Mistral Small 3's
# Tekken vocab is 131072 (shared by the 30-layer cow distillation). This gates out
# unrelated causal LMs that happen to share the 5120-hidden / 40-layer geometry —
# most notably Llama-2-13B (vocab 32000), which otherwise matches the llama.cpp key
# names and geometry and would install as a Mistral encoder producing garbage. We
# use a floor rather than an exact match to tolerate any vocab padding in GGUF.
_MISTRAL_3_MIN_VOCAB_SIZE = 100000

# Wrapper prefixes some FLUX.2 single-file redistributions add to Mistral keys. The
# runtime loader strips these (see ``_strip_known_prefixes`` in the loader); the
# install-time probe below normalizes keys the same way so it recognizes exactly the
# layouts the loader can actually load.
_PROBE_KEY_PREFIXES = ("text_encoder.", "language_model.")


def _normalize_probe_key(key: str) -> str:
    """Strip a single known wrapper prefix, mirroring the loader's ``_strip_known_prefixes``."""
    for prefix in _PROBE_KEY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _has_mistral_keys(state_dict: dict[str | int, Any]) -> bool:
    """Check if a state dict looks like a Mistral causal-LM / multimodal model.

    Supports both:
    - PyTorch/diffusers/transformers format: model.layers.0., model.embed_tokens.weight
      (with optional language_model. / text_encoder. prefixes used by multimodal
      Mistral3ForConditionalGeneration and Comfy-Org single-file redistributions)
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
        normalized = _normalize_probe_key(key)
        if normalized.startswith(pytorch_indicators):
            return True
        if normalized.startswith(gguf_indicators):
            return True
    return False


def _has_ggml_tensors(state_dict: dict[str | int, Any]) -> bool:
    """Check if state dict contains GGML tensors (GGUF quantized)."""
    return any(isinstance(v, GGMLTensor) for v in state_dict.values())


def _count_mistral_layers(state_dict: dict[str | int, Any]) -> int:
    """Count *language* transformer layers in a Mistral state dict.

    Supports both transformers' ``model.layers.N.*`` layout and llama.cpp's
    ``blk.N.*`` layout. Returns 0 if no per-layer keys are present.

    The Mistral3 stack nests a second transformer under ``vision_tower.``, whose layers are
    counted by the same ``.layers.N.`` spelling. Today the vision tower is the shorter of the two
    (Pixtral's 24 against the language tower's 26), so a plain maximum happens to land on the
    right one -- but the count decides the variant, so it should not rest on which tower is
    deeper.
    """
    indices: set[int] = set()
    for key in state_dict.keys():
        if not isinstance(key, str):
            continue
        normalized = _normalize_probe_key(key)
        if normalized.startswith(_MISTRAL3_STACK_PREFIXES):
            continue
        # transformers / diffusers: model.layers.N.* or language_model.model.layers.N.*
        if ".layers." in normalized:
            parts = normalized.split(".layers.", 1)[1].split(".", 1)
            if parts and parts[0].isdigit():
                indices.add(int(parts[0]))
                continue
        # llama.cpp GGUF: blk.N.*
        if normalized.startswith("blk."):
            parts = normalized.split(".", 2)
            if len(parts) >= 2 and parts[1].isdigit():
                indices.add(int(parts[1]))
    return (max(indices) + 1) if indices else 0


def _embed_shape(state_dict: dict[str | int, Any]) -> tuple[int, int] | None:
    """Read the ``(vocab_size, hidden_size)`` of the embedding tensor, or ``None``.

    Scans keys with the loader's prefix normalization so ``text_encoder.``- /
    ``language_model.``-prefixed layouts are recognized too.
    """
    candidate_keys = {
        "model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "token_embd.weight",
    }
    for key, tensor in state_dict.items():
        if not isinstance(key, str) or _normalize_probe_key(key) not in candidate_keys:
            continue
        if isinstance(tensor, GGMLTensor):
            shape = getattr(tensor, "tensor_shape", None) or getattr(tensor, "shape", None)
        else:
            shape = getattr(tensor, "shape", None)
        if shape is not None and len(shape) >= 2:
            return int(shape[0]), int(shape[1])
    return None


def _embed_hidden_size(state_dict: dict[str | int, Any]) -> int | None:
    """Read the embedding hidden size from a Mistral-like state dict, or ``None``."""
    shape = _embed_shape(state_dict)
    return shape[1] if shape is not None else None


def _embed_vocab_size(state_dict: dict[str | int, Any]) -> int | None:
    """Read the embedding vocab size from a Mistral-like state dict, or ``None``."""
    shape = _embed_shape(state_dict)
    return shape[0] if shape is not None else None


def _variant_for_geometry(hidden_size: int | None, num_layers: int | None) -> MistralVariantType | None:
    """Map a ``(hidden_size, num_layers)`` pair onto a variant, or ``None`` if unrecognized.

    Recognized variants:
    - 30-layer + hidden_size=5120 → ``MistralVariantType.Cow`` (BFL distillation)
    - 40-layer + hidden_size=5120 → ``MistralVariantType.Mistral24B`` (BFL canonical / upstream Mistral Small 3.x)
    - 26-layer + hidden_size=3072 → ``MistralVariantType.Ministral3B`` (ERNIE-Image)

    Shared by the state-dict and ``config.json`` probes so the two can never recognize
    different sets of encoders.
    """
    if hidden_size == _MISTRAL_3_HIDDEN_SIZE:
        if num_layers == _COW_NUM_LAYERS:
            return MistralVariantType.Cow
        if num_layers == _MISTRAL_24B_NUM_LAYERS:
            return MistralVariantType.Mistral24B
        return None
    if hidden_size == _MINISTRAL_3B_HIDDEN_SIZE and num_layers == _MINISTRAL_3B_NUM_LAYERS:
        return MistralVariantType.Ministral3B
    return None


# ERNIE-Image ships two files of identical Ministral 3B geometry in one folder: the text encoder
# and the prompt enhancer (`Ministral3ForCausalLM`), which rewrites prompts and conditions nothing.
# Both carry the embedded Tekken vocab and the same 236 `model.*` tensors, so neither the vocab
# floor nor the geometry separates them. What does: the encoder is the full Mistral3 multimodal
# stack and ships the Pixtral vision tower the loader later drops, while the enhancer is a bare
# language model. Requiring the tower is what keeps the enhancer from installing as an encoder and
# then conditioning every prompt with the wrong weights. The cost is that a vision-stripped
# repackaging is not recognized -- a visible failure, where the alternative is a silent one.
_MISTRAL3_STACK_PREFIXES = ("vision_tower.", "multi_modal_projector.")


def _has_mistral3_vision_stack(state_dict: dict[str | int, Any]) -> bool:
    """Whether the file carries the Mistral3 multimodal stack around its language tower."""
    return any(
        isinstance(key, str) and _normalize_probe_key(key).startswith(_MISTRAL3_STACK_PREFIXES) for key in state_dict
    )


PROMPT_ENHANCER_REFUSAL = (
    "this is ERNIE-Image's prompt enhancer, not its text encoder: the same Ministral 3B geometry, "
    "but without the vision tower the encoder ships. The enhancer only runs as part of an "
    "ERNIE-Image diffusers pipeline; for a single-file transformer install "
    "text_encoders/ministral-3-3b.safetensors instead."
)
"""Named so the refusal can be asserted, and because the generic geometry message is actively
misleading here: it would list Ministral's 3072/26 among the *expected* geometries, which is
exactly what this file has."""


def _is_ministral_language_model_only(state_dict: dict[str | int, Any]) -> bool:
    """Ministral 3B geometry with no multimodal stack -- ERNIE-Image's prompt enhancer."""
    vocab_size = _embed_vocab_size(state_dict)
    if vocab_size is None or vocab_size < _MISTRAL_3_MIN_VOCAB_SIZE:
        return False
    geometry = _variant_for_geometry(_embed_hidden_size(state_dict), _count_mistral_layers(state_dict))
    return geometry is MistralVariantType.Ministral3B and not _has_mistral3_vision_stack(state_dict)


def _get_mistral_variant_from_state_dict(state_dict: dict[str | int, Any]) -> MistralVariantType | None:
    """Return the Mistral variant for a state dict, or ``None`` if unrecognized.

    The vocab-size floor rejects unrelated causal LMs (e.g. Llama-2-13B) that share
    the 5120-hidden / 40-layer geometry and llama.cpp key names but are not Mistral
    Small 3 encoders — without it they would install and emit garbage embeddings. Both
    accepted families share the 131072-entry Tekken vocab, so one floor covers them.
    """
    vocab_size = _embed_vocab_size(state_dict)
    if vocab_size is None or vocab_size < _MISTRAL_3_MIN_VOCAB_SIZE:
        return None
    variant = _variant_for_geometry(_embed_hidden_size(state_dict), _count_mistral_layers(state_dict))
    if variant is MistralVariantType.Ministral3B and not _has_mistral3_vision_stack(state_dict):
        # ERNIE-Image's prompt enhancer has this exact geometry; only the tower tells them apart.
        return None
    return variant


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

    return _variant_for_geometry(hidden_size, num_layers)


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
                # ERNIE-Image's released `text_encoder/` declares this one. Without it the folder a
                # user can download straight from `baidu/ERNIE-Image` installs as Unknown, and the
                # Ministral branch of the geometry probe below is unreachable.
                # `Ministral3ForCausalLM` is deliberately absent: that is the prompt enhancer, a
                # different model that happens to share the encoder's geometry.
                "Mistral3Model",
            },
        )

        variant = _get_mistral_variant_from_config(expected_config_path)
        if variant is None:
            raise NotAMatchError(
                f"config.json does not describe a recognized Mistral variant (expected {_ACCEPTED_GEOMETRIES})."
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
        if variant is None and _is_ministral_language_model_only(state_dict):
            raise NotAMatchError(PROMPT_ENHANCER_REFUSAL)
        if variant is None:
            raise NotAMatchError(
                f"unrecognized Mistral geometry (got hidden_size={_embed_hidden_size(state_dict)}, "
                f"vocab_size={_embed_vocab_size(state_dict)}, layers={_count_mistral_layers(state_dict)}). "
                f"Expected vocab_size>={_MISTRAL_3_MIN_VOCAB_SIZE} and {_ACCEPTED_GEOMETRIES}."
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
        if variant is MistralVariantType.Ministral3B:
            # The GGUF loader builds a `MistralModel` with RoPE read from GGUF metadata. Ministral
            # 3B needs `Ministral3Model` -- YaRN scaling plus a position-dependent attention scale --
            # so accepting one here would install a model that loads cleanly and encodes garbage.
            raise NotAMatchError("Ministral 3B GGUF encoders are not supported")
        if variant is None:
            raise NotAMatchError(
                f"unrecognized Mistral geometry (got hidden_size={_embed_hidden_size(state_dict)}, "
                f"vocab_size={_embed_vocab_size(state_dict)}, layers={_count_mistral_layers(state_dict)}). "
                f"Expected vocab_size>={_MISTRAL_3_MIN_VOCAB_SIZE} and {_ACCEPTED_GEOMETRIES}."
            )

        return cls(variant=variant, **override_fields)
