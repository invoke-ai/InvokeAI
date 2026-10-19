from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    InvalidMatchError,
    NotAMatchError,
    get_config_dict_or_raise,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, Qwen3VLVariantType

# The two Qwen3-VL encoders v7 can build, by language-model width. Both have 36 layers, so the
# width alone identifies the variant. Anything else -- including MiniMax H3's truncated
# Qwen3-VL-32B, which has its own config class -- is not one of these and must not install as one:
# the loader builds the architecture from a HuggingFace config chosen by this variant, and a
# mismatch surfaces as a shape error thousands of tensors into the load.
_QWEN3_VL_VARIANT_BY_HIDDEN_SIZE = {
    2560: Qwen3VLVariantType.Qwen3VL_4B,
    4096: Qwen3VLVariantType.Qwen3VL_8B,
}
_QWEN3_VL_NUM_HIDDEN_LAYERS = 36


def _variant_from_hidden_size(hidden_size: Any) -> Qwen3VLVariantType:
    variant = _QWEN3_VL_VARIANT_BY_HIDDEN_SIZE.get(hidden_size) if isinstance(hidden_size, int) else None
    if variant is None:
        supported = ", ".join(str(size) for size in sorted(_QWEN3_VL_VARIANT_BY_HIDDEN_SIZE))
        raise NotAMatchError(f"unsupported Qwen3-VL hidden size {hidden_size} (supported: {supported})")
    return variant


def _variant_from_config(config_path: Path) -> Qwen3VLVariantType:
    config = get_config_dict_or_raise(config_path)
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise NotAMatchError("Qwen3-VL text_config must be an object")
    variant = _variant_from_hidden_size(text_config.get("hidden_size"))
    num_hidden_layers = text_config.get("num_hidden_layers")
    if num_hidden_layers != _QWEN3_VL_NUM_HIDDEN_LAYERS:
        raise NotAMatchError(f"expected {_QWEN3_VL_NUM_HIDDEN_LAYERS} Qwen3-VL layers, got {num_hidden_layers}")
    return variant


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


def _variant_from_checkpoint_shape(state_dict: dict[str | int, Any]) -> Qwen3VLVariantType:
    embed_keys = (
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
        "embed_tokens.weight",
    )
    embed = next((state_dict[key] for key in embed_keys if key in state_dict), None)
    shape = getattr(embed, "shape", ())
    variant = _variant_from_hidden_size(shape[1] if len(shape) >= 2 else None)
    if not any(isinstance(key, str) and f".layers.{_QWEN3_VL_NUM_HIDDEN_LAYERS - 1}." in key for key in state_dict):
        raise NotAMatchError(
            f"a Qwen3-VL encoder checkpoint must contain language-model layer {_QWEN3_VL_NUM_HIDDEN_LAYERS - 1}"
        )
    return variant


class Qwen3VLEncoder_Qwen3VLEncoder_Config(Config_Base):
    """Configuration for standalone Qwen3-VL text encoder models (diffusers-like directory format).

    Used by Krea-2 (4B) and Ideogram 4 (8B), whose text conditioning comes from a Qwen3-VL model
    (``Qwen3VLModel``). The model weights are expected either in a ``text_encoder`` subfolder of the
    model directory or directly at the root (standalone download). This is distinct from the text-only
    ``Qwen3Encoder`` (Z-Image / FLUX.2 Klein) and the Qwen2.5-VL ``QwenVLEncoder`` (Qwen Image).
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.Qwen3VLEncoder] = Field(default=ModelType.Qwen3VLEncoder)
    format: Literal[ModelFormat.Qwen3VLEncoder] = Field(default=ModelFormat.Qwen3VLEncoder)
    variant: Qwen3VLVariantType = Field(
        description="Which Qwen3-VL encoder this is. The consuming architecture is fixed: Krea-2 needs "
        "the 4B, Ideogram 4 the 8B, and the two are not interchangeable."
    )
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
        variant = override_fields.pop("variant", None) or _variant_from_config(expected_config_path)

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

        return cls(**override_fields, variant=variant)


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
    """Configuration for a single-file Qwen3-VL text encoder checkpoint (ComfyUI ``qwen3vl_4b_*`` /
    ``qwen3vl_8b_*``).

    Distinguished from the text-only ``Qwen3Encoder`` checkpoint (Z-Image) by the presence of the
    Qwen3-VL visual tower. Neither the config nor the tokenizer is bundled in a single-file
    checkpoint; the loader supplies both from the copies vendored in ``invokeai.backend.qwen3_vl``,
    selecting the config by the recorded variant.
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.Qwen3VLEncoder] = Field(default=ModelType.Qwen3VLEncoder)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    variant: Qwen3VLVariantType = Field(
        description="Which Qwen3-VL encoder this is. The consuming architecture is fixed: Krea-2 needs "
        "the 4B, Ideogram 4 the 8B, and the two are not interchangeable."
    )
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
        variant = override_fields.pop("variant", None) or _variant_from_checkpoint_shape(state_dict)

        return cls(**override_fields, variant=variant)


_QWEN3_VL_GGUF_ARCHITECTURE = "qwen3vl"
_QWEN3_VL_GGUF_EMBED_KEY = "token_embd.weight"
# llama.cpp writes the Qwen3-VL visual tower to a companion file under the generic "clip"
# architecture; this projector name is what makes it recognisably Qwen3-VL's rather than some other
# model's vision tower.
_QWEN3_VL_MMPROJ_PROJECTOR = "qwen3vl_merger"


def _raise_if_qwen3_vl_mmproj(mod: ModelOnDisk) -> None:
    """Reject the companion visual tower with the reason, rather than letting it pass as unknown.

    Installing the HuggingFace GGUF repo hands the user both files, and the ``mmproj`` one looks like
    a plausible second encoder. Nothing else claims it either, so without this it registers as an
    ``Unknown`` model (``allow_unknown_models`` defaults to True) and the user is left with a dead
    entry and no reason for it.
    """
    if mod.metadata().get("clip.projector_type") != _QWEN3_VL_MMPROJ_PROJECTOR:
        return
    raise InvalidMatchError(
        "this is the Qwen3-VL visual tower ('mmproj'), not a text encoder. Install the language tower "
        "instead (e.g. 'Qwen3VL-4B-Instruct-Q4_K_M.gguf'); Krea-2 and Ideogram 4 condition on text only "
        "and never use the visual tower."
    )


def _variant_from_gguf_state_dict(state_dict: dict[str | int, Any]) -> Qwen3VLVariantType:
    """Derive the variant from the tensors, and reject a file whose tensors are incomplete.

    Deliberately not taken from the ``embedding_length``/``block_count`` metadata, though both are
    present. A llama.cpp multi-part quant (``...-00002-of-00003.gguf``, routine for the larger 8B
    quants) and an interrupted download both carry a complete KV block and only some of the tensors.
    Matching on metadata alone would install such a file, offer it in the encoder picker, and fail
    at the user's first generation rather than at install time.

    Raises ``InvalidMatchError``, not ``NotAMatchError``: the caller runs this only after the
    architecture metadata identified the file as a Qwen3-VL encoder, so "recognised and unusable" is
    the accurate verdict. It is also the only one the user ever sees -- ``allow_unknown_models``
    defaults to True, so a ``NotAMatchError`` here would register the file as an ``Unknown`` model
    and leave the reason in the server log.
    """
    embed = state_dict.get(_QWEN3_VL_GGUF_EMBED_KEY)
    if embed is None:
        raise InvalidMatchError(
            f"a Qwen3-VL encoder GGUF must contain '{_QWEN3_VL_GGUF_EMBED_KEY}'; this file does not. "
            "Install the language tower (e.g. 'Qwen3VL-4B-Instruct-Q4_K_M.gguf'), not a companion file."
        )
    shape = getattr(embed, "shape", ())
    try:
        variant = _variant_from_hidden_size(shape[1] if len(shape) >= 2 else None)
    except NotAMatchError as e:
        raise InvalidMatchError(f"unsupported Qwen3-VL encoder GGUF: {e}") from e

    last_layer = _QWEN3_VL_NUM_HIDDEN_LAYERS - 1
    if not any(isinstance(key, str) and key.startswith(f"blk.{last_layer}.") for key in state_dict):
        raise InvalidMatchError(
            f"this Qwen3-VL encoder GGUF is incomplete: it carries no 'blk.{last_layer}.*' tensors, but a "
            f"Qwen3-VL encoder has {_QWEN3_VL_NUM_HIDDEN_LAYERS} layers. Multi-part quants "
            "('...-00002-of-00003.gguf') must be joined before installing, and an interrupted download "
            "must be re-fetched."
        )
    return variant


class Qwen3VLEncoder_GGUF_Config(Checkpoint_Config_Base, Config_Base):
    """Configuration for a single-file GGUF Qwen3-VL encoder (llama.cpp, e.g.
    ``Qwen3VL-4B-Instruct-Q4_K_M.gguf``).

    llama.cpp splits Qwen3-VL across two files: the language tower -- all Krea-2 and Ideogram 4
    condition on -- and a companion ``mmproj-*.gguf`` holding the visual tower. Only the language
    tower is accepted here; the ``mmproj`` file is turned away with its own reason.

    Identified by the ``general.architecture`` metadata, which is the only honest discriminator: a
    Qwen3-VL language tower is structurally indistinguishable from a text-only Qwen3 of the same
    size (both 36 layers at hidden 2560 for the 4B), so a shape probe would accept a stock Qwen3
    GGUF as a Krea-2 encoder. That file loads without error and produces silently wrong
    conditioning, which is precisely the failure identification exists to prevent.
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.Qwen3VLEncoder] = Field(default=ModelType.Qwen3VLEncoder)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)
    variant: Qwen3VLVariantType = Field(
        description="Which Qwen3-VL encoder this is. The consuming architecture is fixed: Krea-2 needs "
        "the 4B, Ideogram 4 the 8B, and the two are not interchangeable."
    )
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        if mod.path.suffix.lower() != ".gguf":
            raise NotAMatchError(f"expected a .gguf file, got {mod.path.suffix or '(no suffix)'}")

        # `mod.metadata()` reads and caches every GGUF string field, and identification probes many
        # configs against the same ModelOnDisk -- so this costs one shared read rather than a second
        # mmap plus the full GC that closing a reader forces.
        architecture = mod.metadata().get("general.architecture")
        if architecture is None:
            raise NotAMatchError("not a readable GGUF file, or it carries no 'general.architecture' field")
        # Deliberately exact: the MoE Qwen3-VL variants report "qwen3vlmoe" and need a different
        # transformers architecture than the one the loader builds.
        if architecture != _QWEN3_VL_GGUF_ARCHITECTURE:
            _raise_if_qwen3_vl_mmproj(mod)
            raise NotAMatchError(f"GGUF architecture is {architecture!r}, not {_QWEN3_VL_GGUF_ARCHITECTURE!r}")

        # Run unconditionally, even when the caller overrode `variant`. This is the only check that
        # the file holds a whole encoder, and an override says which Qwen3-VL it is -- not that the
        # tensors are all there. Deriving first and letting the override win afterwards keeps both.
        variant = _variant_from_gguf_state_dict(mod.load_state_dict())

        # `override_fields` is one dict shared by every candidate config in this sweep, so it must not
        # be mutated: popping here would erase a user's variant override for every class probed after
        # this one, and the probe order is a set iteration (see `Config_Base.CONFIG_CLASSES`).
        return cls(**{"variant": variant, **override_fields})


_MINIMAX_H3_TE_METADATA_KEY = "minimax_h3_te"
_MINIMAX_H3_TE_HIDDEN_SIZE = 5120


class Qwen3VLEncoder_Checkpoint_MiniMaxH3_Config(Checkpoint_Config_Base, Config_Base):
    """Configuration for MiniMax H3's truncated Qwen3-VL-32B conditioning encoder single files
    (Comfy-Org ``qwen3vl_32b_minimax_h3_*.safetensors`` and mirrors).

    These are NOT complete Qwen3-VL-32B checkpoints: the language stack is truncated to the 50
    layers H3 conditions on (the file's ``minimax_h3_te`` metadata records the contract:
    "unnormalized_hidden_after_layer_50"), the final norm and LM head are omitted, and the
    bf16/int8-convrot repacks quantize only the 50 language layers (vision tower stays bf16).

    Identified primarily by the explicit ``minimax_h3_te`` safetensors metadata; a structural
    fallback covers metadata-stripped re-uploads. ``Qwen3VLEncoder_Checkpoint_Config`` accepts only
    the two widths it can build (2560 and 4096), so neither config can claim the other's files.

    The nvfp4 repacks share this layout and are accepted here, but the loader rejects their
    quantization format early (header-only check) with a clear error - mirroring how the H3
    transformer checkpoint config treats fp8_scaled files.
    """

    base: Literal[BaseModelType.MiniMaxH3] = Field(default=BaseModelType.MiniMaxH3)
    type: Literal[ModelType.Qwen3VLEncoder] = Field(default=ModelType.Qwen3VLEncoder)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        if mod.path.suffix.lower() != ".safetensors":
            raise NotAMatchError(f"expected a .safetensors file, got {mod.path.suffix or '(no suffix)'}")

        state_dict = mod.load_state_dict()
        # The structural minimum holds on BOTH paths: a re-tagged arbitrary file must not install
        # on the strength of its metadata alone and fail only after the ~25 GiB load.
        if (
            "model.layers.0.self_attn.q_proj.weight" not in state_dict
            or "visual.blocks.0.attn.qkv.weight" not in state_dict
        ):
            raise NotAMatchError("state dict does not look like a MiniMax H3 Qwen3-VL-32B encoder")

        if _MINIMAX_H3_TE_METADATA_KEY not in mod.metadata():
            # Structural fallback for metadata-stripped re-uploads: additionally require the
            # 32B hidden size and no layer beyond the H3 truncation point. (A full 64-layer 32B
            # encoder is deliberately NOT matched - H3 conditioning requires the truncated stack
            # or the diffusers folder install.)
            embed = state_dict.get("model.embed_tokens.weight")
            shape = getattr(embed, "shape", ())
            if len(shape) < 2 or shape[1] != _MINIMAX_H3_TE_HIDDEN_SIZE:
                raise NotAMatchError("state dict does not look like a MiniMax H3 Qwen3-VL-32B encoder")
            if any(isinstance(key, str) and ".layers.50." in key for key in state_dict):
                raise NotAMatchError(
                    "state dict looks like a full (untruncated) Qwen3-VL-32B - MiniMax H3 requires the "
                    "50-layer truncated conditioning encoder"
                )

        return cls(**override_fields)
