import json
from pathlib import Path
from typing import Any, Literal, Optional, Self

from pydantic import Field

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    InvalidMatchError,
    NotAMatchError,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
    raise_if_quantized_beyond_fp8,
    state_dict_has_any_keys_ending_with,
    state_dict_has_any_keys_starting_with,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.quantization.sdnq.detection import folder_has_sdnq_keys
from invokeai.backend.t5.t5_tokenizer import T5_VOCAB_SIZE

# The width of T5 v1.1 XXL, the encoder FLUX.1 and SD 3 condition on.
T5_XXL_D_MODEL = 4096


def _safetensors_dir_has_sdnq_keys(directory) -> bool:
    """Return True if the safetensors in ``directory`` look SDNQ-quantized (weight + matching scale).

    Thin alias over the shared detector's key check — the pair is resolved across the union of all
    shards, since sharding routinely separates a weight from its scale.
    """
    return folder_has_sdnq_keys(directory)


class T5Encoder_T5Encoder_Config(Config_Base):
    """Configuration for T5 Encoder models in a bespoke, diffusers-like format. The model weights are expected to be in
    a folder called text_encoder_2 inside the model directory, with a config file named model.safetensors.index.json."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.T5Encoder] = Field(default=ModelType.T5Encoder)
    format: Literal[ModelFormat.T5Encoder] = Field(default=ModelFormat.T5Encoder)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        expected_config_path = mod.path / "text_encoder_2" / "config.json"
        expected_class_name = "T5EncoderModel"
        raise_for_class_name(expected_config_path, expected_class_name)

        cls.raise_if_doesnt_have_unquantized_config_file(mod)

        return cls(**override_fields)

    @classmethod
    def raise_if_doesnt_have_unquantized_config_file(cls, mod: ModelOnDisk) -> None:
        has_unquantized_config = (mod.path / "text_encoder_2" / "model.safetensors.index.json").exists()

        if not has_unquantized_config:
            raise NotAMatchError("missing text_encoder_2/model.safetensors.index.json")


class T5Encoder_BnBLLMint8_Config(Config_Base):
    """Configuration for T5 Encoder models quantized by bitsandbytes' LLM.int8."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.T5Encoder] = Field(default=ModelType.T5Encoder)
    format: Literal[ModelFormat.BnbQuantizedLlmInt8b] = Field(default=ModelFormat.BnbQuantizedLlmInt8b)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        expected_config_path = mod.path / "text_encoder_2" / "config.json"
        expected_class_name = "T5EncoderModel"
        raise_for_class_name(expected_config_path, expected_class_name)

        cls.raise_if_filename_doesnt_look_like_bnb_quantized(mod)

        cls.raise_if_state_dict_doesnt_look_like_bnb_quantized(mod)

        return cls(**override_fields)

    @classmethod
    def raise_if_filename_doesnt_look_like_bnb_quantized(cls, mod: ModelOnDisk) -> None:
        filename_looks_like_bnb = any(x for x in mod.weight_files() if "llm_int8" in x.as_posix())
        if not filename_looks_like_bnb:
            raise NotAMatchError("filename does not look like bnb quantized llm_int8")

    @classmethod
    def raise_if_state_dict_doesnt_look_like_bnb_quantized(cls, mod: ModelOnDisk) -> None:
        has_scb_key_suffix = state_dict_has_any_keys_ending_with(mod.load_state_dict(), "SCB")
        if not has_scb_key_suffix:
            raise NotAMatchError("state dict does not look like bnb quantized llm_int8")


class T5Encoder_SDNQ_Config(Config_Base):
    """Configuration for SDNQ-quantized T5 Encoder models.

    Matches two layouts:

    1. **Standalone T5 bundle**: ``mod.path`` is the pipeline-style root, with
       ``text_encoder_2/`` (and usually ``tokenizer_2/``) as subfolders.
    2. **Inline submodel**: ``mod.path`` *is* the ``text_encoder_2`` folder itself —
       this is how a parent FluxPipeline / similar config registers its T5 submodel
       (``submodels[TextEncoder2].path_or_prefix`` points straight at the folder).

    In both cases, the SDNQ-quantized state lives next to a ``config.json`` declaring
    ``T5EncoderModel`` and is signalled either by ``quantization_config.json`` with
    ``quant_method == "sdnq"`` or by SDNQ-style ``weight`` + ``scale`` key pairs.
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.T5Encoder] = Field(default=ModelType.T5Encoder)
    format: Literal[ModelFormat.SDNQQuantized] = Field(default=ModelFormat.SDNQQuantized)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        te_dir = cls._locate_text_encoder_dir(mod)
        raise_for_class_name(te_dir / "config.json", "T5EncoderModel")

        cls._raise_if_not_sdnq_quantized(te_dir)

        # Every FLUX workflow requests a Tokenizer2 alongside the encoder, and the loader can only load
        # it from a `tokenizer_2/` folder. Reject an install that has no resolvable tokenizer (e.g. a
        # bare inline `text_encoder_2` folder with no sibling `tokenizer_2/`) at identification time so
        # it never registers as a selectable T5 that then fails mid-workflow on the missing tokenizer.
        if cls.resolve_tokenizer_dir(mod.path) is None:
            raise NotAMatchError("no tokenizer_2 folder resolvable for this SDNQ T5 encoder layout")

        return cls(**override_fields)

    @staticmethod
    def resolve_text_encoder_dir(path: Path) -> Optional[Path]:
        """Return the directory holding T5's config.json + safetensors, or None.

        Two layouts: a standalone bundle (``path`` is the pipeline root, T5 under ``text_encoder_2/``)
        or an inline submodel (``path`` *is* the ``text_encoder_2`` folder).
        """
        nested = path / "text_encoder_2"
        if (nested / "config.json").exists():
            return nested
        if (path / "config.json").exists():
            return path
        return None

    @staticmethod
    def resolve_tokenizer_dir(path: Path) -> Optional[Path]:
        """Return the ``tokenizer_2/`` directory for either layout, or None if it doesn't exist.

        In the standalone-bundle layout ``tokenizer_2/`` is a child of the pipeline root; in the
        inline layout (``path`` is the ``text_encoder_2`` folder) it's a *sibling* of that folder.
        The encoder loader picks the encoder dir by the same layout test, so the tokenizer must too —
        using ``path / "tokenizer_2"`` unconditionally is wrong for the inline case.
        """
        if (path / "text_encoder_2" / "config.json").exists():
            candidate = path / "tokenizer_2"
        else:
            candidate = path.parent / "tokenizer_2"
        return candidate if candidate.exists() else None

    @classmethod
    def _locate_text_encoder_dir(cls, mod: ModelOnDisk):
        """Return the directory that actually holds T5's config.json + safetensors."""
        te_dir = cls.resolve_text_encoder_dir(mod.path)
        if te_dir is None:
            raise NotAMatchError("no text_encoder_2/config.json or config.json at model root")
        return te_dir

    @classmethod
    def _raise_if_not_sdnq_quantized(cls, te_dir) -> None:
        quant_config_path = te_dir / "quantization_config.json"
        if quant_config_path.exists():
            try:
                with open(quant_config_path, "r", encoding="utf-8") as f:
                    quant_config = json.load(f)
            except (OSError, ValueError):
                quant_config = {}
            if quant_config.get("quant_method") == "sdnq":
                return

        if _safetensors_dir_has_sdnq_keys(te_dir):
            return

        raise NotAMatchError("text_encoder_2 does not look like an SDNQ-quantized T5 encoder")


class T5Encoder_GGUF_Config(Checkpoint_Config_Base, Config_Base):
    """Configuration for GGUF-quantized T5 text encoder models in a single .gguf file.

    These are conversions like city96/t5-v1_1-xxl-encoder-gguf, which use llama.cpp's T5 encoder
    tensor naming (``enc.blk.N.*``, ``token_embd.weight``, ``enc.output_norm.weight``)."""

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.T5Encoder] = Field(default=ModelType.T5Encoder)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls.raise_if_doesnt_look_like_t5_encoder(mod)

        cls.raise_if_doesnt_look_like_gguf_quantized(mod)

        return cls(**override_fields)

    @classmethod
    def raise_if_doesnt_look_like_t5_encoder(cls, mod: ModelOnDisk) -> None:
        # llama.cpp T5 encoders use the ``enc.`` prefix on their transformer blocks and final norm. This
        # distinguishes them from decoder-only GGUF models (e.g. Qwen3, which uses bare ``blk.*``).
        state_dict = mod.load_state_dict()
        if not state_dict_has_any_keys_starting_with(
            state_dict, "enc.blk."
        ) and not state_dict_has_any_keys_ending_with(state_dict, "enc.output_norm.weight"):
            raise NotAMatchError("state dict does not look like a T5 encoder (no 'enc.blk.*' keys)")

    @classmethod
    def raise_if_doesnt_look_like_gguf_quantized(cls, mod: ModelOnDisk) -> None:
        has_ggml = any(isinstance(v, GGMLTensor) for v in mod.load_state_dict().values())
        if not has_ggml:
            raise NotAMatchError("state dict does not look like GGUF quantized")


class T5Encoder_Checkpoint_Config(Checkpoint_Config_Base, Config_Base):
    """Configuration for a T5 encoder in a single safetensors file with transformers key names.

    This is how ComfyUI distributes T5-XXL (``t5xxl_fp16``, ``t5xxl_fp8_e4m3fn`` and
    ``t5xxl_fp8_e4m3fn_scaled``): the ``T5EncoderModel`` state dict as is, with no config and no tokenizer.
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.T5Encoder] = Field(default=ModelType.T5Encoder)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        if mod.path.suffix != ".safetensors":
            raise NotAMatchError("not a safetensors file")

        state_dict = mod.load_state_dict()
        cls._raise_if_not_t5_encoder(state_dict)
        cls._raise_if_not_t5_xxl(state_dict)
        raise_if_quantized_beyond_fp8(
            mod, state_dict, "T5 encoder", "Use the fp16 or fp8 (scaled) build of T5-XXL instead."
        )

        return cls(**override_fields)

    @classmethod
    def _raise_if_not_t5_encoder(cls, state_dict: dict[str | int, Any]) -> None:
        if "encoder.block.0.layer.0.SelfAttention.q.weight" not in state_dict or not (
            "shared.weight" in state_dict or "encoder.embed_tokens.weight" in state_dict
        ):
            raise NotAMatchError("state dict does not look like a transformers T5 encoder")
        # The first shard of a sharded export has block 0 and the embedding too; only a whole encoder ends in its norm.
        if "encoder.final_layer_norm.weight" not in state_dict:
            raise NotAMatchError("state dict has no encoder.final_layer_norm; not a complete T5 encoder")
        if state_dict_has_any_keys_starting_with(state_dict, "decoder."):
            raise NotAMatchError("state dict carries a T5 decoder; only encoder-only files are supported")
        # UMT5 (Wan's text encoder) shares T5's key names but gives every block its own position bias,
        # where T5 has one in the first block only.
        if "encoder.block.1.layer.0.SelfAttention.relative_attention_bias.weight" in state_dict:
            raise NotAMatchError("state dict looks like UMT5 (a relative attention bias in every block)")
        # T5 v1.0 has a single, ungated input projection; v1.1 (what FLUX.1 and SD 3 use) is gated.
        if "encoder.block.0.layer.1.DenseReluDense.wi_0.weight" not in state_dict:
            raise NotAMatchError("state dict does not have T5 v1.1's gated feed-forward")

    @classmethod
    def _raise_if_not_t5_xxl(cls, state_dict: dict[str | int, Any]) -> None:
        embedding = state_dict.get("shared.weight", state_dict.get("encoder.embed_tokens.weight"))
        vocab_size, d_model = (int(x) for x in getattr(embedding, "shape", (0, 0)))
        if d_model != T5_XXL_D_MODEL or vocab_size != T5_VOCAB_SIZE:
            raise InvalidMatchError(
                f"this T5 encoder has width {d_model} and vocabulary {vocab_size}, but FLUX.1 and SD 3 need "
                f"T5 v1.1 XXL (width {T5_XXL_D_MODEL}, vocabulary {T5_VOCAB_SIZE})."
            )
