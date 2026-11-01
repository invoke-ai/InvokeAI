from typing import (
    Literal,
    Self,
)

import torch
from pydantic import Field
from typing_extensions import Any

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.gemma4_encoder import is_ltx2_gemma4_encoder_folder
from invokeai.backend.model_manager.configs.identification_utils import (
    InvalidMatchError,
    NotAMatchError,
    common_config_paths,
    get_class_name_from_config_dict_or_raise,
    get_config_dict_or_raise,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
)
from invokeai.backend.model_manager.configs.qwen3_5_encoder import is_qwen3_5_vision_language_lm
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelType,
)
from invokeai.backend.quantization.fp8_scaled import FP8_WEIGHT_DTYPES, WEIGHT_SCALE_SUFFIXES, is_scale_metadata_key
from invokeai.backend.quantization.int8_convrot import INT8_TENSORWISE_FORMAT, read_comfy_quant_markers

# Gemma-2-2b's hidden size. Only this size is handled by the dedicated PiD Gemma2 encoder config; larger
# Gemma 2 variants (9B=3584, 27B=4608) are rejected there and must stay classifiable as a generic TextLLM.
_GEMMA2_2B_HIDDEN_SIZE = 2304

# Multimodal architectures the TextLLM loader builds with their vision tower, by the transformers auto class that
# builds them. `AutoModelForCausalLM` maps Gemma-4 onto its conditional-generation class, but Qwen3.5 onto the
# text-only `Qwen3_5ForCausalLM`, so the loader takes `AutoModelForImageTextToText` for the second set. Only listed
# classes are accepted, so a vision-language folder never becomes a TextLLM unless the loader is known to build it.
CAUSAL_LM_MULTIMODAL_TEXT_LLMS = frozenset({"Gemma4ForConditionalGeneration"})
IMAGE_TEXT_TO_TEXT_LLMS = frozenset({"Qwen3_5ForConditionalGeneration"})
_MULTIMODAL_TEXT_LLMS = CAUSAL_LM_MULTIMODAL_TEXT_LLMS | IMAGE_TEXT_TO_TEXT_LLMS

# Files that let `AutoProcessor` build the image side of a multimodal checkpoint.
_PROCESSOR_CONFIG_FILES = ("processor_config.json", "preprocessor_config.json")

_SUPPORTS_IMAGES_DESCRIPTION = (
    "Whether the model has a vision tower and processor, so Expand Prompt can condition on an image"
)

# ComfyUI's w4a8 build of the FLUX.3 prompt upsampler (`asym_w4a8_int8`): packed 4-bit codes beside a codebook and
# per-channel and per-group scales, which no loader here decodes.
_W4A8_SUFFIXES = (".weight_codebook", ".weight_s_channel", ".weight_s_rel")

# mxfp8's per-block scale: one E8M0 exponent per 32 weights, stored as bytes or as the float8 exponent type.
_MXFP8_SCALE_DTYPES = tuple(
    dtype for dtype in (torch.uint8, getattr(torch, "float8_e8m0fnu", None)) if dtype is not None
)


class TextLLM_Diffusers_Config(Diffusers_Config_Base, Config_Base):
    """Model config for a language model folder: a causal LM (Llama, Phi, Qwen, Mistral, ...) or an allow-listed
    vision-language model (Gemma-4, Qwen3.5)."""

    type: Literal[ModelType.TextLLM] = Field(default=ModelType.TextLLM)
    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")
    supports_images: bool = Field(default=False, description=_SUPPORTS_IMAGES_DESCRIPTION)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # Check that the model's architecture is a causal language model.
        # This covers LlamaForCausalLM, PhiForCausalLM, Phi3ForCausalLM, Qwen2ForCausalLM,
        # MistralForCausalLM, GemmaForCausalLM, GPTNeoXForCausalLM, etc.
        config_dict = get_config_dict_or_raise(common_config_paths(mod.path))
        class_name = get_class_name_from_config_dict_or_raise(config_dict)
        if not class_name.endswith("ForCausalLM") and class_name not in _MULTIMODAL_TEXT_LLMS:
            raise NotAMatchError(f"model architecture '{class_name}' is not a causal language model")

        # During *automatic* classification, defer to the dedicated PiD Gemma2 encoder config — but only
        # for the hidden size that config actually accepts (2304 = Gemma-2-2b). Larger Gemma 2 variants
        # (9B=3584, 27B=4608) are rejected by the encoder config, so they must remain classifiable as a
        # generic TextLLM here rather than falling through to Unknown. An explicit `type=text_llm` request
        # always keeps the model as TextLLM (the generic AutoModelForCausalLM loader supports these).
        explicitly_requested_text_llm = override_fields.get("type") == ModelType.TextLLM
        if (
            not explicitly_requested_text_llm
            and class_name == "Gemma2ForCausalLM"
            and config_dict.get("hidden_size") == _GEMMA2_2B_HIDDEN_SIZE
        ):
            raise NotAMatchError(
                "architecture 'Gemma2ForCausalLM' (2304-dim Gemma-2-2b) is handled by the PiD encoder config, not TextLLM"
            )

        # Likewise defer to the LTX-2 Gemma-4 encoder config -- and only where it would actually
        # claim the folder (the LTX single-file layout), so stock sharded Gemma-4-12B stays a TextLLM.
        if not explicitly_requested_text_llm and is_ltx2_gemma4_encoder_folder(mod.path):
            raise NotAMatchError(
                f"architecture '{class_name}' in the LTX-2 single-file layout is handled by the LTX-2 Gemma-4 "
                "encoder config, not TextLLM"
            )

        # Verify tokenizer files exist to avoid runtime failures
        tokenizer_files = {"tokenizer.json", "tokenizer.model", "tokenizer_config.json"}
        if not any((mod.path / f).exists() for f in tokenizer_files):
            raise NotAMatchError(
                f"no tokenizer files found in '{mod.path}' "
                f"(expected at least one of: {', '.join(sorted(tokenizer_files))})"
            )

        # Only an allow-listed multimodal class is known to load with its vision tower through the generic loader.
        supports_images = (
            class_name in _MULTIMODAL_TEXT_LLMS
            and "vision_config" in config_dict
            and any((mod.path / f).exists() for f in _PROCESSOR_CONFIG_FILES)
        )
        return cls(**override_fields, supports_images=supports_images)


class TextLLM_Checkpoint_Config(Checkpoint_Config_Base, Config_Base):
    """Model config for a single-file Qwen3.5 vision-language model, run as a text LLM.

    The layout is transformers' own `Qwen3_5ForConditionalGeneration` state dict, as ComfyUI exports the FLUX.3
    prompt upsampler (bf16, or `int8_tensorwise` with a multi-token-prediction head the loader drops). The file
    carries weights only; the loader supplies config, tokenizer and processor from the vendored Qwen3.5 assets,
    which exist for the 9B. Its LM head sets it apart from the Qwen3.5 text-encoder exports, which have none and
    are `Qwen35Encoder_Checkpoint_Config`'s.
    """

    type: Literal[ModelType.TextLLM] = Field(default=ModelType.TextLLM)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")
    # True for every file this config accepts: it requires the vision tower, and the processor is vendored.
    supports_images: bool = Field(default=True, description=_SUPPORTS_IMAGES_DESCRIPTION)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)
        if mod.path.suffix.lower() != ".safetensors":
            raise NotAMatchError("a single-file Qwen3.5 vision-language model is a .safetensors file")

        raise_for_override_fields(cls, override_fields)

        state_dict = mod.load_state_dict()
        if not is_qwen3_5_vision_language_lm(state_dict):
            raise NotAMatchError(
                "state dict is not a whole Qwen3.5 vision-language model (model.language_model.*, model.visual.*, "
                "lm_head) of a size with a vendored config"
            )
        _raise_for_unsupported_quantization(mod, state_dict)
        return cls(**override_fields)


def _raise_for_unsupported_quantization(mod: ModelOnDisk, state_dict: dict[str | int, Any]) -> None:
    """Refuse every quantized build but ComfyUI's int8_tensorwise: with bf16, all the loader can build.

    `InvalidMatchError`: the file is this model, so the installer shows the reason instead of registering a model
    nothing loads. An int8 layer counts when its weight is int8 and its marker -- read from the file, since
    identification sees dtypes and shapes but no bytes -- says int8_tensorwise; any other scale or marker is a scheme
    the loader refuses.
    """
    keys = [key for key in state_dict if isinstance(key, str)]
    int8_layers = {
        key.removesuffix(".weight")
        for key in keys
        if key.endswith(".weight") and getattr(state_dict[key], "dtype", None) is torch.int8
    }
    scheme = _foreign_quantization(state_dict, keys, int8_layers)
    if scheme is None and int8_layers:
        markers = read_comfy_quant_markers(mod.path)
        unmarked = sorted(
            layer for layer in int8_layers if markers.get(layer, {}).get("format") != INT8_TENSORWISE_FORMAT
        )
        if unmarked:
            scheme = f"int8 without an {INT8_TENSORWISE_FORMAT} marker (e.g. '{unmarked[0]}')"
    if scheme is not None:
        raise InvalidMatchError(
            f"{mod.path.name} is quantized as {scheme}, which InvokeAI cannot load for this model. "
            "Install the bf16 or int8_convrot build instead."
        )


def _foreign_quantization(state_dict: dict[str | int, Any], keys: list[str], int8_layers: set[str]) -> str | None:
    """Name a quantization side channel that belongs to no int8 layer, or None if there is none."""
    if any(key.endswith(_W4A8_SUFFIXES) for key in keys):
        return "ComfyUI's w4a8 (asym_w4a8_int8)"
    if any(key.endswith(".weight_scale_2") for key in keys):
        return "nvfp4"
    for key in keys:
        layer = key.rsplit(".", 1)[0]
        if not is_scale_metadata_key(key) or layer in int8_layers:
            continue
        if key.endswith(WEIGHT_SCALE_SUFFIXES) and getattr(state_dict.get(f"{layer}.weight"), "dtype", None) in (
            FP8_WEIGHT_DTYPES
        ):
            return "mxfp8" if getattr(state_dict[key], "dtype", None) in _MXFP8_SCALE_DTYPES else "scaled fp8"
        return "scaled fp8" if key == "scaled_fp8" else f"an unsupported scheme (it carries '{key}')"
    return None
