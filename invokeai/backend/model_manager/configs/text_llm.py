from typing import (
    Literal,
    Self,
)

from pydantic import Field
from typing_extensions import Any

from invokeai.backend.model_manager.configs.base import Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.gemma4_encoder import is_ltx2_gemma4_encoder_folder
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    common_config_paths,
    get_class_name_from_config_dict_or_raise,
    get_config_dict_or_raise,
    raise_for_override_fields,
    raise_if_not_dir,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelType,
)

# Gemma-2-2b's hidden size. Only this size is handled by the dedicated PiD Gemma2 encoder config; larger
# Gemma 2 variants (9B=3584, 27B=4608) are rejected there and must stay classifiable as a generic TextLLM.
_GEMMA2_2B_HIDDEN_SIZE = 2304

# Multimodal architectures that `AutoModelForCausalLM` still loads (transformers maps their model types
# onto these classes in its causal-LM table). Only listed ones are accepted, so a vision-language folder
# never becomes a TextLLM unless the generic loader is known to build it.
_CONDITIONAL_GENERATION_TEXT_LLMS = {"Gemma4ForConditionalGeneration"}

# Files that let `AutoProcessor` build the image side of a multimodal checkpoint.
_PROCESSOR_CONFIG_FILES = ("processor_config.json", "preprocessor_config.json")


class TextLLM_Diffusers_Config(Diffusers_Config_Base, Config_Base):
    """Model config for text-only causal language models (e.g. Llama, Phi, Qwen, Mistral)."""

    type: Literal[ModelType.TextLLM] = Field(default=ModelType.TextLLM)
    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")
    supports_images: bool = Field(
        default=False,
        description="Whether the model has a vision tower and processor, so Expand Prompt can condition on an image",
    )

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # Check that the model's architecture is a causal language model.
        # This covers LlamaForCausalLM, PhiForCausalLM, Phi3ForCausalLM, Qwen2ForCausalLM,
        # MistralForCausalLM, GemmaForCausalLM, GPTNeoXForCausalLM, etc.
        config_dict = get_config_dict_or_raise(common_config_paths(mod.path))
        class_name = get_class_name_from_config_dict_or_raise(config_dict)
        if not class_name.endswith("ForCausalLM") and class_name not in _CONDITIONAL_GENERATION_TEXT_LLMS:
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
            class_name in _CONDITIONAL_GENERATION_TEXT_LLMS
            and "vision_config" in config_dict
            and any((mod.path / f).exists() for f in _PROCESSOR_CONFIG_FILES)
        )
        return cls(**override_fields, supports_images=supports_images)
