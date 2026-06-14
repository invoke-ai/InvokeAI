import json
from typing import Any, Literal, Self

from pydantic import Field
from safetensors import safe_open

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
    state_dict_has_any_keys_ending_with,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType


def _safetensors_dir_has_sdnq_keys(directory) -> bool:
    """Return True if any safetensors file in ``directory`` looks SDNQ-quantized (weight + matching scale)."""
    for st_file in sorted(directory.glob("*.safetensors")):
        try:
            with safe_open(st_file, framework="pt") as f:
                keys = set(f.keys())
        except Exception:
            continue
        for key in keys:
            if key.endswith(".weight") and f"{key[:-7]}.scale" in keys:
                return True
    return False


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

        return cls(**override_fields)

    @classmethod
    def _locate_text_encoder_dir(cls, mod: ModelOnDisk):
        """Return the directory that actually holds T5's config.json + safetensors."""
        nested = mod.path / "text_encoder_2"
        if (nested / "config.json").exists():
            return nested
        if (mod.path / "config.json").exists():
            return mod.path
        raise NotAMatchError("no text_encoder_2/config.json or config.json at model root")

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
