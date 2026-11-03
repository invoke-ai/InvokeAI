import re
from typing import (
    Literal,
    Self,
)

import torch
from pydantic import Field
from typing_extensions import Any

from invokeai.backend.clip.clip_text_encoder import clip_text_config
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    InvalidMatchError,
    NotAMatchError,
    get_config_dict_or_raise,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
    state_dict_has_any_keys_starting_with,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ClipVariantType,
    ModelFormat,
    ModelType,
)


def get_clip_variant_type_from_config(config: dict[str, Any]) -> ClipVariantType | None:
    try:
        hidden_size = config.get("hidden_size")
        match hidden_size:
            case 1280:
                return ClipVariantType.G
            case 768:
                return ClipVariantType.L
            case _:
                return None
    except Exception:
        return None


class CLIPEmbed_Diffusers_Config_Base(Diffusers_Config_Base):
    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.CLIPEmbed] = Field(default=ModelType.CLIPEmbed)
    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        raise_for_class_name(
            {
                mod.path / "config.json",
                mod.path / "text_encoder" / "config.json",
            },
            {
                "CLIPModel",
                "CLIPTextModel",
                "CLIPTextModelWithProjection",
            },
        )

        cls._validate_variant(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_variant(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model variant does not match this config class."""
        expected_variant = cls.model_fields["variant"].default
        config = get_config_dict_or_raise(
            {
                mod.path / "config.json",
                mod.path / "text_encoder" / "config.json",
            },
        )
        recognized_variant = get_clip_variant_type_from_config(config)

        if recognized_variant is None:
            raise NotAMatchError("unable to determine CLIP variant from config")

        if expected_variant is not recognized_variant:
            raise NotAMatchError(f"variant is {recognized_variant}, not {expected_variant}")


class CLIPEmbed_Diffusers_G_Config(CLIPEmbed_Diffusers_Config_Base, Config_Base):
    variant: Literal[ClipVariantType.G] = Field(default=ClipVariantType.G)


class CLIPEmbed_Diffusers_L_Config(CLIPEmbed_Diffusers_Config_Base, Config_Base):
    variant: Literal[ClipVariantType.L] = Field(default=ClipVariantType.L)


_CLIP_LAYER_INDEX = re.compile(r"^text_model\.encoder\.layers\.(\d+)\.")
_FLOAT_WEIGHT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class CLIPEmbed_Checkpoint_Config_Base(Checkpoint_Config_Base):
    """A CLIP text encoder in a single safetensors file with transformers key names.

    This is how ComfyUI distributes the CLIP-L and CLIP-G text towers (``clip_l``, ``clip_g``): the text model's
    state dict under ``text_model.``, with CLIP-G's ``text_projection`` beside it, and no config or tokenizer.
    """

    base: Literal[BaseModelType.Any] = Field(default=BaseModelType.Any)
    type: Literal[ModelType.CLIPEmbed] = Field(default=ModelType.CLIPEmbed)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    cpu_only: bool | None = Field(default=None, description="Whether this model should run on CPU only")

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        if mod.path.suffix != ".safetensors":
            raise NotAMatchError("not a safetensors file")

        state_dict = mod.load_state_dict()
        if (
            "text_model.encoder.layers.0.self_attn.q_proj.weight" not in state_dict
            or "text_model.embeddings.token_embedding.weight" not in state_dict
        ):
            raise NotAMatchError("state dict does not look like a transformers CLIP text encoder")
        if state_dict_has_any_keys_starting_with(state_dict, "vision_model."):
            raise NotAMatchError("state dict carries a vision tower; only text encoders are supported")

        expected_variant = cls.model_fields["variant"].default
        recognized_variant = cls._variant_from_shapes(state_dict)
        if recognized_variant is not expected_variant:
            raise NotAMatchError(f"variant is {recognized_variant}, not {expected_variant}")

        # Long-CLIP has CLIP-L's width and depth but a longer position table, which the vendored architecture
        # cannot take: recognised and refused, rather than registered to fail at the first generation.
        config = clip_text_config(expected_variant)
        positions = state_dict.get("text_model.embeddings.position_embedding.weight")
        found = (
            int(positions.shape[0]) if positions is not None else 0,
            int(state_dict["text_model.embeddings.token_embedding.weight"].shape[0]),
        )
        if found != (config.max_position_embeddings, config.vocab_size):
            raise InvalidMatchError(
                f"this CLIP text encoder has {found[0]} positions and a {found[1]}-token vocabulary, where "
                f"CLIP-{expected_variant.name} has {config.max_position_embeddings} and {config.vocab_size}. "
                "Extended-context variants such as Long-CLIP are not supported."
            )

        # SD 3 conditions on CLIP-G's projected pooled output, which a file without the projection cannot give.
        if expected_variant is ClipVariantType.G and "text_projection.weight" not in state_dict:
            raise InvalidMatchError(
                "this CLIP-G text encoder has no text_projection, which SD 3 needs. Install a CLIP-G that includes it, "
                "such as the clip_g.safetensors Stability AI and Comfy-Org distribute."
            )

        quantized = sorted(
            key
            for key, value in state_dict.items()
            if isinstance(key, str)
            and key.endswith(".weight")
            and getattr(value, "dtype", None) not in _FLOAT_WEIGHT_DTYPES
        )
        if quantized:
            raise InvalidMatchError(
                f"this CLIP text encoder carries {len(quantized)} quantized weight(s) (e.g. '{quantized[0]}'), "
                "which are not supported. Install the fp16 build."
            )

        return cls(**override_fields)

    @staticmethod
    def _variant_from_shapes(state_dict: dict[str | int, Any]) -> ClipVariantType | None:
        """The variant whose width and depth the file has, or None for any other CLIP text tower."""
        width = int(state_dict["text_model.embeddings.token_embedding.weight"].shape[1])
        depth = 1 + max(
            int(match.group(1))
            for key in state_dict
            if isinstance(key, str) and (match := _CLIP_LAYER_INDEX.match(key))
        )
        for variant in (ClipVariantType.L, ClipVariantType.G):
            config = clip_text_config(variant)
            if (width, depth) == (config.hidden_size, config.num_hidden_layers):
                return variant
        return None


class CLIPEmbed_Checkpoint_G_Config(CLIPEmbed_Checkpoint_Config_Base, Config_Base):
    variant: Literal[ClipVariantType.G] = Field(default=ClipVariantType.G)


class CLIPEmbed_Checkpoint_L_Config(CLIPEmbed_Checkpoint_Config_Base, Config_Base):
    variant: Literal[ClipVariantType.L] = Field(default=ClipVariantType.L)
