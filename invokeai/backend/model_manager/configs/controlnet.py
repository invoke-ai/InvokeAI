from typing import (
    Literal,
    Self,
)

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Any

from invokeai.backend.flux.controlnet.state_dict_utils import (
    is_state_dict_instantx_controlnet,
    is_state_dict_xlabs_controlnet,
)
from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    common_config_paths,
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
    ModelFormat,
    ModelType,
)

MODEL_NAME_TO_PREPROCESSOR = {
    "canny": "canny_image_processor",
    "mlsd": "mlsd_image_processor",
    "depth": "depth_anything_image_processor",
    "bae": "normalbae_image_processor",
    "normal": "normalbae_image_processor",
    "sketch": "pidi_image_processor",
    "scribble": "lineart_image_processor",
    "lineart anime": "lineart_anime_image_processor",
    "lineart_anime": "lineart_anime_image_processor",
    "lineart": "lineart_image_processor",
    "soft": "hed_image_processor",
    "softedge": "hed_image_processor",
    "hed": "hed_image_processor",
    "shuffle": "content_shuffle_image_processor",
    "pose": "dw_openpose_image_processor",
    "mediapipe": "mediapipe_face_processor",
    "pidi": "pidi_image_processor",
    "zoe": "zoe_depth_image_processor",
    "color": "color_map_image_processor",
}


class ControlAdapterDefaultSettings(BaseModel):
    # This could be narrowed to controlnet processor nodes, but they change. Leaving this a string is safer.
    preprocessor: str | None
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_model_name(cls, model_name: str) -> Self:
        for k, v in MODEL_NAME_TO_PREPROCESSOR.items():
            model_name_lower = model_name.lower()
            if k in model_name_lower:
                return cls(preprocessor=v)
        return cls(preprocessor=None)


class ControlNet_Diffusers_Config_Base(Diffusers_Config_Base):
    """Model config for ControlNet models (diffusers version)."""

    type: Literal[ModelType.ControlNet] = Field(default=ModelType.ControlNet)
    format: Literal[ModelFormat.Diffusers] = Field(default=ModelFormat.Diffusers)
    default_settings: ControlAdapterDefaultSettings | None = Field(None)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "ControlNetModel",
                "FluxControlNetModel",
            },
        )

        cls._validate_base(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatchError(f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        config_dict = get_config_dict_or_raise(common_config_paths(mod.path))

        if config_dict.get("_class_name") == "FluxControlNetModel":
            return BaseModelType.Flux

        dimension = config_dict.get("cross_attention_dim")

        match dimension:
            case 768:
                return BaseModelType.StableDiffusion1
            case 1024:
                # No obvious way to distinguish between sd2-base and sd2-768, but we don't really differentiate them
                # anyway.
                return BaseModelType.StableDiffusion2
            case 2048:
                return BaseModelType.StableDiffusionXL
            case _:
                raise NotAMatchError(f"unrecognized cross_attention_dim {dimension}")


class ControlNet_Diffusers_SD1_Config(ControlNet_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class ControlNet_Diffusers_SD2_Config(ControlNet_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class ControlNet_Diffusers_SDXL_Config(ControlNet_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class ControlNet_Diffusers_FLUX_Config(ControlNet_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)


class ControlNet_Checkpoint_Config_Base(Checkpoint_Config_Base):
    """Model config for ControlNet models (diffusers version)."""

    type: Literal[ModelType.ControlNet] = Field(default=ModelType.ControlNet)
    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    default_settings: ControlAdapterDefaultSettings | None = Field(None)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_controlnet(mod)

        cls._validate_base(mod)

        return cls(**override_fields)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatchError(f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _validate_looks_like_controlnet(cls, mod: ModelOnDisk) -> None:
        if not state_dict_has_any_keys_starting_with(
            mod.load_state_dict(),
            {
                "controlnet",
                "control_model",
                "input_blocks",
                # XLabs FLUX ControlNet models have keys starting with "controlnet_blocks."
                # For example: https://huggingface.co/XLabs-AI/flux-controlnet-collections/blob/86ab1e915a389d5857135c00e0d350e9e38a9048/flux-canny-controlnet_v2.safetensors
                # TODO(ryand): This is very fragile. XLabs FLUX ControlNet models also contain keys starting with
                # "double_blocks.", which we check for above. But, I'm afraid to modify this logic because it is so
                # delicate.
                "controlnet_blocks",
            },
        ):
            raise NotAMatchError("state dict does not look like a ControlNet checkpoint")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        state_dict = mod.load_state_dict()

        if is_state_dict_xlabs_controlnet(state_dict) or is_state_dict_instantx_controlnet(state_dict):
            # TODO(ryand): Should I distinguish between XLabs, InstantX and other ControlNet models by implementing
            # get_format()?
            return BaseModelType.Flux

        for key in (
            "control_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight",
            "controlnet_mid_block.bias",
            "input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight",
            "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight",
        ):
            if key not in state_dict:
                continue
            width = state_dict[key].shape[-1]
            match width:
                case 768:
                    return BaseModelType.StableDiffusion1
                case 1024:
                    return BaseModelType.StableDiffusion2
                case 2048:
                    return BaseModelType.StableDiffusionXL
                case 1280:
                    return BaseModelType.StableDiffusionXL
                case _:
                    pass

        raise NotAMatchError("unable to determine base type from state dict")


class ControlNet_Checkpoint_SD1_Config(ControlNet_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class ControlNet_Checkpoint_SD2_Config(ControlNet_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class ControlNet_Checkpoint_SDXL_Config(ControlNet_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class ControlNet_Checkpoint_FLUX_Config(ControlNet_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
