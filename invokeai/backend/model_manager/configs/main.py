from abc import ABC
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from invokeai.backend.model_manager.configs.base import (
    Checkpoint_Config_Base,
    Config_Base,
    Diffusers_Config_Base,
    SubmodelDefinition,
)
from invokeai.backend.model_manager.configs.clip_embed import get_clip_variant_type_from_config
from invokeai.backend.model_manager.configs.identification_utils import (
    NotAMatchError,
    common_config_paths,
    get_config_dict_or_raise,
    raise_for_class_name,
    raise_for_override_fields,
    raise_if_not_dir,
    raise_if_not_file,
    state_dict_has_any_keys_exact,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    FluxVariantType,
    ModelFormat,
    ModelType,
    ModelVariantType,
    SchedulerPredictionType,
    SubModelType,
)
from invokeai.backend.quantization.gguf.ggml_tensor import GGMLTensor
from invokeai.backend.stable_diffusion.schedulers.schedulers import SCHEDULER_NAME_VALUES

DEFAULTS_PRECISION = Literal["fp16", "fp32"]


class MainModelDefaultSettings(BaseModel):
    vae: str | None = Field(default=None, description="Default VAE for this model (model key)")
    vae_precision: DEFAULTS_PRECISION | None = Field(default=None, description="Default VAE precision for this model")
    scheduler: SCHEDULER_NAME_VALUES | None = Field(default=None, description="Default scheduler for this model")
    steps: int | None = Field(default=None, gt=0, description="Default number of steps for this model")
    cfg_scale: float | None = Field(default=None, ge=1, description="Default CFG Scale for this model")
    cfg_rescale_multiplier: float | None = Field(
        default=None, ge=0, lt=1, description="Default CFG Rescale Multiplier for this model"
    )
    width: int | None = Field(default=None, multiple_of=8, ge=64, description="Default width for this model")
    height: int | None = Field(default=None, multiple_of=8, ge=64, description="Default height for this model")
    guidance: float | None = Field(default=None, ge=1, description="Default Guidance for this model")

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_base(cls, base: BaseModelType) -> Self | None:
        match base:
            case BaseModelType.StableDiffusion1:
                return cls(width=512, height=512)
            case BaseModelType.StableDiffusion2:
                return cls(width=768, height=768)
            case BaseModelType.StableDiffusionXL:
                return cls(width=1024, height=1024)
            case _:
                # TODO(psyche): Do we want defaults for other base types?
                return None


class Main_Config_Base(ABC, BaseModel):
    type: Literal[ModelType.Main] = Field(default=ModelType.Main)
    trigger_phrases: set[str] | None = Field(
        default=None,
        description="Set of trigger phrases for this model",
    )
    default_settings: MainModelDefaultSettings | None = Field(
        default=None,
        description="Default settings for this model",
    )


def _has_bnb_nf4_keys(state_dict: dict[str | int, Any]) -> bool:
    bnb_nf4_keys = {
        "double_blocks.0.img_attn.proj.weight.quant_state.bitsandbytes__nf4",
        "model.diffusion_model.double_blocks.0.img_attn.proj.weight.quant_state.bitsandbytes__nf4",
    }
    return any(key in state_dict for key in bnb_nf4_keys)


def _has_ggml_tensors(state_dict: dict[str | int, Any]) -> bool:
    return any(isinstance(v, GGMLTensor) for v in state_dict.values())


def _has_main_keys(state_dict: dict[str | int, Any]) -> bool:
    for key in state_dict.keys():
        if isinstance(key, int):
            continue
        elif key.startswith(
            (
                "cond_stage_model.",
                "first_stage_model.",
                "model.diffusion_model.",
                # Some FLUX checkpoint files contain transformer keys prefixed with "model.diffusion_model".
                # This prefix is typically used to distinguish between multiple models bundled in a single file.
                "model.diffusion_model.double_blocks.",
            )
        ):
            return True
        elif key.startswith("double_blocks.") and "ip_adapter" not in key:
            # FLUX models in the official BFL format contain keys with the "double_blocks." prefix, but we must be
            # careful to avoid false positives on XLabs FLUX IP-Adapter models.
            return True
    return False


class Main_SD_Checkpoint_Config_Base(Checkpoint_Config_Base, Main_Config_Base):
    """Model config for main checkpoint models."""

    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)

    prediction_type: SchedulerPredictionType = Field()
    variant: ModelVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_main_model(mod)

        cls._validate_base(mod)

        prediction_type = override_fields.get("prediction_type") or cls._get_scheduler_prediction_type_or_raise(mod)

        variant = override_fields.get("variant") or cls._get_variant_or_raise(mod)

        return cls(**override_fields, prediction_type=prediction_type, variant=variant)

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatchError(f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        state_dict = mod.load_state_dict()

        key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
        if key_name in state_dict and state_dict[key_name].shape[-1] == 768:
            return BaseModelType.StableDiffusion1
        if key_name in state_dict and state_dict[key_name].shape[-1] == 1024:
            return BaseModelType.StableDiffusion2

        key_name = "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"
        if key_name in state_dict and state_dict[key_name].shape[-1] == 2048:
            return BaseModelType.StableDiffusionXL
        elif key_name in state_dict and state_dict[key_name].shape[-1] == 1280:
            return BaseModelType.StableDiffusionXLRefiner

        raise NotAMatchError("unable to determine base type from state dict")

    @classmethod
    def _get_scheduler_prediction_type_or_raise(cls, mod: ModelOnDisk) -> SchedulerPredictionType:
        base = cls.model_fields["base"].default

        if base is BaseModelType.StableDiffusion2:
            state_dict = mod.load_state_dict()
            key_name = "model.diffusion_model.input_blocks.2.1.transformer_blocks.0.attn2.to_k.weight"
            if key_name in state_dict and state_dict[key_name].shape[-1] == 1024:
                if "global_step" in state_dict:
                    if state_dict["global_step"] == 220000:
                        return SchedulerPredictionType.Epsilon
                    elif state_dict["global_step"] == 110000:
                        return SchedulerPredictionType.VPrediction
            return SchedulerPredictionType.VPrediction
        else:
            return SchedulerPredictionType.Epsilon

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> ModelVariantType:
        base = cls.model_fields["base"].default

        state_dict = mod.load_state_dict()
        key_name = "model.diffusion_model.input_blocks.0.0.weight"

        if key_name not in state_dict:
            raise NotAMatchError("unable to determine model variant from state dict")

        in_channels = state_dict["model.diffusion_model.input_blocks.0.0.weight"].shape[1]

        match in_channels:
            case 4:
                return ModelVariantType.Normal
            case 5:
                # Only SD2 has a depth variant
                assert base is BaseModelType.StableDiffusion2, f"unexpected unet in_channels 5 for base '{base}'"
                return ModelVariantType.Depth
            case 9:
                return ModelVariantType.Inpaint
            case _:
                raise NotAMatchError(f"unrecognized unet in_channels {in_channels} for base '{base}'")

    @classmethod
    def _validate_looks_like_main_model(cls, mod: ModelOnDisk) -> None:
        has_main_model_keys = _has_main_keys(mod.load_state_dict())
        if not has_main_model_keys:
            raise NotAMatchError("state dict does not look like a main model")


class Main_Checkpoint_SD1_Config(Main_SD_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(default=BaseModelType.StableDiffusion1)


class Main_Checkpoint_SD2_Config(Main_SD_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(default=BaseModelType.StableDiffusion2)


class Main_Checkpoint_SDXL_Config(Main_SD_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(default=BaseModelType.StableDiffusionXL)


class Main_Checkpoint_SDXLRefiner_Config(Main_SD_Checkpoint_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXLRefiner] = Field(default=BaseModelType.StableDiffusionXLRefiner)


def _get_flux_variant(state_dict: dict[str | int, Any]) -> FluxVariantType | None:
    # FLUX Model variant types are distinguished by input channels and the presence of certain keys.

    # Input channels are derived from the shape of either "img_in.weight" or "model.diffusion_model.img_in.weight".
    #
    # Known models that use the latter key:
    # - https://civitai.com/models/885098?modelVersionId=990775
    # - https://civitai.com/models/1018060?modelVersionId=1596255
    # - https://civitai.com/models/978314/ultrareal-fine-tune?modelVersionId=1413133
    #
    # Input channels for known FLUX models:
    # - Unquantized Dev and Schnell have in_channels=64
    # - BNB-NF4 Dev and Schnell have in_channels=1
    # - FLUX Fill has in_channels=384
    # - Unsure of quantized FLUX Fill models
    # - Unsure of GGUF-quantized models

    in_channels = None
    for key in {"img_in.weight", "model.diffusion_model.img_in.weight"}:
        if key in state_dict:
            in_channels = state_dict[key].shape[1]
            break

    if in_channels is None:
        # TODO(psyche): Should we have a graceful fallback here? Previously we fell back to the "normal" variant,
        # but this variant is no longer used for FLUX models. If we get here, but the model is definitely a FLUX
        # model, we should figure out a good fallback value.
        return None

    # Because FLUX Dev and Schnell models have the same in_channels, we need to check for the presence of
    # certain keys to distinguish between them.
    is_flux_dev = (
        "guidance_in.out_layer.weight" in state_dict
        or "model.diffusion_model.guidance_in.out_layer.weight" in state_dict
    )

    if is_flux_dev and in_channels == 384:
        return FluxVariantType.DevFill
    elif is_flux_dev:
        return FluxVariantType.Dev
    else:
        # Must be a Schnell model...?
        return FluxVariantType.Schnell


class Main_Checkpoint_FLUX_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for main checkpoint models."""

    format: Literal[ModelFormat.Checkpoint] = Field(default=ModelFormat.Checkpoint)
    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)

    variant: FluxVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_main_model(mod)

        cls._validate_is_flux(mod)

        cls._validate_does_not_look_like_bnb_quantized(mod)

        cls._validate_does_not_look_like_gguf_quantized(mod)

        variant = override_fields.get("variant") or cls._get_variant_or_raise(mod)

        return cls(**override_fields, variant=variant)

    @classmethod
    def _validate_is_flux(cls, mod: ModelOnDisk) -> None:
        if not state_dict_has_any_keys_exact(
            mod.load_state_dict(),
            {
                "double_blocks.0.img_attn.norm.key_norm.scale",
                "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale",
            },
        ):
            raise NotAMatchError("state dict does not look like a FLUX checkpoint")

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> FluxVariantType:
        # FLUX Model variant types are distinguished by input channels and the presence of certain keys.
        state_dict = mod.load_state_dict()
        variant = _get_flux_variant(state_dict)

        if variant is None:
            # TODO(psyche): Should we have a graceful fallback here? Previously we fell back to the "normal" variant,
            # but this variant is no longer used for FLUX models. If we get here, but the model is definitely a FLUX
            # model, we should figure out a good fallback value.
            raise NotAMatchError("unable to determine model variant from state dict")

        return variant

    @classmethod
    def _validate_looks_like_main_model(cls, mod: ModelOnDisk) -> None:
        has_main_model_keys = _has_main_keys(mod.load_state_dict())
        if not has_main_model_keys:
            raise NotAMatchError("state dict does not look like a main model")

    @classmethod
    def _validate_does_not_look_like_bnb_quantized(cls, mod: ModelOnDisk) -> None:
        has_bnb_nf4_keys = _has_bnb_nf4_keys(mod.load_state_dict())
        if has_bnb_nf4_keys:
            raise NotAMatchError("state dict looks like bnb quantized nf4")

    @classmethod
    def _validate_does_not_look_like_gguf_quantized(cls, mod: ModelOnDisk):
        has_ggml_tensors = _has_ggml_tensors(mod.load_state_dict())
        if has_ggml_tensors:
            raise NotAMatchError("state dict looks like GGUF quantized")


class Main_BnBNF4_FLUX_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for main checkpoint models."""

    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
    format: Literal[ModelFormat.BnbQuantizednf4b] = Field(default=ModelFormat.BnbQuantizednf4b)

    variant: FluxVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_main_model(mod)

        cls._validate_model_looks_like_bnb_quantized(mod)

        variant = override_fields.get("variant") or cls._get_variant_or_raise(mod)

        return cls(**override_fields, variant=variant)

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> FluxVariantType:
        # FLUX Model variant types are distinguished by input channels and the presence of certain keys.
        state_dict = mod.load_state_dict()
        variant = _get_flux_variant(state_dict)

        if variant is None:
            # TODO(psyche): Should we have a graceful fallback here? Previously we fell back to the "normal" variant,
            # but this variant is no longer used for FLUX models. If we get here, but the model is definitely a FLUX
            # model, we should figure out a good fallback value.
            raise NotAMatchError("unable to determine model variant from state dict")

        return variant

    @classmethod
    def _validate_looks_like_main_model(cls, mod: ModelOnDisk) -> None:
        has_main_model_keys = _has_main_keys(mod.load_state_dict())
        if not has_main_model_keys:
            raise NotAMatchError("state dict does not look like a main model")

    @classmethod
    def _validate_model_looks_like_bnb_quantized(cls, mod: ModelOnDisk) -> None:
        has_bnb_nf4_keys = _has_bnb_nf4_keys(mod.load_state_dict())
        if not has_bnb_nf4_keys:
            raise NotAMatchError("state dict does not look like bnb quantized nf4")


class Main_GGUF_FLUX_Config(Checkpoint_Config_Base, Main_Config_Base, Config_Base):
    """Model config for main checkpoint models."""

    base: Literal[BaseModelType.Flux] = Field(default=BaseModelType.Flux)
    format: Literal[ModelFormat.GGUFQuantized] = Field(default=ModelFormat.GGUFQuantized)

    variant: FluxVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_file(mod)

        raise_for_override_fields(cls, override_fields)

        cls._validate_looks_like_main_model(mod)

        cls._validate_looks_like_gguf_quantized(mod)

        variant = override_fields.get("variant") or cls._get_variant_or_raise(mod)

        return cls(**override_fields, variant=variant)

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> FluxVariantType:
        # FLUX Model variant types are distinguished by input channels and the presence of certain keys.
        state_dict = mod.load_state_dict()
        variant = _get_flux_variant(state_dict)

        if variant is None:
            # TODO(psyche): Should we have a graceful fallback here? Previously we fell back to the "normal" variant,
            # but this variant is no longer used for FLUX models. If we get here, but the model is definitely a FLUX
            # model, we should figure out a good fallback value.
            raise NotAMatchError("unable to determine model variant from state dict")

        return variant

    @classmethod
    def _validate_looks_like_main_model(cls, mod: ModelOnDisk) -> None:
        has_main_model_keys = _has_main_keys(mod.load_state_dict())
        if not has_main_model_keys:
            raise NotAMatchError("state dict does not look like a main model")

    @classmethod
    def _validate_looks_like_gguf_quantized(cls, mod: ModelOnDisk) -> None:
        has_ggml_tensors = _has_ggml_tensors(mod.load_state_dict())
        if not has_ggml_tensors:
            raise NotAMatchError("state dict does not look like GGUF quantized")


class Main_SD_Diffusers_Config_Base(Diffusers_Config_Base, Main_Config_Base):
    prediction_type: SchedulerPredictionType = Field()
    variant: ModelVariantType = Field()

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        raise_for_class_name(
            common_config_paths(mod.path),
            {
                # SD 1.x and 2.x
                "StableDiffusionPipeline",
                "StableDiffusionInpaintPipeline",
                # SDXL
                "StableDiffusionXLPipeline",
                "StableDiffusionXLInpaintPipeline",
                # SDXL Refiner
                "StableDiffusionXLImg2ImgPipeline",
                # TODO(psyche): Do we actually support LCM models? I don't see using this class anywhere in the codebase.
                "LatentConsistencyModelPipeline",
            },
        )

        cls._validate_base(mod)

        variant = override_fields.get("variant") or cls._get_variant_or_raise(mod)

        prediction_type = override_fields.get("prediction_type") or cls._get_scheduler_prediction_type_or_raise(mod)

        repo_variant = override_fields.get("repo_variant") or cls._get_repo_variant_or_raise(mod)

        return cls(
            **override_fields,
            variant=variant,
            prediction_type=prediction_type,
            repo_variant=repo_variant,
        )

    @classmethod
    def _validate_base(cls, mod: ModelOnDisk) -> None:
        """Raise `NotAMatch` if the model base does not match this config class."""
        expected_base = cls.model_fields["base"].default
        recognized_base = cls._get_base_or_raise(mod)
        if expected_base is not recognized_base:
            raise NotAMatchError(f"base is {recognized_base}, not {expected_base}")

    @classmethod
    def _get_base_or_raise(cls, mod: ModelOnDisk) -> BaseModelType:
        # Handle pipelines with a UNet (i.e SD 1.x, SD2.x, SDXL).
        unet_conf = get_config_dict_or_raise(mod.path / "unet" / "config.json")
        cross_attention_dim = unet_conf.get("cross_attention_dim")
        match cross_attention_dim:
            case 768:
                return BaseModelType.StableDiffusion1
            case 1024:
                return BaseModelType.StableDiffusion2
            case 1280:
                return BaseModelType.StableDiffusionXLRefiner
            case 2048:
                return BaseModelType.StableDiffusionXL
            case _:
                raise NotAMatchError(f"unrecognized cross_attention_dim {cross_attention_dim}")

    @classmethod
    def _get_scheduler_prediction_type_or_raise(cls, mod: ModelOnDisk) -> SchedulerPredictionType:
        scheduler_conf = get_config_dict_or_raise(mod.path / "scheduler" / "scheduler_config.json")

        # TODO(psyche): Is epsilon the right default or should we raise if it's not present?
        prediction_type = scheduler_conf.get("prediction_type", "epsilon")

        match prediction_type:
            case "v_prediction":
                return SchedulerPredictionType.VPrediction
            case "epsilon":
                return SchedulerPredictionType.Epsilon
            case _:
                raise NotAMatchError(f"unrecognized scheduler prediction_type {prediction_type}")

    @classmethod
    def _get_variant_or_raise(cls, mod: ModelOnDisk) -> ModelVariantType:
        base = cls.model_fields["base"].default
        unet_config = get_config_dict_or_raise(mod.path / "unet" / "config.json")
        in_channels = unet_config.get("in_channels")

        match in_channels:
            case 4:
                return ModelVariantType.Normal
            case 5:
                # Only SD2 has a depth variant
                assert base is BaseModelType.StableDiffusion2, f"unexpected unet in_channels 5 for base '{base}'"
                return ModelVariantType.Depth
            case 9:
                return ModelVariantType.Inpaint
            case _:
                raise NotAMatchError(f"unrecognized unet in_channels {in_channels} for base '{base}'")


class Main_Diffusers_SD1_Config(Main_SD_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion1] = Field(BaseModelType.StableDiffusion1)


class Main_Diffusers_SD2_Config(Main_SD_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion2] = Field(BaseModelType.StableDiffusion2)


class Main_Diffusers_SDXL_Config(Main_SD_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXL] = Field(BaseModelType.StableDiffusionXL)


class Main_Diffusers_SDXLRefiner_Config(Main_SD_Diffusers_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusionXLRefiner] = Field(BaseModelType.StableDiffusionXLRefiner)


class Main_Diffusers_SD3_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base):
    base: Literal[BaseModelType.StableDiffusion3] = Field(BaseModelType.StableDiffusion3)
    submodels: dict[SubModelType, SubmodelDefinition] | None = Field(
        description="Loadable submodels in this model",
        default=None,
    )

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # This check implies the base type - no further validation needed.
        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "StableDiffusion3Pipeline",
                "SD3Transformer2DModel",
            },
        )

        submodels = override_fields.get("submodels") or cls._get_submodels_or_raise(mod)

        repo_variant = override_fields.get("repo_variant") or cls._get_repo_variant_or_raise(mod)

        return cls(
            **override_fields,
            submodels=submodels,
            repo_variant=repo_variant,
        )

    @classmethod
    def _get_submodels_or_raise(cls, mod: ModelOnDisk) -> dict[SubModelType, SubmodelDefinition]:
        # Example: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/blob/main/model_index.json
        config = get_config_dict_or_raise(common_config_paths(mod.path))

        submodels: dict[SubModelType, SubmodelDefinition] = {}

        for key, value in config.items():
            # Anything that starts with an underscore is top-level metadata, not a submodel
            if key.startswith("_") or not (isinstance(value, list) and len(value) == 2):
                continue
            # The key is something like "transformer" and is a submodel - it will be in a dir of the same name.
            # The value value is something like ["diffusers", "SD3Transformer2DModel"]
            _library_name, class_name = value

            match class_name:
                case "CLIPTextModelWithProjection":
                    model_type = ModelType.CLIPEmbed
                    path_or_prefix = (mod.path / key).resolve().as_posix()

                    # We need to read the config to determine the variant of the CLIP model.
                    clip_embed_config = get_config_dict_or_raise(
                        {
                            mod.path / key / "config.json",
                            mod.path / key / "model_index.json",
                        }
                    )
                    variant = get_clip_variant_type_from_config(clip_embed_config)
                    submodels[SubModelType(key)] = SubmodelDefinition(
                        path_or_prefix=path_or_prefix,
                        model_type=model_type,
                        variant=variant,
                    )
                case "SD3Transformer2DModel":
                    model_type = ModelType.Main
                    path_or_prefix = (mod.path / key).resolve().as_posix()
                    variant = None
                    submodels[SubModelType(key)] = SubmodelDefinition(
                        path_or_prefix=path_or_prefix,
                        model_type=model_type,
                        variant=variant,
                    )
                case _:
                    pass

        return submodels


class Main_Diffusers_CogView4_Config(Diffusers_Config_Base, Main_Config_Base, Config_Base):
    base: Literal[BaseModelType.CogView4] = Field(BaseModelType.CogView4)

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, Any]) -> Self:
        raise_if_not_dir(mod)

        raise_for_override_fields(cls, override_fields)

        # This check implies the base type - no further validation needed.
        raise_for_class_name(
            common_config_paths(mod.path),
            {
                "CogView4Pipeline",
            },
        )

        repo_variant = override_fields.get("repo_variant") or cls._get_repo_variant_or_raise(mod)

        return cls(
            **override_fields,
            repo_variant=repo_variant,
        )
