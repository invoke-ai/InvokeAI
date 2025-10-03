import logging
from pathlib import Path
from typing import (
    Union,
)

from pydantic import Discriminator, TypeAdapter, ValidationError
from typing_extensions import Annotated, Any

from invokeai.app.services.config.config_default import get_config
from invokeai.app.util.misc import uuid_string
from invokeai.backend.model_hash.model_hash import HASHING_ALGORITHMS
from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.clip_embed import CLIPEmbed_Diffusers_G_Config, CLIPEmbed_Diffusers_L_Config
from invokeai.backend.model_manager.configs.clip_vision import CLIPVision_Diffusers_Config
from invokeai.backend.model_manager.configs.controlnet import (
    ControlNet_Checkpoint_FLUX_Config,
    ControlNet_Checkpoint_SD1_Config,
    ControlNet_Checkpoint_SD2_Config,
    ControlNet_Checkpoint_SDXL_Config,
    ControlNet_Diffusers_FLUX_Config,
    ControlNet_Diffusers_SD1_Config,
    ControlNet_Diffusers_SD2_Config,
    ControlNet_Diffusers_SDXL_Config,
)
from invokeai.backend.model_manager.configs.flux_redux import FLUXRedux_Checkpoint_Config
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.ip_adapter import (
    IPAdapter_Checkpoint_FLUX_Config,
    IPAdapter_Checkpoint_SD1_Config,
    IPAdapter_Checkpoint_SD2_Config,
    IPAdapter_Checkpoint_SDXL_Config,
    IPAdapter_InvokeAI_SD1_Config,
    IPAdapter_InvokeAI_SD2_Config,
    IPAdapter_InvokeAI_SDXL_Config,
)
from invokeai.backend.model_manager.configs.llava_onevision import LlavaOnevision_Diffusers_Config
from invokeai.backend.model_manager.configs.lora import (
    ControlLoRA_LyCORIS_FLUX_Config,
    LoRA_Diffusers_FLUX_Config,
    LoRA_Diffusers_SD1_Config,
    LoRA_Diffusers_SD2_Config,
    LoRA_Diffusers_SDXL_Config,
    LoRA_LyCORIS_FLUX_Config,
    LoRA_LyCORIS_SD1_Config,
    LoRA_LyCORIS_SD2_Config,
    LoRA_LyCORIS_SDXL_Config,
    LoRA_OMI_FLUX_Config,
    LoRA_OMI_SDXL_Config,
)
from invokeai.backend.model_manager.configs.main import (
    Main_BnBNF4_FLUX_Config,
    Main_Checkpoint_FLUX_Config,
    Main_Checkpoint_SD1_Config,
    Main_Checkpoint_SD2_Config,
    Main_Checkpoint_SDXL_Config,
    Main_Checkpoint_SDXLRefiner_Config,
    Main_Diffusers_CogView4_Config,
    Main_Diffusers_SD1_Config,
    Main_Diffusers_SD2_Config,
    Main_Diffusers_SD3_Config,
    Main_Diffusers_SDXL_Config,
    Main_Diffusers_SDXLRefiner_Config,
    Main_ExternalAPI_ChatGPT4o_Config,
    Main_ExternalAPI_FluxKontext_Config,
    Main_ExternalAPI_Gemini2_5_Config,
    Main_ExternalAPI_Imagen3_Config,
    Main_ExternalAPI_Imagen4_Config,
    Main_GGUF_FLUX_Config,
    Video_ExternalAPI_Runway_Config,
    Video_ExternalAPI_Veo3_Config,
)
from invokeai.backend.model_manager.configs.siglip import SigLIP_Diffusers_Config
from invokeai.backend.model_manager.configs.spandrel import Spandrel_Checkpoint_Config
from invokeai.backend.model_manager.configs.t2i_adapter import (
    T2IAdapter_Diffusers_SD1_Config,
    T2IAdapter_Diffusers_SDXL_Config,
)
from invokeai.backend.model_manager.configs.t5_encoder import T5Encoder_BnBLLMint8_Config, T5Encoder_T5Encoder_Config
from invokeai.backend.model_manager.configs.textual_inversion import (
    TI_File_SD1_Config,
    TI_File_SD2_Config,
    TI_File_SDXL_Config,
    TI_Folder_SD1_Config,
    TI_Folder_SD2_Config,
    TI_Folder_SDXL_Config,
)
from invokeai.backend.model_manager.configs.unknown import Unknown_Config
from invokeai.backend.model_manager.configs.vae import (
    VAE_Checkpoint_FLUX_Config,
    VAE_Checkpoint_SD1_Config,
    VAE_Checkpoint_SD2_Config,
    VAE_Checkpoint_SDXL_Config,
    VAE_Diffusers_SD1_Config,
    VAE_Diffusers_SDXL_Config,
)
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelSourceType,
    ModelType,
    variant_type_adapter,
)

logger = logging.getLogger(__name__)
app_config = get_config()


# The types are listed explicitly because IDEs/LSPs can't identify the correct types
# when AnyModelConfig is constructed dynamically using ModelConfigBase.all_config_classes
AnyModelConfig = Annotated[
    Union[
        # Main (Pipeline) - diffusers format
        Annotated[Main_Diffusers_SD1_Config, Main_Diffusers_SD1_Config.get_tag()],
        Annotated[Main_Diffusers_SD2_Config, Main_Diffusers_SD2_Config.get_tag()],
        Annotated[Main_Diffusers_SDXL_Config, Main_Diffusers_SDXL_Config.get_tag()],
        Annotated[Main_Diffusers_SDXLRefiner_Config, Main_Diffusers_SDXLRefiner_Config.get_tag()],
        Annotated[Main_Diffusers_SD3_Config, Main_Diffusers_SD3_Config.get_tag()],
        Annotated[Main_Diffusers_CogView4_Config, Main_Diffusers_CogView4_Config.get_tag()],
        # Main (Pipeline) - checkpoint format
        Annotated[Main_Checkpoint_SD1_Config, Main_Checkpoint_SD1_Config.get_tag()],
        Annotated[Main_Checkpoint_SD2_Config, Main_Checkpoint_SD2_Config.get_tag()],
        Annotated[Main_Checkpoint_SDXL_Config, Main_Checkpoint_SDXL_Config.get_tag()],
        Annotated[Main_Checkpoint_SDXLRefiner_Config, Main_Checkpoint_SDXLRefiner_Config.get_tag()],
        Annotated[Main_Checkpoint_FLUX_Config, Main_Checkpoint_FLUX_Config.get_tag()],
        # Main (Pipeline) - quantized formats
        Annotated[Main_BnBNF4_FLUX_Config, Main_BnBNF4_FLUX_Config.get_tag()],
        Annotated[Main_GGUF_FLUX_Config, Main_GGUF_FLUX_Config.get_tag()],
        # VAE - checkpoint format
        Annotated[VAE_Checkpoint_SD1_Config, VAE_Checkpoint_SD1_Config.get_tag()],
        Annotated[VAE_Checkpoint_SD2_Config, VAE_Checkpoint_SD2_Config.get_tag()],
        Annotated[VAE_Checkpoint_SDXL_Config, VAE_Checkpoint_SDXL_Config.get_tag()],
        Annotated[VAE_Checkpoint_FLUX_Config, VAE_Checkpoint_FLUX_Config.get_tag()],
        # VAE - diffusers format
        Annotated[VAE_Diffusers_SD1_Config, VAE_Diffusers_SD1_Config.get_tag()],
        Annotated[VAE_Diffusers_SDXL_Config, VAE_Diffusers_SDXL_Config.get_tag()],
        # ControlNet - checkpoint format
        Annotated[ControlNet_Checkpoint_SD1_Config, ControlNet_Checkpoint_SD1_Config.get_tag()],
        Annotated[ControlNet_Checkpoint_SD2_Config, ControlNet_Checkpoint_SD2_Config.get_tag()],
        Annotated[ControlNet_Checkpoint_SDXL_Config, ControlNet_Checkpoint_SDXL_Config.get_tag()],
        Annotated[ControlNet_Checkpoint_FLUX_Config, ControlNet_Checkpoint_FLUX_Config.get_tag()],
        # ControlNet - diffusers format
        Annotated[ControlNet_Diffusers_SD1_Config, ControlNet_Diffusers_SD1_Config.get_tag()],
        Annotated[ControlNet_Diffusers_SD2_Config, ControlNet_Diffusers_SD2_Config.get_tag()],
        Annotated[ControlNet_Diffusers_SDXL_Config, ControlNet_Diffusers_SDXL_Config.get_tag()],
        Annotated[ControlNet_Diffusers_FLUX_Config, ControlNet_Diffusers_FLUX_Config.get_tag()],
        # LoRA - LyCORIS format
        Annotated[LoRA_LyCORIS_SD1_Config, LoRA_LyCORIS_SD1_Config.get_tag()],
        Annotated[LoRA_LyCORIS_SD2_Config, LoRA_LyCORIS_SD2_Config.get_tag()],
        Annotated[LoRA_LyCORIS_SDXL_Config, LoRA_LyCORIS_SDXL_Config.get_tag()],
        Annotated[LoRA_LyCORIS_FLUX_Config, LoRA_LyCORIS_FLUX_Config.get_tag()],
        # LoRA - OMI format
        Annotated[LoRA_OMI_SDXL_Config, LoRA_OMI_SDXL_Config.get_tag()],
        Annotated[LoRA_OMI_FLUX_Config, LoRA_OMI_FLUX_Config.get_tag()],
        # LoRA - diffusers format
        Annotated[LoRA_Diffusers_SD1_Config, LoRA_Diffusers_SD1_Config.get_tag()],
        Annotated[LoRA_Diffusers_SD2_Config, LoRA_Diffusers_SD2_Config.get_tag()],
        Annotated[LoRA_Diffusers_SDXL_Config, LoRA_Diffusers_SDXL_Config.get_tag()],
        Annotated[LoRA_Diffusers_FLUX_Config, LoRA_Diffusers_FLUX_Config.get_tag()],
        # ControlLoRA - diffusers format
        Annotated[ControlLoRA_LyCORIS_FLUX_Config, ControlLoRA_LyCORIS_FLUX_Config.get_tag()],
        # T5 Encoder - all formats
        Annotated[T5Encoder_T5Encoder_Config, T5Encoder_T5Encoder_Config.get_tag()],
        Annotated[T5Encoder_BnBLLMint8_Config, T5Encoder_BnBLLMint8_Config.get_tag()],
        # TI - file format
        Annotated[TI_File_SD1_Config, TI_File_SD1_Config.get_tag()],
        Annotated[TI_File_SD2_Config, TI_File_SD2_Config.get_tag()],
        Annotated[TI_File_SDXL_Config, TI_File_SDXL_Config.get_tag()],
        # TI - folder format
        Annotated[TI_Folder_SD1_Config, TI_Folder_SD1_Config.get_tag()],
        Annotated[TI_Folder_SD2_Config, TI_Folder_SD2_Config.get_tag()],
        Annotated[TI_Folder_SDXL_Config, TI_Folder_SDXL_Config.get_tag()],
        # IP Adapter - InvokeAI format
        Annotated[IPAdapter_InvokeAI_SD1_Config, IPAdapter_InvokeAI_SD1_Config.get_tag()],
        Annotated[IPAdapter_InvokeAI_SD2_Config, IPAdapter_InvokeAI_SD2_Config.get_tag()],
        Annotated[IPAdapter_InvokeAI_SDXL_Config, IPAdapter_InvokeAI_SDXL_Config.get_tag()],
        # IP Adapter - checkpoint format
        Annotated[IPAdapter_Checkpoint_SD1_Config, IPAdapter_Checkpoint_SD1_Config.get_tag()],
        Annotated[IPAdapter_Checkpoint_SD2_Config, IPAdapter_Checkpoint_SD2_Config.get_tag()],
        Annotated[IPAdapter_Checkpoint_SDXL_Config, IPAdapter_Checkpoint_SDXL_Config.get_tag()],
        Annotated[IPAdapter_Checkpoint_FLUX_Config, IPAdapter_Checkpoint_FLUX_Config.get_tag()],
        # T2I Adapter - diffusers format
        Annotated[T2IAdapter_Diffusers_SD1_Config, T2IAdapter_Diffusers_SD1_Config.get_tag()],
        Annotated[T2IAdapter_Diffusers_SDXL_Config, T2IAdapter_Diffusers_SDXL_Config.get_tag()],
        # Misc models
        Annotated[Spandrel_Checkpoint_Config, Spandrel_Checkpoint_Config.get_tag()],
        Annotated[CLIPEmbed_Diffusers_G_Config, CLIPEmbed_Diffusers_G_Config.get_tag()],
        Annotated[CLIPEmbed_Diffusers_L_Config, CLIPEmbed_Diffusers_L_Config.get_tag()],
        Annotated[CLIPVision_Diffusers_Config, CLIPVision_Diffusers_Config.get_tag()],
        Annotated[SigLIP_Diffusers_Config, SigLIP_Diffusers_Config.get_tag()],
        Annotated[FLUXRedux_Checkpoint_Config, FLUXRedux_Checkpoint_Config.get_tag()],
        Annotated[LlavaOnevision_Diffusers_Config, LlavaOnevision_Diffusers_Config.get_tag()],
        # Main - external API
        Annotated[Main_ExternalAPI_ChatGPT4o_Config, Main_ExternalAPI_ChatGPT4o_Config.get_tag()],
        Annotated[Main_ExternalAPI_Gemini2_5_Config, Main_ExternalAPI_Gemini2_5_Config.get_tag()],
        Annotated[Main_ExternalAPI_Imagen3_Config, Main_ExternalAPI_Imagen3_Config.get_tag()],
        Annotated[Main_ExternalAPI_Imagen4_Config, Main_ExternalAPI_Imagen4_Config.get_tag()],
        Annotated[Main_ExternalAPI_FluxKontext_Config, Main_ExternalAPI_FluxKontext_Config.get_tag()],
        # Video - external API
        Annotated[Video_ExternalAPI_Veo3_Config, Video_ExternalAPI_Veo3_Config.get_tag()],
        Annotated[Video_ExternalAPI_Runway_Config, Video_ExternalAPI_Runway_Config.get_tag()],
        # Unknown model (fallback)
        Annotated[Unknown_Config, Unknown_Config.get_tag()],
    ],
    Discriminator(Config_Base.get_model_discriminator_value),
]

AnyModelConfigValidator = TypeAdapter[AnyModelConfig](AnyModelConfig)


class ModelConfigFactory:
    @staticmethod
    def from_dict(fields: dict[str, Any]) -> AnyModelConfig:
        """Return the appropriate config object from raw dict values."""
        model = AnyModelConfigValidator.validate_python(fields)
        return model

    @staticmethod
    def build_common_fields(
        mod: ModelOnDisk,
        override_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Builds the common fields for all model configs.

        Args:
            mod: The model on disk to extract fields from.
            overrides: A optional dictionary of fields to override. These fields will take precedence over the values
                extracted from the model on disk.

        - Casts string fields to their Enum types.
        - Does not validate the fields against the model config schema.
        """

        _overrides: dict[str, Any] = override_fields or {}
        fields: dict[str, Any] = {}

        if "type" in _overrides:
            fields["type"] = ModelType(_overrides["type"])

        if "format" in _overrides:
            fields["format"] = ModelFormat(_overrides["format"])

        if "base" in _overrides:
            fields["base"] = BaseModelType(_overrides["base"])

        if "source_type" in _overrides:
            fields["source_type"] = ModelSourceType(_overrides["source_type"])

        if "variant" in _overrides:
            fields["variant"] = variant_type_adapter.validate_strings(_overrides["variant"])

        fields["path"] = mod.path.as_posix()
        fields["source"] = _overrides.get("source") or fields["path"]
        fields["source_type"] = _overrides.get("source_type") or ModelSourceType.Path
        fields["name"] = _overrides.get("name") or mod.name
        fields["hash"] = _overrides.get("hash") or mod.hash()
        fields["key"] = _overrides.get("key") or uuid_string()
        fields["description"] = _overrides.get("description")
        fields["file_size"] = _overrides.get("file_size") or mod.size()

        return fields

    @staticmethod
    def from_model_on_disk(
        mod: str | Path | ModelOnDisk,
        override_fields: dict[str, Any] | None = None,
        hash_algo: HASHING_ALGORITHMS = "blake3_single",
    ) -> AnyModelConfig:
        """
        Returns the best matching ModelConfig instance from a model's file/folder path.
        Raises InvalidModelConfigException if no valid configuration is found.
        Created to deprecate ModelProbe.probe
        """
        if isinstance(mod, Path | str):
            mod = ModelOnDisk(Path(mod), hash_algo)

        # We will always need these fields to build any model config.
        fields = ModelConfigFactory.build_common_fields(mod, override_fields)

        # Store results as a mapping of config class to either an instance of that class or an exception
        # that was raised when trying to build it.
        results: dict[str, AnyModelConfig | Exception] = {}

        # Try to build an instance of each model config class that uses the classify API.
        # Each class will either return an instance of itself or raise NotAMatch if it doesn't match.
        # Other exceptions may be raised if something unexpected happens during matching or building.
        for config_class in Config_Base.CONFIG_CLASSES:
            class_name = config_class.__name__
            try:
                instance = config_class.from_model_on_disk(mod, fields)
                # Technically, from_model_on_disk returns a Config_Base, but in practice it will always be a member of
                # the AnyModelConfig union.
                results[class_name] = instance  # type: ignore
            except NotAMatchError as e:
                results[class_name] = e
                logger.debug(f"No match for {config_class.__name__} on model {mod.name}")
            except ValidationError as e:
                # This means the model matched, but we couldn't create the pydantic model instance for the config.
                # Maybe invalid overrides were provided?
                results[class_name] = e
                logger.warning(f"Schema validation error for {config_class.__name__} on model {mod.name}: {e}")
            except Exception as e:
                results[class_name] = e
                logger.warning(f"Unexpected exception while matching {mod.name} to {config_class.__name__}: {e}")

        matches = [r for r in results.values() if isinstance(r, Config_Base)]

        if not matches and app_config.allow_unknown_models:
            logger.warning(f"Unable to identify model {mod.name}, falling back to Unknown_Config")
            return Unknown_Config(**fields)

        if len(matches) > 1:
            # We have multiple matches, in which case at most 1 is correct. We need to pick one.
            #
            # Known cases:
            # - SD main models can look like a LoRA when they have merged in LoRA weights. Prefer the main model.
            # - SD main models in diffusers format can look like a CLIP Embed; they have a text_encoder folder with
            #   a config.json file. Prefer the main model.

            # Sort the matching according to known special cases.
            def sort_key(m: AnyModelConfig) -> int:
                match m.type:
                    case ModelType.Main:
                        return 0
                    case ModelType.LoRA:
                        return 1
                    case ModelType.CLIPEmbed:
                        return 2
                    case _:
                        return 3

            matches.sort(key=sort_key)
            logger.warning(
                f"Multiple model config classes matched for model {mod.name}: {[type(m).__name__ for m in matches]}. Using {type(matches[0]).__name__}."
            )

        instance = matches[0]
        logger.info(f"Model {mod.name} classified as {type(instance).__name__}")
        return instance
