import logging
from dataclasses import dataclass
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
    ControlAdapterDefaultSettings,
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
    LoraModelDefaultSettings,
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
    Main_GGUF_FLUX_Config,
    MainModelDefaultSettings,
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

# Known model file extensions for sanity checking
_MODEL_EXTENSIONS = {
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".gguf",
    ".onnx",
}

# Known config file names for diffusers/transformers models
_CONFIG_FILES = {
    "model_index.json",
    "config.json",
}

# Maximum number of files in a directory to be considered a model
_MAX_FILES_IN_MODEL_DIR = 50

# Maximum depth to search for model files in directories
_MAX_SEARCH_DEPTH = 2


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
        # Unknown model (fallback)
        Annotated[Unknown_Config, Unknown_Config.get_tag()],
    ],
    Discriminator(Config_Base.get_model_discriminator_value),
]

AnyModelConfigValidator = TypeAdapter[AnyModelConfig](AnyModelConfig)
"""Pydantic TypeAdapter for the AnyModelConfig union, used for parsing and validation.

If you need to parse/validate a dict or JSON into an AnyModelConfig, you should probably use
ModelConfigFactory.from_dict or ModelConfigFactory.from_json instead as they may implement
additional logic in the future.
"""


@dataclass
class ModelClassificationResult:
    """Result of attempting to classify a model on disk into a specific model config.

    Attributes:
        match: The best matching model config, or None if no match was found.
        results: A mapping of model config class names to either an instance of that class (if it matched)
            or an Exception (if it didn't match or an error occurred during matching).
    """

    config: AnyModelConfig | None
    details: dict[str, AnyModelConfig | Exception]

    @property
    def all_matches(self) -> list[AnyModelConfig]:
        """Returns a list of all matching model configs found."""
        return [r for r in self.details.values() if isinstance(r, Config_Base)]

    @property
    def match_count(self) -> int:
        """Returns the number of matching model configs found."""
        return len(self.all_matches)


class ModelConfigFactory:
    @staticmethod
    def from_dict(fields: dict[str, Any]) -> AnyModelConfig:
        """Return the appropriate config object from raw dict values."""
        model = AnyModelConfigValidator.validate_python(fields)
        return model

    @staticmethod
    def from_json(json: str | bytes | bytearray) -> AnyModelConfig:
        """Return the appropriate config object from json."""
        model = AnyModelConfigValidator.validate_json(json)
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
    def _validate_path_looks_like_model(path: Path) -> None:
        """Perform basic sanity checks to ensure a path looks like a model.

        This prevents wasting time trying to identify obviously non-model paths like
        home directories or downloads folders. Raises RuntimeError if the path doesn't
        pass basic checks.

        Args:
            path: The path to validate

        Raises:
            ValueError: If the path doesn't look like a model
        """
        if path.is_file():
            # For files, just check the extension
            if path.suffix.lower() not in _MODEL_EXTENSIONS:
                raise ValueError(
                    f"File extension {path.suffix} is not a recognized model format. "
                    f"Expected one of: {', '.join(sorted(_MODEL_EXTENSIONS))}"
                )
        else:
            # For directories, do a quick file count check with early exit
            total_files = 0
            # Ignore hidden files and directories
            paths_to_check = (
                p
                for p in path.rglob("*")
                if not p.name.startswith(".") and not any(part.startswith(".") for part in p.parts)
            )
            for item in paths_to_check:
                if item.is_file():
                    total_files += 1
                    if total_files > _MAX_FILES_IN_MODEL_DIR:
                        raise ValueError(
                            f"Directory contains more than {_MAX_FILES_IN_MODEL_DIR} files. "
                            "This looks like a general-purpose directory rather than a model. "
                            "Please provide a path to a specific model file or model directory."
                        )

            # Check if it has config files at root (diffusers/transformers marker)
            has_root_config = any((path / config).exists() for config in _CONFIG_FILES)

            if has_root_config:
                # Has a config file, looks like a valid model directory
                return

            # Otherwise, search for model files within depth limit
            def find_model_files(current_path: Path, depth: int) -> bool:
                if depth > _MAX_SEARCH_DEPTH:
                    return False
                try:
                    for item in current_path.iterdir():
                        if item.is_file() and item.suffix.lower() in _MODEL_EXTENSIONS:
                            return True
                        elif item.is_dir() and find_model_files(item, depth + 1):
                            return True
                except PermissionError:
                    pass
                return False

            if not find_model_files(path, 0):
                raise ValueError(
                    f"No model files or config files found in directory {path}. "
                    f"Expected to find model files with extensions: {', '.join(sorted(_MODEL_EXTENSIONS))} "
                    f"or config files: {', '.join(sorted(_CONFIG_FILES))}"
                )

    @staticmethod
    def matches_sort_key(m: AnyModelConfig) -> int:
        """Sort key function to prioritize model config matches in case of multiple matches."""

        # It is possible that we have multiple matches. We need to prioritize them.

        # Known cases where multiple matches can occur:
        # - SD main models can look like a LoRA when they have merged in LoRA weights. Prefer the main model.
        # - SD main models in diffusers format can look like a CLIP Embed; they have a text_encoder folder with
        #   a config.json file. Prefer the main model.

        # Given the above cases, we can prioritize the matches by type. If we find more cases, we may need a more
        # sophisticated approach.
        match m.type:
            case ModelType.Main:
                return 0
            case ModelType.LoRA:
                return 1
            case ModelType.CLIPEmbed:
                return 2
            case _:
                return 3

    @staticmethod
    def from_model_on_disk(
        mod: str | Path | ModelOnDisk,
        override_fields: dict[str, Any] | None = None,
        hash_algo: HASHING_ALGORITHMS = "blake3_single",
        allow_unknown: bool = True,
    ) -> ModelClassificationResult:
        """Classify a model on disk and return the best matching model config.

        Args:
            mod: The model on disk to classify. Can be a path (str or Path) or a ModelOnDisk instance.
            override_fields: Optional dictionary of fields to override. These fields will take precedence
                over the values extracted from the model on disk, but this cannot force a match if the
                model on disk doesn't actually match the config class.
            hash_algo: The hashing algorithm to use when computing the model hash if needed.

        Returns:
            A ModelClassificationResult containing the best matching model config (or None if no match)
            and a mapping of all attempted model config classes to either an instance of that class (if it matched)
            or an Exception (if it didn't match or an error occurred during matching).

        Raises:
            ValueError: If the provided path doesn't look like a model.
        """
        if isinstance(mod, Path | str):
            mod = ModelOnDisk(Path(mod), hash_algo)

        # Perform basic sanity checks before attempting any config matching
        # This rejects obviously non-model paths early, saving time
        ModelConfigFactory._validate_path_looks_like_model(mod.path)

        # We will always need these fields to build any model config.
        fields = ModelConfigFactory.build_common_fields(mod, override_fields)

        # Store results as a mapping of config class to either an instance of that class or an exception
        # that was raised when trying to build it.
        details: dict[str, AnyModelConfig | Exception] = {}

        # Try to build an instance of each model config class that uses the classify API.
        # Each class will either return an instance of itself or raise NotAMatch if it doesn't match.
        # Other exceptions may be raised if something unexpected happens during matching or building.
        for candidate_class in filter(lambda x: x is not Unknown_Config, Config_Base.CONFIG_CLASSES):
            candidate_name = candidate_class.__name__
            try:
                # Technically, from_model_on_disk returns a Config_Base, but in practice it will always be a member of
                # the AnyModelConfig union.
                details[candidate_name] = candidate_class.from_model_on_disk(mod, fields)  # type: ignore
            except NotAMatchError as e:
                # This means the model didn't match this config class. It's not an error, just no match.
                details[candidate_name] = e
            except ValidationError as e:
                # This means the model matched, but we couldn't create the pydantic model instance for the config.
                # Maybe invalid overrides were provided?
                details[candidate_name] = e
            except Exception as e:
                # Some other unexpected error occurred. Store the exception for reporting later.
                details[candidate_name] = e

        # Extract just the successful matches
        matches = [r for r in details.values() if isinstance(r, Config_Base)]

        if not matches:
            if not allow_unknown:
                # No matches and we are not allowed to fall back to Unknown_Config
                return ModelClassificationResult(config=None, details=details)
            else:
                # Fall back to Unknown_Config
                # This should always succeed as Unknown_Config.from_model_on_disk never raises NotAMatch
                config = Unknown_Config.from_model_on_disk(mod, fields)
                details[Unknown_Config.__name__] = config
                return ModelClassificationResult(config=config, details=details)

        matches.sort(key=ModelConfigFactory.matches_sort_key)
        config = matches[0]

        # Now do any post-processing needed for specific model types/bases/etc.
        match config.type:
            case ModelType.Main:
                config.default_settings = MainModelDefaultSettings.from_base(config.base)
            case ModelType.ControlNet | ModelType.T2IAdapter | ModelType.ControlLoRa:
                config.default_settings = ControlAdapterDefaultSettings.from_model_name(config.name)
            case ModelType.LoRA:
                config.default_settings = LoraModelDefaultSettings()
            case _:
                pass

        return ModelClassificationResult(config=config, details=details)


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
