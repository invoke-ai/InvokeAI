from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from invokeai.backend.model_manager.configs.base import Config_Base
from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.model_on_disk import ModelOnDisk
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelSourceType, ModelType

ExternalGenerationMode = Literal["txt2img", "img2img", "inpaint"]
ExternalMaskFormat = Literal["alpha", "binary", "none"]
ExternalPanelControlName = Literal["reference_images", "dimensions", "seed"]


class ExternalImageSize(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    model_config = ConfigDict(extra="forbid")


class ExternalResolutionPreset(BaseModel):
    label: str = Field(min_length=1, description="Display label, e.g. '1:1 (1K)'")
    aspect_ratio: str = Field(min_length=1, description="Aspect ratio string, e.g. '1:1'")
    image_size: str = Field(min_length=1, description="Image size preset, e.g. '1K'")
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    model_config = ConfigDict(extra="forbid")


class ExternalModelCapabilities(BaseModel):
    modes: list[ExternalGenerationMode] = Field(default_factory=lambda: ["txt2img"])
    supports_reference_images: bool = Field(default=False)
    supports_negative_prompt: bool = Field(default=True)
    supports_seed: bool = Field(default=False)
    supports_guidance: bool = Field(default=False)
    supports_steps: bool = Field(default=False)
    max_images_per_request: int | None = Field(default=None, gt=0)
    max_image_size: ExternalImageSize | None = Field(default=None)
    allowed_aspect_ratios: list[str] | None = Field(default=None)
    aspect_ratio_sizes: dict[str, ExternalImageSize] | None = Field(default=None)
    resolution_presets: list[ExternalResolutionPreset] | None = Field(default=None)
    max_reference_images: int | None = Field(default=None, gt=0)
    mask_format: ExternalMaskFormat = Field(default="none")
    input_image_required_for: list[ExternalGenerationMode] | None = Field(default=None)

    model_config = ConfigDict(extra="forbid")


class ExternalApiModelDefaultSettings(BaseModel):
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    num_images: int | None = Field(default=None, gt=0)

    model_config = ConfigDict(extra="forbid")


class ExternalModelPanelControl(BaseModel):
    name: ExternalPanelControlName
    slider_min: float | None = Field(default=None)
    slider_max: float | None = Field(default=None)
    number_input_min: float | None = Field(default=None)
    number_input_max: float | None = Field(default=None)
    fine_step: float | None = Field(default=None)
    coarse_step: float | None = Field(default=None)
    marks: list[float] | None = Field(default=None)

    model_config = ConfigDict(extra="forbid")


class ExternalModelPanelSchema(BaseModel):
    prompts: list[ExternalModelPanelControl] = Field(default_factory=list)
    image: list[ExternalModelPanelControl] = Field(default_factory=list)
    generation: list[ExternalModelPanelControl] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class ExternalApiModelConfig(Config_Base):
    base: Literal[BaseModelType.External] = Field(default=BaseModelType.External)
    type: Literal[ModelType.ExternalImageGenerator] = Field(default=ModelType.ExternalImageGenerator)
    format: Literal[ModelFormat.ExternalApi] = Field(default=ModelFormat.ExternalApi)

    provider_id: str = Field(min_length=1, description="External provider ID")
    provider_model_id: str = Field(min_length=1, description="Provider-specific model ID")
    capabilities: ExternalModelCapabilities = Field(description="Provider capability matrix")
    default_settings: ExternalApiModelDefaultSettings | None = Field(default=None)
    panel_schema: ExternalModelPanelSchema | None = Field(default=None)
    tags: list[str] | None = Field(default=None)
    is_default: bool = Field(default=False)

    source_type: ModelSourceType = Field(default=ModelSourceType.External)
    path: str = Field(default="")
    source: str = Field(default="")
    hash: str = Field(default="")
    file_size: int = Field(default=0, ge=0)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _populate_external_fields(self) -> "ExternalApiModelConfig":
        if not self.path:
            self.path = f"external://{self.provider_id}/{self.provider_model_id}"
        if not self.source:
            self.source = self.path
        if not self.hash:
            self.hash = f"external:{self.provider_id}:{self.provider_model_id}"
        return self

    @classmethod
    def from_model_on_disk(cls, mod: ModelOnDisk, override_fields: dict[str, object]) -> Self:
        raise NotAMatchError("external API models are not probed from disk")
