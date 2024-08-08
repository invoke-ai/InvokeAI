from typing import Any, Optional

from pydantic import BaseModel, Field, TypeAdapter


class StylePresetNotFoundError(Exception):
    """Raised when a style preset is not found"""


class PresetData(BaseModel, extra="forbid"):
    positive_prompt: str = Field(description="Positive prompt")
    negative_prompt: str = Field(description="Negative prompt")


PresetDataValidator = TypeAdapter(PresetData)


class StylePresetChanges(BaseModel, extra="forbid"):
    name: Optional[str] = Field(default=None, description="The style preset's new name.")
    preset_data: Optional[PresetData] = Field(default=None, description="The updated data for style preset.")


class StylePresetWithoutId(BaseModel):
    name: str = Field(description="The name of the style preset.")
    preset_data: PresetData = Field(description="The preset data")


class StylePresetRecordDTO(StylePresetWithoutId):
    id: str = Field(description="The style preset ID.")
    is_default: bool = Field(description="Whether or not the style preset is default")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StylePresetRecordDTO":
        data["preset_data"] = PresetDataValidator.validate_json(data.get("preset_data", ""))
        return StylePresetRecordDTOValidator.validate_python(data)


StylePresetRecordDTOValidator = TypeAdapter(StylePresetRecordDTO)


class StylePresetRecordWithImage(StylePresetRecordDTO):
    image: Optional[str] = Field(description="The path for image")
