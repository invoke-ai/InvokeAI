from typing import Any, Union

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, field_validator


class StylePresetNotFoundError(Exception):
    """Raised when a style preset is not found"""


class PresetData(BaseModel):
    positive_prompt: str = Field(description="Positive prompt")
    negative_prompt: str = Field(description="Negative prompt")


PresetDataValidator = TypeAdapter(PresetData)


class StylePresetWithoutId(BaseModel):
    name: str = Field(description="The name of the style preset.")
    preset_data: PresetData = Field(description="The preset data")


class StylePresetRecordDTO(StylePresetWithoutId):
    id: str = Field(description="The style preset ID.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StylePresetRecordDTO":
        data["preset_data"] = PresetDataValidator.validate_json(data.get("preset_data", ""))
        return StylePresetRecordDTOValidator.validate_python(data)


StylePresetRecordDTOValidator = TypeAdapter(StylePresetRecordDTO)
