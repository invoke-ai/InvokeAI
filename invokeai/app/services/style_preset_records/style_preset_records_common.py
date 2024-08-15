from enum import Enum
from typing import Any, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, TypeAdapter

from invokeai.app.util.metaenum import MetaEnum


class StylePresetNotFoundError(Exception):
    """Raised when a style preset is not found"""


class StylePresetImportValidationError(Exception):
    """Raised when a style preset import is not valid"""


class PresetData(BaseModel, extra="forbid"):
    positive_prompt: str = Field(description="Positive prompt")
    negative_prompt: str = Field(description="Negative prompt")


PresetDataValidator = TypeAdapter(PresetData)


class PresetType(str, Enum, metaclass=MetaEnum):
    User = "user"
    Default = "default"
    Project = "project"


class StylePresetChanges(BaseModel, extra="forbid"):
    name: Optional[str] = Field(default=None, description="The style preset's new name.")
    preset_data: Optional[PresetData] = Field(default=None, description="The updated data for style preset.")


class StylePresetWithoutId(BaseModel):
    name: str = Field(description="The name of the style preset.")
    preset_data: PresetData = Field(description="The preset data")
    type: PresetType = Field(description="The type of style preset")


class StylePresetRecordDTO(StylePresetWithoutId):
    id: str = Field(description="The style preset ID.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StylePresetRecordDTO":
        data["preset_data"] = PresetDataValidator.validate_json(data.get("preset_data", ""))
        return StylePresetRecordDTOValidator.validate_python(data)


StylePresetRecordDTOValidator = TypeAdapter(StylePresetRecordDTO)


class StylePresetRecordWithImage(StylePresetRecordDTO):
    image: Optional[str] = Field(description="The path for image")


class StylePresetImportRow(BaseModel):
    name: str = Field(min_length=1, description="The name of the preset.")
    positive_prompt: str = Field(
        default="",
        description="The positive prompt for the preset.",
        validation_alias=AliasChoices("positive_prompt", "prompt"),
    )
    negative_prompt: str = Field(default="", description="The negative prompt for the preset.")

    model_config = ConfigDict(str_strip_whitespace=True)


StylePresetImportList = list[StylePresetImportRow]
StylePresetImportListTypeAdapter = TypeAdapter(StylePresetImportList)
