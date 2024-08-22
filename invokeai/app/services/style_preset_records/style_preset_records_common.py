import codecs
import csv
import json
from enum import Enum
from typing import Any, Optional

import pydantic
from fastapi import UploadFile
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, TypeAdapter

from invokeai.app.util.metaenum import MetaEnum


class StylePresetNotFoundError(Exception):
    """Raised when a style preset is not found"""


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
    type: Optional[PresetType] = Field(description="The updated type of the style preset")


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

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")


StylePresetImportList = list[StylePresetImportRow]
StylePresetImportListTypeAdapter = TypeAdapter(StylePresetImportList)


class UnsupportedFileTypeError(ValueError):
    """Raised when an unsupported file type is encountered"""

    pass


class InvalidPresetImportDataError(ValueError):
    """Raised when invalid preset import data is encountered"""

    pass


async def parse_presets_from_file(file: UploadFile) -> list[StylePresetWithoutId]:
    """Parses style presets from a file. The file must be a CSV or JSON file.

    If CSV, the file must have the following columns:
    - name
    - prompt (or positive_prompt)
    - negative_prompt

    If JSON, the file must be a list of objects with the following keys:
    - name
    - prompt (or positive_prompt)
    - negative_prompt

    Args:
        file (UploadFile): The file to parse.

    Returns:
        list[StylePresetWithoutId]: The parsed style presets.

    Raises:
        UnsupportedFileTypeError: If the file type is not supported.
        InvalidPresetImportDataError: If the data in the file is invalid.
    """
    if file.content_type not in ["text/csv", "application/json"]:
        raise UnsupportedFileTypeError()

    if file.content_type == "text/csv":
        csv_reader = csv.DictReader(codecs.iterdecode(file.file, "utf-8"))
        data = list(csv_reader)
    else:  # file.content_type == "application/json":
        json_data = await file.read()
        data = json.loads(json_data)

    try:
        imported_presets = StylePresetImportListTypeAdapter.validate_python(data)

        style_presets: list[StylePresetWithoutId] = []

        for imported in imported_presets:
            preset_data = PresetData(positive_prompt=imported.positive_prompt, negative_prompt=imported.negative_prompt)
            style_preset = StylePresetWithoutId(name=imported.name, preset_data=preset_data, type=PresetType.User)
            style_presets.append(style_preset)
    except pydantic.ValidationError as e:
        if file.content_type == "text/csv":
            msg = "Invalid CSV format: must include columns 'name', 'prompt', and 'negative_prompt' and name cannot be blank"
        else:  # file.content_type == "application/json":
            msg = "Invalid JSON format: must be a list of objects with keys 'name', 'prompt', and 'negative_prompt' and name cannot be blank"
        raise InvalidPresetImportDataError(msg) from e
    finally:
        file.file.close()

    return style_presets
