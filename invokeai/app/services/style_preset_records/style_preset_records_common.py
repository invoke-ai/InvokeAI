import csv
import io
from enum import Enum
from typing import Any, Generator, Optional

from fastapi import UploadFile
from pydantic import BaseModel, Field, TypeAdapter

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
    name: str
    prompt: str
    negative_prompt: str


def parse_csv(file: UploadFile) -> Generator[StylePresetImportRow, None, None]:
    """Yield parsed and validated rows from the CSV file."""
    file_content = file.file.read().decode("utf-8")
    csv_reader = csv.DictReader(io.StringIO(file_content))

    for row in csv_reader:
        if "name" not in row or "prompt" not in row or "negative_prompt" not in row:
            raise StylePresetImportValidationError()

        yield StylePresetImportRow(
            name=row["name"].strip(), prompt=row["prompt"].strip(), negative_prompt=row["negative_prompt"].strip()
        )
