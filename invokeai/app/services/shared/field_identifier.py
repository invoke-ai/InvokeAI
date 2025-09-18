from typing import Literal

from pydantic import BaseModel, Field


class FieldIdentifier(BaseModel):
    kind: Literal["input", "output"] = Field(description="The kind of field")
    node_id: str = Field(description="The ID of the node")
    field_name: str = Field(description="The name of the field")
    user_label: str | None = Field(description="The user label of the field, if any")


__all__ = ["FieldIdentifier"]

