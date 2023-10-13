from typing import Generic, TypeVar

from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

GenericBaseModel = TypeVar("GenericBaseModel", bound=BaseModel)


class CursorPaginatedResults(GenericModel, Generic[GenericBaseModel]):
    """Cursor-paginated results"""

    limit: int = Field(..., description="Limit of items to get")
    has_more: bool = Field(..., description="Whether there are more items available")
    items: list[GenericBaseModel] = Field(..., description="Items")
