from typing import Generic, TypeVar

from pydantic import BaseModel, Field

GenericBaseModel = TypeVar("GenericBaseModel", bound=BaseModel)


class CursorPaginatedResults(BaseModel, Generic[GenericBaseModel]):
    """
    Cursor-paginated results
    Generic must be a Pydantic model
    """

    limit: int = Field(..., description="Limit of items to get")
    has_more: bool = Field(..., description="Whether there are more items available")
    items: list[GenericBaseModel] = Field(..., description="Items")


class OffsetPaginatedResults(BaseModel, Generic[GenericBaseModel]):
    """
    Offset-paginated results
    Generic must be a Pydantic model
    """

    limit: int = Field(description="Limit of items to get")
    offset: int = Field(description="Offset from which to retrieve items")
    total: int = Field(description="Total number of items in result")
    items: list[GenericBaseModel] = Field(description="Items")


class PaginatedResults(BaseModel, Generic[GenericBaseModel]):
    """
    Paginated results
    Generic must be a Pydantic model
    """

    page: int = Field(description="Current Page")
    pages: int = Field(description="Total number of pages")
    per_page: int = Field(description="Number of items per page")
    total: int = Field(description="Total number of items in result")
    items: list[GenericBaseModel] = Field(description="Items")
