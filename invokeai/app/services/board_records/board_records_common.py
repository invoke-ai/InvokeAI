from datetime import datetime
from typing import Any, Optional, Union

from attr import dataclass
from pydantic import BaseModel, Field

from invokeai.app.util.misc import get_iso_timestamp
from invokeai.app.util.model_exclude_null import BaseModelExcludeNull

# This query is missing a GROUP BY clause, which is required for the query to be valid.
BASE_UNTERMINATED_AND_MISSING_GROUP_BY_BOARD_RECORDS_QUERY = """
    SELECT b.board_id,
        b.board_name,
        b.created_at,
        b.updated_at,
        b.archived,
        COUNT(
            CASE
                WHEN i.image_category in ('general')
                AND i.is_intermediate = 0 THEN 1
            END
        ) AS image_count,
        COUNT(
            CASE
                WHEN i.image_category in ('control', 'mask', 'user', 'other')
                AND i.is_intermediate = 0 THEN 1
            END
        ) AS asset_count,
        (
            SELECT bi.image_name
            FROM board_images bi
                JOIN images i ON bi.image_name = i.image_name
            WHERE bi.board_id = b.board_id
                AND i.is_intermediate = 0
            ORDER BY i.created_at DESC
            LIMIT 1
        ) AS cover_image_name
    FROM boards b
        LEFT JOIN board_images bi ON b.board_id = bi.board_id
        LEFT JOIN images i ON bi.image_name = i.image_name
    """


@dataclass
class PaginatedBoardRecordsQueries:
    main_query: str
    total_count_query: str


def get_paginated_list_board_records_query(include_archived: bool) -> PaginatedBoardRecordsQueries:
    """Gets a query to retrieve a paginated list of board records."""

    archived_condition = "WHERE b.archived = 0" if not include_archived else ""

    # The GROUP BY must be added _after_ the WHERE clause!
    main_query = f"""
        {BASE_UNTERMINATED_AND_MISSING_GROUP_BY_BOARD_RECORDS_QUERY}
        {archived_condition}
        GROUP BY b.board_id,
            b.board_name,
            b.created_at,
            b.updated_at
        ORDER BY b.created_at DESC
        LIMIT ? OFFSET ?;
        """

    total_count_query = f"""
        SELECT COUNT(*)
        FROM boards b
        {archived_condition};
        """

    return PaginatedBoardRecordsQueries(main_query=main_query, total_count_query=total_count_query)


def get_list_all_board_records_query(include_archived: bool) -> str:
    """Gets a query to retrieve all board records."""

    archived_condition = "WHERE b.archived = 0" if not include_archived else ""

    # The GROUP BY must be added _after_ the WHERE clause!
    return f"""
        {BASE_UNTERMINATED_AND_MISSING_GROUP_BY_BOARD_RECORDS_QUERY}
        {archived_condition}
        GROUP BY b.board_id,
            b.board_name,
            b.created_at,
            b.updated_at
        ORDER BY b.created_at DESC;
        """


def get_board_record_query() -> str:
    """Gets a query to retrieve a board record."""

    return f"""
        {BASE_UNTERMINATED_AND_MISSING_GROUP_BY_BOARD_RECORDS_QUERY}
        WHERE b.board_id = ?;
        """


class BoardRecord(BaseModelExcludeNull):
    """Deserialized board record."""

    board_id: str = Field(description="The unique ID of the board.")
    """The unique ID of the board."""
    board_name: str = Field(description="The name of the board.")
    """The name of the board."""
    created_at: Union[datetime, str] = Field(description="The created timestamp of the board.")
    """The created timestamp of the image."""
    updated_at: Union[datetime, str] = Field(description="The updated timestamp of the board.")
    """The updated timestamp of the image."""
    deleted_at: Optional[Union[datetime, str]] = Field(default=None, description="The deleted timestamp of the board.")
    """The updated timestamp of the image."""
    cover_image_name: Optional[str] = Field(default=None, description="The name of the cover image of the board.")
    """The name of the cover image of the board."""
    archived: bool = Field(description="Whether or not the board is archived.")
    """Whether or not the board is archived."""
    is_private: Optional[bool] = Field(default=None, description="Whether the board is private.")
    """Whether the board is private."""
    image_count: int = Field(description="The number of images in the board.")
    asset_count: int = Field(description="The number of assets in the board.")


def deserialize_board_record(board_dict: dict[str, Any]) -> BoardRecord:
    """Deserializes a board record."""

    # Retrieve all the values, setting "reasonable" defaults if they are not present.

    board_id = board_dict.get("board_id", "unknown")
    board_name = board_dict.get("board_name", "unknown")
    cover_image_name = board_dict.get("cover_image_name", None)
    created_at = board_dict.get("created_at", get_iso_timestamp())
    updated_at = board_dict.get("updated_at", get_iso_timestamp())
    deleted_at = board_dict.get("deleted_at", get_iso_timestamp())
    archived = board_dict.get("archived", False)
    is_private = board_dict.get("is_private", False)
    image_count = board_dict.get("image_count", 0)
    asset_count = board_dict.get("asset_count", 0)

    return BoardRecord(
        board_id=board_id,
        board_name=board_name,
        cover_image_name=cover_image_name,
        created_at=created_at,
        updated_at=updated_at,
        deleted_at=deleted_at,
        archived=archived,
        is_private=is_private,
        image_count=image_count,
        asset_count=asset_count,
    )


class BoardChanges(BaseModel, extra="forbid"):
    board_name: Optional[str] = Field(default=None, description="The board's new name.")
    cover_image_name: Optional[str] = Field(default=None, description="The name of the board's new cover image.")
    archived: Optional[bool] = Field(default=None, description="Whether or not the board is archived")


class BoardRecordNotFoundException(Exception):
    """Raised when an board record is not found."""

    def __init__(self, message: str = "Board record not found"):
        super().__init__(message)


class BoardRecordSaveException(Exception):
    """Raised when an board record cannot be saved."""

    def __init__(self, message: str = "Board record not saved"):
        super().__init__(message)


class BoardRecordDeleteException(Exception):
    """Raised when an board record cannot be deleted."""

    def __init__(self, message: str = "Board record not deleted"):
        super().__init__(message)


class UncategorizedImageCounts(BaseModel):
    image_count: int = Field(description="The number of uncategorized images.")
    asset_count: int = Field(description="The number of uncategorized assets.")
