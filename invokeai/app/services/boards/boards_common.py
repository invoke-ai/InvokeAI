from typing import Optional

from pydantic import Field

from invokeai.app.services.board_records.board_records_common import BoardRecord


class BoardDTO(BoardRecord):
    """Deserialized board record with cover image URL and image count."""

    cover_image_name: Optional[str] = Field(description="The name of the board's cover image.")
    """The URL of the thumbnail of the most recent image in the board."""
    image_count: int = Field(description="The number of images in the board.")
    """The number of images in the board."""


def board_record_to_dto(board_record: BoardRecord, cover_image_name: Optional[str], image_count: int) -> BoardDTO:
    """Converts a board record to a board DTO."""
    return BoardDTO(
        **board_record.model_dump(exclude={"cover_image_name"}),
        cover_image_name=cover_image_name,
        image_count=image_count,
    )
