"""Which board each image is on: at most one, since the image name is the key."""

import functools
from collections.abc import Sequence
from typing import Any, Optional

from sqlalchemy import Connection, Insert, Select, bindparam, case, delete, false, func, literal, select

from invokeai.app.services.image_records.image_records_common import (
    ASSETS_CATEGORIES,
    IMAGE_CATEGORIES,
    ImageCategory,
)
from invokeai.app.services.shared.database.dialect import OrderedJoin, upsert
from invokeai.app.services.shared.database.queries.base import QueryModule, read, write
from invokeai.app.services.shared.database.schema.boards import board_images
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.types import now_text

_BOARD_OF = select(board_images.c.board_id).where(board_images.c.image_name == bindparam("image_name"))
_REMOVE = delete(board_images).where(
    board_images.c.image_name == bindparam("image_name"), board_images.c.board_id == bindparam("board_id")
)


def _in(categories: Sequence[ImageCategory | str]) -> list[Any]:
    # Bound one by one: a list would be an expanding parameter, rendered anew on every execution.
    return [literal(value) for value in sorted({ImageCategory(category).value for category in categories})]


def _counted(categories: list[ImageCategory]) -> Any:
    return func.count(case((images.c.image_category.in_(_in(categories)), 1)))


# Both counts in one pass. From the board's membership to its images, not the other way round: an index on
# `images` would otherwise make every image of the gallery the outer loop of a board's count.
_COUNTS = (
    select(_counted(IMAGE_CATEGORIES), _counted(ASSETS_CATEGORIES))
    .select_from(OrderedJoin(board_images, images, board_images.c.image_name == images.c.image_name))
    .where(images.c.is_intermediate == false(), board_images.c.board_id == bindparam("board_id"))
)


@functools.cache
def _add(dialect_name: str) -> Insert:
    return upsert(dialect_name, board_images, update=("board_id", "updated_at"))


@functools.cache
def _names(off_board: bool, categories: Optional[tuple[str, ...]], by_intermediate: bool, by_user: bool) -> Select[Any]:
    statement = select(images.c.image_name).select_from(
        images.outerjoin(board_images, board_images.c.image_name == images.c.image_name)
    )
    if off_board:
        statement = statement.where(board_images.c.board_id.is_(None))
    else:
        statement = statement.where(board_images.c.board_id == bindparam("board_id"))
    if categories is not None:
        statement = statement.where(images.c.image_category.in_(_in(categories)))
    if by_intermediate:
        statement = statement.where(images.c.is_intermediate == bindparam("is_intermediate"))
    if by_user:
        statement = statement.where(images.c.user_id == bindparam("user_id"))
    return statement


class BoardImageQueries(QueryModule):
    @write
    def add(self, conn: Connection, board_id: str, image_name: str) -> None:
        """Puts the image on the board, taking it off the board it was on."""
        conn.execute(
            _add(conn.dialect.name), {"board_id": board_id, "image_name": image_name, "updated_at": now_text()}
        )

    @write
    def remove(self, conn: Connection, image_name: str, board_id: str) -> bool:
        """Takes the image off this board; whether it was on it."""
        return conn.execute(_REMOVE, {"image_name": image_name, "board_id": board_id}).rowcount > 0

    @read
    def board_of(self, conn: Connection, image_name: str) -> Optional[str]:
        return conn.execute(_BOARD_OF, {"image_name": image_name}).scalar_one_or_none()

    @read
    def image_names(
        self,
        conn: Connection,
        board_id: Optional[str],
        *,
        categories: Optional[list[ImageCategory]],
        is_intermediate: Optional[bool],
        user_id: Optional[str],
    ) -> list[str]:
        """The names of the images on a board, or on none (`board_id` None), that match the filters given."""
        statement = _names(
            board_id is None,
            None if categories is None else tuple(sorted({category.value for category in categories})),
            is_intermediate is not None,
            user_id is not None,
        )
        parameters = {"board_id": board_id, "is_intermediate": is_intermediate, "user_id": user_id}
        return list(conn.execute(statement, parameters).scalars().all())

    @read
    def counts(self, conn: Connection, board_id: str) -> tuple[int, int]:
        """The board's images that are not intermediates: those of the image categories, and those of the asset
        categories."""
        counts = conn.execute(_COUNTS, {"board_id": board_id}).one()
        return counts[0], counts[1]
