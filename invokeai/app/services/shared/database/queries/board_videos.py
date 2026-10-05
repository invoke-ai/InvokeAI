"""Which board each video is on: at most one, since the video name is the key."""

import functools
from typing import Any, Optional

from sqlalchemy import Connection, Insert, Select, bindparam, case, delete, false, func, literal, select

from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.shared.database.dialect import upsert
from invokeai.app.services.shared.database.queries.base import QueryModule, read, write
from invokeai.app.services.shared.database.schema.boards import board_videos
from invokeai.app.services.shared.database.schema.videos import videos
from invokeai.app.services.shared.database.types import now_text

_BOARD_OF = select(board_videos.c.board_id).where(board_videos.c.video_name == bindparam("video_name"))
_REMOVE = delete(board_videos).where(board_videos.c.video_name == bindparam("video_name"))
# Both counts in one pass: every video, and those of a category other than general (assets).
_COUNTS = (
    select(func.count(), func.count(case((videos.c.video_category != literal(ImageCategory.GENERAL.value), 1))))
    .select_from(board_videos.join(videos, board_videos.c.video_name == videos.c.video_name))
    .where(videos.c.is_intermediate == false(), board_videos.c.board_id == bindparam("board_id"))
)


@functools.cache
def _add(dialect_name: str) -> Insert:
    return upsert(dialect_name, board_videos, update=("board_id", "updated_at"))


@functools.cache
def _names(off_board: bool, categories: Optional[tuple[str, ...]], by_intermediate: bool, by_user: bool) -> Select[Any]:
    statement = select(videos.c.video_name).select_from(
        videos.outerjoin(board_videos, board_videos.c.video_name == videos.c.video_name)
    )
    if off_board:
        statement = statement.where(board_videos.c.board_id.is_(None))
    else:
        statement = statement.where(board_videos.c.board_id == bindparam("board_id"))
    if categories is not None:
        # Bound one by one: a list would be an expanding parameter, rendered anew on every execution.
        statement = statement.where(videos.c.video_category.in_([literal(value) for value in categories]))
    if by_intermediate:
        statement = statement.where(videos.c.is_intermediate == bindparam("is_intermediate"))
    if by_user:
        statement = statement.where(videos.c.user_id == bindparam("user_id"))
    return statement


class BoardVideoQueries(QueryModule):
    @write
    def add(self, conn: Connection, board_id: str, video_name: str) -> None:
        """Puts the video on the board, taking it off the board it was on."""
        conn.execute(
            _add(conn.dialect.name), {"board_id": board_id, "video_name": video_name, "updated_at": now_text()}
        )

    @write
    def remove(self, conn: Connection, video_name: str) -> None:
        """Takes the video off whichever board it is on."""
        conn.execute(_REMOVE, {"video_name": video_name})

    @read
    def board_of(self, conn: Connection, video_name: str) -> Optional[str]:
        return conn.execute(_BOARD_OF, {"video_name": video_name}).scalar_one_or_none()

    @read
    def video_names(
        self,
        conn: Connection,
        board_id: Optional[str],
        *,
        categories: Optional[list[ImageCategory]],
        is_intermediate: Optional[bool],
        user_id: Optional[str],
    ) -> list[str]:
        """The names of the videos on a board, or on none (`board_id` None), that match the filters given."""
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
        """The board's videos that are not intermediates: all of them, and those of a category other than general."""
        counts = conn.execute(_COUNTS, {"board_id": board_id}).one()
        return counts[0], counts[1]
