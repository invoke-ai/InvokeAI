"""Canvas projects: their documents, revisions, and the board each one claims."""

import json
from collections.abc import Sequence
from typing import Any, NamedTuple, Optional

from sqlalchemy import (
    ColumnElement,
    Connection,
    Row,
    bindparam,
    delete,
    false,
    insert,
    literal,
    select,
    union_all,
    update,
)

from invokeai.app.services.image_records.image_records_common import ASSETS_CATEGORIES, IMAGE_CATEGORIES
from invokeai.app.services.project_records.project_records_common import (
    ProjectBoardItemDTO,
    ProjectRecordDTO,
    ProjectSummaryDTO,
)
from invokeai.app.services.shared.database.queries.base import QueryModule, locking, mapped, read, write
from invokeai.app.services.shared.database.schema.boards import board_images, board_videos
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.projects import projects
from invokeai.app.services.shared.database.schema.videos import videos


class LockedProject(NamedTuple):
    """A project's row as a transaction locked it."""

    revision: int
    minimum_canvas_schema_version: int
    board_id: str


_SUMMARY_COLUMNS = (
    projects.c.project_id,
    projects.c.board_id,
    projects.c.name,
    projects.c.revision,
    projects.c.minimum_canvas_schema_version,
    projects.c.created_at,
    projects.c.updated_at,
)
_THE_PROJECT = (projects.c.user_id == bindparam("user_id"), projects.c.project_id == bindparam("project_id"))


_GET = select(*_SUMMARY_COLUMNS, projects.c.data).where(*_THE_PROJECT)
_SUMMARY = select(*_SUMMARY_COLUMNS).where(*_THE_PROJECT)
# Oldest first. Projects created in the same millisecond come in the order of their ids.
_SUMMARIES = (
    select(*_SUMMARY_COLUMNS)
    .where(projects.c.user_id == bindparam("user_id"))
    .order_by(projects.c.created_at, projects.c.project_id)
)
_BOARD_ID = select(projects.c.board_id).where(*_THE_PROJECT)
_LOCK = (
    select(projects.c.revision, projects.c.minimum_canvas_schema_version, projects.c.board_id)
    .where(*_THE_PROJECT)
    .with_for_update()
)
_CLAIMANT = select(projects.c.user_id, projects.c.project_id).where(projects.c.board_id == bindparam("board_id"))
_INSERT = insert(projects)
_SAVE = (
    update(projects)
    .where(projects.c.user_id == bindparam("target_user_id"), projects.c.project_id == bindparam("target_project_id"))
    .values(
        name=bindparam("new_name"),
        data=bindparam("new_data"),
        minimum_canvas_schema_version=bindparam("new_minimum_canvas_schema_version"),
        revision=projects.c.revision + 1,
    )
)
_DELETE = delete(projects).where(*_THE_PROJECT)


def _shown_by_the_gallery(category: ColumnElement[str]) -> ColumnElement[bool]:
    # `other`, the canvas's private category, is in neither list: the gallery never shows it on a board.
    return category.in_([literal(each.value) for each in (*IMAGE_CATEGORIES, *ASSETS_CATEGORIES)])


# What the gallery shows on a board, of both kinds. `deleted_at` is not filtered: soft delete is unused, and no
# other board query consults it, so filtering here would make the snapshot disagree with the board's counts.
_ITEMS = union_all(
    select(
        literal("image").label("kind"),
        images.c.image_name.label("name"),
        images.c.image_category.label("category"),
        images.c.starred.label("starred"),
    )
    .select_from(board_images.join(images, board_images.c.image_name == images.c.image_name))
    .where(
        board_images.c.board_id == bindparam("board_id"),
        images.c.is_intermediate == false(),
        _shown_by_the_gallery(images.c.image_category),
    ),
    select(literal("video"), videos.c.video_name, videos.c.video_category, videos.c.starred)
    .select_from(board_videos.join(videos, board_videos.c.video_name == videos.c.video_name))
    .where(
        board_videos.c.board_id == bindparam("board_id"),
        videos.c.is_intermediate == false(),
        _shown_by_the_gallery(videos.c.video_category),
    ),
)
_BOARD_ITEMS = _ITEMS.order_by(_ITEMS.selected_columns.kind, _ITEMS.selected_columns.name).limit(bindparam("limit"))


def _summary(row: Sequence[Any]) -> ProjectSummaryDTO:
    project_id, board_id, name, revision, minimum_canvas_schema_version, created_at, updated_at = row
    return ProjectSummaryDTO(
        project_id=project_id,
        board_id=board_id,
        name=name,
        revision=revision,
        minimum_canvas_schema_version=minimum_canvas_schema_version,
        created_at=created_at,
        updated_at=updated_at,
    )


def _summary_or_none(row: Optional[Sequence[Any]]) -> Optional[ProjectSummaryDTO]:
    return _summary(row) if row is not None else None


def _summaries(rows: Sequence[Sequence[Any]]) -> list[ProjectSummaryDTO]:
    return [_summary(row) for row in rows]


def _record_or_none(row: Optional[Sequence[Any]]) -> Optional[ProjectRecordDTO]:
    if row is None:
        return None
    project_id, board_id, name, revision, minimum_canvas_schema_version, created_at, updated_at, data = row
    return ProjectRecordDTO(
        project_id=project_id,
        board_id=board_id,
        name=name,
        revision=revision,
        minimum_canvas_schema_version=minimum_canvas_schema_version,
        created_at=created_at,
        updated_at=updated_at,
        data=json.loads(data),
    )


def _board_items_or_none(rows: Optional[Sequence[Sequence[Any]]]) -> Optional[list[ProjectBoardItemDTO]]:
    if rows is None:
        return None
    return [
        ProjectBoardItemDTO(kind=kind, name=name, category=category, starred=starred)
        for kind, name, category, starred in rows
    ]


class ProjectQueries(QueryModule):
    @mapped(_record_or_none)
    @read
    def get(self, conn: Connection, user_id: str, project_id: str) -> Optional[Row[Any]]:
        """The project with its document."""
        return conn.execute(_GET, {"user_id": user_id, "project_id": project_id}).first()

    @mapped(_summary_or_none)
    @read
    def summary(self, conn: Connection, user_id: str, project_id: str) -> Optional[Row[Any]]:
        """The project without its document."""
        return conn.execute(_SUMMARY, {"user_id": user_id, "project_id": project_id}).first()

    @mapped(_summaries)
    @read
    def summaries(self, conn: Connection, user_id: str) -> Sequence[Row[Any]]:
        """The account's projects without their documents, oldest first."""
        return conn.execute(_SUMMARIES, {"user_id": user_id}).all()

    @read
    def board_id(self, conn: Connection, user_id: str, project_id: str) -> Optional[str]:
        return conn.execute(_BOARD_ID, {"user_id": user_id, "project_id": project_id}).scalar_one_or_none()

    @locking
    def lock(self, conn: Connection, user_id: str, project_id: str) -> Optional[LockedProject]:
        """Locks the project's row until the transaction ends: what a write of the project decides on, or None when
        there is no such project. Lock it before its board: rows are locked in that order."""
        row: Optional[Sequence[Any]] = conn.execute(_LOCK, {"user_id": user_id, "project_id": project_id}).first()
        if row is None:
            return None
        revision, minimum_canvas_schema_version, board_id = row
        return LockedProject(revision, minimum_canvas_schema_version, board_id)

    @read
    def claimant(self, conn: Connection, board_id: str) -> Optional[tuple[str, str]]:
        """The account and id of the project that claims the board, if one does."""
        row: Optional[Sequence[Any]] = conn.execute(_CLAIMANT, {"board_id": board_id}).first()
        if row is None:
            return None
        user_id, project_id = row
        return user_id, project_id

    @mapped(_board_items_or_none)
    @read
    def board_items(self, conn: Connection, user_id: str, project_id: str, limit: int) -> Optional[Sequence[Row[Any]]]:
        """Up to `limit` of what the gallery shows on the project's board, by kind and then name; None when there
        is no such project."""
        board_id = conn.execute(_BOARD_ID, {"user_id": user_id, "project_id": project_id}).scalar_one_or_none()
        if board_id is None:
            return None
        return conn.execute(_BOARD_ITEMS, {"board_id": board_id, "limit": limit}).all()

    @write
    def insert(
        self,
        conn: Connection,
        *,
        project_id: str,
        user_id: str,
        name: str,
        data: str,
        board_id: str,
        minimum_canvas_schema_version: int,
    ) -> None:
        conn.execute(
            _INSERT,
            {
                "project_id": project_id,
                "user_id": user_id,
                "name": name,
                "data": data,
                "board_id": board_id,
                "minimum_canvas_schema_version": minimum_canvas_schema_version,
            },
        )

    @write
    def save(
        self,
        conn: Connection,
        *,
        user_id: str,
        project_id: str,
        name: str,
        data: str,
        minimum_canvas_schema_version: int,
    ) -> None:
        """Writes the project as its next revision. The caller has locked it and checked the revision it saves over."""
        conn.execute(
            _SAVE,
            {
                "target_user_id": user_id,
                "target_project_id": project_id,
                "new_name": name,
                "new_data": data,
                "new_minimum_canvas_schema_version": minimum_canvas_schema_version,
            },
        )

    @write
    def delete(self, conn: Connection, user_id: str, project_id: str) -> None:
        conn.execute(_DELETE, {"user_id": user_id, "project_id": project_id})
