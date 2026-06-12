"""Central DB wrapper: every SQL query in the application lives in this module.

``DbQueries`` is the single place where SQL statements are built and executed.
Services must not open sessions or build statements themselves — they call
methods on this class (available as ``db.queries``) and keep only business
logic (validation, events, file I/O, config access).

Rules of this module:
- One class, grouped into per-domain sections (see the ``region`` banners).
- Every method owns its complete unit of work: session lifecycle, statement,
  row extraction and conversion to domain objects. Rows never leave a session
  half-loaded.
- Methods raise the same domain exceptions the services raised before the
  queries were centralized, so service behaviour is unchanged.
- Portability: statements must work on SQLite and MySQL/MariaDB (and ideally
  Postgres). Dialect-specific behaviour belongs in ``models.py`` (schema) or
  here (queries) — never in services.

The test ``tests/app/services/test_sqlmodel_services/test_db_queries_centralized.py``
enforces that no service outside this module touches sessions or SQLAlchemy.
"""

import json
from datetime import datetime, timezone
from logging import Logger
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union
from uuid import uuid4

import pydantic
from sqlalchemy import and_, case, delete, func, insert, literal_column, or_, update
from sqlalchemy import select as sa_select
from sqlalchemy.engine import Row
from sqlalchemy.orm import aliased
from sqlmodel import col, select

from invokeai.app.invocations.fields import MetadataField, MetadataFieldValidator
from invokeai.app.services.auth.password_utils import verify_password
from invokeai.app.services.board_records.board_records_common import (
    BoardChanges,
    BoardRecord,
    BoardRecordDeleteException,
    BoardRecordNotFoundException,
    BoardRecordOrderBy,
    BoardRecordSaveException,
    BoardVisibility,
)
from invokeai.app.services.image_records.image_records_common import (
    ASSETS_CATEGORIES,
    IMAGE_CATEGORIES,
    ImageCategory,
    ImageNamesResult,
    ImageRecord,
    ImageRecordChanges,
    ImageRecordDeleteException,
    ImageRecordNotFoundException,
    ImageRecordSaveException,
    ResourceOrigin,
    deserialize_image_record,
)
from invokeai.app.services.model_records.model_records_base import (
    DuplicateModelException,
    ModelRecordOrderBy,
    ModelSummary,
    UnknownModelException,
)
from invokeai.app.services.session_queue.session_queue_common import (
    QUEUE_ITEM_STATUS,
    BatchStatus,
    SessionQueueCountsByDestination,
    SessionQueueItem,
    SessionQueueItemNotFoundError,
    ValueToInsertTuple,
)
from invokeai.app.services.shared.pagination import (
    CursorPaginatedResults,
    OffsetPaginatedResults,
    PaginatedResults,
)
from invokeai.app.services.shared.sqlite.models import (
    AppSettingTable,
    BoardImageTable,
    BoardTable,
    ClientStateTable,
    ImageTable,
    ModelRelationshipTable,
    ModelTable,
    SessionQueueTable,
    StylePresetTable,
    UserTable,
    WorkflowLibraryTable,
)
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.style_preset_records.style_preset_records_common import (
    PresetType,
    StylePresetChanges,
    StylePresetNotFoundError,
    StylePresetRecordDTO,
    StylePresetWithoutId,
)
from invokeai.app.services.users.users_common import UserDTO
from invokeai.app.services.virtual_boards.virtual_boards_common import VirtualSubBoardDTO
from invokeai.app.services.workflow_records.workflow_records_common import (
    WorkflowCategory,
    WorkflowNotFoundError,
    WorkflowRecordDTO,
    WorkflowRecordListItemDTO,
    WorkflowRecordListItemDTOValidator,
    WorkflowRecordOrderBy,
)
from invokeai.app.util.misc import uuid_string
from invokeai.backend.model_manager.configs.factory import AnyModelConfig, ModelConfigFactory
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType

if TYPE_CHECKING:
    from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase

# ------------------------------------------------------------------------------------------------
# Module-level helpers (row converters and statement builders, grouped by domain)
# ------------------------------------------------------------------------------------------------

# --- Boards ---


def _board_to_record(row: BoardTable) -> BoardRecord:
    """Convert a BoardTable row to a BoardRecord. Call while the row is session-bound."""
    try:
        visibility = BoardVisibility(row.board_visibility)
    except ValueError:
        visibility = BoardVisibility.Private

    return BoardRecord(
        board_id=row.board_id,
        board_name=row.board_name,
        user_id=row.user_id,
        cover_image_name=row.cover_image_name,
        created_at=row.created_at,
        updated_at=row.updated_at,
        deleted_at=row.deleted_at,
        archived=row.archived,
        board_visibility=visibility,
    )


# --- Images ---


def _image_to_dict(row: ImageTable) -> dict:
    """Convert an ImageTable row to a dict compatible with deserialize_image_record."""
    return {
        "image_name": row.image_name,
        "image_origin": row.image_origin,
        "image_category": row.image_category,
        "width": row.width,
        "height": row.height,
        "session_id": row.session_id,
        "node_id": row.node_id,
        "metadata": row.metadata_,
        "is_intermediate": row.is_intermediate,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "deleted_at": row.deleted_at,
        "starred": row.starred,
        "has_workflow": row.has_workflow,
        "image_subfolder": row.image_subfolder,
    }


def _apply_image_filters(
    stmt, count_stmt, image_origin, categories, is_intermediate, board_id, search_term, user_id, is_admin
):
    """Apply common image filters to both data and count queries."""
    if image_origin is not None:
        cond = col(ImageTable.image_origin) == image_origin.value
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)

    if categories is not None:
        category_strings = [c.value for c in set(categories)]
        cond = col(ImageTable.image_category).in_(category_strings)
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)

    if is_intermediate is not None:
        cond = col(ImageTable.is_intermediate) == is_intermediate
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)

    if board_id == "none":
        cond = col(BoardImageTable.board_id).is_(None)
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)
        if user_id is not None and not is_admin:
            user_cond = col(ImageTable.user_id) == user_id
            stmt = stmt.where(user_cond)
            count_stmt = count_stmt.where(user_cond)
    elif board_id is not None:
        cond = col(BoardImageTable.board_id) == board_id
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)

    if search_term:
        term = f"%{search_term.lower()}%"
        cond = col(ImageTable.metadata_).like(term) | col(ImageTable.created_at).like(term)
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)

    return stmt, count_stmt


def _image_order_by(stmt, starred_first: bool, order_dir: SQLiteDirection):
    """Apply the standard image gallery ordering."""
    created = (
        col(ImageTable.created_at).desc()
        if order_dir == SQLiteDirection.Descending
        else col(ImageTable.created_at).asc()
    )
    if starred_first:
        return stmt.order_by(col(ImageTable.starred).desc(), created)
    return stmt.order_by(created)


# --- Users ---


def _user_to_dto(row: UserTable) -> UserDTO:
    return UserDTO(
        user_id=row.user_id,
        email=row.email,
        display_name=row.display_name,
        is_admin=row.is_admin,
        is_active=row.is_active,
        created_at=row.created_at,
        updated_at=row.updated_at,
        last_login_at=row.last_login_at,
    )


# --- Style presets ---


def _style_preset_to_dto(row: StylePresetTable) -> StylePresetRecordDTO:
    return StylePresetRecordDTO.from_dict(
        {
            "id": row.id,
            "name": row.name,
            "preset_data": row.preset_data,
            "type": row.type,
            "user_id": row.user_id,
            "is_public": row.is_public,
            "created_at": str(row.created_at),
            "updated_at": str(row.updated_at),
        }
    )


# --- Workflows ---


def _workflow_to_dto(row: WorkflowLibraryTable) -> WorkflowRecordDTO:
    return WorkflowRecordDTO.from_dict(
        {
            "workflow_id": row.workflow_id,
            "workflow": row.workflow,
            "name": row.name,
            "created_at": str(row.created_at),
            "updated_at": str(row.updated_at),
            "opened_at": str(row.opened_at) if row.opened_at else None,
            "user_id": row.user_id,
            "is_public": row.is_public,
        }
    )


def _workflow_to_list_item(row: WorkflowLibraryTable) -> WorkflowRecordListItemDTO:
    return WorkflowRecordListItemDTOValidator.validate_python(
        {
            "workflow_id": row.workflow_id,
            "category": row.category,
            "name": row.name,
            "description": row.description,
            "created_at": str(row.created_at),
            "updated_at": str(row.updated_at),
            "opened_at": str(row.opened_at) if row.opened_at else None,
            "tags": row.tags,
            "user_id": row.user_id,
            "is_public": row.is_public,
        }
    )


def _apply_workflow_filters(stmt, count_stmt, categories, query, tags, has_been_opened, user_id, is_public):
    """Apply common workflow filters to both data and count queries."""
    if categories:
        category_strings = [c.value for c in categories]
        cond = col(WorkflowLibraryTable.category).in_(category_strings)
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)

    if tags:
        for tag in tags:
            cond = col(WorkflowLibraryTable.tags).like(f"%{tag.strip()}%")
            stmt = stmt.where(cond)
            count_stmt = count_stmt.where(cond)

    if has_been_opened is True:
        cond = col(WorkflowLibraryTable.opened_at).is_not(None)
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)
    elif has_been_opened is False:
        cond = col(WorkflowLibraryTable.opened_at).is_(None)
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)

    stripped_query = query.strip() if query else None
    if stripped_query:
        wildcard = f"%{stripped_query}%"
        cond = (
            col(WorkflowLibraryTable.name).like(wildcard)
            | col(WorkflowLibraryTable.description).like(wildcard)
            | col(WorkflowLibraryTable.tags).like(wildcard)
        )
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)

    if user_id is not None:
        cond = (col(WorkflowLibraryTable.user_id) == user_id) | (col(WorkflowLibraryTable.category) == "default")
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)

    if is_public is True:
        cond = col(WorkflowLibraryTable.is_public) == True  # noqa: E712
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)
    elif is_public is False:
        cond = col(WorkflowLibraryTable.is_public) == False  # noqa: E712
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)

    return stmt, count_stmt


def _workflow_order_col(order_by: WorkflowRecordOrderBy):
    if order_by == WorkflowRecordOrderBy.Name:
        return col(WorkflowLibraryTable.name)
    elif order_by == WorkflowRecordOrderBy.Description:
        return col(WorkflowLibraryTable.description)
    elif order_by == WorkflowRecordOrderBy.CreatedAt:
        return col(WorkflowLibraryTable.created_at)
    elif order_by == WorkflowRecordOrderBy.UpdatedAt:
        return col(WorkflowLibraryTable.updated_at)
    elif order_by == WorkflowRecordOrderBy.OpenedAt:
        return col(WorkflowLibraryTable.opened_at)
    else:
        return col(WorkflowLibraryTable.created_at)


# --- Models ---

# Columns to ORDER BY for each ModelRecordOrderBy (the SQLite GENERATED / create_all Computed columns).
_MODEL_ORDER_BY_COLUMNS = {
    ModelRecordOrderBy.Type: ["type"],
    ModelRecordOrderBy.Base: ["base"],
    ModelRecordOrderBy.Name: ["name"],
    ModelRecordOrderBy.Format: ["format"],
}
_MODEL_DEFAULT_ORDER_COLUMNS = ["type", "base", "name", "format"]


def _model_order_columns(order_by: ModelRecordOrderBy, direction: SQLiteDirection):
    """Build the ORDER BY clause for the given key + direction over the generated columns."""
    names = _MODEL_ORDER_BY_COLUMNS.get(order_by, _MODEL_DEFAULT_ORDER_COLUMNS)
    descending = direction == SQLiteDirection.Descending
    return [literal_column(n).desc() if descending else literal_column(n).asc() for n in names]


# --- Session queue ---

_TERMINAL_STATUSES: tuple[str, ...] = ("completed", "failed", "canceled")

_QUEUE_COLUMNS = (
    SessionQueueTable.item_id,
    SessionQueueTable.batch_id,
    SessionQueueTable.queue_id,
    SessionQueueTable.session_id,
    SessionQueueTable.field_values,
    SessionQueueTable.session,
    SessionQueueTable.status,
    SessionQueueTable.priority,
    SessionQueueTable.error_traceback,
    SessionQueueTable.created_at,
    SessionQueueTable.updated_at,
    SessionQueueTable.started_at,
    SessionQueueTable.completed_at,
    SessionQueueTable.error_type,
    SessionQueueTable.error_message,
    SessionQueueTable.origin,
    SessionQueueTable.destination,
    SessionQueueTable.retried_from_item_id,
    SessionQueueTable.user_id,
    SessionQueueTable.status_sequence,
)


def _queue_status_change_values(status: str) -> dict[str, Any]:
    """Extra columns to set on a status-changing UPDATE.

    On SQLite these are maintained by triggers (started_at/completed_at) and by the legacy
    SQL (status_sequence). A fresh MySQL/Postgres DB built from create_all() has no triggers,
    so we set them in Python on every backend. `updated_at` is handled by the model's
    `onupdate`. Harmless on SQLite — the AFTER UPDATE triggers simply re-stamp the same values.
    """
    values: dict[str, Any] = {
        # Monotonic per-item counter the UI uses to order cross-channel status snapshots.
        "status_sequence": func.coalesce(SessionQueueTable.status_sequence, 0) + 1,
    }
    if status == "in_progress":
        values["started_at"] = datetime.utcnow()
    if status in _TERMINAL_STATUSES:
        values["completed_at"] = datetime.utcnow()
    return values


def _queue_row_to_dict(row: Row) -> dict[str, Any]:
    """Convert a Row produced by `_queue_select_item_with_user` to a plain dict
    that `SessionQueueItem.queue_item_from_dict` expects."""
    mapping = dict(row._mapping)
    # Stringify datetime columns so the Pydantic union (`datetime | str`) accepts them
    # consistently across queries that JOIN datetime columns from multiple tables.
    for ts_key in ("created_at", "updated_at", "started_at", "completed_at"):
        ts_value = mapping.get(ts_key)
        if ts_value is not None and not isinstance(ts_value, str):
            mapping[ts_key] = str(ts_value)
    mapping.setdefault("user_display_name", None)
    mapping.setdefault("user_email", None)
    mapping.setdefault("workflow", None)
    return mapping


def _queue_select_item_with_user():
    """Build a SELECT that mirrors `sq.*, u.display_name, u.email` with LEFT JOIN."""
    return (
        sa_select(
            *_QUEUE_COLUMNS,
            SessionQueueTable.workflow,
            UserTable.display_name.label("user_display_name"),
            UserTable.email.label("user_email"),
        )
        .select_from(SessionQueueTable)
        .join(UserTable, SessionQueueTable.user_id == UserTable.user_id, isouter=True)
    )


def _queue_value_tuple_to_dict(t: ValueToInsertTuple) -> dict[str, Any]:
    """Adapt the positional tuple from `prepare_values_to_insert` to a dict that
    SQLAlchemy Core's `insert(...).values([...])` expects."""
    return {
        "queue_id": t[0],
        "session": t[1],
        "session_id": t[2],
        "batch_id": t[3],
        "field_values": t[4],
        "priority": t[5],
        "workflow": t[6],
        "origin": t[7],
        "destination": t[8],
        "retried_from_item_id": t[9],
        "user_id": t[10],
    }


def _queue_cancelable_filter(queue_id: str, user_id: Optional[str], extra: list) -> list:
    """WHERE clauses selecting items that may be bulk-canceled (skips in-progress/terminal)."""
    where = [
        SessionQueueTable.queue_id == queue_id,
        SessionQueueTable.status.notin_(("canceled", "completed", "failed", "in_progress")),
    ]
    if user_id is not None:
        where.append(SessionQueueTable.user_id == user_id)
    where.extend(extra)
    return where


class DbQueries:
    """The application's query catalog: every SQL statement, one class, one file.

    Access via ``db.queries``. See the module docstring for the rules.
    """

    def __init__(self, db: "SqliteDatabase", logger: Logger) -> None:
        self._db = db
        self._logger = logger

    # region: App settings

    def app_settings_get(self, key: str) -> Optional[str]:
        with self._db.get_readonly_session() as session:
            row = session.get(AppSettingTable, key)
            return row.value if row else None

    def app_settings_set(self, key: str, value: str) -> None:
        with self._db.get_session() as session:
            existing = session.get(AppSettingTable, key)
            if existing is not None:
                existing.value = value
                session.add(existing)
            else:
                session.add(AppSettingTable(key=key, value=value))

    # endregion

    # region: Client state

    def client_state_set_by_key(self, user_id: str, key: str, value: str) -> None:
        with self._db.get_session() as session:
            existing = session.get(ClientStateTable, (user_id, key))
            if existing is not None:
                existing.value = value
                session.add(existing)
            else:
                session.add(ClientStateTable(user_id=user_id, key=key, value=value))

    def client_state_get_by_key(self, user_id: str, key: str) -> Optional[str]:
        with self._db.get_readonly_session() as session:
            row = session.get(ClientStateTable, (user_id, key))
            if row is None:
                return None
            return row.value

    def client_state_get_keys_by_prefix(self, user_id: str, prefix: str) -> list[str]:
        # Escape LIKE wildcards (%, _) and the escape char itself so callers can pass
        # arbitrary strings as a literal prefix without accidental pattern matching.
        escaped_prefix = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._db.get_readonly_session() as session:
            stmt = (
                select(ClientStateTable.key)
                .where(
                    col(ClientStateTable.user_id) == user_id,
                    col(ClientStateTable.key).like(f"{escaped_prefix}%", escape="\\"),
                )
                .order_by(col(ClientStateTable.updated_at).desc())
            )
            return list(session.exec(stmt).all())

    def client_state_delete_by_key(self, user_id: str, key: str) -> None:
        with self._db.get_session() as session:
            row = session.get(ClientStateTable, (user_id, key))
            if row is not None:
                session.delete(row)

    def client_state_delete_all(self, user_id: str) -> None:
        with self._db.get_session() as session:
            stmt = select(ClientStateTable).where(col(ClientStateTable.user_id) == user_id)
            rows = session.exec(stmt).all()
            for row in rows:
                session.delete(row)

    # endregion

    # region: Model relationships

    def model_relationships_add(self, model_key_1: str, model_key_2: str) -> None:
        a, b = sorted([model_key_1, model_key_2])
        with self._db.get_session() as session:
            existing = session.get(ModelRelationshipTable, (a, b))
            if existing is None:
                session.add(ModelRelationshipTable(model_key_1=a, model_key_2=b))

    def model_relationships_remove(self, model_key_1: str, model_key_2: str) -> None:
        a, b = sorted([model_key_1, model_key_2])
        with self._db.get_session() as session:
            existing = session.get(ModelRelationshipTable, (a, b))
            if existing is not None:
                session.delete(existing)

    def model_relationships_get_related(self, model_key: str) -> list[str]:
        with self._db.get_readonly_session() as session:
            # Get keys where model_key appears in either column
            stmt1 = select(ModelRelationshipTable.model_key_2).where(
                col(ModelRelationshipTable.model_key_1) == model_key
            )
            stmt2 = select(ModelRelationshipTable.model_key_1).where(
                col(ModelRelationshipTable.model_key_2) == model_key
            )
            results1 = session.exec(stmt1).all()
            results2 = session.exec(stmt2).all()
        return list(set(results1 + results2))

    def model_relationships_get_related_batch(self, model_keys: list[str]) -> list[str]:
        with self._db.get_readonly_session() as session:
            stmt1 = select(ModelRelationshipTable.model_key_2).where(
                col(ModelRelationshipTable.model_key_1).in_(model_keys)
            )
            stmt2 = select(ModelRelationshipTable.model_key_1).where(
                col(ModelRelationshipTable.model_key_2).in_(model_keys)
            )
            results1 = session.exec(stmt1).all()
            results2 = session.exec(stmt2).all()
        return list(set(results1 + results2))

    # endregion

    # region: Style presets

    def style_presets_get(self, style_preset_id: str) -> StylePresetRecordDTO:
        with self._db.get_readonly_session() as session:
            row = session.get(StylePresetTable, style_preset_id)
            if row is None:
                raise StylePresetNotFoundError(f"Style preset with id {style_preset_id} not found")
            return _style_preset_to_dto(row)

    def style_presets_create(self, style_preset: StylePresetWithoutId, user_id: str) -> StylePresetRecordDTO:
        style_preset_id = uuid_string()
        row = StylePresetTable(
            id=style_preset_id,
            name=style_preset.name,
            preset_data=style_preset.preset_data.model_dump_json(),
            type=style_preset.type,
            user_id=user_id,
            is_public=style_preset.is_public,
        )
        with self._db.get_session() as session:
            session.add(row)
        return self.style_presets_get(style_preset_id)

    def style_presets_create_many(self, style_presets: list[StylePresetWithoutId], user_id: str) -> None:
        with self._db.get_session() as session:
            for style_preset in style_presets:
                row = StylePresetTable(
                    id=uuid_string(),
                    name=style_preset.name,
                    preset_data=style_preset.preset_data.model_dump_json(),
                    type=style_preset.type,
                    user_id=user_id,
                    is_public=style_preset.is_public,
                )
                session.add(row)

    def style_presets_update(self, style_preset_id: str, changes: StylePresetChanges) -> StylePresetRecordDTO:
        with self._db.get_session() as session:
            row = session.get(StylePresetTable, style_preset_id)
            if row is None:
                raise StylePresetNotFoundError(f"Style preset with id {style_preset_id} not found")

            if changes.name is not None:
                row.name = changes.name
            if changes.preset_data is not None:
                row.preset_data = changes.preset_data.model_dump_json()

            session.add(row)
        return self.style_presets_get(style_preset_id)

    def style_presets_delete(self, style_preset_id: str) -> None:
        with self._db.get_session() as session:
            row = session.get(StylePresetTable, style_preset_id)
            if row is not None:
                session.delete(row)

    def style_presets_get_many(
        self,
        type: Optional[PresetType] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> list[StylePresetRecordDTO]:
        with self._db.get_readonly_session() as session:
            stmt = select(StylePresetTable)
            if not is_admin:
                # Visible to non-admin: default + public + own.
                visibility = (col(StylePresetTable.type) == "default") | (col(StylePresetTable.is_public) == True)  # noqa: E712
                if user_id is not None:
                    visibility = visibility | (col(StylePresetTable.user_id) == user_id)
                stmt = stmt.where(visibility)
            if type is not None:
                stmt = stmt.where(col(StylePresetTable.type) == type)
            stmt = stmt.order_by(col(StylePresetTable.name).asc())
            rows = session.exec(stmt).all()
            return [_style_preset_to_dto(r) for r in rows]

    def style_presets_delete_defaults(self) -> None:
        """Delete all presets of type 'default' (used by the startup sync)."""
        with self._db.get_session() as session:
            stmt = select(StylePresetTable).where(col(StylePresetTable.type) == "default")
            rows = session.exec(stmt).all()
            for row in rows:
                session.delete(row)

    # endregion

    # region: Boards

    def boards_delete(self, board_id: str) -> None:
        with self._db.get_session() as session:
            try:
                board = session.get(BoardTable, board_id)
                if board:
                    session.delete(board)
            except Exception as e:
                raise BoardRecordDeleteException from e

    def boards_save(self, board_name: str, user_id: str) -> BoardRecord:
        board_id = uuid_string()
        board = BoardTable(board_id=board_id, board_name=board_name, user_id=user_id)
        with self._db.get_session() as session:
            try:
                session.add(board)
                session.flush()
                return _board_to_record(board)
            except Exception as e:
                raise BoardRecordSaveException from e

    def boards_get(self, board_id: str) -> BoardRecord:
        with self._db.get_readonly_session() as session:
            board = session.get(BoardTable, board_id)
            if board is None:
                raise BoardRecordNotFoundException
            return _board_to_record(board)

    def boards_update(self, board_id: str, changes: BoardChanges) -> BoardRecord:
        with self._db.get_session() as session:
            try:
                board = session.get(BoardTable, board_id)
                if board is None:
                    raise BoardRecordNotFoundException

                if changes.board_name is not None:
                    board.board_name = changes.board_name
                if changes.cover_image_name is not None:
                    board.cover_image_name = changes.cover_image_name
                if changes.archived is not None:
                    board.archived = changes.archived
                if changes.board_visibility is not None:
                    board.board_visibility = changes.board_visibility.value

                session.add(board)
                session.flush()
                return _board_to_record(board)
            except BoardRecordNotFoundException:
                raise
            except Exception as e:
                raise BoardRecordSaveException from e

    def boards_get_many(
        self,
        user_id: str,
        is_admin: bool,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        offset: int = 0,
        limit: int = 10,
        include_archived: bool = False,
    ) -> OffsetPaginatedResults[BoardRecord]:
        with self._db.get_readonly_session() as session:
            # Build filter conditions
            conditions = []

            if not is_admin:
                conditions.append(
                    (col(BoardTable.user_id) == user_id) | (col(BoardTable.board_visibility).in_(["shared", "public"]))
                )

            if not include_archived:
                conditions.append(col(BoardTable.archived) == False)  # noqa: E712

            # Count query
            count_stmt = select(func.count()).select_from(BoardTable)
            for cond in conditions:
                count_stmt = count_stmt.where(cond)
            total = session.exec(count_stmt).one()

            # Data query
            stmt = select(BoardTable)
            for cond in conditions:
                stmt = stmt.where(cond)

            # Apply ordering
            order_col = (
                col(BoardTable.created_at) if order_by == BoardRecordOrderBy.CreatedAt else col(BoardTable.board_name)
            )
            stmt = stmt.order_by(order_col.desc() if direction == SQLiteDirection.Descending else order_col.asc())
            stmt = stmt.offset(offset).limit(limit)

            results = session.exec(stmt).all()
            boards = [_board_to_record(r) for r in results]

        return OffsetPaginatedResults[BoardRecord](items=boards, offset=offset, limit=limit, total=total)

    def boards_get_all(
        self,
        user_id: str,
        is_admin: bool,
        order_by: BoardRecordOrderBy,
        direction: SQLiteDirection,
        include_archived: bool = False,
    ) -> list[BoardRecord]:
        with self._db.get_readonly_session() as session:
            stmt = select(BoardTable)

            if not is_admin:
                stmt = stmt.where(
                    (col(BoardTable.user_id) == user_id) | (col(BoardTable.board_visibility).in_(["shared", "public"]))
                )

            if not include_archived:
                stmt = stmt.where(col(BoardTable.archived) == False)  # noqa: E712

            # Apply ordering
            if order_by == BoardRecordOrderBy.Name:
                order_col = col(BoardTable.board_name)
            else:
                order_col = col(BoardTable.created_at)

            stmt = stmt.order_by(order_col.desc() if direction == SQLiteDirection.Descending else order_col.asc())

            results = session.exec(stmt).all()
            boards = [_board_to_record(r) for r in results]

        return boards

    # endregion

    # region: Board images

    def board_images_add(self, board_id: str, image_name: str) -> None:
        with self._db.get_session() as session:
            existing = session.get(BoardImageTable, image_name)
            if existing is not None:
                existing.board_id = board_id
                session.add(existing)
            else:
                session.add(BoardImageTable(board_id=board_id, image_name=image_name))

    def board_images_remove(self, image_name: str) -> None:
        with self._db.get_session() as session:
            existing = session.get(BoardImageTable, image_name)
            if existing is not None:
                session.delete(existing)

    def board_images_get_images_for_board(
        self,
        board_id: str,
        offset: int = 0,
        limit: int = 10,
    ) -> OffsetPaginatedResults[ImageRecord]:
        with self._db.get_readonly_session() as session:
            # Join board_images with images
            stmt = (
                select(ImageTable)
                .join(BoardImageTable, col(BoardImageTable.image_name) == col(ImageTable.image_name))
                .where(col(BoardImageTable.board_id) == board_id)
                .order_by(col(BoardImageTable.updated_at).desc())
            )
            results = session.exec(stmt).all()
            images = [deserialize_image_record(_image_to_dict(r)) for r in results]

            # Total count of all images
            count_stmt = select(func.count()).select_from(ImageTable)
            count = session.exec(count_stmt).one()

        return OffsetPaginatedResults(items=images, offset=offset, limit=limit, total=count)

    def board_images_get_all_image_names_for_board(
        self,
        board_id: str,
        categories: Optional[list[ImageCategory]],
        is_intermediate: Optional[bool],
    ) -> list[str]:
        with self._db.get_readonly_session() as session:
            stmt = select(ImageTable.image_name).outerjoin(
                BoardImageTable, col(BoardImageTable.image_name) == col(ImageTable.image_name)
            )

            if board_id == "none":
                stmt = stmt.where(col(BoardImageTable.board_id).is_(None))
            else:
                stmt = stmt.where(col(BoardImageTable.board_id) == board_id)

            if categories is not None:
                category_strings = [c.value for c in set(categories)]
                stmt = stmt.where(col(ImageTable.image_category).in_(category_strings))

            if is_intermediate is not None:
                stmt = stmt.where(col(ImageTable.is_intermediate) == is_intermediate)

            results = session.exec(stmt).all()
        return list(results)

    def board_images_get_board_for_image(self, image_name: str) -> Optional[str]:
        with self._db.get_readonly_session() as session:
            row = session.get(BoardImageTable, image_name)
            if row is None:
                return None
            return row.board_id

    def board_images_get_image_count(self, board_id: str) -> int:
        category_strings = [c.value for c in set(IMAGE_CATEGORIES)]
        return self._board_images_count_in_categories(board_id, category_strings)

    def board_images_get_asset_count(self, board_id: str) -> int:
        category_strings = [c.value for c in set(ASSETS_CATEGORIES)]
        return self._board_images_count_in_categories(board_id, category_strings)

    def _board_images_count_in_categories(self, board_id: str, category_strings: list[str]) -> int:
        with self._db.get_readonly_session() as session:
            stmt = (
                select(func.count())
                .select_from(BoardImageTable)
                .join(ImageTable, col(BoardImageTable.image_name) == col(ImageTable.image_name))
                .where(
                    col(ImageTable.is_intermediate) == False,  # noqa: E712
                    col(ImageTable.image_category).in_(category_strings),
                    col(BoardImageTable.board_id) == board_id,
                )
            )
            count = session.exec(stmt).one()
        return count

    # endregion

    # region: Users

    def users_create(self, email: str, display_name: Optional[str], password_hash: str, is_admin: bool) -> str:
        """Insert a new user row and return the generated user id."""
        user_id = str(uuid4())
        user = UserTable(
            user_id=user_id,
            email=email,
            display_name=display_name,
            password_hash=password_hash,
            is_admin=is_admin,
        )
        with self._db.get_session() as session:
            session.add(user)
        return user_id

    def users_get(self, user_id: str) -> Optional[UserDTO]:
        with self._db.get_readonly_session() as session:
            row = session.get(UserTable, user_id)
            if row is None:
                return None
            return _user_to_dto(row)

    def users_get_by_email(self, email: str) -> Optional[UserDTO]:
        with self._db.get_readonly_session() as session:
            stmt = select(UserTable).where(col(UserTable.email) == email)
            row = session.exec(stmt).first()
            if row is None:
                return None
            return _user_to_dto(row)

    def users_apply_update(
        self,
        user_id: str,
        display_name: Optional[str],
        password_hash: Optional[str],
        is_admin: Optional[bool],
        is_active: Optional[bool],
    ) -> None:
        """Apply non-None field changes to a user row. The password is already hashed."""
        with self._db.get_session() as session:
            row = session.get(UserTable, user_id)
            if row is None:
                raise ValueError(f"User {user_id} not found")

            if display_name is not None:
                row.display_name = display_name
            if password_hash is not None:
                row.password_hash = password_hash
            if is_admin is not None:
                row.is_admin = is_admin
            if is_active is not None:
                row.is_active = is_active

            session.add(row)

    def users_delete(self, user_id: str) -> None:
        with self._db.get_session() as session:
            row = session.get(UserTable, user_id)
            if row is None:
                raise ValueError(f"User {user_id} not found")
            session.delete(row)

    def users_authenticate(self, email: str, password: str) -> Optional[UserDTO]:
        """Verify credentials and stamp last_login_at in a single transaction."""
        with self._db.get_session() as session:
            stmt = select(UserTable).where(col(UserTable.email) == email)
            row = session.exec(stmt).first()
            if row is None:
                return None

            if not verify_password(password, row.password_hash):
                return None

            row.last_login_at = datetime.now(timezone.utc)
            session.add(row)

            return _user_to_dto(row)

    def users_count_admins(self) -> int:
        with self._db.get_readonly_session() as session:
            stmt = (
                select(func.count())
                .select_from(UserTable)
                .where(
                    col(UserTable.is_admin) == True,  # noqa: E712
                    col(UserTable.is_active) == True,  # noqa: E712
                )
            )
            count = session.exec(stmt).one()
        return count

    def users_get_admin_email(self) -> Optional[str]:
        with self._db.get_readonly_session() as session:
            stmt = (
                select(UserTable)
                .where(
                    col(UserTable.is_admin) == True,  # noqa: E712
                    col(UserTable.is_active) == True,  # noqa: E712
                )
                .order_by(col(UserTable.created_at).asc())
                .limit(1)
            )
            row = session.exec(stmt).first()
            return row.email if row else None

    def users_list(self, limit: int = 100, offset: int = 0) -> list[UserDTO]:
        with self._db.get_readonly_session() as session:
            stmt = select(UserTable).order_by(col(UserTable.created_at).desc()).limit(limit).offset(offset)
            rows = session.exec(stmt).all()
            return [_user_to_dto(r) for r in rows]

    # endregion

    # region: Models

    def models_insert(self, config: AnyModelConfig) -> None:
        row = ModelTable(id=config.key, config=config.model_dump_json())
        try:
            with self._db.get_session() as session:
                session.add(row)
        except Exception as e:
            err_str = str(e)
            # "UNIQUE constraint failed" is SQLite, "Duplicate entry" is MySQL/MariaDB (error 1062).
            if "UNIQUE constraint failed" in err_str or "Duplicate entry" in err_str:
                if "path" in err_str:
                    msg = f"A model with path '{config.path}' is already installed"
                elif "name" in err_str:
                    msg = f"A model with name='{config.name}', type='{config.type}', base='{config.base}' is already installed"
                else:
                    msg = f"A model with key '{config.key}' is already installed"
                raise DuplicateModelException(msg) from e
            raise

    def models_delete(self, key: str) -> None:
        with self._db.get_session() as session:
            row = session.get(ModelTable, key)
            if row is None:
                raise UnknownModelException("model not found")
            session.delete(row)

    def models_update_config_json(self, key: str, config_json: str) -> None:
        with self._db.get_session() as session:
            row = session.get(ModelTable, key)
            if row is None:
                raise UnknownModelException("model not found")
            row.config = config_json
            session.add(row)

    def models_get(self, key: str) -> AnyModelConfig:
        with self._db.get_readonly_session() as session:
            row = session.get(ModelTable, key)
            if row is None:
                raise UnknownModelException("model not found")
            return ModelConfigFactory.from_dict(json.loads(row.config))

    def models_get_by_hash(self, hash: str) -> AnyModelConfig:
        with self._db.get_readonly_session() as session:
            stmt = select(ModelTable).where(literal_column("hash") == hash)
            row = session.exec(stmt).first()
            if row is None:
                raise UnknownModelException("model not found")
            return ModelConfigFactory.from_dict(json.loads(row.config))

    def models_exists(self, key: str) -> bool:
        with self._db.get_readonly_session() as session:
            row = session.get(ModelTable, key)
        return row is not None

    def models_search_by_attr(
        self,
        model_name: Optional[str] = None,
        base_model: Optional[BaseModelType] = None,
        model_type: Optional[ModelType] = None,
        model_format: Optional[ModelFormat] = None,
        order_by: ModelRecordOrderBy = ModelRecordOrderBy.Default,
        direction: SQLiteDirection = SQLiteDirection.Ascending,
    ) -> list[AnyModelConfig]:
        with self._db.get_readonly_session() as session:
            stmt = select(ModelTable)

            if model_name:
                stmt = stmt.where(literal_column("name") == model_name)
            if base_model:
                stmt = stmt.where(literal_column("base") == base_model)
            if model_type:
                stmt = stmt.where(literal_column("type") == model_type)
            if model_format:
                stmt = stmt.where(literal_column("format") == model_format)

            # Apply ordering via the generated columns
            stmt = stmt.order_by(*_model_order_columns(order_by, direction))

            rows = session.exec(stmt).all()
            # Extract config strings while still in the session
            config_strings = [row.config for row in rows]

        results: list[AnyModelConfig] = []
        for config_str in config_strings:
            try:
                model_config = ModelConfigFactory.from_dict(json.loads(config_str))
            except pydantic.ValidationError as e:
                config_preview = f"{config_str[:64]}..." if len(config_str) > 64 else config_str
                try:
                    name = json.loads(config_str).get("name", "<unknown>")
                except Exception:
                    name = "<unknown>"
                self._logger.warning(
                    f"Skipping invalid model config in the database with name {name}. ({config_preview})"
                )
                self._logger.warning(f"Validation error: {e}")
            else:
                results.append(model_config)

        return results

    def models_search_by_path(self, path: Union[str, Path]) -> list[AnyModelConfig]:
        with self._db.get_readonly_session() as session:
            stmt = select(ModelTable).where(literal_column("path") == str(path))
            rows = session.exec(stmt).all()
            configs = [r.config for r in rows]
        return [ModelConfigFactory.from_dict(json.loads(c)) for c in configs]

    def models_search_by_hash(self, hash: str) -> list[AnyModelConfig]:
        with self._db.get_readonly_session() as session:
            stmt = select(ModelTable).where(literal_column("hash") == hash)
            rows = session.exec(stmt).all()
            configs = [r.config for r in rows]
        return [ModelConfigFactory.from_dict(json.loads(c)) for c in configs]

    def models_list_summaries(
        self,
        page: int = 0,
        per_page: int = 10,
        order_by: ModelRecordOrderBy = ModelRecordOrderBy.Default,
        direction: SQLiteDirection = SQLiteDirection.Ascending,
    ) -> PaginatedResults[ModelSummary]:
        with self._db.get_readonly_session() as session:
            total = session.exec(select(func.count()).select_from(ModelTable)).one()
            stmt = (
                select(ModelTable)
                .order_by(*_model_order_columns(order_by, direction))
                .limit(per_page)
                .offset(page * per_page)
            )
            rows = session.exec(stmt).all()
            # Read the generated columns while the rows are still bound to the session.
            summaries = [(r.id, r.type, r.base, r.format, r.name, r.description) for r in rows]

        items: list[ModelSummary] = []
        for key, model_type, base, model_format, name, description in summaries:
            try:
                items.append(
                    ModelSummary(
                        key=key,
                        type=model_type,
                        base=base,
                        format=model_format,
                        name=name,
                        description=description or "",
                        tags=set(),
                    )
                )
            except pydantic.ValidationError:
                # Skip models whose type/base/format are not recognised by the current build.
                self._logger.warning(f"Skipping model summary for {key}: unsupported attributes")
        return PaginatedResults(
            page=page,
            pages=ceil(total / per_page),
            per_page=per_page,
            total=total,
            items=items,
        )

    def models_get_all_config_json(self) -> list[str]:
        """All model config JSON blobs (used by the orphaned-models scan)."""
        with self._db.get_readonly_session() as session:
            return list(session.exec(select(ModelTable.config)).all())

    # endregion

    # region: Workflows

    def workflows_get(self, workflow_id: str) -> WorkflowRecordDTO:
        with self._db.get_readonly_session() as session:
            row = session.get(WorkflowLibraryTable, workflow_id)
            if row is None:
                raise WorkflowNotFoundError(f"Workflow with id {workflow_id} not found")
            return _workflow_to_dto(row)

    def workflows_insert(self, workflow_id: str, workflow_json: str, user_id: str, is_public: bool) -> None:
        row = WorkflowLibraryTable(
            workflow_id=workflow_id,
            workflow=workflow_json,
            user_id=user_id,
            is_public=is_public,
        )
        with self._db.get_session() as session:
            session.add(row)

    def workflows_update_json(self, workflow_id: str, workflow_json: str, user_id: Optional[str]) -> None:
        """Update the workflow JSON of a user-category workflow (no-op if not found)."""
        with self._db.get_session() as session:
            stmt = select(WorkflowLibraryTable).where(
                col(WorkflowLibraryTable.workflow_id) == workflow_id,
                col(WorkflowLibraryTable.category) == "user",
            )
            if user_id is not None:
                stmt = stmt.where(col(WorkflowLibraryTable.user_id) == user_id)

            row = session.exec(stmt).first()
            if row is not None:
                row.workflow = workflow_json
                session.add(row)

    def workflows_delete_user_workflow(self, workflow_id: str, user_id: Optional[str]) -> None:
        """Delete a user-category workflow (no-op if not found)."""
        with self._db.get_session() as session:
            stmt = select(WorkflowLibraryTable).where(
                col(WorkflowLibraryTable.workflow_id) == workflow_id,
                col(WorkflowLibraryTable.category) == "user",
            )
            if user_id is not None:
                stmt = stmt.where(col(WorkflowLibraryTable.user_id) == user_id)

            row = session.exec(stmt).first()
            if row is not None:
                session.delete(row)

    def workflows_set_public(
        self, workflow_id: str, workflow_json: str, is_public: bool, user_id: Optional[str]
    ) -> None:
        """Update workflow JSON + is_public of a user-category workflow (no-op if not found)."""
        with self._db.get_session() as session:
            stmt = select(WorkflowLibraryTable).where(
                col(WorkflowLibraryTable.workflow_id) == workflow_id,
                col(WorkflowLibraryTable.category) == "user",
            )
            if user_id is not None:
                stmt = stmt.where(col(WorkflowLibraryTable.user_id) == user_id)

            row = session.exec(stmt).first()
            if row is not None:
                row.workflow = workflow_json
                row.is_public = is_public
                session.add(row)

    def workflows_get_many(
        self,
        order_by: WorkflowRecordOrderBy,
        direction: SQLiteDirection,
        categories: Optional[list[WorkflowCategory]] = None,
        page: int = 0,
        per_page: Optional[int] = None,
        query: Optional[str] = None,
        tags: Optional[list[str]] = None,
        has_been_opened: Optional[bool] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> PaginatedResults[WorkflowRecordListItemDTO]:
        with self._db.get_readonly_session() as session:
            stmt = select(WorkflowLibraryTable)
            count_stmt = select(func.count()).select_from(WorkflowLibraryTable)

            # Apply filters to both
            stmt, count_stmt = _apply_workflow_filters(
                stmt,
                count_stmt,
                categories,
                query,
                tags,
                has_been_opened,
                user_id,
                is_public,
            )

            # Count
            total = session.exec(count_stmt).one()

            # Ordering
            order_col = _workflow_order_col(order_by)
            stmt = stmt.order_by(order_col.desc() if direction == SQLiteDirection.Descending else order_col.asc())

            # Pagination
            if per_page:
                stmt = stmt.limit(per_page).offset(page * per_page)

            rows = session.exec(stmt).all()
            workflows = [_workflow_to_list_item(r) for r in rows]

        if per_page:
            pages = total // per_page + (total % per_page > 0)
        else:
            pages = 1

        return PaginatedResults(
            items=workflows,
            page=page,
            per_page=per_page if per_page else total,
            pages=pages,
            total=total,
        )

    def workflows_counts_by_tag(
        self,
        tags: list[str],
        categories: Optional[list[WorkflowCategory]] = None,
        has_been_opened: Optional[bool] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> dict[str, int]:
        if not tags:
            return {}

        result: dict[str, int] = {}
        with self._db.get_readonly_session() as session:
            for tag in tags:
                stmt = select(func.count()).select_from(WorkflowLibraryTable)
                stmt, _ = _apply_workflow_filters(
                    stmt, stmt, categories, None, None, has_been_opened, user_id, is_public
                )
                stmt = stmt.where(col(WorkflowLibraryTable.tags).like(f"%{tag.strip()}%"))
                count = session.exec(stmt).one()
                result[tag] = count
        return result

    def workflows_counts_by_category(
        self,
        categories: list[WorkflowCategory],
        has_been_opened: Optional[bool] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        with self._db.get_readonly_session() as session:
            for category in categories:
                stmt = select(func.count()).select_from(WorkflowLibraryTable)
                stmt, _ = _apply_workflow_filters(
                    stmt, stmt, categories, None, None, has_been_opened, user_id, is_public
                )
                stmt = stmt.where(col(WorkflowLibraryTable.category) == category.value)
                count = session.exec(stmt).one()
                result[category.value] = count
        return result

    def workflows_touch_opened_at(self, workflow_id: str, user_id: Optional[str] = None) -> None:
        with self._db.get_session() as session:
            stmt = select(WorkflowLibraryTable).where(col(WorkflowLibraryTable.workflow_id) == workflow_id)
            if user_id is not None:
                stmt = stmt.where(col(WorkflowLibraryTable.user_id) == user_id)
            row = session.exec(stmt).first()
            if row is not None:
                row.opened_at = datetime.utcnow()
                session.add(row)

    def workflows_get_all_tags(
        self,
        categories: Optional[list[WorkflowCategory]] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> list[str]:
        with self._db.get_readonly_session() as session:
            stmt = select(WorkflowLibraryTable.tags).where(
                col(WorkflowLibraryTable.tags).is_not(None),
                col(WorkflowLibraryTable.tags) != "",
            )

            if categories:
                category_strings = [c.value for c in categories]
                stmt = stmt.where(col(WorkflowLibraryTable.category).in_(category_strings))
            if user_id is not None:
                stmt = stmt.where(
                    (col(WorkflowLibraryTable.user_id) == user_id) | (col(WorkflowLibraryTable.category) == "default")
                )
            if is_public is True:
                stmt = stmt.where(col(WorkflowLibraryTable.is_public) == True)  # noqa: E712
            elif is_public is False:
                stmt = stmt.where(col(WorkflowLibraryTable.is_public) == False)  # noqa: E712

            rows = session.exec(stmt).all()

        all_tags: set[str] = set()
        for tags_value in rows:
            if tags_value and isinstance(tags_value, str):
                for tag in tags_value.split(","):
                    tag_stripped = tag.strip()
                    if tag_stripped:
                        all_tags.add(tag_stripped)
        return sorted(all_tags)

    def workflows_apply_default_sync(
        self,
        delete_ids: list[str],
        add_workflows: list[tuple[str, str]],
        update_workflows: list[tuple[str, str]],
    ) -> None:
        """Apply the startup default-workflow sync in one transaction.

        ``add_workflows`` and ``update_workflows`` are (workflow_id, workflow_json) pairs.
        """
        with self._db.get_session() as session:
            for workflow_id in delete_ids:
                row = session.get(WorkflowLibraryTable, workflow_id)
                if row is not None:
                    session.delete(row)

            for workflow_id, workflow_json in add_workflows:
                session.add(
                    WorkflowLibraryTable(
                        workflow_id=workflow_id,
                        workflow=workflow_json,
                    )
                )

            for workflow_id, workflow_json in update_workflows:
                row = session.get(WorkflowLibraryTable, workflow_id)
                if row is not None:
                    row.workflow = workflow_json
                    session.add(row)

    # endregion

    # region: Images

    def images_get(self, image_name: str) -> ImageRecord:
        with self._db.get_readonly_session() as session:
            row = session.get(ImageTable, image_name)
            if row is None:
                raise ImageRecordNotFoundException
            return deserialize_image_record(_image_to_dict(row))

    def images_get_user_id(self, image_name: str) -> Optional[str]:
        with self._db.get_readonly_session() as session:
            row = session.get(ImageTable, image_name)
            if row is None:
                return None
            return row.user_id

    def images_get_metadata(self, image_name: str) -> Optional[MetadataField]:
        with self._db.get_readonly_session() as session:
            row = session.get(ImageTable, image_name)
            if row is None:
                raise ImageRecordNotFoundException
            if row.metadata_ is None:
                return None
            return MetadataFieldValidator.validate_json(row.metadata_)

    def images_update(self, image_name: str, changes: ImageRecordChanges) -> None:
        with self._db.get_session() as session:
            try:
                row = session.get(ImageTable, image_name)
                if row is None:
                    raise ImageRecordNotFoundException

                if changes.image_category is not None:
                    row.image_category = changes.image_category.value
                if changes.session_id is not None:
                    row.session_id = changes.session_id
                if changes.is_intermediate is not None:
                    row.is_intermediate = changes.is_intermediate
                if changes.starred is not None:
                    row.starred = changes.starred

                session.add(row)
            except ImageRecordNotFoundException:
                raise
            except Exception as e:
                raise ImageRecordSaveException from e

    def images_get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        image_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> OffsetPaginatedResults[ImageRecord]:
        with self._db.get_readonly_session() as session:
            # Base query with left join to board_images
            stmt = select(ImageTable).outerjoin(
                BoardImageTable, col(BoardImageTable.image_name) == col(ImageTable.image_name)
            )
            count_stmt = (
                select(func.count())
                .select_from(ImageTable)
                .outerjoin(BoardImageTable, col(BoardImageTable.image_name) == col(ImageTable.image_name))
            )

            # Apply filters
            stmt, count_stmt = _apply_image_filters(
                stmt,
                count_stmt,
                image_origin,
                categories,
                is_intermediate,
                board_id,
                search_term,
                user_id,
                is_admin,
            )

            # Count
            total = session.exec(count_stmt).one()

            # Ordering + pagination
            stmt = _image_order_by(stmt, starred_first, order_dir)
            stmt = stmt.limit(limit).offset(offset)
            results = session.exec(stmt).all()
            images = [deserialize_image_record(_image_to_dict(r)) for r in results]

        return OffsetPaginatedResults(items=images, offset=offset, limit=limit, total=total)

    def images_delete(self, image_name: str) -> None:
        with self._db.get_session() as session:
            try:
                row = session.get(ImageTable, image_name)
                if row is not None:
                    session.delete(row)
            except Exception as e:
                raise ImageRecordDeleteException from e

    def images_delete_many(self, image_names: list[str]) -> None:
        with self._db.get_session() as session:
            try:
                stmt = select(ImageTable).where(col(ImageTable.image_name).in_(image_names))
                rows = session.exec(stmt).all()
                for row in rows:
                    session.delete(row)
            except Exception as e:
                raise ImageRecordDeleteException from e

    def images_get_intermediates_count(self, user_id: Optional[str] = None) -> int:
        with self._db.get_readonly_session() as session:
            stmt = (
                select(func.count())
                .select_from(ImageTable)
                .where(
                    col(ImageTable.is_intermediate) == True  # noqa: E712
                )
            )
            if user_id is not None:
                stmt = stmt.where(col(ImageTable.user_id) == user_id)
            count = session.exec(stmt).one()
        return count

    def images_delete_intermediates(self) -> list[tuple[str, str]]:
        with self._db.get_session() as session:
            try:
                stmt = select(ImageTable).where(col(ImageTable.is_intermediate) == True)  # noqa: E712
                rows = session.exec(stmt).all()
                # Return (image_name, image_subfolder) pairs so the file storage can clean up.
                pairs = [(r.image_name, r.image_subfolder) for r in rows]
                for row in rows:
                    session.delete(row)
            except Exception as e:
                raise ImageRecordDeleteException from e
        return pairs

    def images_save(
        self,
        image_name: str,
        image_origin: ResourceOrigin,
        image_category: ImageCategory,
        width: int,
        height: int,
        has_workflow: bool,
        is_intermediate: Optional[bool] = False,
        starred: Optional[bool] = False,
        session_id: Optional[str] = None,
        node_id: Optional[str] = None,
        metadata: Optional[str] = None,
        user_id: Optional[str] = None,
        image_subfolder: str = "",
    ) -> datetime:
        row = ImageTable(
            image_name=image_name,
            image_origin=image_origin.value,
            image_category=image_category.value,
            width=width,
            height=height,
            session_id=session_id,
            node_id=node_id,
            metadata_=metadata,
            is_intermediate=is_intermediate or False,
            starred=starred or False,
            has_workflow=has_workflow,
            user_id=user_id or "system",
            image_subfolder=image_subfolder,
        )
        with self._db.get_session() as session:
            try:
                session.add(row)
                session.flush()
                # With expire_on_commit=False, row.created_at is still accessible
                return (
                    row.created_at
                    if isinstance(row.created_at, datetime)
                    else datetime.fromisoformat(str(row.created_at))
                )
            except Exception as e:
                raise ImageRecordSaveException from e

    def images_get_most_recent_for_board(self, board_id: str) -> Optional[ImageRecord]:
        with self._db.get_readonly_session() as session:
            stmt = (
                select(ImageTable)
                .join(BoardImageTable, col(ImageTable.image_name) == col(BoardImageTable.image_name))
                .where(
                    col(BoardImageTable.board_id) == board_id,
                    col(ImageTable.is_intermediate) == False,  # noqa: E712
                )
                .order_by(col(ImageTable.starred).desc(), col(ImageTable.created_at).desc())
                .limit(1)
            )
            row = session.exec(stmt).first()
            if row is None:
                return None
            return deserialize_image_record(_image_to_dict(row))

    def images_get_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        image_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> ImageNamesResult:
        with self._db.get_readonly_session() as session:
            # Base query
            stmt = select(ImageTable.image_name).outerjoin(
                BoardImageTable, col(BoardImageTable.image_name) == col(ImageTable.image_name)
            )

            # Count stmt for the starred count (same filters)
            count_stmt = (
                select(func.count())
                .select_from(ImageTable)
                .outerjoin(BoardImageTable, col(BoardImageTable.image_name) == col(ImageTable.image_name))
            )

            stmt, count_stmt = _apply_image_filters(
                stmt,
                count_stmt,
                image_origin,
                categories,
                is_intermediate,
                board_id,
                search_term,
                user_id,
                is_admin,
            )

            # Starred count
            starred_count = 0
            if starred_first:
                starred_stmt = count_stmt.where(col(ImageTable.starred) == True)  # noqa: E712
                starred_count = session.exec(starred_stmt).one()

            stmt = _image_order_by(stmt, starred_first, order_dir)
            results = session.exec(stmt).all()

        return ImageNamesResult(
            image_names=list(results),
            starred_count=starred_count,
            total_count=len(results),
        )

    def images_get_dates(
        self,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> list[VirtualSubBoardDTO]:
        # func.date() yields 'YYYY-MM-DD' on both SQLite (TEXT created_at) and MySQL (DATETIME),
        # so this groups identically across backends.
        date_col = func.date(col(ImageTable.created_at))

        with self._db.get_readonly_session() as session:
            # Correlated subquery: most recent non-intermediate image for each grouped date.
            i2 = aliased(ImageTable)
            cover_subq = (
                select(i2.image_name)
                .where(
                    func.date(col(i2.created_at)) == date_col,
                    col(i2.is_intermediate) == False,  # noqa: E712
                )
                .order_by(col(i2.created_at).desc())
                .limit(1)
                .correlate(ImageTable)
                .scalar_subquery()
            )

            stmt = (
                select(
                    date_col.label("date"),
                    func.sum(case((col(ImageTable.image_category) == "general", 1), else_=0)).label("image_count"),
                    func.sum(case((col(ImageTable.image_category) != "general", 1), else_=0)).label("asset_count"),
                    cover_subq.label("cover_image_name"),
                )
                .where(col(ImageTable.is_intermediate) == False)  # noqa: E712
                .group_by(date_col)
                .order_by(date_col.desc())
            )

            # User isolation for non-admin users
            if user_id is not None and not is_admin:
                stmt = stmt.where(col(ImageTable.user_id) == user_id)

            rows = session.exec(stmt).all()

        return [
            VirtualSubBoardDTO(
                virtual_board_id=f"by_date:{row.date}",
                board_name=str(row.date),
                date=str(row.date),
                image_count=int(row.image_count or 0),
                asset_count=int(row.asset_count or 0),
                cover_image_name=row.cover_image_name,
            )
            for row in rows
        ]

    def images_get_names_by_date(
        self,
        date: str,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        categories: Optional[list[ImageCategory]] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> ImageNamesResult:
        with self._db.get_readonly_session() as session:
            # Shared filter conditions for the count and the data query.
            conditions = [
                func.date(col(ImageTable.created_at)) == date,
                col(ImageTable.is_intermediate) == False,  # noqa: E712
            ]

            if categories is not None:
                category_strings = [c.value for c in set(categories)]
                conditions.append(col(ImageTable.image_category).in_(category_strings))

            # User isolation for non-admin users
            if user_id is not None and not is_admin:
                conditions.append(col(ImageTable.user_id) == user_id)

            if search_term:
                term = f"%{search_term.lower()}%"
                conditions.append(col(ImageTable.metadata_).like(term) | col(ImageTable.created_at).like(term))

            # Starred count
            starred_count = 0
            if starred_first:
                starred_stmt = select(func.count()).select_from(ImageTable).where(*conditions)
                starred_stmt = starred_stmt.where(col(ImageTable.starred) == True)  # noqa: E712
                starred_count = session.exec(starred_stmt).one()

            stmt = select(ImageTable.image_name).where(*conditions)
            stmt = _image_order_by(stmt, starred_first, order_dir)

            results = session.exec(stmt).all()

        return ImageNamesResult(
            image_names=list(results),
            starred_count=starred_count,
            total_count=len(results),
        )

    # endregion

    # region: Session queue

    def queue_set_in_progress_to_canceled(self) -> None:
        """Sets all in_progress queue items to canceled. Run on app startup."""
        with self._db.get_session() as session:
            session.execute(
                update(SessionQueueTable)
                .where(SessionQueueTable.status == "in_progress")
                .values(status="canceled", **_queue_status_change_values("canceled"))
            )

    def queue_prune_terminal_to_limit(self, queue_id: str, keep: int) -> int:
        """Prune terminal items (completed/failed/canceled) to keep at most N most-recent items."""
        terminal_filter = and_(
            SessionQueueTable.queue_id == queue_id,
            SessionQueueTable.status.in_(_TERMINAL_STATUSES),
        )
        # Subquery: ids of the items we want to keep (most recent N)
        keep_ids_stmt = (
            sa_select(SessionQueueTable.item_id)
            .where(terminal_filter)
            .order_by(
                func.coalesce(
                    SessionQueueTable.completed_at,
                    SessionQueueTable.updated_at,
                    SessionQueueTable.created_at,
                ).desc(),
                SessionQueueTable.item_id.desc(),
            )
            .limit(keep)
        )
        with self._db.get_session() as session:
            count_stmt = (
                sa_select(func.count())
                .select_from(SessionQueueTable)
                .where(terminal_filter)
                .where(~SessionQueueTable.item_id.in_(keep_ids_stmt))
            )
            count = session.execute(count_stmt).scalar_one()
            session.execute(
                delete(SessionQueueTable).where(terminal_filter).where(~SessionQueueTable.item_id.in_(keep_ids_stmt))
            )
        return int(count)

    def queue_pending_count(self, queue_id: str) -> int:
        """Gets the current number of pending queue items."""
        with self._db.get_readonly_session() as session:
            count = session.execute(
                sa_select(func.count())
                .select_from(SessionQueueTable)
                .where(
                    SessionQueueTable.queue_id == queue_id,
                    SessionQueueTable.status == "pending",
                )
            ).scalar_one()
        return int(count)

    def queue_highest_pending_priority(self, queue_id: str) -> int:
        """Gets the highest priority value among pending items."""
        with self._db.get_readonly_session() as session:
            priority = session.execute(
                sa_select(func.max(SessionQueueTable.priority)).where(
                    SessionQueueTable.queue_id == queue_id,
                    SessionQueueTable.status == "pending",
                )
            ).scalar()
        return int(priority) if priority is not None else 0

    def queue_count(self, queue_id: str) -> int:
        """Total number of items in the queue (any status)."""
        with self._db.get_readonly_session() as session:
            count = session.execute(
                sa_select(func.count()).select_from(SessionQueueTable).where(SessionQueueTable.queue_id == queue_id)
            ).scalar_one()
        return int(count)

    def queue_insert_values(self, values: list[ValueToInsertTuple]) -> None:
        """Bulk-insert prepared queue item value tuples (used by retry)."""
        if not values:
            return
        with self._db.get_session() as session:
            session.execute(
                insert(SessionQueueTable),
                [_queue_value_tuple_to_dict(v) for v in values],
            )

    def queue_enqueue_values(self, values: list[ValueToInsertTuple], batch_id: str) -> list[int]:
        """Bulk-insert queue items and return the item ids of the batch, in one transaction."""
        with self._db.get_session() as session:
            if values:
                session.execute(
                    insert(SessionQueueTable),
                    [_queue_value_tuple_to_dict(v) for v in values],
                )
            item_ids_rows = session.execute(
                sa_select(SessionQueueTable.item_id)
                .where(SessionQueueTable.batch_id == batch_id)
                .order_by(SessionQueueTable.item_id.desc())
            ).all()
        return [row[0] for row in item_ids_rows]

    def queue_get_next_pending_any_queue(self) -> Optional[SessionQueueItem]:
        """The next pending item across all queues, by priority then item id (dequeue order)."""
        with self._db.get_readonly_session() as session:
            row = session.execute(
                _queue_select_item_with_user()
                .where(SessionQueueTable.status == "pending")
                .order_by(SessionQueueTable.priority.desc(), SessionQueueTable.item_id.asc())
                .limit(1)
            ).first()
        if row is None:
            return None
        return SessionQueueItem.queue_item_from_dict(_queue_row_to_dict(row))

    def queue_get_next_pending(self, queue_id: str) -> Optional[SessionQueueItem]:
        """The next pending item of a queue, by priority then created_at."""
        with self._db.get_readonly_session() as session:
            row = session.execute(
                _queue_select_item_with_user()
                .where(
                    SessionQueueTable.queue_id == queue_id,
                    SessionQueueTable.status == "pending",
                )
                .order_by(SessionQueueTable.priority.desc(), SessionQueueTable.created_at.asc())
                .limit(1)
            ).first()
        if row is None:
            return None
        return SessionQueueItem.queue_item_from_dict(_queue_row_to_dict(row))

    def queue_get_in_progress(self, queue_id: str) -> Optional[SessionQueueItem]:
        """The currently in-progress item of a queue, if any."""
        with self._db.get_readonly_session() as session:
            row = session.execute(
                _queue_select_item_with_user()
                .where(
                    SessionQueueTable.queue_id == queue_id,
                    SessionQueueTable.status == "in_progress",
                )
                .limit(1)
            ).first()
        if row is None:
            return None
        return SessionQueueItem.queue_item_from_dict(_queue_row_to_dict(row))

    def queue_get_item(self, item_id: int) -> SessionQueueItem:
        with self._db.get_readonly_session() as session:
            row = session.execute(_queue_select_item_with_user().where(SessionQueueTable.item_id == item_id)).first()
        if row is None:
            raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}")
        return SessionQueueItem.queue_item_from_dict(_queue_row_to_dict(row))

    def queue_set_status_returning_prior(
        self,
        item_id: int,
        status: QUEUE_ITEM_STATUS,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
    ) -> str:
        """Set an item's status (unless it is already terminal) and return the prior status.

        Items in a terminal status are left untouched — the caller decides, based on the
        returned prior status, whether to emit a status-changed event.
        """
        with self._db.get_session() as session:
            current_status = session.execute(
                sa_select(SessionQueueTable.status).where(SessionQueueTable.item_id == item_id)
            ).scalar()
            if current_status is None:
                raise SessionQueueItemNotFoundError(f"No queue item with id {item_id}")

            # Only update if not already finished (completed, failed or canceled)
            if current_status not in _TERMINAL_STATUSES:
                session.execute(
                    update(SessionQueueTable)
                    .where(SessionQueueTable.item_id == item_id)
                    .values(
                        status=status,
                        error_type=error_type,
                        error_message=error_message,
                        error_traceback=error_traceback,
                        **_queue_status_change_values(status),
                    )
                )
        return current_status

    def queue_delete_item(self, item_id: int) -> None:
        with self._db.get_session() as session:
            session.execute(delete(SessionQueueTable).where(SessionQueueTable.item_id == item_id))

    def queue_update_session_json(self, item_id: int, session_json: str) -> None:
        with self._db.get_session() as db_session:
            db_session.execute(
                update(SessionQueueTable).where(SessionQueueTable.item_id == item_id).values(session=session_json)
            )

    def _queue_count_and_delete(self, where: list) -> int:
        """Count the matching rows, delete them, return the count — one transaction."""
        with self._db.get_session() as session:
            count = session.execute(sa_select(func.count()).select_from(SessionQueueTable).where(*where)).scalar_one()
            session.execute(delete(SessionQueueTable).where(*where))
        return int(count)

    def _queue_count_and_cancel(self, where: list) -> int:
        """Count the matching rows, set them to canceled, return the count — one transaction."""
        with self._db.get_session() as session:
            count = session.execute(sa_select(func.count()).select_from(SessionQueueTable).where(*where)).scalar_one()
            session.execute(
                update(SessionQueueTable)
                .where(*where)
                .values(status="canceled", **_queue_status_change_values("canceled"))
            )
        return int(count)

    def queue_clear(self, queue_id: str, user_id: Optional[str] = None) -> int:
        """Delete all items of a queue (optionally only one user's). Returns the count."""
        where: list = [SessionQueueTable.queue_id == queue_id]
        if user_id is not None:
            where.append(SessionQueueTable.user_id == user_id)
        return self._queue_count_and_delete(where)

    def queue_prune_terminal(self, queue_id: str, user_id: Optional[str] = None) -> int:
        """Delete all terminal items of a queue (optionally only one user's). Returns the count."""
        where: list = [
            SessionQueueTable.queue_id == queue_id,
            SessionQueueTable.status.in_(_TERMINAL_STATUSES),
        ]
        if user_id is not None:
            where.append(SessionQueueTable.user_id == user_id)
        return self._queue_count_and_delete(where)

    def queue_delete_by_destination(self, queue_id: str, destination: str, user_id: Optional[str] = None) -> int:
        where: list = [
            SessionQueueTable.queue_id == queue_id,
            SessionQueueTable.destination == destination,
        ]
        if user_id is not None:
            where.append(SessionQueueTable.user_id == user_id)
        return self._queue_count_and_delete(where)

    def queue_delete_pending(self, queue_id: str, user_id: Optional[str] = None) -> int:
        where: list = [
            SessionQueueTable.queue_id == queue_id,
            SessionQueueTable.status == "pending",
        ]
        if user_id is not None:
            where.append(SessionQueueTable.user_id == user_id)
        return self._queue_count_and_delete(where)

    def queue_cancel_by_batch_ids(self, queue_id: str, batch_ids: list[str], user_id: Optional[str] = None) -> int:
        where = _queue_cancelable_filter(queue_id, user_id, [SessionQueueTable.batch_id.in_(batch_ids)])
        return self._queue_count_and_cancel(where)

    def queue_cancel_by_destination(self, queue_id: str, destination: str, user_id: Optional[str] = None) -> int:
        where = _queue_cancelable_filter(queue_id, user_id, [SessionQueueTable.destination == destination])
        return self._queue_count_and_cancel(where)

    def queue_cancel_by_queue_id(self, queue_id: str) -> int:
        where = _queue_cancelable_filter(queue_id, None, [])
        return self._queue_count_and_cancel(where)

    def queue_cancel_pending(self, queue_id: str, user_id: Optional[str] = None) -> int:
        where: list = [
            SessionQueueTable.queue_id == queue_id,
            SessionQueueTable.status == "pending",
        ]
        if user_id is not None:
            where.append(SessionQueueTable.user_id == user_id)
        return self._queue_count_and_cancel(where)

    def queue_list_items(
        self,
        queue_id: str,
        limit: int,
        priority: int,
        cursor: Optional[int] = None,
        status: Optional[QUEUE_ITEM_STATUS] = None,
        destination: Optional[str] = None,
    ) -> CursorPaginatedResults[SessionQueueItem]:
        # NOTE: this preserves the (somewhat surprising) cursor semantics of the original
        # raw-SQL implementation, including the unparenthesised `AND ... OR ...` precedence.
        item_id = cursor

        stmt = sa_select(*_QUEUE_COLUMNS, SessionQueueTable.workflow).where(SessionQueueTable.queue_id == queue_id)
        if status is not None:
            stmt = stmt.where(SessionQueueTable.status == status)
        if destination is not None:
            stmt = stmt.where(SessionQueueTable.destination == destination)
        if item_id is not None:
            stmt = stmt.where(
                or_(
                    SessionQueueTable.priority < priority,
                    and_(
                        SessionQueueTable.priority == priority,
                        SessionQueueTable.item_id > item_id,
                    ),
                )
            )
        stmt = stmt.order_by(SessionQueueTable.priority.desc(), SessionQueueTable.item_id.asc()).limit(limit + 1)

        with self._db.get_readonly_session() as session:
            rows = session.execute(stmt).all()

        items = [SessionQueueItem.queue_item_from_dict(_queue_row_to_dict(r)) for r in rows]
        has_more = False
        if len(items) > limit:
            items.pop()
            has_more = True
        return CursorPaginatedResults(items=items, limit=limit, has_more=has_more)

    def queue_list_all_items(
        self,
        queue_id: str,
        destination: Optional[str] = None,
    ) -> list[SessionQueueItem]:
        stmt = _queue_select_item_with_user().where(SessionQueueTable.queue_id == queue_id)
        if destination is not None:
            stmt = stmt.where(SessionQueueTable.destination == destination)
        stmt = stmt.order_by(SessionQueueTable.priority.desc(), SessionQueueTable.item_id.asc())
        with self._db.get_readonly_session() as session:
            rows = session.execute(stmt).all()
        return [SessionQueueItem.queue_item_from_dict(_queue_row_to_dict(r)) for r in rows]

    def queue_get_item_ids(
        self,
        queue_id: str,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        user_id: Optional[str] = None,
    ) -> list[int]:
        stmt = sa_select(SessionQueueTable.item_id).where(SessionQueueTable.queue_id == queue_id)
        if user_id is not None:
            stmt = stmt.where(SessionQueueTable.user_id == user_id)
        if order_dir == SQLiteDirection.Descending:
            stmt = stmt.order_by(SessionQueueTable.created_at.desc())
        else:
            stmt = stmt.order_by(SessionQueueTable.created_at.asc())

        with self._db.get_readonly_session() as session:
            rows = session.execute(stmt).all()
        return [row[0] for row in rows]

    def queue_status_counts(self, queue_id: str, user_id: Optional[str] = None) -> dict[str, int]:
        """Item counts per status for a queue (optionally one user's items only)."""
        stmt = (
            sa_select(SessionQueueTable.status, func.count())
            .where(SessionQueueTable.queue_id == queue_id)
            .group_by(SessionQueueTable.status)
        )
        if user_id is not None:
            stmt = stmt.where(SessionQueueTable.user_id == user_id)

        with self._db.get_readonly_session() as session:
            rows = session.execute(stmt).all()
        return {row[0]: int(row[1] or 0) for row in rows}

    def queue_get_batch_status(self, queue_id: str, batch_id: str, user_id: Optional[str] = None) -> BatchStatus:
        stmt = (
            sa_select(
                SessionQueueTable.status,
                func.count(),
                SessionQueueTable.origin,
                SessionQueueTable.destination,
            )
            .where(
                SessionQueueTable.queue_id == queue_id,
                SessionQueueTable.batch_id == batch_id,
            )
            .group_by(SessionQueueTable.status)
        )
        if user_id is not None:
            stmt = stmt.where(SessionQueueTable.user_id == user_id)

        with self._db.get_readonly_session() as session:
            rows = session.execute(stmt).all()

        total = sum(int(row[1] or 0) for row in rows)
        counts: dict[str, int] = {row[0]: int(row[1]) for row in rows}
        origin = rows[0][2] if rows else None
        destination = rows[0][3] if rows else None

        return BatchStatus(
            batch_id=batch_id,
            origin=origin,
            destination=destination,
            queue_id=queue_id,
            pending=counts.get("pending", 0),
            in_progress=counts.get("in_progress", 0),
            completed=counts.get("completed", 0),
            failed=counts.get("failed", 0),
            canceled=counts.get("canceled", 0),
            total=total,
        )

    def queue_get_counts_by_destination(
        self, queue_id: str, destination: str, user_id: Optional[str] = None
    ) -> SessionQueueCountsByDestination:
        stmt = (
            sa_select(SessionQueueTable.status, func.count())
            .where(
                SessionQueueTable.queue_id == queue_id,
                SessionQueueTable.destination == destination,
            )
            .group_by(SessionQueueTable.status)
        )
        if user_id is not None:
            stmt = stmt.where(SessionQueueTable.user_id == user_id)

        with self._db.get_readonly_session() as session:
            rows = session.execute(stmt).all()

        total = sum(int(row[1] or 0) for row in rows)
        counts: dict[str, int] = {row[0]: int(row[1]) for row in rows}

        return SessionQueueCountsByDestination(
            queue_id=queue_id,
            destination=destination,
            pending=counts.get("pending", 0),
            in_progress=counts.get("in_progress", 0),
            completed=counts.get("completed", 0),
            failed=counts.get("failed", 0),
            canceled=counts.get("canceled", 0),
            total=total,
        )

    # endregion
