"""SQLModel table definitions for the InvokeAI database.

These models mirror the schema created by the raw SQL migrations.
The migrations remain the source of truth for schema changes —
these models are used only for querying via SQLModel/SQLAlchemy.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Computed, Integer, Text
from sqlalchemy.dialects import mysql
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import FunctionElement
from sqlalchemy.types import TypeEngine
from sqlmodel import Field, SQLModel


def _blob() -> TypeEngine[str]:
    """Column type for JSON/text blobs.

    Plain ``TEXT`` on SQLite/Postgres (effectively unbounded). On MySQL/MariaDB a bare
    ``str`` would map to ``VARCHAR(255)`` and silently truncate large payloads (model
    configs, session graphs, workflows), so we use ``LONGTEXT`` (up to 4 GiB) there.
    Returns a fresh instance per call — a ``Column`` may not be shared across tables.
    """
    return Text().with_variant(mysql.LONGTEXT(), "mysql")


class _JsonField(FunctionElement):
    """Dialect-aware JSON scalar extraction for GENERATED columns.

    Mirrors the SQLite ``GENERATED ALWAYS AS (json_extract(...))`` columns the migrations
    create. SQLite's ``json_extract`` auto-unquotes scalars; MySQL's does not, so on MySQL we
    wrap it in ``json_unquote`` to get the same plain value (``sdxl`` not ``"sdxl"``). Arrays
    and numbers (``unquote=False``) use the raw ``json_extract`` on both. Only emitted by
    ``create_all()`` on non-SQLite backends — the SQLite schema is owned by the migrations.
    """

    inherit_cache = True

    def __init__(self, source: str, path: str, unquote: bool) -> None:
        self.source = source
        self.path = path
        self.unquote = unquote
        super().__init__()


@compiles(_JsonField)
def _compile_json_field(element: _JsonField, compiler, **kw) -> str:  # type: ignore[no-untyped-def]
    # Default (SQLite/Postgres): json_extract already yields the bare scalar on SQLite.
    return f"json_extract({element.source}, '{element.path}')"


@compiles(_JsonField, "mysql")
def _compile_json_field_mysql(element: _JsonField, compiler, **kw) -> str:  # type: ignore[no-untyped-def]
    inner = f"json_extract({element.source}, '{element.path}')"
    return f"json_unquote({inner})" if element.unquote else inner


def _generated(source: str, path: str, *, unquote: bool = True) -> Computed:
    """A DB-generated column mirroring the SQLite ``GENERATED ALWAYS`` columns, dialect-aware.

    Only emitted by ``create_all()`` (MySQL/MariaDB/Postgres); on SQLite the migrations own
    the table. ``persisted`` (STORED) so Postgres — which has no VIRTUAL generated columns —
    can also compile it. Returns a fresh instance per call (a ``Computed`` may not be shared).
    """
    return Computed(_JsonField(source, path, unquote), persisted=True)


# --- boards ---


class BoardTable(SQLModel, table=True):
    """Mirrors the `boards` table."""

    __tablename__ = "boards"

    board_id: str = Field(primary_key=True)
    board_name: str
    cover_image_name: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow})
    deleted_at: Optional[datetime] = Field(default=None)
    archived: bool = Field(default=False)
    user_id: str = Field(default="system")
    is_public: bool = Field(default=False)
    board_visibility: str = Field(default="private")


class BoardImageTable(SQLModel, table=True):
    """Mirrors the `board_images` junction table."""

    __tablename__ = "board_images"

    image_name: str = Field(primary_key=True)
    board_id: str = Field(foreign_key="boards.board_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow})
    deleted_at: Optional[datetime] = Field(default=None)


class SharedBoardTable(SQLModel, table=True):
    """Mirrors the `shared_boards` table."""

    __tablename__ = "shared_boards"

    board_id: str = Field(primary_key=True, foreign_key="boards.board_id")
    user_id: str = Field(primary_key=True, foreign_key="users.user_id")
    can_edit: bool = Field(default=False)
    shared_at: datetime = Field(default_factory=datetime.utcnow)


# --- images ---


class ImageTable(SQLModel, table=True):
    """Mirrors the `images` table."""

    __tablename__ = "images"

    image_name: str = Field(primary_key=True)
    image_origin: str
    image_category: str
    width: int
    height: int
    session_id: Optional[str] = Field(default=None)
    node_id: Optional[str] = Field(default=None)
    metadata_: Optional[str] = Field(default=None, sa_column=Column("metadata", _blob(), nullable=True))
    is_intermediate: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow})
    deleted_at: Optional[datetime] = Field(default=None)
    starred: bool = Field(default=False)
    has_workflow: bool = Field(default=False)
    user_id: str = Field(default="system")
    image_subfolder: str = Field(default="")  # added by migration_31; subfolder strategy on disk


# --- workflows ---


class WorkflowLibraryTable(SQLModel, table=True):
    """Mirrors the `workflow_library` table."""

    __tablename__ = "workflow_library"

    workflow_id: str = Field(primary_key=True)
    workflow: str = Field(sa_column=Column(_blob(), nullable=False))  # JSON blob
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow})
    opened_at: Optional[datetime] = Field(default=None)
    # DB-generated columns extracted from the `workflow` JSON (category lives at $.meta.category).
    # Declared as Computed so create_all() reproduces them on MySQL/Postgres; on SQLite the
    # migrations own them. tags is a JSON array, kept as raw json_extract text for LIKE search.
    category: Optional[str] = Field(default=None, sa_column=Column(Text, _generated("workflow", "$.meta.category")))
    name: Optional[str] = Field(default=None, sa_column=Column(Text, _generated("workflow", "$.name")))
    description: Optional[str] = Field(default=None, sa_column=Column(Text, _generated("workflow", "$.description")))
    tags: Optional[str] = Field(
        default=None, sa_column=Column(Text, _generated("workflow", "$.tags", unquote=False))
    )
    user_id: str = Field(default="system")
    is_public: bool = Field(default=False)


# --- session queue ---


class SessionQueueTable(SQLModel, table=True):
    """Mirrors the `session_queue` table."""

    __tablename__ = "session_queue"

    item_id: Optional[int] = Field(default=None, primary_key=True)  # AUTOINCREMENT
    batch_id: str
    queue_id: str
    session_id: str = Field(unique=True)
    field_values: Optional[str] = Field(default=None, sa_column=Column(_blob(), nullable=True))
    session: str = Field(sa_column=Column(_blob(), nullable=False))  # JSON blob
    status: str = Field(default="pending")
    priority: int = Field(default=0)
    error_traceback: Optional[str] = Field(default=None, sa_column=Column(_blob(), nullable=True))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow})
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    error_type: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None, sa_column=Column(_blob(), nullable=True))
    origin: Optional[str] = Field(default=None)
    destination: Optional[str] = Field(default=None)
    retried_from_item_id: Optional[int] = Field(default=None)
    user_id: str = Field(default="system")
    workflow: Optional[str] = Field(default=None, sa_column=Column(_blob(), nullable=True))  # JSON blob
    status_sequence: int = Field(default=0)  # added by migration_30; incremented on each status change


# --- models ---


class ModelTable(SQLModel, table=True):
    """Mirrors the `models` table.

    Most columns are GENERATED ALWAYS from the `config` JSON blob.
    We define them here for read access but they should not be set directly.
    """

    __tablename__ = "models"

    id: str = Field(primary_key=True)
    config: str = Field(sa_column=Column(_blob(), nullable=False))  # JSON blob — all model metadata is extracted from this
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow})
    # Columns extracted from the `config` JSON blob. On SQLite the migrations create these as
    # GENERATED ALWAYS columns; here they are declared as Computed so create_all() reproduces
    # them as DB-generated columns on MySQL/MariaDB/Postgres too. Being Computed, SQLAlchemy
    # excludes them from INSERT/UPDATE, so add_model() keeps writing only id+config everywhere.
    hash: Optional[str] = Field(default=None, sa_column=Column(Text, _generated("config", "$.hash")))
    base: Optional[str] = Field(default=None, sa_column=Column(Text, _generated("config", "$.base")))
    type: Optional[str] = Field(default=None, sa_column=Column(Text, _generated("config", "$.type")))
    path: Optional[str] = Field(default=None, sa_column=Column(Text, _generated("config", "$.path")))
    format: Optional[str] = Field(default=None, sa_column=Column(Text, _generated("config", "$.format")))
    name: Optional[str] = Field(default=None, sa_column=Column(Text, _generated("config", "$.name")))
    description: Optional[str] = Field(default=None, sa_column=Column(Text, _generated("config", "$.description")))
    source: Optional[str] = Field(default=None, sa_column=Column(Text, _generated("config", "$.source")))
    source_type: Optional[str] = Field(default=None, sa_column=Column(Text, _generated("config", "$.source_type")))
    source_api_response: Optional[str] = Field(
        default=None, sa_column=Column(Text, _generated("config", "$.source_api_response"))
    )
    trigger_phrases: Optional[str] = Field(
        default=None, sa_column=Column(Text, _generated("config", "$.trigger_phrases", unquote=False))
    )
    file_size: Optional[int] = Field(
        default=None, sa_column=Column(Integer, _generated("config", "$.file_size", unquote=False))
    )


class ModelRelationshipTable(SQLModel, table=True):
    """Mirrors the `model_relationships` table."""

    __tablename__ = "model_relationships"

    model_key_1: str = Field(primary_key=True)
    model_key_2: str = Field(primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# --- style presets ---


class StylePresetTable(SQLModel, table=True):
    """Mirrors the `style_presets` table."""

    __tablename__ = "style_presets"

    id: str = Field(primary_key=True)
    name: str
    preset_data: str = Field(sa_column=Column(_blob(), nullable=False))  # JSON blob
    type: str = Field(default="user")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow})
    user_id: str = Field(default="system")
    is_public: bool = Field(default=False)


# --- users & auth ---


class UserTable(SQLModel, table=True):
    """Mirrors the `users` table."""

    __tablename__ = "users"

    user_id: str = Field(primary_key=True)
    email: str = Field(unique=True)
    display_name: Optional[str] = Field(default=None)
    password_hash: str
    is_admin: bool = Field(default=False)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow})
    last_login_at: Optional[datetime] = Field(default=None)


class UserSessionTable(SQLModel, table=True):
    """Mirrors the `user_sessions` table (added by migration_27)."""

    __tablename__ = "user_sessions"

    session_id: str = Field(primary_key=True)
    user_id: str = Field(foreign_key="users.user_id", index=True)
    token_hash: str = Field(index=True)
    expires_at: datetime = Field(index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity_at: datetime = Field(default_factory=datetime.utcnow)


class UserInvitationTable(SQLModel, table=True):
    """Mirrors the `user_invitations` table (added by migration_27)."""

    __tablename__ = "user_invitations"

    invitation_id: str = Field(primary_key=True)
    email: str = Field(index=True)
    invited_by: str = Field(foreign_key="users.user_id")
    invitation_code: str = Field(unique=True, index=True)
    is_admin: bool = Field(default=False)
    expires_at: datetime = Field(index=True)
    used_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# --- app settings ---


class AppSettingTable(SQLModel, table=True):
    """Mirrors the `app_settings` table."""

    __tablename__ = "app_settings"

    key: str = Field(primary_key=True)
    value: str = Field(sa_column=Column(_blob(), nullable=False))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow})


# --- client state ---


class ClientStateTable(SQLModel, table=True):
    """Mirrors the `client_state` table."""

    __tablename__ = "client_state"

    user_id: str = Field(primary_key=True, foreign_key="users.user_id")
    key: str = Field(primary_key=True)
    value: str = Field(sa_column=Column(_blob(), nullable=False))
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow})
