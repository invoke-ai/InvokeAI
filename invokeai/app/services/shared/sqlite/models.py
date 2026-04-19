"""SQLModel table definitions for the InvokeAI database.

These models mirror the schema created by the raw SQL migrations.
The migrations remain the source of truth for schema changes —
these models are used only for querying via SQLModel/SQLAlchemy.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String
from sqlalchemy.schema import FetchedValue
from sqlmodel import Field, SQLModel

# --- boards ---


class BoardTable(SQLModel, table=True):
    """Mirrors the `boards` table."""

    __tablename__ = "boards"

    board_id: str = Field(primary_key=True)
    board_name: str
    cover_image_name: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
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
    updated_at: datetime = Field(default_factory=datetime.utcnow)
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
    metadata_: Optional[str] = Field(default=None, sa_column_kwargs={"name": "metadata"})
    is_intermediate: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deleted_at: Optional[datetime] = Field(default=None)
    starred: bool = Field(default=False)
    has_workflow: bool = Field(default=False)
    user_id: str = Field(default="system")


# --- workflows ---


class WorkflowLibraryTable(SQLModel, table=True):
    """Mirrors the `workflow_library` table."""

    __tablename__ = "workflow_library"

    workflow_id: str = Field(primary_key=True)
    workflow: str  # JSON blob
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    opened_at: Optional[datetime] = Field(default=None)
    # Generated columns — server-side, excluded from INSERT/UPDATE
    category: Optional[str] = Field(default=None, sa_column=Column(String, FetchedValue(), server_default=None))
    name: Optional[str] = Field(default=None, sa_column=Column(String, FetchedValue(), server_default=None))
    description: Optional[str] = Field(default=None, sa_column=Column(String, FetchedValue(), server_default=None))
    tags: Optional[str] = Field(default=None, sa_column=Column(String, FetchedValue(), server_default=None))
    user_id: str = Field(default="system")
    is_public: bool = Field(default=False)


class WorkflowImageTable(SQLModel, table=True):
    """Mirrors the `workflow_images` junction table."""

    __tablename__ = "workflow_images"

    image_name: str = Field(primary_key=True, foreign_key="images.image_name")
    workflow_id: str = Field(foreign_key="workflow_library.workflow_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deleted_at: Optional[datetime] = Field(default=None)


# --- session queue ---


class SessionQueueTable(SQLModel, table=True):
    """Mirrors the `session_queue` table."""

    __tablename__ = "session_queue"

    item_id: Optional[int] = Field(default=None, primary_key=True)  # AUTOINCREMENT
    batch_id: str
    queue_id: str
    session_id: str = Field(unique=True)
    field_values: Optional[str] = Field(default=None)
    session: str  # JSON blob
    status: str = Field(default="pending")
    priority: int = Field(default=0)
    error_traceback: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    error_type: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    origin: Optional[str] = Field(default=None)
    destination: Optional[str] = Field(default=None)
    retried_from_item_id: Optional[int] = Field(default=None)
    user_id: str = Field(default="system")


# --- models ---


class ModelTable(SQLModel, table=True):
    """Mirrors the `models` table.

    Most columns are GENERATED ALWAYS from the `config` JSON blob.
    We define them here for read access but they should not be set directly.
    """

    __tablename__ = "models"

    id: str = Field(primary_key=True)
    config: str  # JSON blob — all model metadata is extracted from this via GENERATED ALWAYS columns
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    # NOTE: The `models` table has many GENERATED ALWAYS columns (hash, base, type, path, format, name, etc.)
    # that are automatically extracted from the `config` JSON blob by SQLite.
    # We intentionally do NOT define them here because SQLAlchemy would try to include them in
    # INSERT/UPDATE statements, which fails on GENERATED columns.
    # To query by these columns, use raw text filters or the `text()` function.
    # The ModelRecordServiceSqlModel extracts all needed data from the `config` JSON blob directly.


class ModelManagerMetadataTable(SQLModel, table=True):
    """Mirrors the `model_manager_metadata` table."""

    __tablename__ = "model_manager_metadata"

    metadata_key: str = Field(primary_key=True)
    metadata_value: str


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
    preset_data: str  # JSON blob
    type: str = Field(default="user")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
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
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login_at: Optional[datetime] = Field(default=None)


# --- app settings ---


class AppSettingTable(SQLModel, table=True):
    """Mirrors the `app_settings` table."""

    __tablename__ = "app_settings"

    key: str = Field(primary_key=True)
    value: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# --- client state ---


class ClientStateTable(SQLModel, table=True):
    """Mirrors the `client_state` table."""

    __tablename__ = "client_state"

    user_id: str = Field(primary_key=True, foreign_key="users.user_id")
    key: str = Field(primary_key=True)
    value: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)
