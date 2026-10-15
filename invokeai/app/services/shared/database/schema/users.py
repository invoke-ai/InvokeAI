"""Accounts, and the session and invitation tables of an earlier design that nothing uses."""

from sqlalchemy import Boolean, Column, ForeignKey, Index

from invokeai.app.services.shared.database.schema.metadata import (
    USER_ID_LENGTH,
    default,
    inserted_at,
    table,
    updated_at,
)
from invokeai.app.services.shared.database.types import BigInt, Key, LongText, Timestamp

users = table(
    "users",
    Column("user_id", Key(USER_ID_LENGTH), primary_key=True),
    Column("email", Key(), nullable=False, unique=True),
    Column("display_name", LongText()),
    Column("password_hash", LongText(), nullable=False),
    Column("is_admin", Boolean(), nullable=False, server_default=default(False)),
    Column("is_active", Boolean(), nullable=False, server_default=default(True)),
    inserted_at(),
    updated_at(),
    Column("last_login_at", Timestamp()),
    # Raised to revoke every token issued to the account so far.
    Column("token_epoch", BigInt(), nullable=False, server_default=default(0)),
)

# The unique constraint covers it. Only SQLite has it, where a migration created it.
Index("idx_users_email", users.c.email).ddl_if(dialect="sqlite")
Index("idx_users_is_active", users.c.is_active)
Index("idx_users_is_admin", users.c.is_admin)

user_sessions = table(
    "user_sessions",
    Column("session_id", Key(), primary_key=True),
    Column("user_id", Key(USER_ID_LENGTH), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False),
    Column("token_hash", Key(), nullable=False),
    Column("expires_at", Timestamp(), nullable=False),
    inserted_at(),
    inserted_at("last_activity_at"),
)

Index("idx_user_sessions_expires_at", user_sessions.c.expires_at)
Index("idx_user_sessions_token_hash", user_sessions.c.token_hash)
Index("idx_user_sessions_user_id", user_sessions.c.user_id)

user_invitations = table(
    "user_invitations",
    Column("invitation_id", Key(), primary_key=True),
    Column("email", Key(), nullable=False),
    Column("invited_by", Key(USER_ID_LENGTH), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False),
    Column("invitation_code", Key(), nullable=False, unique=True),
    Column("is_admin", Boolean(), nullable=False, server_default=default(False)),
    Column("expires_at", Timestamp(), nullable=False),
    Column("used_at", Timestamp()),
    inserted_at(),
)

Index("idx_user_invitations_email", user_invitations.c.email)
Index("idx_user_invitations_expires_at", user_invitations.c.expires_at)
# The unique constraint covers it. Only SQLite has it, where a migration created it.
Index("idx_user_invitations_invitation_code", user_invitations.c.invitation_code).ddl_if(dialect="sqlite")
