"""Accounts."""

import itertools
from collections.abc import Collection, Sequence
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import Connection, Row, bindparam, delete, func, insert, select, true, update

from invokeai.app.services.shared.database.dialect import fixed_limit
from invokeai.app.services.shared.database.queries.base import IN_CHUNK, QueryModule, mapped, read, write
from invokeai.app.services.shared.database.schema.users import users
from invokeai.app.services.users.users_common import UserDTO

# Every query that yields a `UserDTO` selects these columns and maps them with `_user`. A projection copied
# per query drifts: `get_many` once missed `token_epoch` and reported every account it returned as never
# having rotated its password.
_USER_COLUMNS = (
    users.c.user_id,
    users.c.email,
    users.c.display_name,
    users.c.is_admin,
    users.c.is_active,
    users.c.created_at,
    users.c.updated_at,
    users.c.last_login_at,
    users.c.token_epoch,
)
_IS_ACTIVE_ADMIN = (users.c.is_admin == true(), users.c.is_active == true())

_GET = select(*_USER_COLUMNS).where(users.c.user_id == bindparam("user_id"))
_GET_MANY = select(*_USER_COLUMNS).where(users.c.user_id.in_(bindparam("user_ids", expanding=True)))
_GET_BY_EMAIL = select(*_USER_COLUMNS).where(users.c.email == bindparam("email"))
_GET_CREDENTIALS = select(*_USER_COLUMNS, users.c.password_hash).where(users.c.email == bindparam("email"))
_PAGE = (
    select(*_USER_COLUMNS)
    .order_by(users.c.created_at.desc(), users.c.user_id)
    .limit(bindparam("limit"))
    .offset(bindparam("offset"))
)
_COUNT_ACTIVE_ADMINS = select(func.count()).select_from(users).where(*_IS_ACTIVE_ADMIN)
_FIRST_ACTIVE_ADMIN_EMAIL = (
    select(users.c.email).where(*_IS_ACTIVE_ADMIN).order_by(users.c.created_at, users.c.user_id).limit(fixed_limit(1))
)
_INSERT = insert(users)
# An UPDATE reserves bind names that equal column names for its SET clause.
_RECORD_LOGIN = update(users).where(users.c.user_id == bindparam("target_user_id"))
_DELETE = delete(users).where(users.c.user_id == bindparam("user_id"))


def _user(row: Sequence[Any]) -> UserDTO:
    # Unpacked rather than read by name: a lookup by name on a row costs ten times one by position.
    user_id, email, display_name, is_admin, is_active, created_at, updated_at, last_login_at, token_epoch = row
    return UserDTO(
        user_id=user_id,
        email=email,
        display_name=display_name,
        is_admin=is_admin,
        is_active=is_active,
        created_at=datetime.fromisoformat(created_at),
        updated_at=datetime.fromisoformat(updated_at),
        last_login_at=datetime.fromisoformat(last_login_at) if last_login_at else None,
        token_epoch=token_epoch,
    )


def _user_or_none(row: Optional[Sequence[Any]]) -> Optional[UserDTO]:
    return _user(row) if row is not None else None


def _users(rows: Sequence[Sequence[Any]]) -> list[UserDTO]:
    return [_user(row) for row in rows]


def _users_by_id(rows: Sequence[Sequence[Any]]) -> dict[str, UserDTO]:
    return {user.user_id: user for user in map(_user, rows)}


def _credentials(row: Optional[Sequence[Any]]) -> Optional[tuple[UserDTO, str]]:
    return (_user(row[:-1]), row[-1]) if row is not None else None


class UserQueries(QueryModule):
    @mapped(_user_or_none)
    @read
    def get(self, conn: Connection, user_id: str) -> Optional[Row[Any]]:
        return conn.execute(_GET, {"user_id": user_id}).first()

    @mapped(_users_by_id)
    @read
    def get_many(self, conn: Connection, user_ids: Collection[str]) -> list[Row[Any]]:
        """The accounts with these ids, keyed by id; unknown ids are absent."""
        rows: list[Row[Any]] = []
        for chunk in itertools.batched(dict.fromkeys(user_ids), IN_CHUNK):
            rows.extend(conn.execute(_GET_MANY, {"user_ids": list(chunk)}).all())
        return rows

    @mapped(_user_or_none)
    @read
    def get_by_email(self, conn: Connection, email: str) -> Optional[Row[Any]]:
        return conn.execute(_GET_BY_EMAIL, {"email": email}).first()

    @mapped(_credentials)
    @read
    def get_credentials(self, conn: Connection, email: str) -> Optional[Row[Any]]:
        """The account with this email and its password hash."""
        return conn.execute(_GET_CREDENTIALS, {"email": email}).first()

    @mapped(_users)
    @read
    def page(self, conn: Connection, limit: int, offset: int) -> Sequence[Row[Any]]:
        """Accounts, newest first."""
        return conn.execute(_PAGE, {"limit": limit, "offset": offset}).all()

    @read
    def count_active_admins(self, conn: Connection) -> int:
        return conn.execute(_COUNT_ACTIVE_ADMINS).scalar_one()

    @read
    def first_active_admin_email(self, conn: Connection) -> Optional[str]:
        return conn.execute(_FIRST_ACTIVE_ADMIN_EMAIL).scalar_one_or_none()

    @write
    def insert(
        self,
        conn: Connection,
        *,
        user_id: str,
        email: str,
        display_name: Optional[str],
        password_hash: str,
        is_admin: bool,
    ) -> None:
        conn.execute(
            _INSERT,
            {
                "user_id": user_id,
                "email": email,
                "display_name": display_name,
                "password_hash": password_hash,
                "is_admin": is_admin,
            },
        )

    @write
    def update(
        self,
        conn: Connection,
        user_id: str,
        *,
        display_name: Optional[str] = None,
        password_hash: Optional[str] = None,
        is_admin: Optional[bool] = None,
        is_active: Optional[bool] = None,
    ) -> None:
        """Changes the given fields; None leaves a field as it is. A new password hash also raises the account's
        token epoch, which revokes every token issued before it."""
        changes: dict[str, Any] = {}
        if display_name is not None:
            changes["display_name"] = display_name
        if password_hash is not None:
            changes["password_hash"] = password_hash
            # In SQL, so concurrent changes cannot read, increment and write over each other.
            changes["token_epoch"] = users.c.token_epoch + 1
        if is_admin is not None:
            changes["is_admin"] = is_admin
        if is_active is not None:
            changes["is_active"] = is_active
        if changes:
            conn.execute(update(users).where(users.c.user_id == user_id).values(changes))

    @write
    def record_login(self, conn: Connection, user_id: str, at: datetime) -> None:
        # ISO 8601 with its UTC offset, as last_login_at has always been stored: the API reports it as an aware
        # time.
        conn.execute(_RECORD_LOGIN, {"target_user_id": user_id, "last_login_at": at.isoformat()})

    @write
    def delete(self, conn: Connection, user_id: str) -> None:
        conn.execute(_DELETE, {"user_id": user_id})
