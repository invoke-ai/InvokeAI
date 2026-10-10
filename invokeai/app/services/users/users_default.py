"""Default implementation of the user service."""

from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import uuid4

from invokeai.app.services.auth.password_utils import hash_password, validate_password_strength, verify_password
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import UniqueViolation
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries.locks import DatabaseLock
from invokeai.app.services.users.users_base import UserServiceBase
from invokeai.app.services.users.users_common import (
    LAST_ADMIN_DETAIL,
    SYSTEM_USER_ID,
    SYSTEM_USER_PROTECTED_DETAIL,
    LastAdministratorError,
    SystemUserProtectedError,
    UserCreateRequest,
    UserDTO,
    UserUpdateRequest,
)


class UserService(UserServiceBase):
    """User service on the application database."""

    def __init__(self, database: Database):
        self._queries = database.queries

    def create(self, user_data: UserCreateRequest, strict_password_checking: bool = True) -> UserDTO:
        """Create a new user."""
        return self._create(user_data, strict_password_checking=strict_password_checking, require_no_admin=False)

    def _create(
        self,
        user_data: UserCreateRequest,
        strict_password_checking: bool,
        require_no_admin: bool,
    ) -> UserDTO:
        """Insert a user, optionally conditional on no administrator existing yet.

        `require_no_admin` is checked in the transaction that inserts, so that the check and the write are one
        atomic step - see `create_admin` for why that matters.
        """
        # Validate password strength
        if strict_password_checking:
            is_valid, error_msg = validate_password_strength(user_data.password)
            if not is_valid:
                raise ValueError(error_msg)
        elif not user_data.password:
            raise ValueError("Password cannot be empty")

        # Check if email already exists
        if self._queries.users.get_by_email(user_data.email) is not None:
            raise ValueError(f"User with email {user_data.email} already exists")

        user_id = str(uuid4())
        # Hash before opening the transaction: hashing is deliberately slow, and on SQLite a transaction holds
        # the database's process-wide lock.
        password_hash = hash_password(user_data.password)

        def insert_user(q: Queries) -> UserDTO | None:
            if user_data.is_admin:
                # A setup that runs meanwhile waits here until this transaction ends, then counts its admin.
                q.locks.acquire(DatabaseLock.ADMIN_ACCOUNTS)
                if require_no_admin and q.users.count_active_admins() > 0:
                    raise ValueError("Admin user already exists")
            try:
                q.users.insert(
                    user_id=user_id,
                    email=user_data.email,
                    display_name=user_data.display_name,
                    password_hash=password_hash,
                    is_admin=user_data.is_admin,
                )
            except UniqueViolation as e:
                raise ValueError(f"Failed to create user: {e}") from e
            return q.users.get(user_id)

        user = self._queries.run(insert_user)
        if user is None:
            raise RuntimeError("Failed to retrieve created user")
        return user

    def get(self, user_id: str) -> UserDTO | None:
        """Get user by ID."""
        return self._queries.users.get(user_id)

    def get_many(self, user_ids: Sequence[str]) -> dict[str, UserDTO]:
        """Get users by ID, keyed by user_id. Unknown ids are absent from the result."""
        if not user_ids:
            return {}
        return self._queries.users.get_many(user_ids)

    def get_by_email(self, email: str) -> UserDTO | None:
        """Get user by email."""
        return self._queries.users.get_by_email(email)

    def update(self, user_id: str, changes: UserUpdateRequest, strict_password_checking: bool = True) -> UserDTO:
        """Update user."""
        # Check if user exists
        user = self.get(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")

        self._assert_system_user_protected(
            user_id, is_admin=changes.is_admin, is_active=changes.is_active, password=changes.password
        )

        # Validate password if provided
        if changes.password is not None:
            if strict_password_checking:
                is_valid, error_msg = validate_password_strength(changes.password)
                if not is_valid:
                    raise ValueError(error_msg)
            elif not changes.password:
                raise ValueError("Password cannot be empty")

        if (changes.display_name, changes.password, changes.is_admin, changes.is_active) == (None, None, None, None):
            return user

        # Rotating the password revokes every token issued under the old one (the query raises the token
        # epoch). A JWT is self-contained, so without that a stolen token survives the password change meant
        # to evict the thief - and sliding-window refresh renews it indefinitely.
        password_hash = hash_password(changes.password) if changes.password is not None else None

        def apply_changes(q: Queries) -> UserDTO | None:
            if changes.is_admin is not None or changes.is_active is not None:
                q.locks.acquire(DatabaseLock.ADMIN_ACCOUNTS)
                self._assert_not_last_admin(q, user_id, is_admin=changes.is_admin, is_active=changes.is_active)
            q.users.update(
                user_id,
                display_name=changes.display_name,
                password_hash=password_hash,
                is_admin=changes.is_admin,
                is_active=changes.is_active,
            )
            return q.users.get(user_id)

        updated_user = self._queries.run(apply_changes)
        if updated_user is None:
            raise RuntimeError("Failed to retrieve updated user")
        return updated_user

    def delete(self, user_id: str) -> None:
        """Delete user."""
        user = self.get(user_id)
        if user is None:
            raise ValueError(f"User {user_id} not found")

        self._assert_system_user_protected(user_id, is_deleting=True)

        def delete_user(q: Queries) -> None:
            q.locks.acquire(DatabaseLock.ADMIN_ACCOUNTS)
            self._assert_not_last_admin(q, user_id, is_deleting=True)
            q.users.delete(user_id)
            # Projects and client state cascade with the account; the media references they held
            # would otherwise protect a departed account's intermediates from every cleanup forever.
            # Workflows and quarantined projects do not cascade, so their references stay with them.
            q.media_references.delete_owned_by(user_id, "project", "client_state")

        self._queries.run(delete_user)

    def authenticate(self, email: str, password: str) -> UserDTO | None:
        """Authenticate user credentials."""
        credentials = self._queries.users.get_credentials(email)
        if credentials is None:
            return None
        user, password_hash = credentials

        # The system account is not a login account. It owns everything migrated from
        # before multiuser support, so a token bearing `user_id="system"` reads and writes
        # all of it — and `_assert_system_user_protected` only stops a password being set
        # from *now on*. An instance that set one through the old hole still carries a
        # usable hash, and the migration that clears it cannot reach a row damaged by
        # direct SQL afterwards. Refusing here makes "system cannot authenticate" hold
        # regardless of what the row contains.
        if user.user_id == SYSTEM_USER_ID:
            return None

        if not verify_password(password, password_hash):
            return None

        last_login_at = datetime.now(timezone.utc)
        self._queries.users.record_login(user.user_id, last_login_at)

        # Report the login just recorded rather than the one the row was read with.
        return user.model_copy(update={"last_login_at": last_login_at})

    def has_admin(self) -> bool:
        """Check if any admin user exists."""
        return self._queries.users.count_active_admins() > 0

    def create_admin(self, user_data: UserCreateRequest, strict_password_checking: bool = True) -> UserDTO:
        """Create the first admin user (for initial setup).

        The "no admin exists yet" condition is enforced inside the INSERT's own transaction rather
        than by a preceding has_admin() call. `POST /auth/setup` is necessarily unauthenticated
        during the first-run window and runs in the threadpool, so two concurrent requests can both
        pass a check made in a separate transaction and both create an administrator. Callers may
        still check has_admin() first for a friendly error; this is the backstop that decides.
        """
        # Force is_admin to True
        admin_data = UserCreateRequest(
            email=user_data.email,
            display_name=user_data.display_name,
            password=user_data.password,
            is_admin=True,
        )
        return self._create(admin_data, strict_password_checking=strict_password_checking, require_no_admin=True)

    def list_users(self, limit: int = 100, offset: int = 0) -> list[UserDTO]:
        """List all users."""
        return self._queries.users.page(limit, offset)

    def get_admin_email(self) -> str | None:
        """Get the email address of the first active admin user."""
        return self._queries.users.first_active_admin_email()

    def count_admins(self) -> int:
        """Count active admin users."""
        return self._queries.users.count_active_admins()

    def _assert_not_last_admin(
        self,
        q: Queries,
        user_id: str,
        *,
        is_deleting: bool = False,
        is_admin: bool | None = None,
        is_active: bool | None = None,
    ) -> None:
        """Reject a change that would drop the number of active administrators to zero.

        Called in the transaction that performs the write, which holds the `ADMIN_ACCOUNTS` lock: every change
        to who is an active administrator takes it first. Reading the count in a separate transaction is what
        made this a TOCTOU: two callers could each observe two admins and each proceed to remove one. On SQLite
        the transaction alone serializes them; on a server the lock does.

        ``is_admin``/``is_active`` are the *requested* values, where ``None`` means "not being
        changed" — deletion is signalled separately by ``is_deleting`` rather than by both
        being ``None``, which would also describe a rename. Only a change that actually
        revokes administrator status is checked, so renaming the last admin stays allowed.
        """
        user = q.users.get(user_id)
        if user is None:
            return

        # An admin who is already inactive is not counted, so removing them changes nothing.
        if not (user.is_admin and user.is_active):
            return

        if not (is_deleting or is_admin is False or is_active is False):
            return

        if q.users.count_active_admins() <= 1:
            raise LastAdministratorError(LAST_ADMIN_DETAIL)

    def _assert_system_user_protected(
        self,
        user_id: str,
        *,
        is_deleting: bool = False,
        is_admin: bool | None = None,
        is_active: bool | None = None,
        password: str | None = None,
    ) -> None:
        """Reject changes to the ``system`` account that no legitimate operation needs.

        The system row owns every board, image, workflow, and queue item carried over from
        before multiuser support. Deleting or deactivating it strands all of that: queued
        items are rejected at dequeue and media reads and saves raise ``PermissionError``.

        Promotion and password-setting are refused for a different reason. The system row
        is active but has an empty password hash, so it can never authenticate — yet
        ``count_admins()`` and ``has_admin()`` count any active admin row. Promoting it
        therefore inflates the administrator count with an administrator nobody can log in
        as, which is enough to satisfy the last-admin guard while the real administrator is
        demoted, leaving the instance with no usable administration and no way back in.
        Keeping the system row permanently non-admin is what makes that count mean
        "administrators who can actually log in". Giving it a password would turn the owner
        of all pre-multiuser content into a login account, which is the same hole from the
        other end.

        Lives in the service rather than only in the routes so the ``invoke-usermod`` /
        ``invoke-userdel`` CLIs, which construct :class:`UserService` directly, are covered
        too — the same reasoning that moved the last-admin invariant down here.
        """
        if user_id != SYSTEM_USER_ID:
            return
        if is_deleting or is_active is False or is_admin is True or password is not None:
            raise SystemUserProtectedError(SYSTEM_USER_PROTECTED_DETAIL)
