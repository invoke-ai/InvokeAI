"""Return the ``system`` account to non-administrator, in case it was ever promoted.

The system row (created by migration_27) owns everything carried over from before
multiuser support. It is deliberately not an administrator and has an empty password
hash, so it can never authenticate — but until the service-level guard landed, an
administrator could ``PATCH /auth/users/system {"is_admin": true}``.

That is worth undoing rather than merely preventing, because the last-administrator
guard counts rows where ``is_admin AND is_active``, and this row satisfies both while
being unable to log in. A promoted system row therefore inflates the count by one
permanently: it makes the guard willing to demote the last *real* administrator, and it
keeps ``has_admin()`` true afterwards, so ``/auth/setup`` stays closed and no
authenticated path back exists.

Demoting it restores the count's meaning — "administrators who can actually log in" —
and, on an instance already in that state, lets ``/auth/setup`` reopen so the operator
can recover. Nobody is logged out: no token can carry ``user_id="system"`` in the first
place, since authenticating as it is impossible.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class DemoteSystemUserCallback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("UPDATE users SET is_admin = FALSE WHERE user_id = 'system' AND is_admin = TRUE;")


def build_migration() -> Migration:
    """Clear ``is_admin`` on the ``system`` row.

    Depends on migration_27, which creates the users table and seeds that row.
    """
    return Migration(
        id="2026_08_08_demote_system_user",
        depends_on="migration_27",
        callback=DemoteSystemUserCallback(),
    )
