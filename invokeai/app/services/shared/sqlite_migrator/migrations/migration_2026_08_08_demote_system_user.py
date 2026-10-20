"""Return the ``system`` account to its intended shape: not an administrator, and unable
to authenticate.

The system row (created by migration_27) owns everything carried over from before
multiuser support. It is seeded as a non-administrator with an empty password hash, so
it can never authenticate — but until the service-level guard landed, an administrator
could ``PATCH /auth/users/system`` and both promote it and give it a password.

Promotion is worth undoing rather than merely preventing, because the last-administrator
guard counts rows where ``is_admin AND is_active``, and this row satisfies both while
being unable to log in. A promoted system row therefore inflates the count by one
permanently: it makes the guard willing to demote the last *real* administrator, and it
keeps ``has_admin()`` true afterwards, so ``/auth/setup`` stays closed and no
authenticated path back exists.

A password is worse. The system row's email is fixed and public (``system@system.invokeai``),
so a hash left behind by that hole is a standing login for the account that owns every
pre-multiuser board, image, workflow, and queue item — and its tokens are indistinguishable
from any other user's. Clearing the hash restores the seeded state: ``verify_password``
against ``""`` fails for every input.

Neither change logs anybody out, because no token can carry ``user_id="system"`` unless the
instance was in exactly this damaged state, and such a token is meant to stop working.

This migration cannot be the only defense, in two directions. A row damaged *after* it runs
— by direct SQL, or by a database that applied an earlier revision of this same migration id
— would slip through, since migrations run once; :meth:`UserService.authenticate` refuses
the system account outright for that reason. And a token *already issued* is not reached by
clearing the credential at all: the row is deliberately left active and its epoch untouched,
so nothing else would reject it and sliding-window refresh would renew it forever;
``resolve_authorized_user`` refuses the id for that reason. Between them the invariant holds
regardless of what the row contains or what is already in the wild; this migration's job is
to remove the standing credential itself.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class DemoteSystemUserCallback:
    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute(
            "UPDATE users SET is_admin = FALSE, password_hash = '' "
            "WHERE user_id = 'system' AND (is_admin = TRUE OR password_hash != '');"
        )


def build_migration() -> Migration:
    """Clear ``is_admin`` and any password hash on the ``system`` row.

    Depends on migration_27, which creates the users table and seeds that row.
    """
    return Migration(
        id="2026_08_08_demote_system_user",
        depends_on="migration_27",
        callback=DemoteSystemUserCallback(),
    )
