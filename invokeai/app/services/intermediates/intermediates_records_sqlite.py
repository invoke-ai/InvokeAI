"""Storage queries of the intermediates manager.

One classification expression decides what every intermediate is under the cleanup policy, and it
is the same SQL whether it aggregates a summary, describes a preview, pages an operation or guards
a delete:

- ``active``: produced or referenced by a pending, waiting or running queue item, or a
  completed child of an active root workflow.
- ``recent``: created inside the grace window, so an in-flight browser upload is never collected
  before it is promoted, referenced or enqueued; or a cached output whose consuming session ended
  inside that window, since a cache hit reuses an old row.
- ``referenced``: named by a saved document (`media_references`).
- ``safe``: none of the above.

A preview freezes the instant its recency is judged at; the operation it confirms and the guard
on every deleting transaction classify with that same recency cutoff and a live clock for
everything else. Anything created after the preview is newer than its cutoff, so it stays
``recent`` for the whole operation. Cached-output holds are judged the same way by the operation
itself, but the hold table is shared with live-clock callers (summaries, cache hits) that may
sweep a hold once its grace has passed live; that only ever drops a row whose grace is over.

Queue inputs are found by scanning each active item's stored session for media-name keys; the
scan is cached per item so a queue of a thousand pending items is parsed once, not per query.
"""

import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Collection, Iterator, Literal, NamedTuple, Optional, Sequence, cast

from invokeai.app.services.intermediates.intermediates_common import (
    BROWSER_HOLD_TTL_SECONDS,
    MAX_BROWSER_HOLD_LEASES_PER_USER,
    RECENT_GRACE_SECONDS,
    IntermediatesCleanupMode,
    IntermediatesKindCounts,
)
from invokeai.app.services.shared.intermediate_delete import IntermediateDeleteGuard
from invokeai.app.services.shared.media_references import IMAGE_NAME_KEYS, VIDEO_NAME_KEYS, MediaReferences
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase

MediaKind = Literal["image", "video"]
Classification = Literal["safe", "referenced", "active", "recent"]
# One row of the manager: (owner, project), project None for the owner's unassigned intermediates.
ScopeTarget = tuple[str, Optional[str]]

ACTIVE_QUEUE_STATUSES = ("pending", "in_progress", "waiting")


def _protected_queue_items_sql() -> str:
    statuses = ", ".join(f"'{status}'" for status in ACTIVE_QUEUE_STATUSES)
    # A child becomes completed before its outputs reach its waiting parent. Keep the
    # whole completed subtree protected until the root finishes, including nested calls.
    # CROSS JOIN pins the active roots first, then indexed child lookups, rather than
    # scanning all completed history. Session JSON is fetched only for changed rows.
    return f"""
        SELECT item_id, session_id, status, session_revision FROM session_queue
        WHERE status IN ({statuses})
        UNION ALL
        SELECT child.item_id, child.session_id, child.status, child.session_revision
        FROM session_queue root
        CROSS JOIN session_queue child ON child.root_item_id = root.item_id
        WHERE root.status IN ({statuses}) AND child.status = 'completed'
    """


# Keeps the OR-chain of a selection scope, and every IN list, under SQLITE_MAX_VARIABLE_NUMBER.
_MAX_SQL_VARIABLES = 500
_TARGETS_PER_STATEMENT = 200


def _name_pattern(keys: frozenset[str]) -> re.Pattern[str]:
    alternatives = "|".join(sorted(re.escape(key) for key in keys))
    return re.compile(rf'"(?:{alternatives})"\s*:\s*"([^"\\]{{1,255}})"')


_IMAGE_NAME_RE = _name_pattern(IMAGE_NAME_KEYS)
_VIDEO_NAME_RE = _name_pattern(VIDEO_NAME_KEYS)

_TABLES: dict[MediaKind, tuple[str, str, str]] = {
    "image": ("images", "image_name", "image_subfolder"),
    "video": ("videos", "video_name", "video_subfolder"),
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _sql_timestamp(moment: datetime) -> str:
    """The text `STRFTIME('%Y-%m-%d %H:%M:%f', ...)` stores, so bound instants compare with stored ones as strings."""
    return moment.strftime("%Y-%m-%d %H:%M:%S.") + f"{moment.microsecond // 1000:03d}"


class _Clock(NamedTuple):
    """One instant for every statement of a classification: SQLite fixes `'NOW'` per statement, not per transaction."""

    now: str
    recent_cutoff: str


def _clock(recent_cutoff: Optional[str] = None) -> _Clock:
    """The live clock, or the live clock judging recency as of an earlier preview."""
    now = _utc_now()
    live_cutoff = _sql_timestamp(now - timedelta(seconds=RECENT_GRACE_SECONDS))
    return _Clock(_sql_timestamp(now), min(live_cutoff, recent_cutoff) if recent_cutoff is not None else live_cutoff)


class ReferenceOwner(NamedTuple):
    """A saved document naming a media item, as `media_references` keys it."""

    owner_kind: str
    user_id: str
    owner_id: str


# Media name → the documents naming it.
ReferenceOwners = dict[str, set[ReferenceOwner]]


def _add_count(counts: IntermediatesKindCounts, classification: Classification, n: int) -> None:
    if classification == "safe":
        counts.safe += n
    elif classification == "referenced":
        counts.referenced += n
    elif classification == "active":
        counts.active += n
    else:
        counts.recent += n


@dataclass
class ScopeCounts:
    """Aggregated intermediates of one (owner, project) row for one media kind."""

    counts: IntermediatesKindCounts = field(default_factory=IntermediatesKindCounts)
    safe_bytes: int = 0
    referenced_bytes: int = 0
    unknown_size_count: int = 0


@dataclass
class DeletableTotals:
    count: int = 0
    referenced: int = 0
    measured_bytes: int = 0
    unknown_size_count: int = 0


@dataclass
class ScopePreview:
    """A preview's scope classified once, on one transaction and one clock, so its figures agree."""

    clock: _Clock
    # (owner, project) rows holding any intermediate in scope.
    rows: set[tuple[str, Optional[str]]] = field(default_factory=set)
    counts: dict[MediaKind, IntermediatesKindCounts] = field(
        default_factory=lambda: {"image": IntermediatesKindCounts(), "video": IntermediatesKindCounts()}
    )
    deletable: dict[MediaKind, DeletableTotals] = field(
        default_factory=lambda: {"image": DeletableTotals(), "video": DeletableTotals()}
    )
    # Force mode: the documents naming the deletable referenced items, with how many each names.
    acknowledged: dict[ReferenceOwner, int] = field(default_factory=dict)
    acknowledged_overflow: bool = False


class IntermediatesRecordsSqlite:
    def __init__(self, db: SqliteDatabase) -> None:
        self._db = db
        # Media names an active queue item's session names, keyed by item id and stamped with the
        # row's session revision, which a trigger bumps on every session rewrite: a session
        # rewritten while the item stays active (a workflow-call parent resuming with its child's
        # outputs) is re-scanned, a status change alone is not. Entries live as long as the item is
        # active. This cache is derived from committed rows only, so it is safe across a
        # rolled-back transaction; the temp table is not, which is why it is rebuilt on every call
        # rather than skipped on an unchanged set.
        self._active_inputs: dict[int, tuple[int, set[str], set[str]]] = {}

    @staticmethod
    def _prepare_session_holds(cursor: sqlite3.Cursor, clock: _Clock) -> None:
        # These holds share the cache's process lifetime. After restart the cache is empty and
        # recovered queue sessions protect their serialized inputs through the normal scan.
        # A cache hit reuses an old row, so the recency grace a fresh output gets is counted from
        # when the consuming session is first seen finished instead.
        cursor.execute(
            "CREATE TEMP TABLE IF NOT EXISTS intermediates_session_media "
            "(session_id TEXT, kind TEXT, name TEXT, released_at TEXT, PRIMARY KEY(session_id, kind, name)) "
            "WITHOUT ROWID;"
        )
        cursor.execute(
            "UPDATE temp.intermediates_session_media SET released_at = ? "
            "WHERE released_at IS NULL AND session_id NOT IN "
            f"(SELECT session_id FROM ({_protected_queue_items_sql()}));",
            (clock.now,),
        )
        cursor.execute("DELETE FROM temp.intermediates_session_media WHERE released_at <= ?;", (clock.recent_cutoff,))

    def hold_cached_media(self, session_id: str, references: MediaReferences) -> bool:
        if references.is_empty():
            return True
        with self._db.transaction() as cursor:
            self._prepare_session_holds(cursor, _clock())
            statuses = ", ".join(f"'{status}'" for status in ACTIVE_QUEUE_STATUSES)
            cursor.execute(
                f"SELECT 1 FROM session_queue WHERE session_id = ? AND status IN ({statuses});", (session_id,)
            )
            if cursor.fetchone() is None:
                return False
            held: list[tuple[str, str, str]] = []
            for kind, names in (("image", references.images), ("video", references.videos)):
                table, name_column, _ = _TABLES[cast(MediaKind, kind)]
                ordered = sorted(names)
                for start in range(0, len(ordered), _MAX_SQL_VARIABLES):
                    chunk = ordered[start : start + _MAX_SQL_VARIABLES]
                    placeholders = ",".join("?" for _ in chunk)
                    cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE {name_column} IN ({placeholders});", chunk)
                    if cursor.fetchone()[0] != len(chunk):
                        # Deletion won the race after the cache lookup. Do not return a stale
                        # output; invoking the node again will create fresh media instead.
                        return False
                held.extend((session_id, kind, name) for name in ordered)
            # The existence check and hold commit share the deleting transaction's lock. Cleanup
            # either deletes first (a cache miss), or observes this active session's hold.
            cursor.executemany("INSERT OR REPLACE INTO temp.intermediates_session_media VALUES (?, ?, ?, NULL);", held)
        return True

    def replace_browser_hold(self, user_id: str, lease_id: str, images: Sequence[str], videos: Sequence[str]) -> None:
        with self._db.transaction() as cursor:
            # Indexed on expires_at, so the sweep touches only what expired.
            cursor.execute(
                "DELETE FROM intermediates_browser_holds WHERE expires_at <= STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW');"
            )
            cursor.execute(
                "DELETE FROM intermediates_browser_holds WHERE user_id = ? AND lease_id = ?;", (user_id, lease_id)
            )
            cursor.execute("CREATE TEMP TABLE IF NOT EXISTS intermediates_hold_names (name TEXT PRIMARY KEY);")
            for kind, names in (("image", images), ("video", videos)):
                cursor.execute("DELETE FROM temp.intermediates_hold_names;")
                cursor.executemany(
                    "INSERT OR IGNORE INTO temp.intermediates_hold_names(name) VALUES (?);",
                    [(name,) for name in names],
                )
                table, name_column, _ = _TABLES[cast(MediaKind, kind)]
                cursor.execute(
                    f"""--sql
                    INSERT INTO intermediates_browser_holds
                    (user_id, lease_id, media_kind, media_name, expires_at)
                    SELECT ?, ?, ?, m.{name_column},
                           STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', '+{BROWSER_HOLD_TTL_SECONDS} seconds')
                    FROM {table} m
                    JOIN temp.intermediates_hold_names h ON h.name = m.{name_column}
                    WHERE m.user_id = ? AND m.is_intermediate = TRUE;
                    """,
                    (user_id, lease_id, kind, user_id),
                )
            # Past the cap, the leases refreshed longest ago stop protecting their media; an editor
            # that is still open restores its lease on its next refresh.
            cursor.execute(
                """--sql
                WITH leases AS (
                    SELECT lease_id, MAX(expires_at) AS refreshed
                    FROM intermediates_browser_holds
                    WHERE user_id = ?
                    GROUP BY lease_id
                ), lapsed AS (
                    SELECT lease_id FROM leases
                    WHERE lease_id != ?
                    ORDER BY refreshed DESC, lease_id
                    LIMIT -1 OFFSET ?
                )
                DELETE FROM intermediates_browser_holds
                WHERE user_id = ? AND lease_id IN (SELECT lease_id FROM lapsed);
                """,
                (user_id, lease_id, MAX_BROWSER_HOLD_LEASES_PER_USER - 1, user_id),
            )

    def release_browser_hold(self, user_id: str, lease_id: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                "DELETE FROM intermediates_browser_holds WHERE user_id = ? AND lease_id = ?;", (user_id, lease_id)
            )

    # region policy expression

    @staticmethod
    def _classification_sql(kind: MediaKind) -> str:
        """Binds a `_Clock`'s two fields, in order, ahead of any later parameters of the statement."""
        table, name_column, _ = _TABLES[kind]
        return f"""
            CASE
                WHEN (
                    m.session_id IS NOT NULL
                    AND m.session_id IN (SELECT session_id FROM ({_protected_queue_items_sql()}))
                ) OR m.{name_column} IN (
                    SELECT name FROM temp.intermediates_active_media WHERE kind = '{kind}'
                ) OR EXISTS (
                    SELECT 1 FROM intermediates_browser_holds h
                    WHERE h.media_kind = '{kind}' AND h.media_name = m.{name_column}
                      AND h.expires_at > ?
                ) THEN 'active'
                WHEN m.created_at > ?
                    OR m.{name_column} IN (
                        SELECT name FROM temp.intermediates_session_media
                        WHERE kind = '{kind}' AND released_at IS NOT NULL
                    ) THEN 'recent'
                WHEN EXISTS (
                    SELECT 1 FROM media_references r WHERE r.media_kind = '{kind}' AND r.media_name = m.{name_column}
                ) THEN 'referenced'
                ELSE 'safe'
            END
        """

    @staticmethod
    def _deletable_sql(
        kind: MediaKind, mode: IntermediatesCleanupMode, *, is_admin: bool, caller_user_id: str
    ) -> tuple[str, list[object]]:
        """What a cleanup may delete, over the classified rows of `_scoped_sql`.

        Force mode adds referenced items, but another account's document is never the caller's to
        break unless the caller administers the instance. Applied in SQL so a bounded page or
        batch is a page of what the caller can actually delete.
        """
        if mode == "safe":
            return "cls = 'safe'", []
        if is_admin:
            return "cls IN ('safe', 'referenced')", []
        return (
            "(cls = 'safe' OR (cls = 'referenced' AND NOT EXISTS ("
            "SELECT 1 FROM media_references r WHERE r.media_kind = ? AND r.media_name = name AND r.user_id != ?)))",
            [kind, caller_user_id],
        )

    def _prepare(self, cursor: sqlite3.Cursor, clock: Optional[_Clock] = None) -> _Clock:
        """Makes the transaction's view of active work current, and returns the instant its classifications use.

        Runs on the caller's transaction so a classification and the enqueue it might race are
        ordered by the database lock, never by a stale cache.
        """
        clock = clock or _clock()
        self._prepare_session_holds(cursor, clock)
        cursor.execute(
            """--sql
            CREATE TEMP TABLE IF NOT EXISTS intermediates_active_media (
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                PRIMARY KEY (kind, name)
            ) WITHOUT ROWID;
            """
        )
        cursor.execute(
            f"""--sql
            SELECT item_id, COALESCE(session_revision, 0)
            FROM ({_protected_queue_items_sql()});
            """
        )
        active = {cast(int, row[0]): cast(int, row[1]) for row in cursor.fetchall()}

        for stale in [item_id for item_id in self._active_inputs if item_id not in active]:
            del self._active_inputs[stale]
        changed = [
            item_id
            for item_id, stamp in active.items()
            if item_id not in self._active_inputs or self._active_inputs[item_id][0] != stamp
        ]
        for start in range(0, len(changed), _MAX_SQL_VARIABLES):
            chunk = changed[start : start + _MAX_SQL_VARIABLES]
            placeholders = ",".join("?" for _ in chunk)
            cursor.execute(
                f"""--sql
                SELECT item_id, COALESCE(session_revision, 0), session
                FROM session_queue WHERE item_id IN ({placeholders});
                """,
                chunk,
            )
            for item_id, stamp, session_json in cursor.fetchall():
                session_text = session_json if isinstance(session_json, str) else ""
                self._active_inputs[cast(int, item_id)] = (
                    cast(int, stamp),
                    set(_IMAGE_NAME_RE.findall(session_text)),
                    set(_VIDEO_NAME_RE.findall(session_text)),
                )

        cursor.execute("DELETE FROM temp.intermediates_active_media;")
        rows: set[tuple[str, str]] = set()
        for _, images, videos in self._active_inputs.values():
            rows.update(("image", name) for name in images)
            rows.update(("video", name) for name in videos)
        if rows:
            cursor.executemany(
                "INSERT OR IGNORE INTO temp.intermediates_active_media (kind, name) VALUES (?, ?);", sorted(rows)
            )
        cursor.execute(
            "INSERT OR IGNORE INTO temp.intermediates_active_media (kind, name) "
            "SELECT kind, name FROM temp.intermediates_session_media WHERE released_at IS NULL;"
        )
        return clock

    # endregion

    # region summary

    @classmethod
    def _scoped_sql(cls, kind: MediaKind) -> str:
        """Every intermediate of ``kind`` with its row and classification; callers append ``AND`` scope filters.

        Media whose project no longer exists, or that never had one, belong to the owner's
        unassigned row (``project_key`` NULL).
        """
        table, name_column, _ = _TABLES[kind]
        return f"""
            SELECT m.rowid AS rowid, m.{name_column} AS name, m.user_id AS user_id,
                   CASE WHEN p.project_id IS NULL THEN NULL ELSE m.project_id END AS project_key,
                   m.file_size_bytes AS file_size_bytes,
                   {cls._classification_sql(kind)} AS cls
            FROM {table} m
            LEFT JOIN projects p ON p.user_id = m.user_id AND p.project_id = m.project_id
            WHERE m.is_intermediate = TRUE
        """

    @classmethod
    def _aggregate(
        cls, cursor: sqlite3.Cursor, clock: _Clock, kind: MediaKind, scope_clause: str, params: Sequence[object]
    ) -> Iterator[tuple[str, Optional[str], Classification, int, int, int]]:
        """(owner, project, classification, count, measured bytes, unmeasured count) groups within a scope."""
        cursor.execute(
            f"""--sql
            SELECT user_id, project_key, cls, COUNT(*),
                   SUM(COALESCE(file_size_bytes, 0)),
                   SUM(CASE WHEN file_size_bytes IS NULL THEN 1 ELSE 0 END)
            FROM ({cls._scoped_sql(kind)} {scope_clause})
            GROUP BY user_id, project_key, cls;
            """,
            [*clock, *params],
        )
        for owner, project_key, classification, n, total_bytes, unknown in cursor.fetchall():
            yield (
                cast(str, owner),
                cast(Optional[str], project_key),
                cast(Classification, classification),
                cast(int, n),
                cast(int, total_bytes),
                cast(int, unknown),
            )

    def summarize(
        self, user_ids: Optional[Collection[str]], kinds: Sequence[MediaKind] = ("image", "video")
    ) -> dict[tuple[str, Optional[str]], dict[MediaKind, ScopeCounts]]:
        """Aggregates the ``kinds`` intermediates of ``user_ids`` (None: everyone) by (owner, project) and classification."""
        rows: dict[tuple[str, Optional[str]], dict[MediaKind, ScopeCounts]] = defaultdict(
            lambda: {"image": ScopeCounts(), "video": ScopeCounts()}
        )
        owners = sorted(user_ids) if user_ids is not None else None
        # Owner chunks partition the GROUP BY, so their rows never need merging.
        clauses: list[tuple[str, list[object]]] = (
            [
                (f"AND m.user_id IN ({','.join('?' for _ in chunk)})", list(chunk))
                for chunk in (
                    owners[start : start + _MAX_SQL_VARIABLES] for start in range(0, len(owners), _MAX_SQL_VARIABLES)
                )
            ]
            if owners is not None
            else [("", [])]
        )
        with self._db.transaction() as cursor:
            clock = self._prepare(cursor)
            for kind in kinds:
                for clause, params in clauses:
                    for owner, project_key, cls, n, total_bytes, unknown in self._aggregate(
                        cursor, clock, kind, clause, params
                    ):
                        scope = rows[(owner, project_key)][kind]
                        _add_count(scope.counts, cls, n)
                        if cls == "safe":
                            scope.safe_bytes += total_bytes
                            scope.unknown_size_count += unknown
                        elif cls == "referenced":
                            scope.referenced_bytes += total_bytes
                            scope.unknown_size_count += unknown
        return dict(rows)

    def get_projects(self, user_id: Optional[str]) -> dict[tuple[str, str], tuple[str, Optional[str]]]:
        """Maps (owner, project) to (name, cover image): the newest durable image on the project's board."""
        owner_filter = "WHERE p.user_id = ?" if user_id is not None else ""
        params: list[object] = [user_id] if user_id is not None else []
        with self._db.transaction() as cursor:
            cursor.execute(
                f"""--sql
                SELECT p.user_id, p.project_id, p.name,
                    (
                        SELECT bi.image_name FROM board_images bi
                        INNER JOIN images i ON i.image_name = bi.image_name
                        WHERE bi.board_id = p.board_id AND i.is_intermediate = FALSE
                        ORDER BY bi.created_at DESC LIMIT 1
                    )
                FROM projects p {owner_filter};
                """,
                params,
            )
            return {
                (cast(str, r[0]), cast(str, r[1])): (cast(str, r[2]), cast(Optional[str], r[3]))
                for r in cursor.fetchall()
            }

    def has_unmeasured_intermediates(self) -> bool:
        with self._db.transaction() as cursor:
            cursor.execute(
                "CREATE TEMP TABLE IF NOT EXISTS intermediates_unmeasurable "
                "(kind TEXT, name TEXT, PRIMARY KEY(kind, name));"
            )
            for kind, (table, name_column, _) in _TABLES.items():
                cursor.execute(
                    f"SELECT 1 FROM {table} m WHERE m.is_intermediate = TRUE AND m.file_size_bytes IS NULL "
                    f"AND NOT EXISTS (SELECT 1 FROM temp.intermediates_unmeasurable u "
                    f"WHERE u.kind = ? AND u.name = m.{name_column}) LIMIT 1;",
                    (kind,),
                )
                if cursor.fetchone() is not None:
                    return True
        return False

    def next_unmeasured(self, kind: MediaKind, limit: int, *, min_age_seconds: int = 0) -> list[tuple[str, str]]:
        """Unmeasured intermediates in insertion order; rows younger than ``min_age_seconds`` are left for later.

        A row is written before its file, so a brand-new one would measure as missing. Failures
        are held in a temporary table instead of a bounded IN list, so later rows remain reachable.
        """
        table, name_column, subfolder_column = _TABLES[kind]
        with self._db.transaction() as cursor:
            cursor.execute(
                "CREATE TEMP TABLE IF NOT EXISTS intermediates_unmeasurable (kind TEXT, name TEXT, PRIMARY KEY(kind, name));"
            )
            cursor.execute(
                f"""--sql
                SELECT m.{name_column}, m.{subfolder_column} FROM {table} m
                WHERE m.is_intermediate = TRUE AND m.file_size_bytes IS NULL
                  AND m.created_at <= STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', ?)
                  AND NOT EXISTS (SELECT 1 FROM temp.intermediates_unmeasurable u
                                  WHERE u.kind = ? AND u.name = m.{name_column})
                ORDER BY m.rowid ASC
                LIMIT ?;
                """,
                [f"-{min_age_seconds} seconds", kind, limit],
            )
            return [(cast(str, r[0]), cast(str, r[1])) for r in cursor.fetchall()]

    def mark_unmeasurable(self, kind: MediaKind, names: Sequence[str]) -> None:
        if not names:
            return
        with self._db.transaction() as cursor:
            cursor.execute(
                "CREATE TEMP TABLE IF NOT EXISTS intermediates_unmeasurable (kind TEXT, name TEXT, PRIMARY KEY(kind, name));"
            )
            cursor.executemany(
                "INSERT OR IGNORE INTO temp.intermediates_unmeasurable(kind, name) VALUES (?, ?);",
                [(kind, name) for name in names],
            )

    # endregion

    # region scope

    @classmethod
    def _scope_clauses(
        cls,
        *,
        user_id: Optional[str],
        targets: Optional[Sequence[ScopeTarget]],
        per_statement: int = _TARGETS_PER_STATEMENT,
    ) -> list[tuple[str, list[object]]]:
        """``AND`` filters (with their parameters) that together cover a scope; each stays under the variable limit."""
        if targets is not None:
            clauses: list[tuple[str, list[object]]] = []
            for start in range(0, len(targets), per_statement):
                predicate, params = cls._scope_predicate(targets[start : start + per_statement])
                if predicate:
                    clauses.append((f"AND ({predicate})", params))
            return clauses
        if user_id is not None:
            return [("AND m.user_id = ?", [user_id])]
        return [("", [])]

    @staticmethod
    def _scope_predicate(targets: Sequence[ScopeTarget]) -> tuple[str, list[object]]:
        clauses: list[str] = []
        params: list[object] = []
        for user_id, project_id in targets:
            if project_id is None:
                clauses.append("(m.user_id = ? AND p.project_id IS NULL)")
                params.append(user_id)
            else:
                clauses.append("(m.user_id = ? AND m.project_id = ? AND p.project_id IS NOT NULL)")
                params.extend((user_id, project_id))
        return " OR ".join(clauses), params

    def preview_scope(
        self,
        *,
        user_id: Optional[str],
        targets: Optional[Sequence[ScopeTarget]],
        mode: IntermediatesCleanupMode,
        is_admin: bool,
        caller_user_id: str,
        max_acknowledged: int,
    ) -> ScopePreview:
        """Counts every intermediate in scope and what a cleanup of ``mode`` may delete.

        ``targets`` narrows to selected rows; otherwise ``user_id`` narrows to an owner, and None
        means everyone. One transaction, one instant and one view of active work serve every
        figure, so the kept counts, the deletable totals and the acknowledged documents agree.
        """
        clauses = self._scope_clauses(user_id=user_id, targets=targets)
        with self._db.transaction() as cursor:
            result = ScopePreview(clock=self._prepare(cursor))
            for kind in _TABLES:
                deletable_sql, deletable_params = self._deletable_sql(
                    kind, mode, is_admin=is_admin, caller_user_id=caller_user_id
                )
                totals = result.deletable[kind]
                for clause, params in clauses:
                    # Parameters bind by textual position: the deletable predicate sits in the
                    # SELECT list, ahead of the classified subquery's clock and scope parameters.
                    cursor.execute(
                        f"""--sql
                        SELECT user_id, project_key, cls, {deletable_sql} AS deletable, COUNT(*),
                               SUM(COALESCE(file_size_bytes, 0)),
                               SUM(CASE WHEN file_size_bytes IS NULL THEN 1 ELSE 0 END)
                        FROM ({self._scoped_sql(kind)} {clause})
                        GROUP BY user_id, project_key, cls, deletable;
                        """,
                        [*deletable_params, *result.clock, *params],
                    )
                    for owner, project_key, cls, deletable, n, total_bytes, unknown in cursor.fetchall():
                        result.rows.add((cast(str, owner), cast(Optional[str], project_key)))
                        _add_count(result.counts[kind], cast(Classification, cls), cast(int, n))
                        if deletable:
                            totals.count += cast(int, n)
                            totals.measured_bytes += cast(int, total_bytes)
                            totals.unknown_size_count += cast(int, unknown)
                            if cls == "referenced":
                                totals.referenced += cast(int, n)
                    if mode == "force":
                        cursor.execute(
                            f"""--sql
                            SELECT r.owner_kind, r.user_id, r.owner_id, COUNT(*)
                            FROM ({self._scoped_sql(kind)} {clause}) s
                            JOIN media_references r ON r.media_kind = ? AND r.media_name = s.name
                            WHERE s.cls = 'referenced' AND {deletable_sql}
                            GROUP BY r.owner_kind, r.user_id, r.owner_id;
                            """,
                            [*result.clock, *params, kind, *deletable_params],
                        )
                        for owner_kind, owner_user, owner_id, n in cursor.fetchall():
                            owner = ReferenceOwner(str(owner_kind), str(owner_user), str(owner_id))
                            if owner not in result.acknowledged and len(result.acknowledged) >= max_acknowledged:
                                result.acknowledged_overflow = True
                                continue
                            result.acknowledged[owner] = result.acknowledged.get(owner, 0) + cast(int, n)
        return result

    @classmethod
    def _window_sql(cls, kind: MediaKind, scope_clause: str, deletable_sql: str) -> str:
        """One window of classified rows in insertion order. Binds ``[*deletable, *clock, *scope, after_rowid, limit]``."""
        return f"""--sql
            SELECT rowid, name, file_size_bytes, {deletable_sql} AS deletable
            FROM ({cls._scoped_sql(kind)} {scope_clause})
            WHERE rowid > ?
            ORDER BY rowid
            LIMIT ?;
        """

    def iter_deletable_batches(
        self,
        kind: MediaKind,
        *,
        user_id: Optional[str],
        targets: Optional[Sequence[ScopeTarget]],
        mode: IntermediatesCleanupMode,
        is_admin: bool,
        caller_user_id: str,
        recent_cutoff: Optional[str],
        limit: int,
    ) -> Iterator[list[tuple[str, Optional[int]]]]:
        """The deletable intermediates of a scope in insertion order, a bounded window at a time.

        Each window classifies ``limit`` rows on one short transaction and yields the deletable
        ones (possibly none, so the caller can re-check its authority between windows), so a long
        protected stretch never holds the database for more than one window, and a row is
        classified once per operation however many are kept. Explicit rows are paged one at a
        time: one (owner, project) per statement is an index seek in rowid order, where an OR over
        many rows would sort the whole remaining scope on every window. ``recent_cutoff`` freezes
        recency at the preview that was confirmed; None judges it live.
        """
        deletable_sql, deletable_params = self._deletable_sql(
            kind, mode, is_admin=is_admin, caller_user_id=caller_user_id
        )
        window_sql = None
        for clause, params in self._scope_clauses(user_id=user_id, targets=targets, per_statement=1):
            window_sql = self._window_sql(kind, clause, deletable_sql)
            after_rowid = 0
            while True:
                with self._db.transaction() as cursor:
                    clock = self._prepare(cursor, _clock(recent_cutoff))
                    cursor.execute(window_sql, [*deletable_params, *clock, *params, after_rowid, limit])
                    rows = cursor.fetchall()
                if not rows:
                    break
                after_rowid = cast(int, rows[-1][0])
                yield [(cast(str, row[1]), cast(Optional[int], row[2])) for row in rows if row[3]]

    @staticmethod
    def _reference_owners(cursor: sqlite3.Cursor, kind: MediaKind, names: Sequence[str]) -> ReferenceOwners:
        owners: ReferenceOwners = {}
        for start in range(0, len(names), _MAX_SQL_VARIABLES - 1):
            chunk = list(names[start : start + _MAX_SQL_VARIABLES - 1])
            placeholders = ",".join("?" for _ in chunk)
            cursor.execute(
                f"SELECT media_name, owner_kind, user_id, owner_id FROM media_references "
                f"WHERE media_kind = ? AND media_name IN ({placeholders});",
                [kind, *chunk],
            )
            for name, owner_kind, user_id, owner_id in cursor.fetchall():
                owners.setdefault(str(name), set()).add(ReferenceOwner(str(owner_kind), str(user_id), str(owner_id)))
        return owners

    def get_document_names(self, owners: Collection[ReferenceOwner]) -> dict[ReferenceOwner, str]:
        """Names of the referencing documents, one query per kind and chunk; client state has none."""
        queries = {
            "project": "SELECT user_id, project_id, name FROM projects WHERE project_id IN ({});",
            "quarantined_project": (
                "SELECT user_id, project_id, name FROM orphaned_projects_2026_08_06 WHERE project_id IN ({});"
            ),
            # Workflow ids are globally unique, so the owner is not part of the match.
            "workflow": "SELECT NULL, workflow_id, name FROM workflow_library WHERE workflow_id IN ({});",
        }
        wanted: dict[str, set[ReferenceOwner]] = defaultdict(set)
        for owner in owners:
            if owner.owner_kind in queries:
                wanted[owner.owner_kind].add(owner)
        names: dict[ReferenceOwner, str] = {}
        with self._db.transaction() as cursor:
            for owner_kind, kind_owners in wanted.items():
                ids = sorted({owner.owner_id for owner in kind_owners})
                found: dict[tuple[Optional[str], str], str] = {}
                for start in range(0, len(ids), _MAX_SQL_VARIABLES):
                    chunk = ids[start : start + _MAX_SQL_VARIABLES]
                    cursor.execute(queries[owner_kind].format(",".join("?" for _ in chunk)), chunk)
                    for user_id, owner_id, name in cursor.fetchall():
                        if name is not None:
                            found[(cast(Optional[str], user_id), cast(str, owner_id))] = cast(str, name)
                for owner in kind_owners:
                    name = found.get((None if owner_kind == "workflow" else owner.user_id, owner.owner_id))
                    if name is not None:
                        names[owner] = name
        return names

    # endregion

    # region delete guard

    def make_delete_guard(
        self,
        kind: MediaKind,
        *,
        mode: IntermediatesCleanupMode,
        allowed_user_ids: Optional[frozenset[str]],
        caller_user_id: str,
        is_admin: bool,
        recent_cutoff: Optional[str] = None,
        acknowledged: Optional[frozenset[ReferenceOwner]] = None,
    ) -> IntermediateDeleteGuard:
        """The final check of a cleanup batch, run on the deleting transaction.

        Re-applies the policy at the moment of deletion, judging recency as of ``recent_cutoff``:
        anything that became active, referenced (safe mode), non-intermediate or another
        account's since the batch was read is kept. A force clear deletes a referenced item only
        while every document naming it is in ``acknowledged``. ``allowed_user_ids`` None means
        every account.
        """
        table, name_column, _ = _TABLES[kind]
        deletable_sql, deletable_params = self._deletable_sql(
            kind, mode, is_admin=is_admin, caller_user_id=caller_user_id
        )
        acknowledged_documents = acknowledged or frozenset()

        def guard(cursor: sqlite3.Cursor, names: Sequence[str]) -> list[str]:
            clock = self._prepare(cursor, _clock(recent_cutoff))
            kept: list[str] = []
            for start in range(0, len(names), _MAX_SQL_VARIABLES):
                chunk = list(names[start : start + _MAX_SQL_VARIABLES])
                placeholders = ",".join("?" for _ in chunk)
                current_references = self._reference_owners(cursor, kind, chunk) if mode == "force" else {}
                cursor.execute(
                    f"""--sql
                    SELECT name, user_id FROM (
                        SELECT m.{name_column} AS name, m.user_id AS user_id,
                               {self._classification_sql(kind)} AS cls
                        FROM {table} m
                        WHERE m.{name_column} IN ({placeholders}) AND m.is_intermediate = TRUE
                    ) WHERE {deletable_sql};
                    """,
                    [*clock, *chunk, *deletable_params],
                )
                # Ownership is re-read here rather than trusted from the preview.
                kept.extend(
                    cast(str, row[0])
                    for row in cursor.fetchall()
                    if allowed_user_ids is None or cast(str, row[1]) in allowed_user_ids
                    if current_references.get(cast(str, row[0]), set()).issubset(acknowledged_documents)
                )
            return kept

        return guard

    # endregion
