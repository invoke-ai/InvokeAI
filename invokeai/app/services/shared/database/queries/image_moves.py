"""Image moves: jobs that move image files between subfolders, journaled per image so that a move interrupted
anywhere can be finished, and the images' records repointed, after a restart.

A job is active until it is committed or failed (`error`); there is at most one active job.
"""

from collections.abc import Sequence
from typing import Any, NamedTuple, Optional

from sqlalchemy import (
    Connection,
    Row,
    and_,
    bindparam,
    func,
    insert,
    literal,
    or_,
    select,
    update,
)

from invokeai.app.services.shared.database.dialect import fixed_limit
from invokeai.app.services.shared.database.queries.base import QueryModule, mapped, read, write
from invokeai.app.services.shared.database.schema.image_moves import (
    image_subfolder_move_items,
    image_subfolder_move_jobs,
)
from invokeai.app.services.shared.database.schema.images import images

_J = image_subfolder_move_jobs.c
_M = image_subfolder_move_items.c
_I = images.c

_FINISHED_JOB = _J.state.in_([literal("committed"), literal("error")])
_FINISHED_ITEM = _M.state.in_([literal("committed"), literal("error")])

_RECOVERABLE_JOB_IDS = (
    select(_J.id).where(_J.state.in_([literal("planned"), literal("moving"), literal("moved")])).order_by(_J.id)
)
_ACTIVE_JOB_ID = select(_J.id).where(~_FINISHED_JOB).order_by(_J.id).limit(fixed_limit(1))
_JOB = select(_J.id, _J.state, _J.error_message).where(_J.id == bindparam("job"))
_LATEST_JOB = select(_J.id, _J.state, _J.error_message).order_by(_J.id.desc()).limit(fixed_limit(1))
_INSERT_JOB = insert(image_subfolder_move_jobs)
_INSERT_ITEMS = insert(image_subfolder_move_items)
_SET_JOB_STATE = update(image_subfolder_move_jobs).where(_J.id == bindparam("job")).values(state=bindparam("new_state"))
_SET_JOB_ERROR = (
    update(image_subfolder_move_jobs).where(_J.id == bindparam("job")).values(error_message=bindparam("message"))
)
_FAIL_JOB = (
    update(image_subfolder_move_jobs)
    .where(_J.id == bindparam("job"))
    .values(state=literal("error"), error_message=bindparam("message"))
)
_FINISH_JOB = (
    update(image_subfolder_move_jobs)
    .where(_J.id == bindparam("job"))
    .values(state=bindparam("new_state"), error_message=bindparam("message"))
)

_THE_ITEM = and_(_M.job_id == bindparam("job"), _M.image_name == bindparam("image"))
_ITEM_COLUMNS = (_M.image_name, _M.old_subfolder, _M.new_subfolder, _M.is_intermediate)
_ITEMS = select(*_ITEM_COLUMNS).where(_M.job_id == bindparam("job")).order_by(_M.image_name)
_UNFINISHED_ITEMS = _ITEMS.where(~_FINISHED_ITEM)
_MARK_ITEM_MOVED = update(image_subfolder_move_items).where(_THE_ITEM).values(state=literal("moved"))
_FAIL_ITEM = (
    update(image_subfolder_move_items)
    .where(_THE_ITEM)
    .values(state=literal("error"), error_message=bindparam("message"))
)
_FAIL_ITEMS = (
    update(image_subfolder_move_items)
    .where(_M.job_id == bindparam("job"))
    .values(state=literal("error"), error_message=bindparam("message"))
)
_ITEM_ERRORS = (
    select(func.count())
    .select_from(image_subfolder_move_items)
    .where(_M.job_id == bindparam("job"), _M.state == literal("error"))
)
_ERROR_MESSAGES = (
    select(_M.error_message).where(_M.job_id == bindparam("job"), _M.state == literal("error")).order_by(_M.image_name)
)
_MOVED_COUNT = (
    select(func.count())
    .select_from(image_subfolder_move_items)
    .where(_M.job_id == bindparam("job"), _M.state == literal("moved"))
)
_COMMIT_ITEMS = (
    update(image_subfolder_move_items)
    .where(_M.job_id == bindparam("job"), _M.state == literal("moved"))
    .values(state=literal("committed"))
)
# The moved items whose image is gone, deleted or not where the move put it.
_INVALID_MOVES = (
    select(func.count())
    .select_from(image_subfolder_move_items.outerjoin(images, _I.image_name == _M.image_name))
    .where(
        _M.job_id == bindparam("job"),
        _M.state == literal("moved"),
        or_(_I.image_name.is_(None), _I.deleted_at.is_not(None), _I.image_subfolder != _M.new_subfolder),
    )
)
_ACTIVE_JOB_FOR_IMAGE = (
    select(literal(1))
    .select_from(image_subfolder_move_items.join(image_subfolder_move_jobs, _J.id == _M.job_id))
    .where(_M.image_name == bindparam("image"), ~_FINISHED_JOB)
    .limit(fixed_limit(1))
)

_LIVE_IMAGE = _I.deleted_at.is_(None)
_PLACEMENT = (_I.image_name, _I.image_subfolder, _I.image_category, _I.is_intermediate, _I.created_at)
_PLACEMENTS = select(*_PLACEMENT).where(_LIVE_IMAGE)
_PLACEMENTS_AFTER = (
    select(*_PLACEMENT)
    .where(_I.image_name > bindparam("after"), _LIVE_IMAGE)
    .order_by(_I.image_name)
    .limit(bindparam("limit"))
)
_NEXT_IMAGE_NAME = (
    select(_I.image_name)
    .where(_I.image_name > bindparam("after"), _LIVE_IMAGE)
    .order_by(_I.image_name)
    .limit(fixed_limit(1))
)
# Repoints an image the move found at its new subfolder, unless something else repointed it meanwhile.
_REPOINT = (
    update(images)
    .where(_I.image_name == bindparam("target"), _I.image_subfolder == bindparam("old_subfolder"))
    .values(image_subfolder=bindparam("new_subfolder"))
)
# The same for every image a job moved, in one statement that goes from the job's items to their images by key
# (`UPDATE ... FROM` on SQLite, a multi-table UPDATE on a server). With subqueries on the items instead, MariaDB
# scans every image; with a statement per image, a server takes a round trip per image.
_REPOINT_MOVED = (
    update(images)
    .where(
        _M.job_id == bindparam("job"),
        _M.state == literal("moved"),
        _I.image_name == _M.image_name,
        _I.image_subfolder == _M.old_subfolder,
    )
    .values(image_subfolder=_M.new_subfolder)
)


class MoveJob(NamedTuple):
    id: int
    state: str
    error_message: Optional[str]


class MoveItem(NamedTuple):
    image_name: str
    old_subfolder: str
    new_subfolder: str
    is_intermediate: bool
    old_path: str
    new_path: str
    old_thumbnail_path: str
    new_thumbnail_path: str


# Where an image's files are, and what decides where they belong: (image name, subfolder, category, intermediate,
# created at). Plain tuples, as status polls read every image.
Placement = tuple[str, str, str, bool, str]


def _placements(rows: Sequence[Row[Any]]) -> list[Placement]:
    return [(row[0], row[1], row[2], bool(row[3]), row[4]) for row in rows]


def _items(rows: Sequence[Row[Any]]) -> list[tuple[str, str, str, bool]]:
    return [(row[0], row[1], row[2], bool(row[3])) for row in rows]


def _item_values(job_id: int, item: MoveItem, state: str, error_message: Optional[str] = None) -> dict[str, Any]:
    return {**item._asdict(), "job_id": job_id, "state": state, "error_message": error_message}


class ImageMoveQueries(QueryModule):
    @read
    def recoverable_job_ids(self, conn: Connection) -> list[int]:
        """The jobs a restart finishes: planned, or interrupted while moving or before their records were repointed."""
        return [int(job_id) for job_id in conn.execute(_RECOVERABLE_JOB_IDS).scalars().all()]

    @read
    def active_job_id(self, conn: Connection) -> Optional[int]:
        return conn.execute(_ACTIVE_JOB_ID).scalar()

    @read
    def job(self, conn: Connection, job_id: int) -> Optional[MoveJob]:
        row = conn.execute(_JOB, {"job": job_id}).first()
        return MoveJob(row[0], row[1], row[2]) if row is not None else None

    @read
    def latest_job(self, conn: Connection) -> Optional[MoveJob]:
        row = conn.execute(_LATEST_JOB).first()
        return MoveJob(row[0], row[1], row[2]) if row is not None else None

    @mapped(_placements)
    @read
    def placements(self, conn: Connection) -> Sequence[Row[Any]]:
        """Every live image's placement."""
        return conn.execute(_PLACEMENTS).all()

    @mapped(_placements)
    @read
    def placements_after(self, conn: Connection, after: str, limit: int) -> Sequence[Row[Any]]:
        """Up to `limit` live images' placements, by name, after the name `after`."""
        return conn.execute(_PLACEMENTS_AFTER, {"after": after, "limit": limit}).all()

    @read
    def next_image_name(self, conn: Connection, after: str) -> Optional[str]:
        return conn.execute(_NEXT_IMAGE_NAME, {"after": after}).scalar()

    @write
    def create_job(self, conn: Connection, items: Sequence[MoveItem]) -> Optional[int]:
        """Journals a planned job of these items; its id, or None while another job is active. The move service
        serialises its callers, so the check and the insert need no lock."""
        if conn.execute(_ACTIVE_JOB_ID).first() is not None:
            return None
        job_id = int(conn.execute(_INSERT_JOB, {"state": "planned"}).inserted_primary_key[0])
        conn.execute(_INSERT_ITEMS, [_item_values(job_id, item, "planned") for item in items])
        return job_id

    @write
    def create_failed_job(self, conn: Connection, item: MoveItem, message: str) -> int:
        """Journals a job that failed before it started, for its one item."""
        job_id = int(conn.execute(_INSERT_JOB, {"state": "error", "error_message": message}).inserted_primary_key[0])
        conn.execute(_INSERT_ITEMS, [_item_values(job_id, item, "error", message)])
        return job_id

    @mapped(_items)
    @read
    def items(self, conn: Connection, job_id: int, *, unfinished_only: bool) -> Sequence[Row[Any]]:
        """The job's items as (image name, old subfolder, new subfolder, intermediate), by name; with
        `unfinished_only`, those neither committed nor failed."""
        return conn.execute(_UNFINISHED_ITEMS if unfinished_only else _ITEMS, {"job": job_id}).all()

    @read
    def has_active_job_for_image(self, conn: Connection, image_name: str) -> bool:
        return conn.execute(_ACTIVE_JOB_FOR_IMAGE, {"image": image_name}).first() is not None

    @read
    def error_count(self, conn: Connection, job_id: int) -> int:
        return int(conn.execute(_ITEM_ERRORS, {"job": job_id}).scalar_one())

    @write
    def set_job_state(self, conn: Connection, job_id: int, state: str) -> None:
        conn.execute(_SET_JOB_STATE, {"job": job_id, "new_state": state})

    @write
    def set_job_error_message(self, conn: Connection, job_id: int, message: str) -> None:
        conn.execute(_SET_JOB_ERROR, {"job": job_id, "message": message})

    @write
    def fail_job(self, conn: Connection, job_id: int, message: str) -> None:
        """Fails the job and every one of its items with the message."""
        conn.execute(_FAIL_JOB, {"job": job_id, "message": message})
        conn.execute(_FAIL_ITEMS, {"job": job_id, "message": message})

    @write
    def mark_item_moved(self, conn: Connection, job_id: int, image_name: str) -> None:
        conn.execute(_MARK_ITEM_MOVED, {"job": job_id, "image": image_name})

    @write
    def fail_item(self, conn: Connection, job_id: int, image_name: str, message: str) -> None:
        conn.execute(_FAIL_ITEM, {"job": job_id, "image": image_name, "message": message})

    @write
    def repoint_image(self, conn: Connection, image_name: str, old_subfolder: str, new_subfolder: str) -> None:
        """Points the image's record at its new subfolder, if it still points at the old one."""
        conn.execute(_REPOINT, {"target": image_name, "old_subfolder": old_subfolder, "new_subfolder": new_subfolder})

    @write
    def repoint_moved_images(self, conn: Connection, job_id: int) -> int:
        """Points the records of the job's moved images at their new subfolders, unless something else repointed
        them meanwhile; how many items the job moved."""
        conn.execute(_REPOINT_MOVED, {"job": job_id})
        return int(conn.execute(_MOVED_COUNT, {"job": job_id}).scalar_one())

    @read
    def invalid_move_count(self, conn: Connection, job_id: int) -> int:
        """How many of the job's moved items do not have their image where the move put it."""
        return int(conn.execute(_INVALID_MOVES, {"job": job_id}).scalar_one())

    @read
    def error_messages(self, conn: Connection, job_id: int) -> list[Optional[str]]:
        """The messages of the job's failed items, by image name; None for one without a message."""
        return list(conn.execute(_ERROR_MESSAGES, {"job": job_id}).scalars().all())

    @write
    def finish_job(self, conn: Connection, job_id: int, *, error_message: Optional[str]) -> None:
        """Commits the job's moved items, and the job, or fails the job with the message."""
        conn.execute(_COMMIT_ITEMS, {"job": job_id})
        state = "committed" if error_message is None else "error"
        conn.execute(_FINISH_JOB, {"job": job_id, "new_state": state, "message": error_message})
