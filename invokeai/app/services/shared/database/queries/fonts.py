"""The font catalog: fonts users uploaded (private to their owner, or shared), and the fonts of the configured font
directory, which are shared and read-only."""

import functools
import itertools
import json
from collections.abc import Sequence
from typing import Any, NamedTuple, Optional

from sqlalchemy import (
    ColumnElement,
    Connection,
    Row,
    Select,
    and_,
    bindparam,
    delete,
    func,
    insert,
    literal,
    or_,
    select,
    update,
)

from invokeai.app.services.fonts.fonts_common import FontAxis, FontInstance, FontRecord, FontScope, FontSource
from invokeai.app.services.shared.database.dialect import (
    CaseInsensitiveLike,
    CaseInsensitiveOrder,
    fixed_limit,
    like_contains,
)
from invokeai.app.services.shared.database.queries.base import IN_CHUNK, QueryModule, mapped, read, write
from invokeai.app.services.shared.database.schema.fonts import fonts
from invokeai.app.services.shared.database.schema.metadata import PATH_LENGTH
from invokeai.app.services.shared.database.types import now_text

_F = fonts.c
_COLUMNS = (
    _F.id,
    _F.owner_id,
    _F.family,
    _F.label,
    _F.style,
    _F.weight,
    _F.content_hash,
    _F.scope,
    _F.source,
    _F.filename,
    _F.byte_size,
    _F.axes_json,
    _F.instances_json,
    _F.storage_path,
    _F.source_path,
)

_UPLOADED = _F.source == literal("uploaded")
_DIRECTORY = _F.source == literal("directory")
_PRIVATE = _F.scope == literal("private")
_SHARED = _F.scope == literal("shared")
_OWN_PRIVATE = and_(_UPLOADED, _PRIVATE, _F.owner_id == bindparam("owner"))
_SHARED_UPLOAD = and_(_UPLOADED, _SHARED, _F.owner_id.is_(None))

_FONT = select(*_COLUMNS).where(_F.id == bindparam("font"))
_PRIVATE_DUPLICATE = select(*_COLUMNS).where(_OWN_PRIVATE, _F.content_hash == bindparam("hash")).limit(fixed_limit(1))
_SHARED_DUPLICATE = select(*_COLUMNS).where(_SHARED_UPLOAD, _F.content_hash == bindparam("hash")).limit(fixed_limit(1))
_PRIVATE_BYTES = select(func.coalesce(func.sum(_F.byte_size), 0)).where(_OWN_PRIVATE)
_SHARED_BYTES = select(func.coalesce(func.sum(_F.byte_size), 0)).where(_SHARED_UPLOAD)
_INSERT = insert(fonts)
_DELETE_UPLOAD = delete(fonts).where(_F.id == bindparam("font"), _UPLOADED)

_DIRECTORY_FONTS = select(*_COLUMNS).where(_DIRECTORY)
_DIRECTORY_IDS = select(_F.id).where(_DIRECTORY)
_UPDATE_DIRECTORY_FONT = (
    update(fonts)
    .where(_F.id == bindparam("font"), _DIRECTORY)
    .values(
        filename=bindparam("filename"),
        source_path=bindparam("source_path"),
        family=bindparam("family"),
        label=bindparam("label"),
        style=bindparam("style"),
        weight=bindparam("weight"),
        content_hash=bindparam("content_hash"),
        byte_size=bindparam("byte_size"),
        axes_json=bindparam("axes_json"),
        instances_json=bindparam("instances_json"),
        updated_at=bindparam("now"),
    )
)
_DELETE_DIRECTORY_FONTS = delete(fonts).where(_DIRECTORY, _F.id.in_(bindparam("ids", expanding=True)))
_DELETE_ALL_DIRECTORY_FONTS = delete(fonts).where(_DIRECTORY)

_STORAGE_PATHS = select(_F.storage_path).where(_UPLOADED, _F.storage_path.is_not(None))
_PRIVATE_STORAGE_PATHS = select(_F.storage_path).where(_OWN_PRIVATE, _F.storage_path.is_not(None))

# The longest path, relative to the font directory, that the catalog holds. (A server holds a key column only up
# to its declared length.)
MAX_SOURCE_PATH_LENGTH = PATH_LENGTH


class DirectoryFont(NamedTuple):
    """A font file the directory scan found, as the catalog stores it."""

    id: str
    filename: str
    source_path: str
    family: str
    label: str
    style: str
    weight: int
    content_hash: str
    byte_size: int
    axes: tuple[FontAxis, ...]
    instances: tuple[FontInstance, ...]


class UploadedFont(NamedTuple):
    id: str
    owner_id: Optional[str]
    scope: FontScope
    filename: str
    storage_path: str
    family: str
    label: str
    style: str
    weight: int
    content_hash: str
    byte_size: int
    axes: tuple[FontAxis, ...]
    instances: tuple[FontInstance, ...]


class _Shape(NamedTuple):
    scope: FontScope
    by_hash: bool
    searched: bool


def _json_axes(axes: tuple[FontAxis, ...]) -> str:
    return json.dumps([axis.__dict__ for axis in axes], separators=(",", ":"), sort_keys=True)


def _json_instances(instances: tuple[FontInstance, ...]) -> str:
    return json.dumps([instance.__dict__ for instance in instances], separators=(",", ":"), sort_keys=True)


_SCOPES = {scope.value: scope for scope in FontScope}
_SOURCES = {source.value: source for source in FontSource}
# What `_json_axes(())` and `_json_instances(())` write, and the columns' default: most fonts are not variable.
_NONE = "[]"


def _record(row: Row[Any]) -> FontRecord:
    # A page builds a hundred of these, which costs more than reading their rows.
    (
        font_id,
        owner_id,
        family,
        label,
        style,
        weight,
        content_hash,
        scope,
        source,
        filename,
        byte_size,
        axes_json,
        instances_json,
        storage_path,
        source_path,
    ) = row
    return FontRecord(
        id=font_id,
        family=family,
        label=label,
        style=style,
        weight=int(weight),
        content_hash=content_hash,
        scope=_SCOPES[scope],
        source=_SOURCES[source],
        filename=filename,
        byte_size=int(byte_size),
        axes=() if axes_json == _NONE else tuple(FontAxis(**axis) for axis in json.loads(axes_json)),
        instances=(
            () if instances_json == _NONE else tuple(FontInstance(**item) for item in json.loads(instances_json))
        ),
        owner_id=owner_id,
        storage_path=storage_path,
        source_path=source_path,
    )


def _record_or_none(row: Optional[Row[Any]]) -> Optional[FontRecord]:
    return _record(row) if row is not None else None


def _records(rows: Sequence[Row[Any]]) -> list[FontRecord]:
    return [_record(row) for row in rows]


def _upload_state(state: tuple[Optional[Row[Any]], int]) -> tuple[Optional[FontRecord], int]:
    duplicate, used_bytes = state
    return _record_or_none(duplicate), used_bytes


def _page(page: tuple[Sequence[Row[Any]], int]) -> tuple[list[FontRecord], int]:
    rows, total = page
    return _records(rows), total


def _metadata_values(font: DirectoryFont | UploadedFont) -> dict[str, Any]:
    return {
        "filename": font.filename,
        "family": font.family,
        "label": font.label,
        "style": font.style,
        "weight": font.weight,
        "content_hash": font.content_hash,
        "byte_size": font.byte_size,
        "axes_json": _json_axes(font.axes),
        "instances_json": _json_instances(font.instances),
    }


def _conditions(shape: _Shape) -> list[ColumnElement[bool]]:
    # Directory fonts and shared uploads are everyone's; a private upload only its owner's.
    conditions: list[ColumnElement[bool]] = [or_(_DIRECTORY, _SHARED, and_(_PRIVATE, _F.owner_id == bindparam("user")))]
    if shape.scope == FontScope.PRIVATE:
        conditions.append(and_(_UPLOADED, _PRIVATE, _F.owner_id == bindparam("user")))
    elif shape.scope == FontScope.SHARED:
        conditions.append(or_(_DIRECTORY, and_(_UPLOADED, _SHARED)))
    if shape.by_hash:
        conditions.append(_F.content_hash == bindparam("hash"))
    if shape.searched:
        pattern = bindparam("pattern")
        conditions.append(
            or_(
                CaseInsensitiveLike(_F.family, pattern),
                CaseInsensitiveLike(_F.label, pattern),
                CaseInsensitiveLike(_F.filename, pattern),
            )
        )
    return conditions


@functools.cache
def _listing(shape: _Shape) -> Select[Any]:
    return (
        select(*_COLUMNS)
        .where(*_conditions(shape))
        .order_by(CaseInsensitiveOrder(_F.family), CaseInsensitiveOrder(_F.label), _F.id)
        .limit(bindparam("limit"))
        .offset(bindparam("offset"))
    )


@functools.cache
def _count(shape: _Shape) -> Select[Any]:
    return select(func.count()).select_from(fonts).where(*_conditions(shape))


class FontQueries(QueryModule):
    @mapped(_record_or_none)
    @read
    def get(self, conn: Connection, font_id: str) -> Optional[Row[Any]]:
        """The font, whoever may see it."""
        return conn.execute(_FONT, {"font": font_id}).first()

    @mapped(_page)
    @read
    def page(
        self,
        conn: Connection,
        *,
        user_id: str,
        scope: FontScope,
        content_hash: Optional[str],
        search: Optional[str],
        offset: int,
        limit: int,
    ) -> tuple[Sequence[Row[Any]], int]:
        """`limit` of the fonts the account may see, from `offset`, by family and label ignoring case, then id; and
        how many there are. `scope` keeps only the account's private uploads, or only the shared fonts; `search`
        keeps the fonts whose family, label or file name contains it, ignoring case."""
        shape = _Shape(scope, content_hash is not None, bool(search))
        parameters: dict[str, Any] = {"user": user_id}
        if content_hash is not None:
            parameters["hash"] = content_hash
        if search:
            parameters["pattern"] = like_contains(search)
        rows = conn.execute(_listing(shape), {**parameters, "limit": limit, "offset": offset}).all()
        if len(rows) < limit and (rows or offset == 0):
            # The page ends the listing, so it tells how many there are.
            return rows, offset + len(rows)
        return rows, int(conn.execute(_count(shape), parameters).scalar_one())

    @mapped(_upload_state)
    @read
    def upload_state(
        self, conn: Connection, *, owner_id: Optional[str], content_hash: str
    ) -> tuple[Optional[Row[Any]], int]:
        """The upload of the same content in the same place (the owner's private uploads, or the shared ones with
        no owner), if there is one; and how many bytes the uploads in that place take."""
        if owner_id is None:
            duplicate = conn.execute(_SHARED_DUPLICATE, {"hash": content_hash}).first()
            used = conn.execute(_SHARED_BYTES).scalar_one()
        else:
            duplicate = conn.execute(_PRIVATE_DUPLICATE, {"owner": owner_id, "hash": content_hash}).first()
            used = conn.execute(_PRIVATE_BYTES, {"owner": owner_id}).scalar_one()
        return duplicate, int(used)

    @write
    def insert_upload(self, conn: Connection, font: UploadedFont) -> None:
        conn.execute(
            _INSERT,
            {
                "id": font.id,
                "owner_id": font.owner_id,
                "scope": font.scope.value,
                "source": FontSource.UPLOADED.value,
                "storage_path": font.storage_path,
                "source_path": None,
                **_metadata_values(font),
            },
        )

    @write
    def delete_upload(self, conn: Connection, font_id: str) -> bool:
        """Deletes the uploaded font; whether there was one."""
        return conn.execute(_DELETE_UPLOAD, {"font": font_id}).rowcount == 1

    @mapped(_records)
    @read
    def directory_fonts(self, conn: Connection) -> Sequence[Row[Any]]:
        return conn.execute(_DIRECTORY_FONTS).all()

    @write
    def replace_directory_fonts(
        self, conn: Connection, changed: Sequence[DirectoryFont], keep_ids: frozenset[str]
    ) -> None:
        """Makes the directory fonts those of `keep_ids`: inserts or updates the `changed` ones, and deletes the
        others."""
        existing = set(conn.execute(_DIRECTORY_IDS).scalars().all())
        now = now_text()
        inserted = [font for font in changed if font.id not in existing]
        updated = [font for font in changed if font.id in existing]
        if inserted:
            conn.execute(
                _INSERT,
                [
                    {
                        "id": font.id,
                        "owner_id": None,
                        "scope": FontScope.SHARED.value,
                        "source": FontSource.DIRECTORY.value,
                        "storage_path": None,
                        "source_path": font.source_path,
                        **_metadata_values(font),
                    }
                    for font in inserted
                ],
            )
        if updated:
            conn.execute(
                _UPDATE_DIRECTORY_FONT,
                [
                    {"font": font.id, "source_path": font.source_path, "now": now, **_metadata_values(font)}
                    for font in updated
                ],
            )
        for chunk in itertools.batched(sorted(existing - keep_ids), IN_CHUNK):
            conn.execute(_DELETE_DIRECTORY_FONTS, {"ids": list(chunk)})

    @write
    def delete_directory_fonts(self, conn: Connection) -> int:
        """Deletes every directory font; how many there were."""
        return conn.execute(_DELETE_ALL_DIRECTORY_FONTS).rowcount

    @read
    def storage_paths(self, conn: Connection) -> set[str]:
        """The files of every uploaded font, relative to the managed storage."""
        return set(conn.execute(_STORAGE_PATHS).scalars().all())

    @read
    def private_storage_paths(self, conn: Connection, owner_id: str) -> tuple[str, ...]:
        """The files of the account's private uploads."""
        return tuple(conn.execute(_PRIVATE_STORAGE_PATHS, {"owner": owner_id}).scalars().all())
