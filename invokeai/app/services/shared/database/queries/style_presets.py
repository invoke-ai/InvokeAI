"""Style presets: the accounts' saved prompt templates, and the bundled ones (type `default`)."""

import functools
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from sqlalchemy import (
    Connection,
    Row,
    Select,
    Update,
    bindparam,
    delete,
    insert,
    literal,
    or_,
    select,
    true,
    update,
)

from invokeai.app.services.shared.database.dialect import CaseInsensitiveOrder
from invokeai.app.services.shared.database.queries.base import QueryModule, mapped, read, write
from invokeai.app.services.shared.database.schema.style_presets import style_presets
from invokeai.app.services.style_preset_records.style_preset_records_common import (
    PresetType,
    StylePresetChanges,
    StylePresetRecordDTO,
    StylePresetWithoutId,
)

_P = style_presets.c
_COLUMNS = (_P.id, _P.name, _P.preset_data, _P.type, _P.user_id, _P.is_public)
_NAMES = tuple(column.name for column in _COLUMNS)

_GET = select(*_COLUMNS).where(_P.id == bindparam("style_preset_id"))
_INSERT = insert(style_presets)
_DELETE = delete(style_presets).where(_P.id == bindparam("style_preset_id"))
_DELETE_BUNDLED = delete(style_presets).where(_P.type == literal(PresetType.Default.value))


@functools.cache
def _update(fields: tuple[str, ...]) -> Update:
    """Sets these fields of the preset; one statement per set of fields, of which there are few."""
    return (
        update(style_presets)
        .where(_P.id == bindparam("target_style_preset_id"))
        .values({field: bindparam(f"new_{field}") for field in fields})
    )


@functools.cache
def _list(visible_to_all: bool, scoped: bool, typed: bool) -> Select[Any]:
    """The presets an account sees, in name order: an administrator every one, anyone else the bundled and shared
    ones, and with `scoped` also its own."""
    statement = select(*_COLUMNS)
    if not visible_to_all:
        visible = [_P.type == literal(PresetType.Default.value), _P.is_public == true()]
        if scoped:
            visible.append(_P.user_id == bindparam("user_id"))
        statement = statement.where(or_(*visible))
    if typed:
        statement = statement.where(_P.type == bindparam("type"))
    # The id breaks ties, so that equal names keep one order.
    return statement.order_by(CaseInsensitiveOrder(_P.name), _P.id)


def _preset(row: Sequence[Any]) -> StylePresetRecordDTO:
    return StylePresetRecordDTO.from_dict(dict(zip(_NAMES, row, strict=True)))


def _preset_or_none(row: Optional[Sequence[Any]]) -> Optional[StylePresetRecordDTO]:
    return _preset(row) if row is not None else None


def _presets(rows: Sequence[Sequence[Any]]) -> list[StylePresetRecordDTO]:
    return [_preset(row) for row in rows]


class StylePresetQueries(QueryModule):
    @mapped(_preset_or_none)
    @read
    def get(self, conn: Connection, style_preset_id: str) -> Optional[Row[Any]]:
        return conn.execute(_GET, {"style_preset_id": style_preset_id}).first()

    @mapped(_presets)
    @read
    def all(
        self, conn: Connection, *, type: Optional[PresetType], user_id: Optional[str], is_admin: bool
    ) -> Sequence[Row[Any]]:
        """The presets the account sees, of `type` if given, in name order."""
        statement = _list(is_admin, user_id is not None, type is not None)
        return conn.execute(statement, {"user_id": user_id, "type": type}).all()

    @write
    def insert(self, conn: Connection, presets: Mapping[str, StylePresetWithoutId], user_id: str) -> None:
        """Adds the presets, keyed by their new ids, as the account's."""
        if presets:
            conn.execute(
                _INSERT,
                [
                    {
                        "id": style_preset_id,
                        "name": preset.name,
                        "preset_data": preset.preset_data.model_dump_json(),
                        "type": preset.type.value,
                        "user_id": user_id,
                        "is_public": preset.is_public,
                    }
                    for style_preset_id, preset in presets.items()
                ],
            )

    @write
    def update(self, conn: Connection, style_preset_id: str, changes: StylePresetChanges) -> None:
        """Applies the changes to the preset's name, data and visibility; its type stays as it is."""
        fields: dict[str, Any] = {}
        if changes.name is not None:
            fields["name"] = changes.name
        if changes.preset_data is not None:
            fields["preset_data"] = changes.preset_data.model_dump_json()
        if changes.is_public is not None:
            fields["is_public"] = changes.is_public
        if fields:
            values = {f"new_{field}": value for field, value in fields.items()}
            conn.execute(_update(tuple(sorted(fields))), {"target_style_preset_id": style_preset_id, **values})

    @write
    def delete(self, conn: Connection, style_preset_id: str) -> None:
        conn.execute(_DELETE, {"style_preset_id": style_preset_id})

    @write
    def delete_bundled(self, conn: Connection) -> None:
        conn.execute(_DELETE_BUNDLED)
