"""System prompts for the prompt-expansion models: the accounts' own, and shared ones."""

import functools
from collections.abc import Sequence
from typing import Any, Optional

from sqlalchemy import (
    Connection,
    Delete,
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
from invokeai.app.services.shared.database.schema.system_prompts import system_prompts
from invokeai.app.services.shared.database.types import now_text
from invokeai.app.services.system_prompt_records.system_prompt_records_common import (
    SystemPromptChanges,
    SystemPromptRecordDTO,
    SystemPromptWithoutId,
)

_S = system_prompts.c
_COLUMNS = (_S.id, _S.name, _S.content, _S.max_tokens, _S.user_id, _S.is_public, _S.created_at, _S.updated_at)
_NAMES = tuple(column.name for column in _COLUMNS)

_GET = select(*_COLUMNS).where(_S.id == bindparam("system_prompt_id"))
_INSERT = insert(system_prompts)
_ALL = select(*_COLUMNS).order_by(CaseInsensitiveOrder(_S.name), _S.id)
# An account's own prompts and the shared ones. The id breaks ties of equal names.
_VISIBLE = (
    select(*_COLUMNS)
    .where(or_(_S.user_id == bindparam("user_id"), _S.is_public == true()))
    .order_by(CaseInsensitiveOrder(_S.name), _S.id)
)


@functools.cache
def _owned(scoped: bool) -> Select[Any]:
    """The prompt, if it exists and, with `scoped`, the account owns it."""
    statement = select(literal(1)).where(_S.id == bindparam("system_prompt_id"))
    return statement.where(_S.user_id == bindparam("user_id")) if scoped else statement


@functools.cache
def _update(fields: tuple[str, ...], scoped: bool) -> Update:
    """Sets these fields of the prompt, with `scoped` only if the account owns it; few sets of fields are possible."""
    statement = (
        update(system_prompts)
        .where(_S.id == bindparam("target_system_prompt_id"))
        .values({field: bindparam(f"new_{field}") for field in fields})
    )
    return statement.where(_S.user_id == bindparam("owner_id")) if scoped else statement


@functools.cache
def _delete(scoped: bool) -> Delete:
    statement = delete(system_prompts).where(_S.id == bindparam("system_prompt_id"))
    return statement.where(_S.user_id == bindparam("user_id")) if scoped else statement


def _prompt(row: Sequence[Any]) -> SystemPromptRecordDTO:
    return SystemPromptRecordDTO.from_dict(dict(zip(_NAMES, row, strict=True)))


def _prompt_or_none(row: Optional[Sequence[Any]]) -> Optional[SystemPromptRecordDTO]:
    return _prompt(row) if row is not None else None


def _prompts(rows: Sequence[Sequence[Any]]) -> list[SystemPromptRecordDTO]:
    return [_prompt(row) for row in rows]


class SystemPromptQueries(QueryModule):
    @mapped(_prompt_or_none)
    @read
    def get(self, conn: Connection, system_prompt_id: str) -> Optional[Row[Any]]:
        return conn.execute(_GET, {"system_prompt_id": system_prompt_id}).first()

    @mapped(_prompts)
    @read
    def all(self, conn: Connection, user_id: Optional[str]) -> Sequence[Row[Any]]:
        """Every prompt, in name order; with `user_id`, the account's own and the shared ones."""
        if user_id is None:
            return conn.execute(_ALL).all()
        return conn.execute(_VISIBLE, {"user_id": user_id}).all()

    @write
    def insert(
        self, conn: Connection, system_prompt_id: str, prompt: SystemPromptWithoutId, user_id: str, is_public: bool
    ) -> str:
        """Adds the prompt; the time it was created at, which is also the time it was last updated."""
        now = now_text()
        conn.execute(
            _INSERT,
            {
                "id": system_prompt_id,
                "name": prompt.name,
                "content": prompt.content,
                "max_tokens": prompt.max_tokens,
                "user_id": user_id,
                "is_public": is_public,
                "created_at": now,
                "updated_at": now,
            },
        )
        return now

    @write
    def update(
        self, conn: Connection, system_prompt_id: str, changes: SystemPromptChanges, user_id: Optional[str]
    ) -> bool:
        """Applies the changes, with `user_id` only if the account owns the prompt; whether the prompt exists (and
        the account owns it). Without changes, nothing is written."""
        fields: dict[str, Any] = {}
        if changes.name is not None:
            fields["name"] = changes.name
        if changes.content is not None:
            fields["content"] = changes.content
        if changes.is_public is not None:
            fields["is_public"] = changes.is_public
        # `max_tokens` is the one field whose null is a value rather than "no change": it means "drop back to the
        # endpoint default". `is not None` would make a cap unclearable once set, so the presence of the key in the
        # request body decides instead.
        if "max_tokens" in changes.model_fields_set:
            fields["max_tokens"] = changes.max_tokens
        scoped = user_id is not None
        if not fields:
            parameters = {"system_prompt_id": system_prompt_id, "user_id": user_id}
            return conn.execute(_owned(scoped), parameters).first() is not None
        values = {f"new_{field}": value for field, value in fields.items()}
        parameters = {"target_system_prompt_id": system_prompt_id, "owner_id": user_id, **values}
        return conn.execute(_update(tuple(sorted(fields)), scoped), parameters).rowcount > 0

    @write
    def delete(self, conn: Connection, system_prompt_id: str, user_id: Optional[str]) -> bool:
        """Deletes the prompt, with `user_id` only if the account owns it; whether one was deleted."""
        parameters = {"system_prompt_id": system_prompt_id, "user_id": user_id}
        return conn.execute(_delete(user_id is not None), parameters).rowcount > 0
