"""Every query of the application, one module per domain.

A query module subclasses `QueryModule` (see `base`). Each of its methods takes the connection right after
`self` and is decorated with `read` or `write`, which supply it, so callers never see a connection:

    user = db.queries.users.get(user_id)  # a transaction of its own
    with db.queries.transaction() as q:  # one transaction for several calls
        q.client_state.set(user_id, key, value)
        q.media_references.replace(...)

A query method returns plain values, or rows read completely that `mapped` turns into DTOs once its own
transaction has ended. It does not call other query modules -- composition belongs to the caller's
transaction -- and has no effects outside the database, because a call that loses a race is run again from
the start. It passes execution options per statement (`conn.execute(statement, execution_options=...)`), never
to the connection: on SQLite every transaction shares one connection object, so an option set on it would
stay for all later transactions.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cached_property
from typing import TYPE_CHECKING, Optional, Self, TypeVar

from invokeai.app.services.shared.database.errors import NestedTransactionError, TransactionFailedError
from invokeai.app.services.shared.database.queries.app_settings import AppSettingQueries
from invokeai.app.services.shared.database.queries.base import OwnTransaction, QueryScope, SharedTransaction
from invokeai.app.services.shared.database.queries.board_images import BoardImageQueries
from invokeai.app.services.shared.database.queries.board_videos import BoardVideoQueries
from invokeai.app.services.shared.database.queries.boards import BoardQueries
from invokeai.app.services.shared.database.queries.client_state import ClientStateQueries
from invokeai.app.services.shared.database.queries.fonts import FontQueries
from invokeai.app.services.shared.database.queries.gallery import GalleryQueries
from invokeai.app.services.shared.database.queries.image_index import ImageIndexQueries
from invokeai.app.services.shared.database.queries.image_moves import ImageMoveQueries
from invokeai.app.services.shared.database.queries.images import ImageQueries
from invokeai.app.services.shared.database.queries.intermediates import IntermediateQueries
from invokeai.app.services.shared.database.queries.locks import LockQueries
from invokeai.app.services.shared.database.queries.media_references import MediaReferenceQueries
from invokeai.app.services.shared.database.queries.model_relationships import ModelRelationshipQueries
from invokeai.app.services.shared.database.queries.models import ModelQueries
from invokeai.app.services.shared.database.queries.projects import ProjectQueries
from invokeai.app.services.shared.database.queries.session_queue import SessionQueueQueries
from invokeai.app.services.shared.database.queries.style_presets import StylePresetQueries
from invokeai.app.services.shared.database.queries.system_prompts import SystemPromptQueries
from invokeai.app.services.shared.database.queries.users import UserQueries
from invokeai.app.services.shared.database.queries.videos import VideoQueries
from invokeai.app.services.shared.database.queries.wildcards import WildcardQueries
from invokeai.app.services.shared.database.queries.workflows import WorkflowQueries

if TYPE_CHECKING:
    from invokeai.app.services.shared.database.database import Database

R = TypeVar("R")


class Queries:
    """Every query of the application, grouped by domain as `queries.<domain>.<method>(...)`.

    `Database.queries` runs each call in a transaction of its own. The queries yielded by `transaction()`
    share one transaction, which commits when the block exits. A domain's query module is created on first
    use, once per `Queries`.
    """

    def __init__(self, database: "Database", scope: Optional[QueryScope] = None) -> None:
        self._database = database
        self._scope: QueryScope = scope if scope is not None else OwnTransaction(database)

    @cached_property
    def app_settings(self) -> AppSettingQueries:
        return AppSettingQueries(self._scope)

    @cached_property
    def board_images(self) -> BoardImageQueries:
        return BoardImageQueries(self._scope)

    @cached_property
    def board_videos(self) -> BoardVideoQueries:
        return BoardVideoQueries(self._scope)

    @cached_property
    def boards(self) -> BoardQueries:
        return BoardQueries(self._scope)

    @cached_property
    def client_state(self) -> ClientStateQueries:
        return ClientStateQueries(self._scope)

    @cached_property
    def fonts(self) -> FontQueries:
        return FontQueries(self._scope)

    @cached_property
    def gallery(self) -> GalleryQueries:
        return GalleryQueries(self._scope)

    @cached_property
    def image_index(self) -> ImageIndexQueries:
        return ImageIndexQueries(self._scope)

    @cached_property
    def image_moves(self) -> ImageMoveQueries:
        return ImageMoveQueries(self._scope)

    @cached_property
    def images(self) -> ImageQueries:
        return ImageQueries(self._scope)

    @cached_property
    def intermediates(self) -> IntermediateQueries:
        return IntermediateQueries(self._scope)

    @cached_property
    def locks(self) -> LockQueries:
        return LockQueries(self._scope)

    @cached_property
    def media_references(self) -> MediaReferenceQueries:
        return MediaReferenceQueries(self._scope)

    @cached_property
    def model_relationships(self) -> ModelRelationshipQueries:
        return ModelRelationshipQueries(self._scope)

    @cached_property
    def models(self) -> ModelQueries:
        return ModelQueries(self._scope)

    @cached_property
    def projects(self) -> ProjectQueries:
        return ProjectQueries(self._scope)

    @cached_property
    def session_queue(self) -> SessionQueueQueries:
        return SessionQueueQueries(self._scope)

    @cached_property
    def style_presets(self) -> StylePresetQueries:
        return StylePresetQueries(self._scope)

    @cached_property
    def system_prompts(self) -> SystemPromptQueries:
        return SystemPromptQueries(self._scope)

    @cached_property
    def users(self) -> UserQueries:
        return UserQueries(self._scope)

    @cached_property
    def videos(self) -> VideoQueries:
        return VideoQueries(self._scope)

    @cached_property
    def wildcards(self) -> WildcardQueries:
        return WildcardQueries(self._scope)

    @cached_property
    def workflows(self) -> WorkflowQueries:
        return WorkflowQueries(self._scope)

    @contextmanager
    def transaction(self, *, read_only: bool = False) -> Iterator[Self]:
        """Runs every call on the yielded queries in one transaction, committed when the block exits normally.

        A read-only transaction sees one consistent snapshot and refuses writes. The block is not retried
        (`run()` retries work that has no effects outside the database). After a statement failed, the
        transaction takes no further calls and does not commit: leave the block by raising (a caught
        `DatabaseError` can be turned into a domain error there).
        """
        if isinstance(self._scope, SharedTransaction):
            raise NestedTransactionError("These queries already belong to a transaction")
        with self._database.begin(write=not read_only) as conn:
            scope = SharedTransaction(conn, writable=not read_only, dialect_name=self._database.dialect_name)
            try:
                yield type(self)(self._database, scope)
                if scope.failed:
                    raise TransactionFailedError(
                        "A statement of this transaction failed and the error was not raised further; "
                        "the transaction was rolled back"
                    )
            finally:
                scope.close()

    def run(self, work: Callable[[Self], R], *, read_only: bool = False) -> R:
        """Runs `work` with the queries of one transaction, as `transaction()` does, and again from the start
        when the transaction loses a race (`ConflictError`).

        On MySQL and MariaDB, transactions that write the same rows can deadlock, and the server aborts one of
        them. `work` must have no effects outside the database, since a retry repeats it.
        """

        def transaction() -> R:
            with self.transaction(read_only=read_only) as q:
                return work(q)

        return self._database.retry_conflicts(transaction)
