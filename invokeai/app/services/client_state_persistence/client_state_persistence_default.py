from invokeai.app.services.client_state_persistence.client_state_persistence_base import ClientStatePersistenceABC
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries.locks import DatabaseLock
from invokeai.app.services.shared.media_references import extract_media_references_from_json


class ClientStatePersistence(ClientStatePersistenceABC):
    """
    Client state persistence on the application database.
    This class stores client state data per user to prevent data leakage between users.
    """

    def __init__(self, database: Database) -> None:
        super().__init__()
        self._queries = database.queries

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def set_by_key(self, user_id: str, key: str, value: str) -> str:
        # The legacy editor keeps its canvas layers and reference images only here, as
        # intermediates; indexing them keeps cleanup from treating them as unused. Parsed before
        # the transaction, which on SQLite holds the lock every other writer shares.
        references = extract_media_references_from_json(value)

        def save(q: Queries) -> None:
            # Shared with every write that makes media protected, exclusive for the intermediates cleanup's check and
            # delete: the media this names cannot be deleted between that check and this commit.
            q.locks.acquire(DatabaseLock.MEDIA_PROTECTION, shared=True)
            q.client_state.set(user_id, key, value)
            q.media_references.replace(owner_kind="client_state", user_id=user_id, owner_id=key, references=references)

        self._queries.run(save)
        return value

    def get_by_key(self, user_id: str, key: str) -> str | None:
        return self._queries.client_state.get(user_id, key)

    def get_keys_by_prefix(self, user_id: str, prefix: str) -> list[str]:
        return self._queries.client_state.keys_with_prefix(user_id, prefix)

    def delete_by_key(self, user_id: str, key: str) -> None:
        def delete_value(q: Queries) -> None:
            # Only references of a value this transaction deleted: on a server, a value another transaction writes
            # for the first time meanwhile keeps the references it writes.
            if q.client_state.delete(user_id, key):
                q.media_references.delete(owner_kind="client_state", user_id=user_id, owner_id=key)

        self._queries.run(delete_value)

    def delete(self, user_id: str) -> None:
        def delete_all(q: Queries) -> None:
            keys = q.client_state.delete_all(user_id)
            q.media_references.delete_many(owner_kind="client_state", user_id=user_id, owner_ids=keys)

        self._queries.run(delete_all)
