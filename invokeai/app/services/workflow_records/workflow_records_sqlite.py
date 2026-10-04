import sqlite3
import uuid
from pathlib import Path
from typing import Optional

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.media_references import (
    MediaReferences,
    extract_media_references_from_json,
    replace_media_references,
)
from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.workflow_records.workflow_records_base import WorkflowRecordsStorageBase
from invokeai.app.services.workflow_records.workflow_records_common import (
    WORKFLOW_LIBRARY_DEFAULT_USER_ID,
    Workflow,
    WorkflowAccessDeniedError,
    WorkflowCategory,
    WorkflowIdConflictError,
    WorkflowImmutableError,
    WorkflowNotFoundError,
    WorkflowRecordDTO,
    WorkflowRecordListItemDTO,
    WorkflowRecordListItemDTOValidator,
    WorkflowRecordOrderBy,
    WorkflowRevisionConflictError,
    WorkflowValidator,
    WorkflowWithoutID,
)
from invokeai.app.util.misc import uuid_string

SQL_TIME_FORMAT = "%Y-%m-%d %H:%M:%f"

_RECORD_COLUMNS = (
    "workflow_id, workflow, name, created_at, updated_at, opened_at, last_run_at, user_id, is_public, revision"
)


class SqliteWorkflowRecordsStorage(WorkflowRecordsStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._sync_default_workflows()

    def get(self, workflow_id: str) -> WorkflowRecordDTO:
        """Gets a workflow by ID."""
        with self._db.transaction() as cursor:
            return self._get_on_cursor(cursor, workflow_id)

    @staticmethod
    def _get_on_cursor(cursor: sqlite3.Cursor, workflow_id: str) -> WorkflowRecordDTO:
        """Reads one record on the caller's transaction; `get()` on an open transaction would commit it early."""
        cursor.execute(
            f"SELECT {_RECORD_COLUMNS} FROM workflow_library WHERE workflow_id = ?;",
            (workflow_id,),
        )
        row = cursor.fetchone()
        if row is None:
            raise WorkflowNotFoundError(f"Workflow with id {workflow_id} not found")
        return WorkflowRecordDTO.from_dict(dict(row))

    def create(
        self,
        workflow: WorkflowWithoutID,
        user_id: str = WORKFLOW_LIBRARY_DEFAULT_USER_ID,
        is_public: bool = False,
        workflow_id: Optional[str] = None,
    ) -> WorkflowRecordDTO:
        if workflow.meta.category is WorkflowCategory.Default:
            raise ValueError("Default workflows cannot be created via this method")
        if workflow_id is not None:
            try:
                uuid.UUID(workflow_id)
            except ValueError as e:
                raise ValueError("A reserved workflow id must be a UUID") from e

        workflow_with_id = Workflow(**workflow.model_dump(), id=workflow_id or uuid_string())
        document_json = workflow_with_id.model_dump_json()
        references = extract_media_references_from_json(document_json)
        with self._db.transaction() as cursor:
            if workflow_id is not None:
                existing = self._match_reserved_record(cursor, workflow_with_id, user_id)
                if existing is not None:
                    return existing
            try:
                cursor.execute(
                    """--sql
                    INSERT INTO workflow_library (
                        workflow_id,
                        workflow,
                        user_id,
                        is_public,
                        revision
                    )
                    VALUES (?, ?, ?, ?, 1);
                    """,
                    (workflow_with_id.id, document_json, user_id, is_public),
                )
            except sqlite3.IntegrityError as e:
                # Only a reserved id can collide; a generated one is unique for practical purposes.
                raise WorkflowIdConflictError(workflow_with_id.id) from e
            self._index_references(cursor, workflow_with_id.id, references, user_id=user_id)
            return self._get_on_cursor(cursor, workflow_with_id.id)

    @staticmethod
    def _match_reserved_record(cursor: sqlite3.Cursor, workflow: Workflow, user_id: str) -> Optional[WorkflowRecordDTO]:
        """A retried creation is accepted only when the record it finds is this owner's identical submission.

        Any other record under the id is a conflict; its content and owner are not revealed to the caller.
        """
        cursor.execute(
            f"SELECT {_RECORD_COLUMNS} FROM workflow_library WHERE workflow_id = ?;",
            (workflow.id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        existing = WorkflowRecordDTO.from_dict(dict(row))
        if existing.user_id != user_id or existing.workflow.model_dump() != workflow.model_dump():
            raise WorkflowIdConflictError(workflow.id)
        return existing

    def update(
        self, workflow: Workflow, user_id: Optional[str] = None, expected_revision: Optional[int] = None
    ) -> WorkflowRecordDTO:
        if workflow.meta.category is WorkflowCategory.Default:
            raise ValueError("A workflow cannot be updated into the default category")

        document_json = workflow.model_dump_json()
        references = extract_media_references_from_json(document_json)
        with self._db.transaction() as cursor:
            # `category` is generated from the stored JSON, so it still describes the record as it is, not as
            # the request would rewrite it.
            cursor.execute(
                "SELECT category, user_id, revision FROM workflow_library WHERE workflow_id = ?;",
                (workflow.id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise WorkflowNotFoundError(f"Workflow with id {workflow.id} not found")
            stored_category, owner_id, current_revision = row
            if stored_category == WorkflowCategory.Default.value:
                raise WorkflowImmutableError(workflow.id)
            owner = str(owner_id) if owner_id is not None else WORKFLOW_LIBRARY_DEFAULT_USER_ID
            if user_id is not None and owner != user_id:
                raise WorkflowAccessDeniedError(workflow.id)
            if expected_revision is not None and current_revision != expected_revision:
                raise WorkflowRevisionConflictError(workflow.id, expected_revision, current_revision)

            cursor.execute(
                """--sql
                UPDATE workflow_library
                SET workflow = ?, revision = revision + 1
                WHERE workflow_id = ? AND revision = ?;
                """,
                (document_json, workflow.id, current_revision),
            )
            if cursor.rowcount == 0:
                # Another connection won between the read and this compare-and-swap.
                cursor.execute("SELECT revision FROM workflow_library WHERE workflow_id = ?;", (workflow.id,))
                raced = cursor.fetchone()
                if raced is None:
                    raise WorkflowNotFoundError(f"Workflow with id {workflow.id} not found")
                raise WorkflowRevisionConflictError(workflow.id, expected_revision or current_revision, raced[0])
            self._index_references(cursor, workflow.id, references, user_id=owner)
            return self._get_on_cursor(cursor, workflow.id)

    def delete(self, workflow_id: str, user_id: Optional[str] = None) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                "SELECT category, user_id FROM workflow_library WHERE workflow_id = ?;",
                (workflow_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise WorkflowNotFoundError(f"Workflow with id {workflow_id} not found")
            stored_category, owner_id = row
            if stored_category == WorkflowCategory.Default.value:
                raise WorkflowImmutableError(workflow_id)
            owner = str(owner_id) if owner_id is not None else WORKFLOW_LIBRARY_DEFAULT_USER_ID
            if user_id is not None and owner != user_id:
                raise WorkflowAccessDeniedError(workflow_id)
            cursor.execute("DELETE from workflow_library WHERE workflow_id = ?;", (workflow_id,))
            # The row is gone, so its owner is the only fact left to delete by; every owner's
            # rows for this id are dropped since workflow ids are globally unique.
            cursor.execute(
                "DELETE FROM media_references WHERE owner_kind = 'workflow' AND owner_id = ?;",
                (workflow_id,),
            )
        return None

    def update_is_public(self, workflow_id: str, is_public: bool, user_id: Optional[str] = None) -> WorkflowRecordDTO:
        """Updates the is_public field of a workflow and manages the 'shared' tag automatically."""
        # Read and rewrite the current document under one transaction. A concurrent workflow edit
        # must not land between the read and this full-document write: its reference index would
        # then describe the edit while the stored workflow names the previous assets.
        with self._db.transaction() as cursor:
            if user_id is not None:
                cursor.execute(
                    "SELECT workflow FROM workflow_library WHERE workflow_id = ? AND category = 'user' AND user_id = ?;",
                    (workflow_id, user_id),
                )
            else:
                cursor.execute(
                    "SELECT workflow FROM workflow_library WHERE workflow_id = ? AND category = 'user';", (workflow_id,)
                )
            row = cursor.fetchone()
            if row is not None:
                workflow = Workflow.model_validate_json(row[0])
                tags_list = [t.strip() for t in workflow.tags.split(",") if t.strip()] if workflow.tags else []
                if is_public and "shared" not in tags_list:
                    tags_list.append("shared")
                elif not is_public and "shared" in tags_list:
                    tags_list.remove("shared")
                updated_workflow = workflow.model_copy(update={"tags": ", ".join(tags_list)})
                # Visibility is bookkeeping: the `shared` tag rewrite does not advance the content revision, so
                # an editor holding the previous revision is not asked to resolve a conflict it cannot see.
                cursor.execute(
                    """--sql
                    UPDATE workflow_library
                    SET workflow = ?, is_public = ? WHERE workflow_id = ?;
                    """,
                    (updated_workflow.model_dump_json(), is_public, workflow_id),
                )
            return self._get_on_cursor(cursor, workflow_id)

    @staticmethod
    def _index_references(
        cursor: sqlite3.Cursor, workflow_id: str, references: MediaReferences, *, user_id: str
    ) -> None:
        """Records the media a library workflow names, on the caller's transaction."""
        replace_media_references(
            cursor, owner_kind="workflow", user_id=user_id, owner_id=workflow_id, references=references
        )

    def get_many(
        self,
        order_by: WorkflowRecordOrderBy,
        direction: SQLiteDirection,
        categories: Optional[list[WorkflowCategory]],
        page: int = 0,
        per_page: Optional[int] = None,
        query: Optional[str] = None,
        tags: Optional[list[str]] = None,
        has_been_opened: Optional[bool] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> PaginatedResults[WorkflowRecordListItemDTO]:
        with self._db.transaction() as cursor:
            # sanitize!
            assert order_by in WorkflowRecordOrderBy
            assert direction in SQLiteDirection

            # We will construct the query dynamically based on the query params

            # The main query to get the workflows / counts
            main_query = """
                    SELECT
                        workflow_id,
                        category,
                        name,
                        description,
                        created_at,
                        updated_at,
                        opened_at,
                        last_run_at,
                        tags,
                        user_id,
                        is_public,
                        revision
                    FROM workflow_library
                    """
            count_query = "SELECT COUNT(*) FROM workflow_library"

            # Start with an empty list of conditions and params
            conditions: list[str] = []
            params: list[str | int] = []

            if categories:
                # Categories is a list of WorkflowCategory enum values, and a single string in the DB

                # Ensure all categories are valid (is this necessary?)
                assert all(c in WorkflowCategory for c in categories)

                # Construct a placeholder string for the number of categories
                placeholders = ", ".join("?" for _ in categories)

                # Construct the condition string & params
                category_condition = f"category IN ({placeholders})"
                category_params = [category.value for category in categories]

                conditions.append(category_condition)
                params.extend(category_params)

            if tags:
                # Tags is a list of strings, and a single string in the DB
                # The string in the DB has no guaranteed format

                # Construct a list of conditions for each tag
                tags_conditions = ["tags LIKE ?" for _ in tags]
                tags_conditions_joined = " OR ".join(tags_conditions)
                tags_condition = f"({tags_conditions_joined})"

                # And the params for the tags, case-insensitive
                tags_params = [f"%{t.strip()}%" for t in tags]

                conditions.append(tags_condition)
                params.extend(tags_params)

            if has_been_opened:
                conditions.append("opened_at IS NOT NULL")
            elif has_been_opened is False:
                conditions.append("opened_at IS NULL")

            # Ignore whitespace in the query
            stripped_query = query.strip() if query else None
            if stripped_query:
                # Construct a wildcard query for the name, description, and tags
                wildcard_query = "%" + stripped_query + "%"
                query_condition = "(name LIKE ? OR description LIKE ? OR tags LIKE ?)"

                conditions.append(query_condition)
                params.extend([wildcard_query, wildcard_query, wildcard_query])

            if user_id is not None:
                # Scope to the given user but always include default workflows
                conditions.append("(user_id = ? OR category = 'default')")
                params.append(user_id)

            if is_public is True:
                conditions.append("is_public = TRUE")
            elif is_public is False:
                conditions.append("is_public = FALSE")

            if conditions:
                # If there are conditions, add a WHERE clause and then join the conditions
                main_query += " WHERE "
                count_query += " WHERE "

                all_conditions = " AND ".join(conditions)
                main_query += all_conditions
                count_query += all_conditions

            # After this point, the query and params differ for the main query and the count query
            main_params = params.copy()
            count_params = params.copy()

            # Main query also gets ORDER BY and LIMIT/OFFSET
            main_query += f" ORDER BY {order_by.value} {direction.value}"

            if per_page:
                main_query += " LIMIT ? OFFSET ?"
                main_params.extend([per_page, page * per_page])

            # Put a ring on it
            main_query += ";"
            count_query += ";"

            cursor.execute(main_query, main_params)
            rows = cursor.fetchall()
            workflows = [WorkflowRecordListItemDTOValidator.validate_python(dict(row)) for row in rows]

            cursor.execute(count_query, count_params)
            total = cursor.fetchone()[0]

        if per_page:
            pages = total // per_page + (total % per_page > 0)
        else:
            pages = 1  # If no pagination, there is only one page

        return PaginatedResults(
            items=workflows,
            page=page,
            per_page=per_page if per_page else total,
            pages=pages,
            total=total,
        )

    def counts_by_tag(
        self,
        tags: list[str],
        categories: Optional[list[WorkflowCategory]] = None,
        has_been_opened: Optional[bool] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> dict[str, int]:
        if not tags:
            return {}

        with self._db.transaction() as cursor:
            result: dict[str, int] = {}
            # Base conditions for categories and selected tags
            base_conditions: list[str] = []
            base_params: list[str | int] = []

            # Add category conditions
            if categories:
                assert all(c in WorkflowCategory for c in categories)
                placeholders = ", ".join("?" for _ in categories)
                base_conditions.append(f"category IN ({placeholders})")
                base_params.extend([category.value for category in categories])

            if has_been_opened:
                base_conditions.append("opened_at IS NOT NULL")
            elif has_been_opened is False:
                base_conditions.append("opened_at IS NULL")

            if user_id is not None:
                # Scope to the given user but always include default workflows
                base_conditions.append("(user_id = ? OR category = 'default')")
                base_params.append(user_id)

            if is_public is True:
                base_conditions.append("is_public = TRUE")
            elif is_public is False:
                base_conditions.append("is_public = FALSE")

            # For each tag to count, run a separate query
            for tag in tags:
                # Start with the base conditions
                conditions = base_conditions.copy()
                params = base_params.copy()

                # Add this specific tag condition
                conditions.append("tags LIKE ?")
                params.append(f"%{tag.strip()}%")

                # Construct the full query
                stmt = """--sql
                    SELECT COUNT(*)
                    FROM workflow_library
                    """

                if conditions:
                    stmt += " WHERE " + " AND ".join(conditions)

                cursor.execute(stmt, params)
                count = cursor.fetchone()[0]
                result[tag] = count

        return result

    def counts_by_category(
        self,
        categories: list[WorkflowCategory],
        has_been_opened: Optional[bool] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> dict[str, int]:
        with self._db.transaction() as cursor:
            result: dict[str, int] = {}
            # Base conditions for categories
            base_conditions: list[str] = []
            base_params: list[str | int] = []

            # Add category conditions
            if categories:
                assert all(c in WorkflowCategory for c in categories)
                placeholders = ", ".join("?" for _ in categories)
                base_conditions.append(f"category IN ({placeholders})")
                base_params.extend([category.value for category in categories])

            if has_been_opened:
                base_conditions.append("opened_at IS NOT NULL")
            elif has_been_opened is False:
                base_conditions.append("opened_at IS NULL")

            if user_id is not None:
                # Scope to the given user but always include default workflows
                base_conditions.append("(user_id = ? OR category = 'default')")
                base_params.append(user_id)

            if is_public is True:
                base_conditions.append("is_public = TRUE")
            elif is_public is False:
                base_conditions.append("is_public = FALSE")

            # For each category to count, run a separate query
            for category in categories:
                # Start with the base conditions
                conditions = base_conditions.copy()
                params = base_params.copy()

                # Add this specific category condition
                conditions.append("category = ?")
                params.append(category.value)

                # Construct the full query
                stmt = """--sql
                    SELECT COUNT(*)
                    FROM workflow_library
                    """

                if conditions:
                    stmt += " WHERE " + " AND ".join(conditions)

                cursor.execute(stmt, params)
                count = cursor.fetchone()[0]
                result[category.value] = count

        return result

    def update_opened_at(self, workflow_id: str, user_id: Optional[str] = None) -> None:
        with self._db.transaction() as cursor:
            if user_id is not None:
                cursor.execute(
                    f"""--sql
                    UPDATE workflow_library
                    SET opened_at = STRFTIME('{SQL_TIME_FORMAT}', 'NOW')
                    WHERE workflow_id = ? AND user_id = ?;
                    """,
                    (workflow_id, user_id),
                )
            else:
                cursor.execute(
                    f"""--sql
                    UPDATE workflow_library
                    SET opened_at = STRFTIME('{SQL_TIME_FORMAT}', 'NOW')
                    WHERE workflow_id = ?;
                    """,
                    (workflow_id,),
                )

    def update_last_run_at(self, workflow_id: str, user_id: Optional[str] = None) -> None:
        with self._db.transaction() as cursor:
            if user_id is not None:
                cursor.execute(
                    f"""--sql
                    UPDATE workflow_library
                    SET last_run_at = STRFTIME('{SQL_TIME_FORMAT}', 'NOW')
                    WHERE workflow_id = ? AND user_id = ?;
                    """,
                    (workflow_id, user_id),
                )
            else:
                cursor.execute(
                    f"""--sql
                    UPDATE workflow_library
                    SET last_run_at = STRFTIME('{SQL_TIME_FORMAT}', 'NOW')
                    WHERE workflow_id = ?;
                    """,
                    (workflow_id,),
                )

    def get_all_tags(
        self,
        categories: Optional[list[WorkflowCategory]] = None,
        user_id: Optional[str] = None,
        is_public: Optional[bool] = None,
    ) -> list[str]:
        with self._db.transaction() as cursor:
            conditions: list[str] = []
            params: list[str] = []

            # Only get workflows that have tags
            conditions.append("tags IS NOT NULL AND tags != ''")

            if categories:
                assert all(c in WorkflowCategory for c in categories)
                placeholders = ", ".join("?" for _ in categories)
                conditions.append(f"category IN ({placeholders})")
                params.extend([category.value for category in categories])

            if user_id is not None:
                # Scope to the given user but always include default workflows
                conditions.append("(user_id = ? OR category = 'default')")
                params.append(user_id)

            if is_public is True:
                conditions.append("is_public = TRUE")
            elif is_public is False:
                conditions.append("is_public = FALSE")

            stmt = """--sql
                SELECT DISTINCT tags
                FROM workflow_library
                """

            if conditions:
                stmt += " WHERE " + " AND ".join(conditions)

            cursor.execute(stmt, params)
            rows = cursor.fetchall()

            # Parse comma-separated tags and collect unique tags
            all_tags: set[str] = set()

            for row in rows:
                tags_value = row[0]
                if tags_value and isinstance(tags_value, str):
                    # Tags are stored as comma-separated string
                    for tag in tags_value.split(","):
                        tag_stripped = tag.strip()
                        if tag_stripped:
                            all_tags.add(tag_stripped)

            return sorted(all_tags)

    def _sync_default_workflows(self) -> None:
        """Syncs default workflows to the database. Internal use only."""

        """
        An enhancement might be to only update workflows that have changed. This would require stable
        default workflow IDs, and properly incrementing the workflow version.

        It's much simpler to just replace them all with whichever workflows are in the directory.

        The downside is that the `updated_at` and `opened_at` timestamps for default workflows are
        meaningless, as they are overwritten every time the server starts.
        """

        with self._db.transaction() as cursor:
            workflows_from_file: list[Workflow] = []
            workflows_to_update: list[Workflow] = []
            workflows_to_add: list[Workflow] = []
            workflows_dir = Path(__file__).parent / Path("default_workflows")
            workflow_paths = workflows_dir.glob("*.json")
            for path in workflow_paths:
                bytes_ = path.read_bytes()
                workflow_from_file = WorkflowValidator.validate_json(bytes_)

                assert workflow_from_file.id.startswith("default_"), (
                    f'Invalid default workflow ID (must start with "default_"): {workflow_from_file.id}'
                )

                assert workflow_from_file.meta.category is WorkflowCategory.Default, (
                    f"Invalid default workflow category: {workflow_from_file.meta.category}"
                )

                workflows_from_file.append(workflow_from_file)

                # Read on this cursor: `get()` would open a nested transaction and commit this one early.
                cursor.execute("SELECT workflow FROM workflow_library WHERE workflow_id = ?;", (workflow_from_file.id,))
                row = cursor.fetchone()
                if row is None:
                    self._invoker.services.logger.debug(
                        f"Adding missing default workflow {workflow_from_file.name} ({workflow_from_file.id})"
                    )
                    workflows_to_add.append(workflow_from_file)
                    continue
                if workflow_from_file != WorkflowValidator.validate_json(row[0]):
                    self._invoker.services.logger.debug(
                        f"Updating library workflow {workflow_from_file.name} ({workflow_from_file.id})"
                    )
                    workflows_to_update.append(workflow_from_file)

            cursor.execute("SELECT workflow_id, name FROM workflow_library WHERE category = 'default';")
            library_workflows_from_db = cursor.fetchall()

            workflows_from_file_ids = [w.id for w in workflows_from_file]

            for workflow_id, name in library_workflows_from_db:
                if workflow_id not in workflows_from_file_ids:
                    self._invoker.services.logger.debug(f"Deleting obsolete default workflow {name} ({workflow_id})")
                    # We cannot use the `delete` method here, as it only deletes non-default workflows
                    cursor.execute(
                        """--sql
                        DELETE from workflow_library
                        WHERE workflow_id = ?;
                        """,
                        (workflow_id,),
                    )

            for w in workflows_to_add:
                # We cannot use the `create` method here, as it only creates non-default workflows
                cursor.execute(
                    """--sql
                    INSERT INTO workflow_library (
                        workflow_id,
                        workflow,
                        revision
                    )
                    VALUES (?, ?, 1);
                    """,
                    (w.id, w.model_dump_json()),
                )

            for w in workflows_to_update:
                # We cannot use the `update` method here, as it refuses default workflows. A changed bundle is
                # new content, so its revision advances like any other write.
                cursor.execute(
                    """--sql
                    UPDATE workflow_library
                    SET workflow = ?, revision = revision + 1
                    WHERE workflow_id = ?;
                    """,
                    (w.model_dump_json(), w.id),
                )
