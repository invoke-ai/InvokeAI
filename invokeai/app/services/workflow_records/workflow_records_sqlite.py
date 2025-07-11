from pathlib import Path
from typing import Optional

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import PaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.workflow_records.workflow_records_base import WorkflowRecordsStorageBase
from invokeai.app.services.workflow_records.workflow_records_common import (
    Workflow,
    WorkflowCategory,
    WorkflowNotFoundError,
    WorkflowRecordDTO,
    WorkflowRecordListItemDTO,
    WorkflowRecordListItemDTOValidator,
    WorkflowRecordOrderBy,
    WorkflowValidator,
    WorkflowWithoutID,
)
from invokeai.app.util.misc import uuid_string

SQL_TIME_FORMAT = "%Y-%m-%d %H:%M:%f"


class SqliteWorkflowRecordsStorage(WorkflowRecordsStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker
        self._sync_default_workflows()

    def get(self, workflow_id: str) -> WorkflowRecordDTO:
        """Gets a workflow by ID. Updates the opened_at column."""
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT workflow_id, workflow, name, created_at, updated_at, opened_at
                FROM workflow_library
                WHERE workflow_id = ?;
                """,
                (workflow_id,),
            )
            row = cursor.fetchone()
        if row is None:
            raise WorkflowNotFoundError(f"Workflow with id {workflow_id} not found")
        return WorkflowRecordDTO.from_dict(dict(row))

    def create(self, workflow: WorkflowWithoutID) -> WorkflowRecordDTO:
        if workflow.meta.category is WorkflowCategory.Default:
            raise ValueError("Default workflows cannot be created via this method")

        with self._db.transaction() as cursor:
            workflow_with_id = Workflow(**workflow.model_dump(), id=uuid_string())
            cursor.execute(
                """--sql
                INSERT OR IGNORE INTO workflow_library (
                    workflow_id,
                    workflow
                )
                VALUES (?, ?);
                """,
                (workflow_with_id.id, workflow_with_id.model_dump_json()),
            )
        return self.get(workflow_with_id.id)

    def update(self, workflow: Workflow) -> WorkflowRecordDTO:
        if workflow.meta.category is WorkflowCategory.Default:
            raise ValueError("Default workflows cannot be updated")

        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                UPDATE workflow_library
                SET workflow = ?
                WHERE workflow_id = ? AND category = 'user';
                """,
                (workflow.model_dump_json(), workflow.id),
            )
        return self.get(workflow.id)

    def delete(self, workflow_id: str) -> None:
        if self.get(workflow_id).workflow.meta.category is WorkflowCategory.Default:
            raise ValueError("Default workflows cannot be deleted")

        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                DELETE from workflow_library
                WHERE workflow_id = ? AND category = 'user';
                """,
                (workflow_id,),
            )
        return None

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
        is_published: Optional[bool] = None,
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
                        tags
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
        is_published: Optional[bool] = None,
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
        is_published: Optional[bool] = None,
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

    def update_opened_at(self, workflow_id: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                f"""--sql
                UPDATE workflow_library
                SET opened_at = STRFTIME('{SQL_TIME_FORMAT}', 'NOW')
                WHERE workflow_id = ?;
                """,
                (workflow_id,),
            )

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

                try:
                    workflow_from_db = self.get(workflow_from_file.id).workflow
                    if workflow_from_file != workflow_from_db:
                        self._invoker.services.logger.debug(
                            f"Updating library workflow {workflow_from_file.name} ({workflow_from_file.id})"
                        )
                        workflows_to_update.append(workflow_from_file)
                    continue
                except WorkflowNotFoundError:
                    self._invoker.services.logger.debug(
                        f"Adding missing default workflow {workflow_from_file.name} ({workflow_from_file.id})"
                    )
                    workflows_to_add.append(workflow_from_file)
                    continue

            library_workflows_from_db = self.get_many(
                order_by=WorkflowRecordOrderBy.Name,
                direction=SQLiteDirection.Ascending,
                categories=[WorkflowCategory.Default],
            ).items

            workflows_from_file_ids = [w.id for w in workflows_from_file]

            for w in library_workflows_from_db:
                if w.workflow_id not in workflows_from_file_ids:
                    self._invoker.services.logger.debug(
                        f"Deleting obsolete default workflow {w.name} ({w.workflow_id})"
                    )
                    # We cannot use the `delete` method here, as it only deletes non-default workflows
                    cursor.execute(
                        """--sql
                        DELETE from workflow_library
                        WHERE workflow_id = ?;
                        """,
                        (w.workflow_id,),
                    )

            for w in workflows_to_add:
                # We cannot use the `create` method here, as it only creates non-default workflows
                cursor.execute(
                    """--sql
                    INSERT INTO workflow_library (
                        workflow_id,
                        workflow
                    )
                    VALUES (?, ?);
                    """,
                    (w.id, w.model_dump_json()),
                )

            for w in workflows_to_update:
                # We cannot use the `update` method here, as it only updates non-default workflows
                cursor.execute(
                    """--sql
                    UPDATE workflow_library
                    SET workflow = ?
                    WHERE workflow_id = ?;
                    """,
                    (w.model_dump_json(), w.id),
                )
