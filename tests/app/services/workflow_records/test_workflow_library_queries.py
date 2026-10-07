"""Listing, searching and counting the workflow library, on every database backend."""

from pathlib import Path
from typing import Any, Optional

import pytest
from sqlalchemy import update

from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import workflows as workflow_queries
from invokeai.app.services.shared.database.schema.workflows import workflow_library
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.workflow_records import workflow_records_default
from invokeai.app.services.workflow_records.workflow_records_common import (
    Workflow,
    WorkflowCategory,
    WorkflowMeta,
    WorkflowRecordOrderBy,
    WorkflowWithoutID,
)
from invokeai.app.services.workflow_records.workflow_records_default import WorkflowRecordsStorage

USER = "user-1"
OTHER = "user-2"


def _workflow(name: str, *, description: str = "", tags: str = "") -> WorkflowWithoutID:
    return WorkflowWithoutID(
        name=name,
        author="",
        description=description,
        version="1.0.0",
        contact="",
        tags=tags,
        notes="",
        exposedFields=[],
        meta=WorkflowMeta(version="3.0.0", category=WorkflowCategory.User),
        nodes=[],
        edges=[],
    )


def _set(database: Database, workflow_id: str, **values: Any) -> None:
    with database.begin(write=True) as conn:
        conn.execute(update(workflow_library).where(workflow_library.c.workflow_id == workflow_id).values(**values))


def _names(
    records: WorkflowRecordsStorage,
    *,
    query: Optional[str] = None,
    tags: Optional[list[str]] = None,
    user_id: Optional[str] = None,
    is_public: Optional[bool] = None,
    has_been_opened: Optional[bool] = None,
) -> list[str]:
    listed = records.get_many(
        order_by=WorkflowRecordOrderBy.Name,
        direction=SQLiteDirection.Ascending,
        categories=[WorkflowCategory.User],
        query=query,
        tags=tags,
        user_id=user_id,
        is_public=is_public,
        has_been_opened=has_been_opened,
    )
    return [item.name for item in listed.items]


def test_a_search_looks_in_the_name_description_and_tags_ignoring_case(
    workflow_records: WorkflowRecordsStorage,
) -> None:
    workflow_records.create(_workflow("Hires FIX"), user_id=USER)
    workflow_records.create(_workflow("Faces", description="Fixes faces"), user_id=USER)
    workflow_records.create(_workflow("Tagged", tags="fixer"), user_id=USER)
    workflow_records.create(_workflow("Unrelated"), user_id=USER)

    assert _names(workflow_records, query="fix") == ["Faces", "Hires FIX", "Tagged"]
    assert _names(workflow_records, query="  FIX ") == ["Faces", "Hires FIX", "Tagged"]


def test_search_texts_and_tags_match_wildcard_characters_only_as_themselves(
    workflow_records: WorkflowRecordsStorage,
) -> None:
    for name in ("50% done", "500 done", "my_flow", "my-flow", "back\\slash", "backslash"):
        workflow_records.create(_workflow(name, tags=name), user_id=USER)

    assert _names(workflow_records, query="50%") == ["50% done"]
    assert _names(workflow_records, query="my_flow") == ["my_flow"]
    assert _names(workflow_records, query="back\\slash") == ["back\\slash"]
    assert _names(workflow_records, tags=["my_flow"]) == ["my_flow"]


def test_tag_counts_agree_with_the_tag_filter(workflow_records: WorkflowRecordsStorage) -> None:
    for name, tags in (("A", "Beta"), ("B", "50% off"), ("C", "500 off"), ("D", "my_flow"), ("E", "my-flow")):
        workflow_records.create(_workflow(name, tags=tags), user_id=USER)

    for tag in ("beta", "50%", "my_flow"):
        assert len(_names(workflow_records, tags=[tag])) == 1
        assert workflow_records.counts_by_tag([tag], categories=[WorkflowCategory.User]) == {tag: 1}


def test_a_tag_filter_keeps_the_workflows_with_any_of_its_tags(workflow_records: WorkflowRecordsStorage) -> None:
    workflow_records.create(_workflow("A", tags="alpha, common"), user_id=USER)
    workflow_records.create(_workflow("B", tags="Beta"), user_id=USER)
    workflow_records.create(_workflow("C", tags="gamma"), user_id=USER)

    assert _names(workflow_records, tags=["alpha", "beta"]) == ["A", "B"]
    # Three tags take four slots of a statement: the spare one matches nothing.
    assert _names(workflow_records, tags=["alpha", "beta", "missing"]) == ["A", "B"]
    assert _names(workflow_records, tags=["common"]) == ["A"]


def test_more_tags_than_a_statement_has_slots_are_listed_and_counted_like_few(
    workflow_records: WorkflowRecordsStorage,
) -> None:
    # Zero-padded, so that no tag contains another.
    workflow_records.create(_workflow("Late", tags="t099"), user_id=USER)
    workflow_records.create(_workflow("Early", tags="t000"), user_id=USER)
    tags = [f"t{i:03d}" for i in range(100)]

    assert _names(workflow_records, tags=tags) == ["Early", "Late"]
    counts = workflow_records.counts_by_tag(tags, categories=[WorkflowCategory.User])
    assert (counts["t000"], counts["t050"], counts["t099"]) == (1, 0, 1)


def test_an_account_sees_its_own_and_the_bundled_workflows(workflow_records: WorkflowRecordsStorage) -> None:
    mine = workflow_records.create(_workflow("Mine"), user_id=USER)
    theirs = workflow_records.create(_workflow("Theirs"), user_id=OTHER, is_public=True)

    listed = workflow_records.get_many(
        order_by=WorkflowRecordOrderBy.CreatedAt,
        direction=SQLiteDirection.Descending,
        categories=None,
        user_id=USER,
    ).items

    ids = {item.workflow_id for item in listed}
    assert mine.workflow_id in ids and theirs.workflow_id not in ids
    assert {item.category for item in listed} == {WorkflowCategory.User, WorkflowCategory.Default}
    assert _names(workflow_records, is_public=True) == ["Theirs"]
    assert _names(workflow_records, is_public=False) == ["Mine"]


def test_listings_tell_opened_workflows_apart(workflow_records: WorkflowRecordsStorage) -> None:
    opened = workflow_records.create(_workflow("Opened"), user_id=USER)
    workflow_records.create(_workflow("Never opened"), user_id=USER)
    # Opening by another account changes nothing: it is scoped to the owner.
    workflow_records.update_opened_at(opened.workflow_id, user_id=OTHER)
    assert _names(workflow_records, has_been_opened=True) == []

    workflow_records.update_opened_at(opened.workflow_id, user_id=USER)

    assert _names(workflow_records, has_been_opened=True) == ["Opened"]
    assert _names(workflow_records, has_been_opened=False) == ["Never opened"]


@pytest.mark.parametrize("direction", [SQLiteDirection.Ascending, SQLiteDirection.Descending])
def test_pages_of_equal_names_keep_one_order(
    workflow_records: WorkflowRecordsStorage, direction: SQLiteDirection
) -> None:
    created = [workflow_records.create(_workflow("Same"), user_id=USER).workflow_id for _ in range(5)]

    pages = [
        workflow_records.get_many(
            order_by=WorkflowRecordOrderBy.Name,
            direction=direction,
            categories=[WorkflowCategory.User],
            page=page,
            per_page=2,
        )
        for page in range(3)
    ]

    listed = [item.workflow_id for page in pages for item in page.items]
    assert listed == sorted(created, reverse=direction == SQLiteDirection.Descending)
    assert [(page.total, page.pages, page.per_page) for page in pages] == [(5, 3, 2)] * 3


@pytest.mark.parametrize("order_by", list(WorkflowRecordOrderBy))
def test_each_order_key_sorts_by_its_column(
    workflow_records: WorkflowRecordsStorage, database: Database, order_by: WorkflowRecordOrderBy
) -> None:
    # Each column puts the three workflows in a different order.
    rows: dict[str, dict[str, Any]] = {
        "c": {
            "created_at": "2026-01-01 00:00:00.000",
            "updated_at": "2026-01-03 00:00:00.000",
            "opened_at": "2026-01-02 00:00:00.000",
            "is_public": True,
        },
        "a": {
            "created_at": "2026-01-02 00:00:00.000",
            "updated_at": "2026-01-01 00:00:00.000",
            "opened_at": "2026-01-03 00:00:00.000",
            "is_public": False,
        },
        "b": {
            "created_at": "2026-01-03 00:00:00.000",
            "updated_at": "2026-01-02 00:00:00.000",
            "opened_at": "2026-01-01 00:00:00.000",
            "is_public": False,
        },
    }
    ids = {name: workflow_records.create(_workflow(name), user_id=USER).workflow_id for name in rows}
    for name, values in rows.items():
        _set(database, ids[name], **values)

    listed = workflow_records.get_many(order_by, SQLiteDirection.Ascending, [WorkflowCategory.User]).items

    def key(name: str) -> tuple[Any, str]:
        return (name if order_by == WorkflowRecordOrderBy.Name else rows[name][order_by.value]), ids[name]

    assert [item.name for item in listed] == sorted(rows, key=key)


def test_tags_are_counted_one_by_one_under_the_filters(workflow_records: WorkflowRecordsStorage) -> None:
    workflow_records.create(_workflow("A", tags="alpha, beta"), user_id=USER)
    workflow_records.create(_workflow("B", tags="beta"), user_id=USER, is_public=True)
    workflow_records.create(_workflow("C", tags="alpha"), user_id=OTHER)

    counts = workflow_records.counts_by_tag(["alpha", "beta", "missing"], categories=[WorkflowCategory.User])
    assert counts == {"alpha": 2, "beta": 2, "missing": 0}
    assert workflow_records.counts_by_tag(["alpha", "beta"], categories=[WorkflowCategory.User], user_id=USER) == {
        "alpha": 1,
        "beta": 2,
    }
    assert workflow_records.counts_by_tag(["beta"], categories=[WorkflowCategory.User], is_public=True) == {"beta": 1}
    assert workflow_records.counts_by_tag([]) == {}


def test_counts_and_tags_follow_the_visibility_and_opened_filters(workflow_records: WorkflowRecordsStorage) -> None:
    shared = workflow_records.create(_workflow("S", tags="alpha, shared-only"), user_id=USER, is_public=True)
    workflow_records.create(_workflow("P", tags="alpha"), user_id=USER)
    workflow_records.update_opened_at(shared.workflow_id)

    assert workflow_records.counts_by_category([WorkflowCategory.User], is_public=True) == {"user": 1}
    user = [WorkflowCategory.User]
    assert workflow_records.counts_by_tag(["alpha"], categories=user, has_been_opened=True) == {"alpha": 1}
    assert workflow_records.get_all_tags(categories=[WorkflowCategory.User], is_public=False) == ["alpha"]


def test_categories_are_counted_one_by_one_under_the_filters(workflow_records: WorkflowRecordsStorage) -> None:
    bundled = len(list((Path(workflow_records_default.__file__).parent / "default_workflows").glob("*.json")))
    workflow_records.create(_workflow("Mine"), user_id=USER)
    workflow_records.create(_workflow("Theirs"), user_id=OTHER)

    assert workflow_records.counts_by_category([WorkflowCategory.User, WorkflowCategory.Default]) == {
        "user": 2,
        "default": bundled,
    }
    assert workflow_records.counts_by_category([WorkflowCategory.User], user_id=USER) == {"user": 1}
    assert workflow_records.counts_by_category([WorkflowCategory.User], has_been_opened=True) == {"user": 0}
    assert workflow_records.counts_by_category([]) == {}


def test_all_tags_are_split_trimmed_and_sorted(workflow_records: WorkflowRecordsStorage) -> None:
    workflow_records.create(_workflow("A", tags="beta, alpha"), user_id=USER)
    workflow_records.create(_workflow("B", tags=" alpha ,gamma,"), user_id=USER)
    workflow_records.create(_workflow("C"), user_id=USER)
    workflow_records.create(_workflow("D", tags="theirs"), user_id=OTHER)

    assert workflow_records.get_all_tags(categories=[WorkflowCategory.User], user_id=USER) == ["alpha", "beta", "gamma"]


@pytest.mark.parametrize("write", ["save", "open", "run", "share"])
def test_every_write_renews_when_the_workflow_was_updated(
    workflow_records: WorkflowRecordsStorage, database: Database, write: str
) -> None:
    created = workflow_records.create(_workflow("Written"), user_id=USER)
    _set(database, created.workflow_id, updated_at="2000-01-01 00:00:00.000")

    if write == "save":
        workflow_records.update(Workflow(**_workflow("Written again").model_dump(), id=created.workflow_id))
    elif write == "open":
        workflow_records.update_opened_at(created.workflow_id, user_id=USER)
    elif write == "run":
        workflow_records.update_last_run_at(created.workflow_id, user_id=USER)
    else:
        workflow_records.update_is_public(created.workflow_id, True, user_id=USER)

    assert str(workflow_records.get(created.workflow_id).updated_at) > "2000-01-01 00:00:00.000"


def test_a_flag_passed_as_a_number_filters_as_the_flag(workflow_records: WorkflowRecordsStorage) -> None:
    """Statements are cached by their conditions' shape, in which 1 equals True: a number must not stand for a
    statement without the condition."""
    workflow_queries._list.cache_clear()
    workflow_records.create(_workflow("Shared"), user_id=USER, is_public=True)
    workflow_records.create(_workflow("Private"), user_id=USER)

    assert _names(workflow_records, is_public=1) == ["Shared"]  # type: ignore[arg-type]
    assert _names(workflow_records, is_public=True) == ["Shared"]
    assert _names(workflow_records, is_public=0) == ["Private"]  # type: ignore[arg-type]
    assert _names(workflow_records, has_been_opened=0) == ["Private", "Shared"]  # type: ignore[arg-type]


def test_statement_shapes_stay_few_whatever_a_client_sends(
    workflow_records: WorkflowRecordsStorage, database: Database
) -> None:
    """SQLAlchemy keeps every statement it compiled: how many tags or categories a client sends must not decide how
    many there are."""
    builders = (workflow_queries._list, workflow_queries._count, workflow_queries._tag_counts)
    for builder in builders:
        builder.cache_clear()
    compiled = database.engine._compiled_cache
    assert compiled is not None
    compiled_before = len(compiled)

    for count in range(1, 101):
        tags = [f"tag {i}" for i in range(count)]
        categories = [WorkflowCategory.User] * count
        workflow_records.get_many(WorkflowRecordOrderBy.Name, SQLiteDirection.Ascending, categories, tags=tags)
        workflow_records.counts_by_tag(tags, categories=categories)

    # Tag slots 1, 2, 4 ... 32: six shapes each. A listing for more tags is compiled for its call alone.
    assert [builder.cache_info().currsize for builder in builders] == [6, 6, 6]
    assert len(compiled) - compiled_before <= 3 * 6 + 5
