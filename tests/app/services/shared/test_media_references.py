"""The media reference extractor and index writer shared by project, workflow and migration code."""

import sqlite3

import pytest

from invokeai.app.services.shared.media_references import (
    MediaReferences,
    create_media_references_table,
    delete_media_references,
    extract_media_references,
    replace_media_references,
)


def test_collects_names_under_image_and_video_keys_at_any_depth() -> None:
    document = {
        "layers": [{"image": {"image_name": "layer.png"}}, {"reference": {"imageName": "ref.png"}}],
        "queue": {"items": [{"outputs": [{"imageName": "staged.png"}, {"videoName": "clip.mp4"}]}]},
        "video": {"video_name": "source.mp4", "poster": {"image_name": "poster.png"}},
        "soundtrack": {"source_video_name": "soundtrack-source.mp4"},
        "ignored": {"name": "not-a-ref.png", "image_name": "", "video_name": "x" * 300, "imageName": 12},
    }

    references = extract_media_references(document)

    assert references.images == {"layer.png", "ref.png", "staged.png", "poster.png"}
    assert references.videos == {"clip.mp4", "source.mp4", "soundtrack-source.mp4"}


def test_non_object_documents_reference_nothing() -> None:
    for document in ("image_name", 3, None, ["image_name"], {"nested": [None, 1, "image_name"]}):
        assert extract_media_references(document).is_empty()


@pytest.fixture
def cursor() -> sqlite3.Cursor:
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()
    create_media_references_table(cursor)
    return cursor


def _rows(cursor: sqlite3.Cursor) -> set[tuple[str, str, str, str, str]]:
    cursor.execute("SELECT owner_kind, user_id, owner_id, media_kind, media_name FROM media_references;")
    return set(cursor.fetchall())


def test_replace_makes_the_owner_rows_equal_to_the_new_references(cursor: sqlite3.Cursor) -> None:
    replace_media_references(
        cursor,
        owner_kind="project",
        user_id="u1",
        owner_id="p1",
        references=MediaReferences(images={"old.png", "kept.png"}, videos={"old.mp4"}),
    )
    replace_media_references(
        cursor,
        owner_kind="project",
        user_id="u2",
        owner_id="p1",
        references=MediaReferences(images={"other-account.png"}),
    )

    replace_media_references(
        cursor,
        owner_kind="project",
        user_id="u1",
        owner_id="p1",
        references=MediaReferences(images={"kept.png", "new.png"}),
    )

    # The other account's project with the same id is untouched: project ids are per user.
    assert _rows(cursor) == {
        ("project", "u1", "p1", "image", "kept.png"),
        ("project", "u1", "p1", "image", "new.png"),
        ("project", "u2", "p1", "image", "other-account.png"),
    }


def test_delete_removes_only_that_owner(cursor: sqlite3.Cursor) -> None:
    replace_media_references(
        cursor, owner_kind="project", user_id="u1", owner_id="p1", references=MediaReferences(images={"a.png"})
    )
    replace_media_references(
        cursor, owner_kind="workflow", user_id="u1", owner_id="p1", references=MediaReferences(images={"a.png"})
    )

    delete_media_references(cursor, owner_kind="project", user_id="u1", owner_id="p1")

    assert _rows(cursor) == {("workflow", "u1", "p1", "image", "a.png")}
