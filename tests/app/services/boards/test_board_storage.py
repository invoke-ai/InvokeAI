"""Board records and board memberships on every database backend."""

import pytest
from sqlalchemy import insert, select

from invokeai.app.services.board_image_records.board_image_records_default import BoardImageRecordStorage
from invokeai.app.services.board_records.board_records_common import (
    BoardChanges,
    BoardRecordNotFoundException,
    BoardRecordOrderBy,
    BoardRecordProjectOwnedException,
    BoardVisibility,
)
from invokeai.app.services.board_records.board_records_default import BoardRecordStorage
from invokeai.app.services.board_video_records.board_video_records_default import BoardVideoRecordStorage
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import board_images as board_image_queries
from invokeai.app.services.shared.database.queries import board_videos as board_video_queries
from invokeai.app.services.shared.database.queries import boards as board_queries
from invokeai.app.services.shared.database.schema.boards import board_images as board_images_table
from invokeai.app.services.shared.database.schema.boards import board_videos as board_videos_table
from invokeai.app.services.shared.database.schema.boards import boards as boards_table
from invokeai.app.services.shared.database.schema.boards import shared_boards
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.projects import projects
from invokeai.app.services.shared.database.schema.users import users
from invokeai.app.services.shared.database.schema.videos import videos
from invokeai.app.services.shared.pagination import SQLiteDirection

ALICE = "alice"
BOB = "bob"
CAROL = "carol"


@pytest.fixture
def board_records(database: Database) -> BoardRecordStorage:
    with database.begin(write=True) as conn:
        conn.execute(
            insert(users),
            [{"user_id": user, "email": f"{user}@test.com", "password_hash": "-"} for user in (ALICE, BOB, CAROL)],
        )
    return BoardRecordStorage(database)


@pytest.fixture
def board_images(database: Database) -> BoardImageRecordStorage:
    return BoardImageRecordStorage(database)


@pytest.fixture
def board_videos(database: Database) -> BoardVideoRecordStorage:
    return BoardVideoRecordStorage(database)


def _claim(database: Database, board_id: str, project_id: str = "project") -> None:
    with database.begin(write=True) as conn:
        conn.execute(
            insert(projects).values(project_id=project_id, user_id=ALICE, name="P", data="{}", board_id=board_id)
        )


def _image(
    database: Database,
    name: str,
    *,
    category: ImageCategory = ImageCategory.GENERAL,
    intermediate: bool = False,
    user_id: str = ALICE,
) -> str:
    with database.begin(write=True) as conn:
        conn.execute(
            insert(images).values(
                image_name=name,
                image_origin="internal",
                image_category=category.value,
                width=1,
                height=1,
                is_intermediate=intermediate,
                user_id=user_id,
            )
        )
    return name


def _video(
    database: Database,
    name: str,
    *,
    category: ImageCategory = ImageCategory.GENERAL,
    intermediate: bool = False,
    user_id: str = ALICE,
) -> str:
    with database.begin(write=True) as conn:
        conn.execute(
            insert(videos).values(
                video_name=name,
                video_origin="internal",
                video_category=category.value,
                width=1,
                height=1,
                is_intermediate=intermediate,
                user_id=user_id,
            )
        )
    return name


# region board records


def test_a_saved_board_is_read_back_with_the_project_that_claims_it(
    database: Database, board_records: BoardRecordStorage
) -> None:
    free = board_records.save("Free", ALICE)
    claimed = board_records.save("Claimed", ALICE)
    _claim(database, claimed.board_id)

    assert board_records.get(free.board_id) == free
    assert board_records.get_with_project_id(free.board_id) == (free, None)
    assert board_records.get_with_project_id(claimed.board_id) == (claimed, "project")
    assert board_records.get_project_ids_for_boards([free.board_id, claimed.board_id]) == {claimed.board_id: "project"}
    with pytest.raises(BoardRecordNotFoundException):
        board_records.get("missing")
    with pytest.raises(BoardRecordNotFoundException):
        board_records.get_with_project_id("missing")


def test_project_ids_are_looked_up_in_chunks(
    database: Database, board_records: BoardRecordStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(board_queries, "IN_CHUNK", 2)
    board_ids = [board_records.save(f"Board {index}", ALICE).board_id for index in range(5)]
    for index, board_id in enumerate(board_ids):
        _claim(database, board_id, f"project {index}")

    assert board_records.get_project_ids_for_boards(board_ids) == {
        board_id: f"project {index}" for index, board_id in enumerate(board_ids)
    }


def test_users_list_their_own_shared_and_public_boards(database: Database, board_records: BoardRecordStorage) -> None:
    board_records.save("Own", ALICE)
    shared_with_alice = board_records.save("Shared with Alice", BOB).board_id
    shared_with_carol = board_records.save("Shared with Carol", BOB).board_id
    shared = board_records.save("Shared", BOB).board_id
    public = board_records.save("Public", BOB).board_id
    board_records.save("Private", BOB)
    archived = board_records.save("Archived", ALICE).board_id
    board_records.update(shared, BoardChanges(board_visibility=BoardVisibility.Shared))
    board_records.update(public, BoardChanges(board_visibility=BoardVisibility.Public))
    board_records.update(archived, BoardChanges(archived=True))
    with database.begin(write=True) as conn:
        conn.execute(
            insert(shared_boards),
            [{"board_id": shared_with_alice, "user_id": ALICE}, {"board_id": shared_with_carol, "user_id": CAROL}],
        )

    def listed(*, is_admin: bool, include_archived: bool = False) -> set[str]:
        boards = board_records.get_all(
            ALICE, is_admin, BoardRecordOrderBy.CreatedAt, SQLiteDirection.Descending, include_archived
        )
        page = board_records.get_many(
            ALICE, is_admin, BoardRecordOrderBy.CreatedAt, SQLiteDirection.Descending, 0, 100, include_archived
        )
        assert [board.board_id for board in page.items] == [board.board_id for board in boards]
        assert page.total == len(boards)
        return {board.board_name for board in boards}

    assert listed(is_admin=False) == {"Own", "Shared with Alice", "Shared", "Public"}
    assert listed(is_admin=False, include_archived=True) == {"Own", "Shared with Alice", "Shared", "Public", "Archived"}
    assert listed(is_admin=True) == {"Own", "Shared with Alice", "Shared with Carol", "Shared", "Public", "Private"}
    assert board_records.is_board_shared_with_user(shared_with_carol, CAROL) is True
    assert board_records.is_board_shared_with_user(shared_with_carol, ALICE) is False
    assert board_records.get_shared_user_ids(shared_with_carol) == [CAROL]


def _insert_boards(database: Database, boards: list[tuple[str, str, str]]) -> None:
    """Boards of the first account, as (board_id, board_name, created_at)."""
    with database.begin(write=True) as conn:
        conn.execute(
            insert(boards_table),
            [
                {"board_id": board_id, "board_name": name, "user_id": ALICE, "created_at": created_at}
                for board_id, name, created_at in boards
            ],
        )


def test_boards_list_by_name_ignoring_case(database: Database, board_records: BoardRecordStorage) -> None:
    for name in ("cherry", "Banana", "apple"):
        board_records.save(name, ALICE)

    by_name = board_records.get_all(ALICE, False, BoardRecordOrderBy.Name, SQLiteDirection.Ascending)

    assert [board.board_name for board in by_name] == ["apple", "Banana", "cherry"]


def test_boards_list_by_age(database: Database, board_records: BoardRecordStorage) -> None:
    # Ages that agree with neither the order of insertion nor that of the ids.
    _insert_boards(
        database,
        [
            ("board-b", "second", "2026-01-02 00:00:00.000"),
            ("board-c", "third", "2026-01-01 00:00:00.000"),
            ("board-a", "first", "2026-01-03 00:00:00.000"),
        ],
    )

    by_age = board_records.get_all(ALICE, False, BoardRecordOrderBy.CreatedAt, SQLiteDirection.Descending)

    assert [board.board_id for board in by_age] == ["board-a", "board-b", "board-c"]


def test_boards_of_one_age_list_in_the_order_of_their_ids_on_every_page(
    database: Database, board_records: BoardRecordStorage
) -> None:
    # Inserted out of id order, with names in yet another order.
    _insert_boards(
        database,
        [
            ("board-b", "Banana", "2026-01-01 00:00:00.000"),
            ("board-c", "apple", "2026-01-01 00:00:00.000"),
            ("board-a", "cherry", "2026-01-01 00:00:00.000"),
        ],
    )

    by_age = board_records.get_all(ALICE, False, BoardRecordOrderBy.CreatedAt, SQLiteDirection.Descending)
    pages = [
        board.board_id
        for offset in range(3)
        for board in board_records.get_many(
            ALICE, False, BoardRecordOrderBy.CreatedAt, SQLiteDirection.Descending, offset, 1
        ).items
    ]

    assert [board.board_id for board in by_age] == ["board-c", "board-b", "board-a"]
    assert pages == ["board-c", "board-b", "board-a"]


def test_a_change_keeps_what_it_does_not_name(database: Database, board_records: BoardRecordStorage) -> None:
    board = board_records.save("Board", ALICE).board_id
    _image(database, "cover.png")
    board_records.update(board, BoardChanges(cover_image_name="cover.png"))

    changed = board_records.update(board, BoardChanges(board_name="Renamed", archived=True))

    assert (changed.board_name, changed.archived, changed.cover_image_name) == ("Renamed", True, "cover.png")


def test_a_project_board_keeps_what_its_project_owns(database: Database, board_records: BoardRecordStorage) -> None:
    board = board_records.save("Project", ALICE)
    _claim(database, board.board_id)
    _image(database, "cover.png")

    for changes in (
        BoardChanges(board_name="Renamed"),
        BoardChanges(archived=True),
        BoardChanges(board_visibility=BoardVisibility.Public),
    ):
        with pytest.raises(BoardRecordProjectOwnedException):
            board_records.update(board.board_id, changes)
    updated = board_records.update(board.board_id, BoardChanges(cover_image_name="cover.png"))

    assert (updated.board_name, updated.archived, updated.board_visibility) == (
        "Project",
        False,
        BoardVisibility.Private,
    )
    assert updated.cover_image_name == "cover.png"
    with pytest.raises(BoardRecordNotFoundException):
        board_records.update("missing", BoardChanges(board_name="Renamed"))


def test_only_a_board_no_project_claims_is_deleted(database: Database, board_records: BoardRecordStorage) -> None:
    free = board_records.save("Free", ALICE).board_id
    claimed = board_records.save("Claimed", ALICE).board_id
    _claim(database, claimed)

    assert board_records.delete_if_unclaimed(free) is True
    assert board_records.delete_if_unclaimed(claimed) is False
    with pytest.raises(BoardRecordNotFoundException):
        board_records.get(free)
    assert board_records.get(claimed).board_id == claimed


# endregion

# region board memberships


def test_an_image_is_on_one_board_at_a_time(
    database: Database, board_records: BoardRecordStorage, board_images: BoardImageRecordStorage
) -> None:
    first = board_records.save("First", ALICE).board_id
    second = board_records.save("Second", ALICE).board_id
    image = _image(database, "image.png")

    board_images.add_image_to_board(first, image)
    board_images.add_image_to_board(second, image)

    assert board_images.get_board_for_image(image) == second
    assert board_images.remove_image_from_board(image, first) == 0
    assert board_images.remove_image_from_board(image, second) == 1
    assert board_images.get_board_for_image(image) is None


def test_moving_to_another_board_renews_when_the_membership_was_updated(
    database: Database,
    board_records: BoardRecordStorage,
    board_images: BoardImageRecordStorage,
    board_videos: BoardVideoRecordStorage,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = board_records.save("First", ALICE).board_id
    second = board_records.save("Second", ALICE).board_id
    image = _image(database, "moved.png")
    video = _video(database, "moved.mp4")
    for module in (board_image_queries, board_video_queries):
        monkeypatch.setattr(module, "now_text", lambda: "2030-01-01 00:00:00.000")
    board_images.add_image_to_board(first, image)
    board_videos.add_video_to_board(first, video)

    for module in (board_image_queries, board_video_queries):
        monkeypatch.setattr(module, "now_text", lambda: "2030-01-02 00:00:00.000")
    board_images.add_image_to_board(second, image)
    board_videos.add_video_to_board(second, video)

    with database.begin(write=False) as conn:
        image_row = conn.execute(select(board_images_table.c.board_id, board_images_table.c.updated_at)).one()
        video_row = conn.execute(select(board_videos_table.c.board_id, board_videos_table.c.updated_at)).one()
    assert tuple(image_row) == (second, "2030-01-02 00:00:00.000")
    assert tuple(video_row) == (second, "2030-01-02 00:00:00.000")


def test_image_names_and_counts_follow_the_filters(
    database: Database, board_records: BoardRecordStorage, board_images: BoardImageRecordStorage
) -> None:
    board = board_records.save("Board", ALICE).board_id
    on_board = [
        _image(database, "general.png"),
        _image(database, "control.png", category=ImageCategory.CONTROL),
        _image(database, "mask.png", category=ImageCategory.MASK),
        _image(database, "intermediate.png", intermediate=True),
        _image(database, "bobs.png", user_id=BOB),
    ]
    for name in on_board:
        board_images.add_image_to_board(board, name)
    _image(database, "loose.png")

    def names(board_id: str, **filters: object) -> set[str]:
        filters = {"categories": None, "is_intermediate": None, **filters}
        return set(board_images.get_all_board_image_names_for_board(board_id, **filters))  # type: ignore[arg-type]

    assert names(board) == set(on_board)
    assert names("none") == {"loose.png"}
    assert names(board, categories=[ImageCategory.CONTROL, ImageCategory.MASK]) == {"control.png", "mask.png"}
    assert names(board, is_intermediate=True) == {"intermediate.png"}
    assert names(board, user_id=BOB) == {"bobs.png"}


def test_image_counts_are_per_board_and_per_kind(
    database: Database, board_records: BoardRecordStorage, board_images: BoardImageRecordStorage
) -> None:
    board = board_records.save("Board", ALICE).board_id
    other = board_records.save("Other", ALICE).board_id
    for name, category, intermediate in (
        ("general.png", ImageCategory.GENERAL, False),
        ("control.png", ImageCategory.CONTROL, False),
        ("mask.png", ImageCategory.MASK, False),
        ("user.png", ImageCategory.USER, False),
        ("intermediate.png", ImageCategory.CONTROL, True),
    ):
        board_images.add_image_to_board(board, _image(database, name, category=category, intermediate=intermediate))
    board_images.add_image_to_board(other, _image(database, "elsewhere.png"))

    assert board_images.get_counts_for_board(board) == (1, 3)
    assert board_images.get_counts_for_board(other) == (1, 0)


def test_a_video_is_on_one_board_at_a_time_and_counted_by_category(
    database: Database, board_records: BoardRecordStorage, board_videos: BoardVideoRecordStorage
) -> None:
    first = board_records.save("First", ALICE).board_id
    second = board_records.save("Second", ALICE).board_id
    general = _video(database, "general.mp4")
    asset = _video(database, "asset.mp4", category=ImageCategory.USER)
    intermediate = _video(database, "intermediate.mp4", intermediate=True)
    for name in (general, asset, intermediate):
        board_videos.add_video_to_board(first, name)
    board_videos.add_video_to_board(second, general)

    assert board_videos.get_board_for_video(general) == second
    assert board_videos.get_counts_for_board(first) == (1, 1)  # asset.mp4
    assert board_videos.get_counts_for_board(second) == (1, 0)

    board_videos.remove_video_from_board(general)

    assert board_videos.get_board_for_video(general) is None


def test_video_names_follow_the_filters(
    database: Database, board_records: BoardRecordStorage, board_videos: BoardVideoRecordStorage
) -> None:
    """Deleting a board with its media takes the names from here: the account filter keeps the videos of another
    account out of it."""
    board = board_records.save("Board", ALICE).board_id
    for name, category, intermediate, user in (
        ("general.mp4", ImageCategory.GENERAL, False, ALICE),
        ("asset.mp4", ImageCategory.USER, False, ALICE),
        ("intermediate.mp4", ImageCategory.GENERAL, True, ALICE),
        ("bobs.mp4", ImageCategory.GENERAL, False, BOB),
    ):
        video = _video(database, name, category=category, intermediate=intermediate, user_id=user)
        board_videos.add_video_to_board(board, video)
    _video(database, "alices_loose.mp4")
    _video(database, "bobs_loose.mp4", user_id=BOB)

    def names(board_id: str, **filters: object) -> set[str]:
        filters = {"categories": None, "is_intermediate": None, **filters}
        return set(board_videos.get_all_board_video_names_for_board(board_id, **filters))  # type: ignore[arg-type]

    assert names(board) == {"general.mp4", "asset.mp4", "intermediate.mp4", "bobs.mp4"}
    assert names("none") == {"alices_loose.mp4", "bobs_loose.mp4"}
    assert names("none", user_id=BOB) == {"bobs_loose.mp4"}
    assert names(board, user_id=BOB) == {"bobs.mp4"}
    assert names(board, categories=[ImageCategory.USER]) == {"asset.mp4"}
    assert names(board, is_intermediate=True) == {"intermediate.mp4"}


# endregion
