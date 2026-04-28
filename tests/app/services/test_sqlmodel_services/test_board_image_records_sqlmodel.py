"""Tests for SqlModelBoardImageRecordStorage."""

import pytest

from invokeai.app.services.board_image_records.board_image_records_sqlmodel import SqlModelBoardImageRecordStorage
from invokeai.app.services.board_records.board_records_sqlmodel import SqlModelBoardRecordStorage
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_sqlmodel import SqlModelImageRecordStorage
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


@pytest.fixture
def boards(db: SqliteDatabase) -> SqlModelBoardRecordStorage:
    return SqlModelBoardRecordStorage(db=db)


@pytest.fixture
def images(db: SqliteDatabase) -> SqlModelImageRecordStorage:
    return SqlModelImageRecordStorage(db=db)


@pytest.fixture
def storage(db: SqliteDatabase) -> SqlModelBoardImageRecordStorage:
    return SqlModelBoardImageRecordStorage(db=db)


def _create_image(images: SqlModelImageRecordStorage, name: str, category: ImageCategory = ImageCategory.GENERAL):
    images.save(
        image_name=name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=category,
        width=512,
        height=512,
        has_workflow=False,
        user_id="user1",
    )


def test_add_image_to_board(storage, boards, images):
    board = boards.save("Board", "user1")
    _create_image(images, "img1")
    storage.add_image_to_board(board.board_id, "img1")
    assert storage.get_board_for_image("img1") == board.board_id


def test_remove_image_from_board(storage, boards, images):
    board = boards.save("Board", "user1")
    _create_image(images, "img1")
    storage.add_image_to_board(board.board_id, "img1")
    storage.remove_image_from_board("img1")
    assert storage.get_board_for_image("img1") is None


def test_move_image_between_boards(storage, boards, images):
    board1 = boards.save("Board 1", "user1")
    board2 = boards.save("Board 2", "user1")
    _create_image(images, "img1")
    storage.add_image_to_board(board1.board_id, "img1")
    storage.add_image_to_board(board2.board_id, "img1")
    assert storage.get_board_for_image("img1") == board2.board_id


def test_get_board_for_unassigned_image(storage, images):
    _create_image(images, "img1")
    assert storage.get_board_for_image("img1") is None


def test_get_image_count_for_board(storage, boards, images):
    board = boards.save("Board", "user1")
    _create_image(images, "img1", ImageCategory.GENERAL)
    _create_image(images, "img2", ImageCategory.GENERAL)
    _create_image(images, "img3", ImageCategory.MASK)
    storage.add_image_to_board(board.board_id, "img1")
    storage.add_image_to_board(board.board_id, "img2")
    storage.add_image_to_board(board.board_id, "img3")
    # IMAGE_CATEGORIES = [GENERAL], so count should be 2
    assert storage.get_image_count_for_board(board.board_id) == 2


def test_get_asset_count_for_board(storage, boards, images):
    board = boards.save("Board", "user1")
    _create_image(images, "img1", ImageCategory.GENERAL)
    _create_image(images, "img2", ImageCategory.MASK)
    _create_image(images, "img3", ImageCategory.CONTROL)
    storage.add_image_to_board(board.board_id, "img1")
    storage.add_image_to_board(board.board_id, "img2")
    storage.add_image_to_board(board.board_id, "img3")
    # ASSETS_CATEGORIES = [CONTROL, MASK, USER, OTHER], so count should be 2
    assert storage.get_asset_count_for_board(board.board_id) == 2


def test_get_all_board_image_names(storage, boards, images):
    board = boards.save("Board", "user1")
    _create_image(images, "img1")
    _create_image(images, "img2")
    storage.add_image_to_board(board.board_id, "img1")
    storage.add_image_to_board(board.board_id, "img2")
    names = storage.get_all_board_image_names_for_board(board.board_id, categories=None, is_intermediate=None)
    assert set(names) == {"img1", "img2"}


def test_get_all_board_image_names_uncategorized(storage, images):
    _create_image(images, "img1")
    names = storage.get_all_board_image_names_for_board("none", categories=None, is_intermediate=None)
    assert "img1" in names
