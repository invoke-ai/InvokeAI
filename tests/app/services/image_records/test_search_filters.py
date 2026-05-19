from invokeai.app.services.board_image_records.board_image_records_sqlite import SqliteBoardImageRecordStorage
from invokeai.app.services.board_records.board_records_sqlite import SqliteBoardRecordStorage
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage


def _seed(
    image_records: SqliteImageRecordStorage,
    board_records: SqliteBoardRecordStorage,
    board_images: SqliteBoardImageRecordStorage,
):
    board_1 = board_records.save("Board 1", "system")
    board_2 = board_records.save("Board 2", "system")

    image_records.save(
        image_name="cat_512.png",
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=512,
        height=512,
        has_workflow=False,
        metadata='{"prompt":"cat"}',
    )
    image_records.save(
        image_name="dog_1024.png",
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=1024,
        height=768,
        has_workflow=False,
        starred=True,
        metadata='{"prompt":"dog"}',
    )
    image_records.save(
        image_name="forest_768.png",
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=768,
        height=1024,
        has_workflow=False,
        metadata='{"prompt":"forest"}',
    )

    board_images.add_image_to_board(board_1.board_id, "cat_512.png")
    board_images.add_image_to_board(board_2.board_id, "dog_1024.png")

    return {"board_1_id": board_1.board_id, "board_2_id": board_2.board_id}


def test_get_many_search_filters(mock_invoker):
    db = mock_invoker.services.board_records._db
    image_records = SqliteImageRecordStorage(db)
    board_records = SqliteBoardRecordStorage(db)
    board_images = SqliteBoardImageRecordStorage(db)
    seed = _seed(image_records, board_records, board_images)

    result = image_records.get_many(
        limit=50,
        file_name_term="cat",
        metadata_term="cat",
        width_exact=512,
        height_min=256,
        height_max=700,
        board_ids=[seed["board_1_id"]],
    )

    assert [i.image_name for i in result.items] == ["cat_512.png"]


def test_get_image_names_search_filters(mock_invoker):
    db = mock_invoker.services.board_records._db
    image_records = SqliteImageRecordStorage(db)
    board_records = SqliteBoardRecordStorage(db)
    board_images = SqliteBoardImageRecordStorage(db)
    seed = _seed(image_records, board_records, board_images)

    result = image_records.get_image_names(
        file_name_term=".png",
        width_min=700,
        width_max=1100,
        board_ids=[seed["board_2_id"], "none"],
    )

    assert set(result.image_names) == {"dog_1024.png", "forest_768.png"}


def test_get_many_starred_mode_filter(mock_invoker):
    db = mock_invoker.services.board_records._db
    image_records = SqliteImageRecordStorage(db)
    board_records = SqliteBoardRecordStorage(db)
    board_images = SqliteBoardImageRecordStorage(db)
    _seed(image_records, board_records, board_images)

    only_starred = image_records.get_many(limit=50, starred_mode="only")
    assert [i.image_name for i in only_starred.items] == ["dog_1024.png"]

    exclude_starred = image_records.get_many(limit=50, starred_mode="exclude")
    assert "dog_1024.png" not in [i.image_name for i in exclude_starred.items]
