"""Tests for the image index records service: embedding storage, eligibility, access scoping, projections."""

import numpy as np
import pytest
from sqlalchemy import insert, update

from invokeai.app.services.board_image_records.board_image_records_default import BoardImageRecordStorage
from invokeai.app.services.board_records.board_records_common import BoardChanges, BoardVisibility
from invokeai.app.services.board_records.board_records_default import BoardRecordStorage
from invokeai.app.services.board_video_records.board_video_records_default import BoardVideoRecordStorage
from invokeai.app.services.image_index.image_index_common import (
    IndexedItem,
    blob_to_coords,
    blob_to_embedding,
    coords_to_blob,
    embedding_to_blob,
)
from invokeai.app.services.image_index.image_index_records_default import ImageIndexRecords
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries.locks import DatabaseLock
from invokeai.app.services.shared.database.schema.boards import shared_boards
from invokeai.app.services.shared.database.schema.image_index import image_embeddings, image_projections
from invokeai.app.services.shared.database.schema.images import images
from invokeai.app.services.shared.database.schema.videos import videos
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_default import UserService
from invokeai.app.services.video_records.video_records_default import VideoRecordStorage
from tests.fixtures.races import while_in_flight

SYSTEM_USER_ID = "system"
MODEL_ID = "model-hash-1"
OTHER_MODEL_ID = "model-hash-2"
DIM = 8


@pytest.fixture
def image_records(database: Database) -> ImageRecordStorage:
    return ImageRecordStorage(database)


@pytest.fixture
def board_records(database: Database) -> BoardRecordStorage:
    return BoardRecordStorage(database)


@pytest.fixture
def board_image_records(database: Database) -> BoardImageRecordStorage:
    return BoardImageRecordStorage(database)


@pytest.fixture
def video_records(database: Database) -> VideoRecordStorage:
    return VideoRecordStorage(database)


@pytest.fixture
def board_video_records(database: Database) -> BoardVideoRecordStorage:
    return BoardVideoRecordStorage(database)


@pytest.fixture
def index_records(database: Database) -> ImageIndexRecords:
    return ImageIndexRecords(database)


@pytest.fixture
def other_user_id(database: Database) -> str:
    users = UserService(database)
    user = users.create(
        UserCreateRequest(email="other@example.com", display_name="Other", password="TestPass123", is_admin=False)
    )
    return user.user_id


def _save_image(
    image_records: ImageRecordStorage,
    image_name: str,
    user_id: str = SYSTEM_USER_ID,
    is_intermediate: bool = False,
    image_category: ImageCategory = ImageCategory.GENERAL,
) -> None:
    image_records.save(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=image_category,
        width=64,
        height=64,
        has_workflow=False,
        is_intermediate=is_intermediate,
        user_id=user_id,
    )


def _save_video(
    video_records: VideoRecordStorage,
    video_name: str,
    user_id: str = SYSTEM_USER_ID,
    is_intermediate: bool = False,
    video_category: ImageCategory = ImageCategory.GENERAL,
) -> None:
    video_records.save(
        video_name=video_name,
        video_origin=ResourceOrigin.INTERNAL,
        video_category=video_category,
        width=64,
        height=64,
        duration=2.0,
        fps=24.0,
        has_workflow=False,
        is_intermediate=is_intermediate,
        user_id=user_id,
    )


def _set_created_at(database: Database, kind: str, name: str, created_at: str) -> None:
    media, column = (images, "image_name") if kind == "image" else (videos, "video_name")
    with database.begin(write=True) as conn:
        conn.execute(update(media).where(media.c[column] == name).values(created_at=created_at))


def _share_board(database: Database, board_id: str, user_id: str) -> None:
    """No service writes shared_boards yet."""
    with database.begin(write=True) as conn:
        conn.execute(insert(shared_boards).values(board_id=board_id, user_id=user_id, can_edit=False))


def imgs(*names: str) -> list[IndexedItem]:
    """The image-namespace items for these names."""
    return [IndexedItem("image", name) for name in names]


def vids(*names: str) -> list[IndexedItem]:
    """The video-namespace items for these names."""
    return [IndexedItem("video", name) for name in names]


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v / np.linalg.norm(v)


# --- Blob helpers ---


def test_embedding_blob_roundtrip_is_bit_exact() -> None:
    v = _vec(1)
    assert np.array_equal(blob_to_embedding(embedding_to_blob(v), DIM), v)


def test_embedding_blob_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        embedding_to_blob(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        blob_to_embedding(embedding_to_blob(_vec(1)), DIM + 1)


def test_embedding_blob_rejects_non_finite_values() -> None:
    # A NaN/inf row poisons every similarity and projection computation it later lands in, and
    # cannot be attributed after the fact — so it must be refused at the boundary.
    for bad in (np.nan, np.inf, -np.inf):
        v = _vec(1)
        v[0] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            embedding_to_blob(v)

    # A float64 magnitude that overflows when narrowed to float32 is the same defect, arriving
    # via dtype conversion rather than as a literal inf.
    with pytest.raises(ValueError, match="NaN or infinite"):
        embedding_to_blob(np.array([1e40] * DIM, dtype=np.float64))


def test_embedding_blob_rejects_zero_length() -> None:
    # dim=0 stores cleanly but then fails every batch it appears in, because get_embeddings
    # requires one consistent dim across the result set.
    with pytest.raises(ValueError, match="zero-length"):
        embedding_to_blob(np.zeros(0, dtype=np.float32))


def test_embedding_blob_rejects_all_zero_vector() -> None:
    # Not L2-normalizable, and it yields NaN in every cosine similarity it takes part in.
    with pytest.raises(ValueError, match="all-zero"):
        embedding_to_blob(np.zeros(DIM, dtype=np.float32))

    # Same defect arriving by underflow: float64 components too small to survive the narrowing.
    with pytest.raises(ValueError, match="all-zero"):
        embedding_to_blob(np.full(DIM, 1e-320, dtype=np.float64))


def test_embedding_blob_rejects_non_floating_dtypes() -> None:
    # These must raise the documented ValueError, not TypeError from the cast or a silent
    # coercion that stores meaningless numbers.
    for bad in (
        np.zeros(2, dtype=[("a", "f4"), ("b", "f4")]),
        np.array([1 + 1j, 2 + 0j], dtype=np.complex128),
        np.arange(DIM, dtype=np.int64),
        np.ones(DIM, dtype=bool),
    ):
        with pytest.raises(ValueError, match="floating-point"):
            embedding_to_blob(bad)


def test_embedding_blob_survives_global_numpy_error_state() -> None:
    # A process-wide np.seterr must not turn the documented ValueError into FloatingPointError.
    old = np.seterr(all="raise")
    try:
        with pytest.raises(ValueError, match="NaN or infinite"):
            embedding_to_blob(np.array([1e40] * DIM, dtype=np.float64))
        with pytest.raises(ValueError, match="all-zero"):
            embedding_to_blob(np.full(DIM, 1e-320, dtype=np.float64))
    finally:
        np.seterr(**old)


def test_embedding_blob_narrows_float64_input() -> None:
    v64 = (np.arange(DIM, dtype=np.float64) + 1.0) / 10.0
    assert np.array_equal(blob_to_embedding(embedding_to_blob(v64), DIM), v64.astype(np.float32))


def test_coords_blob_roundtrip_and_validation() -> None:
    coords = np.arange(10, dtype=np.float32).reshape(5, 2)
    assert np.array_equal(blob_to_coords(coords_to_blob(coords), 5), coords)
    with pytest.raises(ValueError):
        coords_to_blob(np.zeros((5, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        blob_to_coords(coords_to_blob(coords), 4)


def test_coords_from_blob_are_writable() -> None:
    coords = blob_to_coords(coords_to_blob(np.zeros((2, 2), dtype=np.float32)), 2)
    coords[0, 0] = 5.0  # must not raise
    assert coords[0, 0] == 5.0


# --- Embedding CRUD ---


def test_upsert_and_get_roundtrip(image_records: ImageRecordStorage, index_records: ImageIndexRecords) -> None:
    _save_image(image_records, "a.png")
    _save_image(image_records, "b.png")
    va, vb = _vec(1), _vec(2)
    index_records.upsert_embedding(IndexedItem("image", "a.png"), MODEL_ID, va)
    index_records.upsert_embedding(IndexedItem("image", "b.png"), MODEL_ID, vb)

    names, matrix = index_records.get_embeddings(imgs("b.png", "a.png", "missing.png"), MODEL_ID)

    assert names == imgs("b.png", "a.png")
    assert matrix.dtype == np.float32
    assert np.array_equal(matrix[0], vb)
    assert np.array_equal(matrix[1], va)


def test_upsert_replaces_existing_embedding(
    image_records: ImageRecordStorage, index_records: ImageIndexRecords
) -> None:
    _save_image(image_records, "a.png")
    index_records.upsert_embedding(IndexedItem("image", "a.png"), MODEL_ID, _vec(1))
    replacement = _vec(99)
    index_records.upsert_embedding(IndexedItem("image", "a.png"), MODEL_ID, replacement)

    names, matrix = index_records.get_embeddings(imgs("a.png"), MODEL_ID)
    assert names == imgs("a.png")
    assert np.array_equal(matrix[0], replacement)


def test_get_embeddings_empty_input_and_no_matches(
    image_records: ImageRecordStorage, index_records: ImageIndexRecords
) -> None:
    names, matrix = index_records.get_embeddings([], MODEL_ID)
    assert names == []
    assert matrix.shape == (0, 0)

    _save_image(image_records, "a.png")
    index_records.upsert_embedding(IndexedItem("image", "a.png"), MODEL_ID, _vec(1))
    names, matrix = index_records.get_embeddings(imgs("a.png"), OTHER_MODEL_ID)
    assert names == []
    assert matrix.shape == (0, 0)


def test_get_embeddings_returns_a_large_request_in_the_callers_order(
    image_records: ImageRecordStorage, video_records: VideoRecordStorage, index_records: ImageIndexRecords
) -> None:
    # More names than an IN list would bind, of both kinds, requested interleaved and against the order the
    # database reads them in: the rows must come back aligned with the request.
    count = 600
    requested = []
    for i in range(count):
        if i % 2:
            _save_video(video_records, f"item-{i:04d}.mp4")
            requested.append(IndexedItem("video", f"item-{i:04d}.mp4"))
        else:
            _save_image(image_records, f"item-{i:04d}.png")
            requested.append(IndexedItem("image", f"item-{i:04d}.png"))
        index_records.upsert_embedding(requested[-1], MODEL_ID, _vec(i))
    requested.reverse()

    names, matrix = index_records.get_embeddings(requested, MODEL_ID)

    assert names == requested
    assert matrix.shape == (count, DIM)
    assert np.array_equal(matrix[0], _vec(count - 1))
    assert np.array_equal(matrix[1], _vec(count - 2))
    assert np.array_equal(matrix[-1], _vec(0))


def test_get_embeddings_rejects_inconsistent_dims(
    image_records: ImageRecordStorage, index_records: ImageIndexRecords, database: Database
) -> None:
    # The ABC promises a failure on mixed dims under one model_id. Write a short vector behind
    # the service's back, since upsert_embedding alone cannot produce the inconsistency.
    _save_image(image_records, "a.png")
    _save_image(image_records, "b.png")
    index_records.upsert_embedding(IndexedItem("image", "a.png"), MODEL_ID, _vec(1))
    short = np.ones(DIM // 2, dtype=np.float32)
    with database.begin(write=True) as conn:
        conn.execute(
            insert(image_embeddings).values(
                image_name="b.png", model_id=MODEL_ID, dim=DIM // 2, embedding=embedding_to_blob(short)
            )
        )

    with pytest.raises(ValueError, match="Inconsistent embedding dims"):
        index_records.get_embeddings(imgs("a.png", "b.png"), MODEL_ID)


def test_get_embeddings_deduplicates_input_names(
    image_records: ImageRecordStorage, index_records: ImageIndexRecords
) -> None:
    _save_image(image_records, "a.png")
    index_records.upsert_embedding(IndexedItem("image", "a.png"), MODEL_ID, _vec(1))

    names, matrix = index_records.get_embeddings(imgs("a.png", "a.png"), MODEL_ID)

    assert names == imgs("a.png")
    assert matrix.shape == (1, DIM)


def test_upsert_embedding_for_deleted_image_is_noop(index_records: ImageIndexRecords) -> None:
    # The image was deleted (or never existed) by the time the write lands;
    # the foreign key must not fail the write.
    index_records.upsert_embedding(IndexedItem("image", "gone.png"), MODEL_ID, _vec(1))
    assert index_records.get_embeddings(imgs("gone.png"), MODEL_ID)[0] == []


def test_set_projection_for_deleted_user_is_noop(index_records: ImageIndexRecords) -> None:
    index_records.set_projection(
        "no-such-user", MODEL_ID, "hash-1", "{}", imgs("a.png"), np.zeros((1, 2), dtype=np.float32)
    )
    assert index_records.get_projection("no-such-user", MODEL_ID) is None


def test_a_skipped_write_leaves_the_next_one_working(
    image_records: ImageRecordStorage, index_records: ImageIndexRecords
) -> None:
    # The missing parent is checked before the write, not provoked as a foreign-key error and swallowed.
    index_records.upsert_embedding(IndexedItem("image", "gone.png"), MODEL_ID, _vec(1))
    index_records.set_projection(
        "no-such-user", MODEL_ID, "hash-1", "{}", imgs("a.png"), np.zeros((1, 2), dtype=np.float32)
    )

    _save_image(image_records, "real.png")
    index_records.upsert_embedding(IndexedItem("image", "real.png"), MODEL_ID, _vec(2))
    assert index_records.get_embeddings(imgs("real.png"), MODEL_ID)[0] == imgs("real.png")


def test_an_image_deleted_while_its_embedding_is_written_gets_none(
    database: Database, image_records: ImageRecordStorage, index_records: ImageIndexRecords
) -> None:
    # On a server the image's deletion is in flight when the embedding is written: the write waits for it and then
    # finds no image, rather than failing on the foreign key.
    _save_image(image_records, "a.png")

    errors = while_in_flight(
        database,
        lambda q: q.images.delete("a.png"),
        lambda: index_records.upsert_embedding(IndexedItem("image", "a.png"), MODEL_ID, _vec(1)),
    )

    assert errors == []
    assert index_records.get_embeddings(imgs("a.png"), MODEL_ID)[0] == []


def test_an_account_deleted_while_its_projection_is_written_gets_none(
    database: Database, index_records: ImageIndexRecords, other_user_id: str
) -> None:
    errors = while_in_flight(
        database,
        lambda q: q.users.delete(other_user_id),
        lambda: index_records.set_projection(
            other_user_id, MODEL_ID, "hash-1", "{}", imgs("a.png"), np.zeros((1, 2), dtype=np.float32)
        ),
    )

    assert errors == []
    assert index_records.get_projection(other_user_id, MODEL_ID) is None


def test_delete_embedding_removes_all_models(
    image_records: ImageRecordStorage, index_records: ImageIndexRecords
) -> None:
    _save_image(image_records, "a.png")
    index_records.upsert_embedding(IndexedItem("image", "a.png"), MODEL_ID, _vec(1))
    index_records.upsert_embedding(IndexedItem("image", "a.png"), OTHER_MODEL_ID, _vec(2))

    index_records.delete_embedding(IndexedItem("image", "a.png"))

    assert index_records.get_embeddings(imgs("a.png"), MODEL_ID)[0] == []
    assert index_records.get_embeddings(imgs("a.png"), OTHER_MODEL_ID)[0] == []


def test_image_delete_cascades_to_embeddings(
    database: Database, image_records: ImageRecordStorage, index_records: ImageIndexRecords
) -> None:
    _save_image(image_records, "a.png")
    index_records.upsert_embedding(IndexedItem("image", "a.png"), MODEL_ID, _vec(1))

    image_records.delete("a.png")

    assert index_records.get_embeddings(imgs("a.png"), MODEL_ID)[0] == []


def test_delete_embeddings_for_other_models(
    image_records: ImageRecordStorage, index_records: ImageIndexRecords
) -> None:
    _save_image(image_records, "a.png")
    index_records.upsert_embedding(IndexedItem("image", "a.png"), MODEL_ID, _vec(1))
    index_records.upsert_embedding(IndexedItem("image", "a.png"), OTHER_MODEL_ID, _vec(2))

    deleted = index_records.delete_embeddings_for_other_models(MODEL_ID)

    assert deleted == 1
    assert index_records.get_embeddings(imgs("a.png"), MODEL_ID)[0] == imgs("a.png")
    assert index_records.get_embeddings(imgs("a.png"), OTHER_MODEL_ID)[0] == []


# --- Eligibility: backfill listing and status counts ---


def test_list_unembedded_skips_ineligible_and_embedded(
    image_records: ImageRecordStorage, index_records: ImageIndexRecords
) -> None:
    _save_image(image_records, "eligible.png")
    _save_image(image_records, "embedded.png")
    _save_image(image_records, "intermediate.png", is_intermediate=True)
    _save_image(image_records, "mask.png", image_category=ImageCategory.MASK)
    index_records.upsert_embedding(IndexedItem("image", "embedded.png"), MODEL_ID, _vec(1))

    unembedded = index_records.list_unembedded_items(MODEL_ID, limit=10)

    assert unembedded == imgs("eligible.png")


def test_list_unembedded_respects_limit_and_returns_oldest_first(
    image_records: ImageRecordStorage, index_records: ImageIndexRecords, database: Database
) -> None:
    # `created_at` has millisecond resolution, so five back-to-back saves almost always tie and
    # the `image_name ASC` tie-break alone decides the result — which would let a reversed
    # `created_at` ordering pass unnoticed. Stamp distinct timestamps in the *reverse* of
    # alphabetical order so the two orderings disagree and only `created_at` can satisfy this.
    for i in range(5):
        _save_image(image_records, f"img-{i}.png")
        _set_created_at(database, "image", f"img-{i}.png", f"2026-01-0{5 - i} 00:00:00.000")

    batch = index_records.list_unembedded_items(MODEL_ID, limit=3)

    # Which three matters, not just how many: backfill walks oldest-first, and a reversed order
    # would silently re-scan the newest images forever while the oldest never got embedded.
    assert batch == imgs("img-4.png", "img-3.png", "img-2.png")


def test_an_embedding_under_another_model_leaves_an_item_to_embed(
    image_records: ImageRecordStorage, video_records: VideoRecordStorage, index_records: ImageIndexRecords
) -> None:
    # After a model switch every item is work again, until the old model's rows are deleted.
    _save_image(image_records, "a.png")
    _save_video(video_records, "clip.mp4")
    index_records.upsert_embedding(IndexedItem("image", "a.png"), OTHER_MODEL_ID, _vec(1))
    index_records.upsert_embedding(IndexedItem("video", "clip.mp4"), OTHER_MODEL_ID, _vec(2))

    assert index_records.list_unembedded_items(MODEL_ID, limit=10) == [
        IndexedItem("image", "a.png"),
        IndexedItem("video", "clip.mp4"),
    ]
    status = index_records.count_index_status(MODEL_ID)
    assert (status.total, status.embedded) == (2, 0)


def test_list_unembedded_rejects_negative_limit(index_records: ImageIndexRecords) -> None:
    # SQLite reads a negative LIMIT as unbounded, which would turn a bounded backfill batch into
    # a full-table load.
    with pytest.raises(ValueError, match="non-negative"):
        index_records.list_unembedded_items(MODEL_ID, limit=-1)


def test_list_unembedded_zero_limit_returns_nothing(
    image_records: ImageRecordStorage, index_records: ImageIndexRecords
) -> None:
    _save_image(image_records, "a.png")
    assert index_records.list_unembedded_items(MODEL_ID, limit=0) == []


def test_count_index_status(image_records: ImageRecordStorage, index_records: ImageIndexRecords) -> None:
    _save_image(image_records, "a.png")
    _save_image(image_records, "b.png")
    _save_image(image_records, "intermediate.png", is_intermediate=True)
    index_records.upsert_embedding(IndexedItem("image", "a.png"), MODEL_ID, _vec(1))

    status = index_records.count_index_status(MODEL_ID)

    assert status.total == 2
    assert status.embedded == 1
    assert status.pending == 1


# --- Access scoping ---


def test_accessible_images_scoping(
    image_records: ImageRecordStorage,
    board_records: BoardRecordStorage,
    board_image_records: BoardImageRecordStorage,
    index_records: ImageIndexRecords,
    other_user_id: str,
) -> None:
    # System user's images: one unboarded, one on a private board, one shared, one public.
    for seed, name in enumerate(["own-unboarded.png", "own-private.png", "own-shared.png", "own-public.png"]):
        _save_image(image_records, name, user_id=SYSTEM_USER_ID)
        index_records.upsert_embedding(IndexedItem("image", name), MODEL_ID, _vec(seed))
    # Other user's unboarded image.
    _save_image(image_records, "theirs-unboarded.png", user_id=other_user_id)
    index_records.upsert_embedding(IndexedItem("image", "theirs-unboarded.png"), MODEL_ID, _vec(5))
    # An intermediate image never shows up even for its owner.
    _save_image(image_records, "own-intermediate.png", user_id=SYSTEM_USER_ID, is_intermediate=True)

    private_board = board_records.save("Private", SYSTEM_USER_ID).board_id
    shared_board = board_records.save("Shared", SYSTEM_USER_ID).board_id
    public_board = board_records.save("Public", SYSTEM_USER_ID).board_id
    board_records.update(shared_board, BoardChanges(board_visibility=BoardVisibility.Shared))
    board_records.update(public_board, BoardChanges(board_visibility=BoardVisibility.Public))
    board_image_records.add_image_to_board(private_board, "own-private.png")
    board_image_records.add_image_to_board(shared_board, "own-shared.png")
    board_image_records.add_image_to_board(public_board, "own-public.png")

    # Owner sees their unboarded image and everything on active boards they own.
    assert index_records.list_accessible_embedded_items(SYSTEM_USER_ID, MODEL_ID) == sorted(
        imgs("own-unboarded.png", "own-private.png", "own-shared.png", "own-public.png")
    )

    # The other user sees their own image plus shared/public board images — never the
    # system user's private-board or unboarded images.
    assert index_records.list_accessible_embedded_items(other_user_id, MODEL_ID) == sorted(
        imgs("theirs-unboarded.png", "own-shared.png", "own-public.png")
    )

    # Admin scope (None) sees everything embedded.
    assert index_records.list_accessible_embedded_items(None, MODEL_ID) == sorted(
        imgs(
            "own-unboarded.png",
            "own-private.png",
            "own-shared.png",
            "own-public.png",
            "theirs-unboarded.png",
        )
    )


def test_accessible_images_includes_individually_shared_boards(
    database: Database,
    image_records: ImageRecordStorage,
    board_records: BoardRecordStorage,
    board_image_records: BoardImageRecordStorage,
    index_records: ImageIndexRecords,
    other_user_id: str,
) -> None:
    # A private board individually shared with other_user via shared_boards
    # must expose its images to them — mirroring the board-listing access
    # model. No service writes shared_boards yet, so insert the row directly.
    _save_image(image_records, "own-individually-shared.png", user_id=SYSTEM_USER_ID)
    index_records.upsert_embedding(IndexedItem("image", "own-individually-shared.png"), MODEL_ID, _vec(1))
    board_id = board_records.save("Individually shared", SYSTEM_USER_ID).board_id
    board_image_records.add_image_to_board(board_id, "own-individually-shared.png")
    _share_board(database, board_id, other_user_id)

    assert index_records.list_accessible_embedded_items(other_user_id, MODEL_ID) == imgs("own-individually-shared.png")
    # A third party without the share still cannot see it.
    users = UserService(database)
    third_user = users.create(
        UserCreateRequest(email="third@example.com", display_name="Third", password="TestPass123", is_admin=False)
    )
    assert index_records.list_accessible_embedded_items(third_user.user_id, MODEL_ID) == []


def test_accessible_images_includes_boards_owned_by_user(
    image_records: ImageRecordStorage,
    board_records: BoardRecordStorage,
    board_image_records: BoardImageRecordStorage,
    index_records: ImageIndexRecords,
    other_user_id: str,
) -> None:
    # An image uploaded by another user onto a board the system user OWNS is
    # accessible to the board owner — matching the gallery "all" listing
    # (image_records_default), which grants access via boards.user_id even
    # without shared/public visibility or a shared_boards row.
    _save_image(image_records, "theirs-on-my-board.png", user_id=other_user_id)
    index_records.upsert_embedding(IndexedItem("image", "theirs-on-my-board.png"), MODEL_ID, _vec(1))
    my_board = board_records.save("Mine", SYSTEM_USER_ID).board_id
    board_image_records.add_image_to_board(my_board, "theirs-on-my-board.png")

    assert index_records.list_accessible_embedded_items(SYSTEM_USER_ID, MODEL_ID) == imgs("theirs-on-my-board.png")
    # The uploader placed it on a private board they neither own nor were
    # granted; like the gallery "all" listing, they no longer see it.
    assert index_records.list_accessible_embedded_items(other_user_id, MODEL_ID) == []


def test_accessible_images_excludes_archived_boards(
    image_records: ImageRecordStorage,
    board_records: BoardRecordStorage,
    board_image_records: BoardImageRecordStorage,
    index_records: ImageIndexRecords,
    other_user_id: str,
) -> None:
    # Archived boards hide their images from every scope, mirroring the
    # gallery "all" listing: even the owner and the administrative scope.
    _save_image(image_records, "own-archived.png", user_id=SYSTEM_USER_ID)
    _save_image(image_records, "own-unboarded.png", user_id=SYSTEM_USER_ID)
    _save_image(image_records, "shared-archived.png", user_id=SYSTEM_USER_ID)
    for seed, name in enumerate(["own-archived.png", "own-unboarded.png", "shared-archived.png"]):
        index_records.upsert_embedding(IndexedItem("image", name), MODEL_ID, _vec(seed))

    archived_board = board_records.save("Archived", SYSTEM_USER_ID).board_id
    archived_shared_board = board_records.save("Archived shared", SYSTEM_USER_ID).board_id
    board_records.update(archived_shared_board, BoardChanges(board_visibility=BoardVisibility.Shared))
    board_image_records.add_image_to_board(archived_board, "own-archived.png")
    board_image_records.add_image_to_board(archived_shared_board, "shared-archived.png")
    board_records.update(archived_board, BoardChanges(archived=True))
    board_records.update(archived_shared_board, BoardChanges(archived=True))

    # Owner: archived-board images are hidden; unboarded images are unaffected.
    assert index_records.list_accessible_embedded_items(SYSTEM_USER_ID, MODEL_ID) == imgs("own-unboarded.png")
    # An archived shared board grants nothing to other users either.
    assert index_records.list_accessible_embedded_items(other_user_id, MODEL_ID) == []
    # The administrative scope also hides archived-board images.
    assert index_records.list_accessible_embedded_items(None, MODEL_ID) == imgs("own-unboarded.png")


def test_accessible_images_returns_boarded_images_once(
    image_records: ImageRecordStorage,
    board_records: BoardRecordStorage,
    board_image_records: BoardImageRecordStorage,
    index_records: ImageIndexRecords,
) -> None:
    # `board_images` has PRIMARY KEY (image_name), so an image is on at most one board and the
    # board join cannot multiply rows. This pins that the join stays single-valued: if
    # board_images ever became many-to-many, the result would duplicate and the scope hash
    # (derived from this list) would change without the accessible set changing.
    _save_image(image_records, "a.png")
    index_records.upsert_embedding(IndexedItem("image", "a.png"), MODEL_ID, _vec(1))
    board_a = board_records.save("A", SYSTEM_USER_ID).board_id
    board_image_records.add_image_to_board(board_a, "a.png")

    assert index_records.list_accessible_embedded_items(SYSTEM_USER_ID, MODEL_ID) == imgs("a.png")


def test_an_embedded_item_that_left_the_gallery_is_not_accessible(
    database: Database,
    image_records: ImageRecordStorage,
    video_records: VideoRecordStorage,
    index_records: ImageIndexRecords,
) -> None:
    # Nothing deletes an embedding when its item becomes an intermediate or leaves the general category, so the
    # listing itself must keep such items off the map and out of search, in every scope.
    for name in ("kept.png", "now-intermediate.png", "now-mask.png"):
        _save_image(image_records, name)
        index_records.upsert_embedding(IndexedItem("image", name), MODEL_ID, _vec(len(name)))
    _save_video(video_records, "now-intermediate.mp4")
    index_records.upsert_embedding(IndexedItem("video", "now-intermediate.mp4"), MODEL_ID, _vec(1))
    with database.begin(write=True) as conn:
        conn.execute(update(images).where(images.c.image_name == "now-intermediate.png").values(is_intermediate=True))
        conn.execute(update(images).where(images.c.image_name == "now-mask.png").values(image_category="mask"))
        conn.execute(update(videos).where(videos.c.video_name == "now-intermediate.mp4").values(is_intermediate=True))

    for user_id in (SYSTEM_USER_ID, None):
        assert index_records.list_accessible_embedded_items(user_id, MODEL_ID) == imgs("kept.png")


def test_accessible_images_are_filtered_by_model_id(
    image_records: ImageRecordStorage,
    board_records: BoardRecordStorage,
    board_image_records: BoardImageRecordStorage,
    index_records: ImageIndexRecords,
    other_user_id: str,
) -> None:
    # The scope hash is derived from this listing while the embedding matrix is fetched by
    # model_id. If the listing ignored model_id, a model switch would put images into the scope
    # whose embeddings get_embeddings cannot return, desynchronizing names from coordinates.
    _save_image(image_records, "current.png")
    _save_image(image_records, "stale.png")
    index_records.upsert_embedding(IndexedItem("image", "current.png"), MODEL_ID, _vec(1))
    index_records.upsert_embedding(IndexedItem("image", "stale.png"), OTHER_MODEL_ID, _vec(2))

    shared = board_records.save("Shared", SYSTEM_USER_ID).board_id
    board_records.update(shared, BoardChanges(board_visibility=BoardVisibility.Shared))
    board_image_records.add_image_to_board(shared, "current.png")
    board_image_records.add_image_to_board(shared, "stale.png")

    # Every scope must apply the filter, not just the owner's.
    assert index_records.list_accessible_embedded_items(SYSTEM_USER_ID, MODEL_ID) == imgs("current.png")
    assert index_records.list_accessible_embedded_items(other_user_id, MODEL_ID) == imgs("current.png")
    assert index_records.list_accessible_embedded_items(None, MODEL_ID) == imgs("current.png")
    assert index_records.list_accessible_embedded_items(None, OTHER_MODEL_ID) == imgs("stale.png")


def test_accessible_images_exclude_individually_shared_archived_board(
    image_records: ImageRecordStorage,
    board_records: BoardRecordStorage,
    board_image_records: BoardImageRecordStorage,
    index_records: ImageIndexRecords,
    database: Database,
    other_user_id: str,
) -> None:
    # Archiving is tested for visibility-shared boards; the shared_boards grant is a separate
    # branch of the access clause and must be archived-gated too.
    _save_image(image_records, "granted.png")
    index_records.upsert_embedding(IndexedItem("image", "granted.png"), MODEL_ID, _vec(1))
    board_id = board_records.save("Private", SYSTEM_USER_ID).board_id
    board_image_records.add_image_to_board(board_id, "granted.png")
    _share_board(database, board_id, other_user_id)

    assert index_records.list_accessible_embedded_items(other_user_id, MODEL_ID) == imgs("granted.png")

    board_records.update(board_id, BoardChanges(archived=True))

    assert index_records.list_accessible_embedded_items(other_user_id, MODEL_ID) == []


# --- Projections ---


def test_projection_roundtrip(index_records: ImageIndexRecords) -> None:
    coords = np.array([[0.5, -1.5], [2.0, 3.0]], dtype=np.float32)
    index_records.set_projection(
        SYSTEM_USER_ID, MODEL_ID, "hash-1", '{"n_neighbors": 15}', imgs("a.png", "b.png"), coords
    )

    record = index_records.get_projection(SYSTEM_USER_ID, MODEL_ID)

    assert record is not None
    assert record.scope_hash == "hash-1"
    assert record.params == '{"n_neighbors": 15}'
    assert record.point_count == 2
    assert record.items == imgs("a.png", "b.png")
    assert np.array_equal(record.coords, coords)


def test_projection_upsert_replaces(index_records: ImageIndexRecords) -> None:
    index_records.set_projection(
        SYSTEM_USER_ID, MODEL_ID, "hash-1", "{}", imgs("a.png"), np.zeros((1, 2), dtype=np.float32)
    )
    index_records.set_projection(
        SYSTEM_USER_ID, MODEL_ID, "hash-2", '{"n": 2}', imgs("a.png", "b.png"), np.ones((2, 2), dtype=np.float32)
    )

    record = index_records.get_projection(SYSTEM_USER_ID, MODEL_ID)

    assert record is not None
    assert record.scope_hash == "hash-2"
    assert record.params == '{"n": 2}'
    assert record.point_count == 2
    assert record.items == imgs("a.png", "b.png")
    assert np.array_equal(record.coords, np.ones((2, 2), dtype=np.float32))


def test_replacing_a_projection_stamps_when_and_keeps_since_when(
    database: Database, index_records: ImageIndexRecords
) -> None:
    index_records.set_projection(
        SYSTEM_USER_ID, MODEL_ID, "hash-1", "{}", imgs("a.png"), np.zeros((1, 2), dtype=np.float32)
    )
    old = "2020-01-01 00:00:00.000"
    with database.begin(write=True) as conn:
        conn.execute(update(image_projections).values(created_at=old, updated_at=old))

    index_records.set_projection(
        SYSTEM_USER_ID, MODEL_ID, "hash-2", "{}", imgs("a.png"), np.zeros((1, 2), dtype=np.float32)
    )

    record = index_records.get_projection(SYSTEM_USER_ID, MODEL_ID)
    assert record is not None
    assert record.created_at == old
    assert record.updated_at > old


def test_projection_missing_and_delete_idempotent(index_records: ImageIndexRecords) -> None:
    assert index_records.get_projection(SYSTEM_USER_ID, MODEL_ID) is None
    index_records.delete_projection(SYSTEM_USER_ID, MODEL_ID)  # no-op

    index_records.set_projection(
        SYSTEM_USER_ID, MODEL_ID, "hash-1", "{}", imgs("a.png"), np.zeros((1, 2), dtype=np.float32)
    )
    index_records.delete_projection(SYSTEM_USER_ID, MODEL_ID)
    assert index_records.get_projection(SYSTEM_USER_ID, MODEL_ID) is None


def test_projection_rejects_mismatched_lengths(index_records: ImageIndexRecords) -> None:
    with pytest.raises(ValueError):
        index_records.set_projection(
            SYSTEM_USER_ID, MODEL_ID, "hash-1", "{}", imgs("a.png"), np.zeros((2, 2), dtype=np.float32)
        )


def test_custom_vocab_terms_roundtrip_sorted_with_replace_semantics(index_records: ImageIndexRecords) -> None:
    assert index_records.get_custom_vocab_terms() == []

    index_records.set_custom_vocab_terms(["zebra", "aardvark"])
    assert index_records.get_custom_vocab_terms() == ["aardvark", "zebra"]

    # Replacement, not merge: what is passed is what is stored.
    index_records.set_custom_vocab_terms(["okapi"])
    assert index_records.get_custom_vocab_terms() == ["okapi"]

    index_records.set_custom_vocab_terms([])
    assert index_records.get_custom_vocab_terms() == []


def test_custom_vocab_case_variant_duplicates_cannot_fail_the_replace(index_records: ImageIndexRecords) -> None:
    # Callers normalize, but the NOCASE primary key is a backstop: a
    # case-variant pair must collapse to one row, not roll back the replace.
    index_records.set_custom_vocab_terms(["Zebra", "zebra"])
    assert index_records.get_custom_vocab_terms() == ["Zebra"]


def test_a_replace_of_the_vocabulary_waits_for_one_in_flight(
    database: Database, index_records: ImageIndexRecords
) -> None:
    # On a server two replaces side by side would each delete none of the other's uncommitted terms and keep both
    # lists; the replace in flight here holds the vocabulary's lock, which the other must wait for.
    index_records.set_custom_vocab_terms(["alpha"])

    errors = while_in_flight(
        database,
        lambda q: q.locks.acquire(DatabaseLock.IMAGE_INDEX_VOCABULARY),
        lambda: index_records.set_custom_vocab_terms(["beta"]),
    )

    assert errors == []
    assert index_records.get_custom_vocab_terms() == ["beta"]


# --- Videos ---


def test_videos_are_stored_counted_and_listed_alongside_images(
    image_records: ImageRecordStorage,
    video_records: VideoRecordStorage,
    index_records: ImageIndexRecords,
) -> None:
    _save_image(image_records, "a.png")
    _save_video(video_records, "clip.mp4")
    index_records.upsert_embedding(IndexedItem("image", "a.png"), MODEL_ID, _vec(1))
    index_records.upsert_embedding(IndexedItem("video", "clip.mp4"), MODEL_ID, _vec(2))

    status = index_records.count_index_status(MODEL_ID)
    assert (status.total, status.embedded) == (2, 2)

    items, matrix = index_records.get_embeddings(
        [IndexedItem("video", "clip.mp4"), IndexedItem("image", "a.png")], MODEL_ID
    )
    # Rows follow the caller's order across both namespaces, not one namespace after the other.
    assert items == [IndexedItem("video", "clip.mp4"), IndexedItem("image", "a.png")]
    assert np.array_equal(matrix[0], _vec(2))
    assert np.array_equal(matrix[1], _vec(1))


def test_an_image_and_a_video_are_different_items(
    image_records: ImageRecordStorage,
    video_records: VideoRecordStorage,
    index_records: ImageIndexRecords,
) -> None:
    # Names are generated with kind-specific extensions, so this cannot happen by accident —
    # but the namespaces must not be able to read or overwrite each other's rows.
    _save_image(image_records, "same.png")
    _save_video(video_records, "same.png")
    index_records.upsert_embedding(IndexedItem("image", "same.png"), MODEL_ID, _vec(1))
    index_records.upsert_embedding(IndexedItem("video", "same.png"), MODEL_ID, _vec(2))

    assert np.array_equal(index_records.get_embeddings(imgs("same.png"), MODEL_ID)[1][0], _vec(1))
    assert np.array_equal(index_records.get_embeddings(vids("same.png"), MODEL_ID)[1][0], _vec(2))

    index_records.delete_embedding(IndexedItem("image", "same.png"))
    assert index_records.get_embeddings(imgs("same.png"), MODEL_ID)[0] == []
    assert index_records.get_embeddings(vids("same.png"), MODEL_ID)[0] == vids("same.png")


def test_ineligible_videos_are_not_indexable(
    video_records: VideoRecordStorage, index_records: ImageIndexRecords
) -> None:
    _save_video(video_records, "eligible.mp4")
    _save_video(video_records, "intermediate.mp4", is_intermediate=True)
    _save_video(video_records, "mask.mp4", video_category=ImageCategory.MASK)

    assert index_records.list_unembedded_items(MODEL_ID, limit=10) == vids("eligible.mp4")
    assert index_records.count_index_status(MODEL_ID).total == 1


def test_unembedded_listing_merges_both_kinds_oldest_first(
    database: Database,
    image_records: ImageRecordStorage,
    video_records: VideoRecordStorage,
    index_records: ImageIndexRecords,
) -> None:
    # Interleaved creation times: listing one namespace and then the other would put every
    # image before any video, so a video generated hours ago would wait behind newer images.
    # Timestamps are set explicitly because four rows saved in a row land in the same
    # millisecond, and then only the name tiebreak would be under test.
    _save_image(image_records, "first.png")
    _save_video(video_records, "second.mp4")
    _save_image(image_records, "third.png")
    _save_video(video_records, "fourth.mp4")
    for kind, name, created_at in (
        ("image", "first.png", "2026-01-01 00:00:01.000"),
        ("video", "second.mp4", "2026-01-01 00:00:02.000"),
        ("image", "third.png", "2026-01-01 00:00:03.000"),
        ("video", "fourth.mp4", "2026-01-01 00:00:04.000"),
    ):
        _set_created_at(database, kind, name, created_at)

    batch = index_records.list_unembedded_items(MODEL_ID, limit=3)

    assert batch == [
        IndexedItem("image", "first.png"),
        IndexedItem("video", "second.mp4"),
        IndexedItem("image", "third.png"),
    ]


def test_deleting_a_video_deletes_its_embedding(
    video_records: VideoRecordStorage, index_records: ImageIndexRecords
) -> None:
    _save_video(video_records, "clip.mp4")
    index_records.upsert_embedding(IndexedItem("video", "clip.mp4"), MODEL_ID, _vec(1))

    video_records.delete("clip.mp4")

    assert index_records.get_embeddings(vids("clip.mp4"), MODEL_ID)[0] == []


def test_embedding_a_missing_video_is_a_noop(index_records: ImageIndexRecords) -> None:
    # The video may be deleted between being scheduled and embedded; the write is dropped
    # rather than raising a foreign-key error at the worker.
    index_records.upsert_embedding(IndexedItem("video", "gone.mp4"), MODEL_ID, _vec(1))
    assert index_records.get_embeddings(vids("gone.mp4"), MODEL_ID)[0] == []


def test_video_access_scoping_follows_board_membership(
    video_records: VideoRecordStorage,
    board_records: BoardRecordStorage,
    board_video_records: BoardVideoRecordStorage,
    index_records: ImageIndexRecords,
    other_user_id: str,
) -> None:
    _save_video(video_records, "own-unboarded.mp4")
    _save_video(video_records, "own-shared.mp4")
    _save_video(video_records, "own-private.mp4")
    for seed, name in enumerate(["own-unboarded.mp4", "own-shared.mp4", "own-private.mp4"]):
        index_records.upsert_embedding(IndexedItem("video", name), MODEL_ID, _vec(seed))

    shared_board = board_records.save("Shared", SYSTEM_USER_ID).board_id
    private_board = board_records.save("Private", SYSTEM_USER_ID).board_id
    board_records.update(shared_board, BoardChanges(board_visibility=BoardVisibility.Shared))
    board_video_records.add_video_to_board(shared_board, "own-shared.mp4")
    board_video_records.add_video_to_board(private_board, "own-private.mp4")

    assert index_records.list_accessible_embedded_items(SYSTEM_USER_ID, MODEL_ID) == sorted(
        vids("own-unboarded.mp4", "own-shared.mp4", "own-private.mp4")
    )
    # The other user sees the shared board's video and nothing else of ours.
    assert index_records.list_accessible_embedded_items(other_user_id, MODEL_ID) == vids("own-shared.mp4")


def test_videos_on_archived_boards_are_hidden_from_every_scope(
    video_records: VideoRecordStorage,
    board_records: BoardRecordStorage,
    board_video_records: BoardVideoRecordStorage,
    index_records: ImageIndexRecords,
) -> None:
    _save_video(video_records, "archived.mp4")
    index_records.upsert_embedding(IndexedItem("video", "archived.mp4"), MODEL_ID, _vec(1))
    board = board_records.save("Archived", SYSTEM_USER_ID).board_id
    board_video_records.add_video_to_board(board, "archived.mp4")
    board_records.update(board, BoardChanges(archived=True))

    assert index_records.list_accessible_embedded_items(SYSTEM_USER_ID, MODEL_ID) == []
    assert index_records.list_accessible_embedded_items(None, MODEL_ID) == []


def test_accessible_listing_orders_images_before_videos(
    image_records: ImageRecordStorage,
    video_records: VideoRecordStorage,
    index_records: ImageIndexRecords,
) -> None:
    # The listing feeds the projection scope hash and the search matrix, both of which need a
    # stable order across calls.
    _save_video(video_records, "a.mp4")
    _save_image(image_records, "z.png")
    index_records.upsert_embedding(IndexedItem("video", "a.mp4"), MODEL_ID, _vec(1))
    index_records.upsert_embedding(IndexedItem("image", "z.png"), MODEL_ID, _vec(2))

    assert index_records.list_accessible_embedded_items(SYSTEM_USER_ID, MODEL_ID) == [
        IndexedItem("image", "z.png"),
        IndexedItem("video", "a.mp4"),
    ]


def test_projection_roundtrips_mixed_items(index_records: ImageIndexRecords) -> None:
    coords = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    items = [IndexedItem("image", "a.png"), IndexedItem("video", "clip.mp4")]

    index_records.set_projection(SYSTEM_USER_ID, MODEL_ID, "hash-1", "{}", items, coords)

    record = index_records.get_projection(SYSTEM_USER_ID, MODEL_ID)
    assert record is not None
    assert record.items == items


def test_projection_without_stored_kinds_reads_as_images(database: Database, index_records: ImageIndexRecords) -> None:
    # Rows written before videos were indexable have no kinds column. They are all-image
    # projections, and must keep serving rather than being discarded on upgrade.
    coords = coords_to_blob(np.array([[1.0, 2.0]], dtype=np.float32))
    with database.begin(write=True) as conn:
        conn.execute(
            insert(image_projections).values(
                user_id=SYSTEM_USER_ID,
                model_id=MODEL_ID,
                scope_hash="hash-1",
                params="{}",
                point_count=1,
                image_names='["legacy.png"]',
                coords=coords,
            )
        )

    record = index_records.get_projection(SYSTEM_USER_ID, MODEL_ID)

    assert record is not None
    assert record.items == imgs("legacy.png")


def test_delete_embeddings_for_other_models_covers_both_kinds(
    image_records: ImageRecordStorage,
    video_records: VideoRecordStorage,
    index_records: ImageIndexRecords,
) -> None:
    _save_image(image_records, "a.png")
    _save_video(video_records, "clip.mp4")
    index_records.upsert_embedding(IndexedItem("image", "a.png"), OTHER_MODEL_ID, _vec(1))
    index_records.upsert_embedding(IndexedItem("video", "clip.mp4"), OTHER_MODEL_ID, _vec(2))
    index_records.upsert_embedding(IndexedItem("video", "clip.mp4"), MODEL_ID, _vec(3))

    # A video indexed by a since-replaced model is as dead as an image indexed by one.
    assert index_records.delete_embeddings_for_other_models(MODEL_ID) == 2
    assert index_records.get_embeddings(imgs("a.png"), OTHER_MODEL_ID)[0] == []
    assert index_records.get_embeddings(vids("clip.mp4"), OTHER_MODEL_ID)[0] == []
    assert index_records.get_embeddings(vids("clip.mp4"), MODEL_ID)[0] == vids("clip.mp4")
