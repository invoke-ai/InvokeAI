"""Tests for the image index records service: embedding storage, eligibility, access scoping, projections."""

import sqlite3

import numpy as np
import pytest

from invokeai.app.services.board_image_records.board_image_records_sqlite import SqliteBoardImageRecordStorage
from invokeai.app.services.board_records.board_records_common import BoardChanges, BoardVisibility
from invokeai.app.services.board_records.board_records_sqlite import SqliteBoardRecordStorage
from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.image_index.image_index_common import (
    blob_to_coords,
    blob_to_embedding,
    coords_to_blob,
    embedding_to_blob,
)
from invokeai.app.services.image_index.image_index_records_sqlite import (
    _IN_CLAUSE_CHUNK,
    ImageIndexRecordsSqlite,
)
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_default import UserService
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database

SYSTEM_USER_ID = "system"
MODEL_ID = "model-hash-1"
OTHER_MODEL_ID = "model-hash-2"
DIM = 8


@pytest.fixture
def db() -> SqliteDatabase:
    config = InvokeAIAppConfig(use_memory_db=True)
    return create_mock_sqlite_database(config=config, logger=InvokeAILogger.get_logger())


@pytest.fixture
def image_records(db: SqliteDatabase) -> SqliteImageRecordStorage:
    return SqliteImageRecordStorage(db=db)


@pytest.fixture
def board_records(db: SqliteDatabase) -> SqliteBoardRecordStorage:
    return SqliteBoardRecordStorage(db=db)


@pytest.fixture
def board_image_records(db: SqliteDatabase) -> SqliteBoardImageRecordStorage:
    return SqliteBoardImageRecordStorage(db=db)


@pytest.fixture
def index_records(db: SqliteDatabase) -> ImageIndexRecordsSqlite:
    return ImageIndexRecordsSqlite(db=db)


@pytest.fixture
def other_user_id(db: SqliteDatabase) -> str:
    users = UserService(db=db)
    user = users.create(
        UserCreateRequest(email="other@example.com", display_name="Other", password="TestPass123", is_admin=False)
    )
    return user.user_id


def _save_image(
    image_records: SqliteImageRecordStorage,
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


def test_upsert_and_get_roundtrip(
    image_records: SqliteImageRecordStorage, index_records: ImageIndexRecordsSqlite
) -> None:
    _save_image(image_records, "a.png")
    _save_image(image_records, "b.png")
    va, vb = _vec(1), _vec(2)
    index_records.upsert_embedding("a.png", MODEL_ID, va)
    index_records.upsert_embedding("b.png", MODEL_ID, vb)

    names, matrix = index_records.get_embeddings(["b.png", "a.png", "missing.png"], MODEL_ID)

    assert names == ["b.png", "a.png"]
    assert matrix.dtype == np.float32
    assert np.array_equal(matrix[0], vb)
    assert np.array_equal(matrix[1], va)


def test_upsert_replaces_existing_embedding(
    image_records: SqliteImageRecordStorage, index_records: ImageIndexRecordsSqlite
) -> None:
    _save_image(image_records, "a.png")
    index_records.upsert_embedding("a.png", MODEL_ID, _vec(1))
    replacement = _vec(99)
    index_records.upsert_embedding("a.png", MODEL_ID, replacement)

    names, matrix = index_records.get_embeddings(["a.png"], MODEL_ID)
    assert names == ["a.png"]
    assert np.array_equal(matrix[0], replacement)


def test_get_embeddings_empty_input_and_no_matches(
    image_records: SqliteImageRecordStorage, index_records: ImageIndexRecordsSqlite
) -> None:
    names, matrix = index_records.get_embeddings([], MODEL_ID)
    assert names == []
    assert matrix.shape == (0, 0)

    _save_image(image_records, "a.png")
    index_records.upsert_embedding("a.png", MODEL_ID, _vec(1))
    names, matrix = index_records.get_embeddings(["a.png"], OTHER_MODEL_ID)
    assert names == []
    assert matrix.shape == (0, 0)


def test_get_embeddings_chunks_large_requests(
    image_records: SqliteImageRecordStorage, index_records: ImageIndexRecordsSqlite
) -> None:
    # Straddle the IN-clause chunk boundary so the multi-chunk path really runs. Derived from
    # the constant rather than hardcoded, so raising the chunk size cannot silently reduce this
    # to a single-chunk test.
    count = _IN_CLAUSE_CHUNK + 1
    for i in range(count):
        _save_image(image_records, f"img-{i:04d}.png")
        index_records.upsert_embedding(f"img-{i:04d}.png", MODEL_ID, _vec(i))

    # Reverse order: SQLite would return each chunk in its own order, so this fails unless the
    # implementation reorders results back to the caller's sequence within every chunk.
    requested = [f"img-{i:04d}.png" for i in reversed(range(count))]
    names, matrix = index_records.get_embeddings(requested, MODEL_ID)
    assert names == requested
    assert matrix.shape == (count, DIM)
    # Rows must still align with the names after chunking and reordering.
    assert np.array_equal(matrix[0], _vec(count - 1))
    assert np.array_equal(matrix[-1], _vec(0))


def test_get_embeddings_rejects_inconsistent_dims(
    image_records: SqliteImageRecordStorage, index_records: ImageIndexRecordsSqlite, db: SqliteDatabase
) -> None:
    # The ABC promises a failure on mixed dims under one model_id. Write a short vector behind
    # the service's back, since upsert_embedding alone cannot produce the inconsistency.
    _save_image(image_records, "a.png")
    _save_image(image_records, "b.png")
    index_records.upsert_embedding("a.png", MODEL_ID, _vec(1))
    short = np.ones(DIM // 2, dtype=np.float32)
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO image_embeddings (image_name, model_id, dim, embedding) VALUES (?, ?, ?, ?);",
            ("b.png", MODEL_ID, DIM // 2, embedding_to_blob(short)),
        )

    with pytest.raises(ValueError, match="Inconsistent embedding dims"):
        index_records.get_embeddings(["a.png", "b.png"], MODEL_ID)


def test_get_embeddings_deduplicates_input_names(
    image_records: SqliteImageRecordStorage, index_records: ImageIndexRecordsSqlite
) -> None:
    _save_image(image_records, "a.png")
    index_records.upsert_embedding("a.png", MODEL_ID, _vec(1))

    names, matrix = index_records.get_embeddings(["a.png", "a.png"], MODEL_ID)

    assert names == ["a.png"]
    assert matrix.shape == (1, DIM)


def test_upsert_embedding_for_deleted_image_is_noop(index_records: ImageIndexRecordsSqlite) -> None:
    # The image was deleted (or never existed) by the time the write lands;
    # the FK violation must not escape as a raw sqlite3 error.
    index_records.upsert_embedding("gone.png", MODEL_ID, _vec(1))
    assert index_records.get_embeddings(["gone.png"], MODEL_ID)[0] == []


def test_set_projection_for_deleted_user_is_noop(index_records: ImageIndexRecordsSqlite) -> None:
    index_records.set_projection(
        "no-such-user", MODEL_ID, "hash-1", "{}", ["a.png"], np.zeros((1, 2), dtype=np.float32)
    )
    assert index_records.get_projection("no-such-user", MODEL_ID) is None


def test_skipped_write_raises_nothing_and_leaves_the_connection_usable(
    image_records: SqliteImageRecordStorage, index_records: ImageIndexRecordsSqlite, db: SqliteDatabase
) -> None:
    # The missing-parent case is handled by `WHERE EXISTS`, not by provoking and swallowing an
    # IntegrityError. That matters because `db.transaction()` commits/rolls back the whole
    # shared connection: a swallowed error would leave the connection mid-statement-failure, and
    # an escaping one would roll back. Neither may happen, and the next write must still work.
    index_records.upsert_embedding("gone.png", MODEL_ID, _vec(1))
    index_records.set_projection(
        "no-such-user", MODEL_ID, "hash-1", "{}", ["a.png"], np.zeros((1, 2), dtype=np.float32)
    )

    assert db._conn.in_transaction is False

    _save_image(image_records, "real.png")
    index_records.upsert_embedding("real.png", MODEL_ID, _vec(2))
    assert index_records.get_embeddings(["real.png"], MODEL_ID)[0] == ["real.png"]


def test_skipped_write_does_not_raise_integrity_error(index_records: ImageIndexRecordsSqlite) -> None:
    # Pin the mechanism, not just the outcome: if the guard regressed to relying on the foreign
    # key, this would raise instead of no-opping.
    try:
        index_records.upsert_embedding("gone.png", MODEL_ID, _vec(1))
    except sqlite3.IntegrityError as exc:  # pragma: no cover - the assertion is the point
        pytest.fail(f"missing-parent write must be a no-op, not an IntegrityError: {exc}")


def test_delete_embedding_removes_all_models(
    image_records: SqliteImageRecordStorage, index_records: ImageIndexRecordsSqlite
) -> None:
    _save_image(image_records, "a.png")
    index_records.upsert_embedding("a.png", MODEL_ID, _vec(1))
    index_records.upsert_embedding("a.png", OTHER_MODEL_ID, _vec(2))

    index_records.delete_embedding("a.png")

    assert index_records.get_embeddings(["a.png"], MODEL_ID)[0] == []
    assert index_records.get_embeddings(["a.png"], OTHER_MODEL_ID)[0] == []


def test_image_delete_cascades_to_embeddings(
    db: SqliteDatabase, image_records: SqliteImageRecordStorage, index_records: ImageIndexRecordsSqlite
) -> None:
    _save_image(image_records, "a.png")
    index_records.upsert_embedding("a.png", MODEL_ID, _vec(1))

    image_records.delete("a.png")

    assert index_records.get_embeddings(["a.png"], MODEL_ID)[0] == []


def test_delete_embeddings_for_other_models(
    image_records: SqliteImageRecordStorage, index_records: ImageIndexRecordsSqlite
) -> None:
    _save_image(image_records, "a.png")
    index_records.upsert_embedding("a.png", MODEL_ID, _vec(1))
    index_records.upsert_embedding("a.png", OTHER_MODEL_ID, _vec(2))

    deleted = index_records.delete_embeddings_for_other_models(MODEL_ID)

    assert deleted == 1
    assert index_records.get_embeddings(["a.png"], MODEL_ID)[0] == ["a.png"]
    assert index_records.get_embeddings(["a.png"], OTHER_MODEL_ID)[0] == []


# --- Eligibility: backfill listing and status counts ---


def test_list_unembedded_skips_ineligible_and_embedded(
    image_records: SqliteImageRecordStorage, index_records: ImageIndexRecordsSqlite
) -> None:
    _save_image(image_records, "eligible.png")
    _save_image(image_records, "embedded.png")
    _save_image(image_records, "intermediate.png", is_intermediate=True)
    _save_image(image_records, "mask.png", image_category=ImageCategory.MASK)
    index_records.upsert_embedding("embedded.png", MODEL_ID, _vec(1))

    unembedded = index_records.list_unembedded_image_names(MODEL_ID, limit=10)

    assert unembedded == ["eligible.png"]


def test_list_unembedded_respects_limit_and_returns_oldest_first(
    image_records: SqliteImageRecordStorage, index_records: ImageIndexRecordsSqlite, db: SqliteDatabase
) -> None:
    # `created_at` has millisecond resolution, so five back-to-back saves almost always tie and
    # the `image_name ASC` tie-break alone decides the result — which would let a reversed
    # `created_at` ordering pass unnoticed. Stamp distinct timestamps in the *reverse* of
    # alphabetical order so the two orderings disagree and only `created_at` can satisfy this.
    for i in range(5):
        _save_image(image_records, f"img-{i}.png")
    with db.transaction() as cursor:
        for i in range(5):
            cursor.execute(
                "UPDATE images SET created_at = ? WHERE image_name = ?;",
                (f"2026-01-0{5 - i} 00:00:00.000", f"img-{i}.png"),
            )

    batch = index_records.list_unembedded_image_names(MODEL_ID, limit=3)

    # Which three matters, not just how many: backfill walks oldest-first, and a reversed order
    # would silently re-scan the newest images forever while the oldest never got embedded.
    assert batch == ["img-4.png", "img-3.png", "img-2.png"]


def test_list_unembedded_rejects_negative_limit(index_records: ImageIndexRecordsSqlite) -> None:
    # SQLite reads a negative LIMIT as unbounded, which would turn a bounded backfill batch into
    # a full-table load.
    with pytest.raises(ValueError, match="non-negative"):
        index_records.list_unembedded_image_names(MODEL_ID, limit=-1)


def test_list_unembedded_zero_limit_returns_nothing(
    image_records: SqliteImageRecordStorage, index_records: ImageIndexRecordsSqlite
) -> None:
    _save_image(image_records, "a.png")
    assert index_records.list_unembedded_image_names(MODEL_ID, limit=0) == []


def test_count_index_status(image_records: SqliteImageRecordStorage, index_records: ImageIndexRecordsSqlite) -> None:
    _save_image(image_records, "a.png")
    _save_image(image_records, "b.png")
    _save_image(image_records, "intermediate.png", is_intermediate=True)
    index_records.upsert_embedding("a.png", MODEL_ID, _vec(1))

    status = index_records.count_index_status(MODEL_ID)

    assert status.total == 2
    assert status.embedded == 1
    assert status.pending == 1


# --- Access scoping ---


def test_accessible_images_scoping(
    image_records: SqliteImageRecordStorage,
    board_records: SqliteBoardRecordStorage,
    board_image_records: SqliteBoardImageRecordStorage,
    index_records: ImageIndexRecordsSqlite,
    other_user_id: str,
) -> None:
    # System user's images: one unboarded, one on a private board, one shared, one public.
    for seed, name in enumerate(["own-unboarded.png", "own-private.png", "own-shared.png", "own-public.png"]):
        _save_image(image_records, name, user_id=SYSTEM_USER_ID)
        index_records.upsert_embedding(name, MODEL_ID, _vec(seed))
    # Other user's unboarded image.
    _save_image(image_records, "theirs-unboarded.png", user_id=other_user_id)
    index_records.upsert_embedding("theirs-unboarded.png", MODEL_ID, _vec(5))
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
    assert index_records.list_accessible_embedded_images(SYSTEM_USER_ID, MODEL_ID) == sorted(
        ["own-unboarded.png", "own-private.png", "own-shared.png", "own-public.png"]
    )

    # The other user sees their own image plus shared/public board images — never the
    # system user's private-board or unboarded images.
    assert index_records.list_accessible_embedded_images(other_user_id, MODEL_ID) == sorted(
        ["theirs-unboarded.png", "own-shared.png", "own-public.png"]
    )

    # Admin scope (None) sees everything embedded.
    assert index_records.list_accessible_embedded_images(None, MODEL_ID) == sorted(
        [
            "own-unboarded.png",
            "own-private.png",
            "own-shared.png",
            "own-public.png",
            "theirs-unboarded.png",
        ]
    )


def test_accessible_images_includes_individually_shared_boards(
    db: SqliteDatabase,
    image_records: SqliteImageRecordStorage,
    board_records: SqliteBoardRecordStorage,
    board_image_records: SqliteBoardImageRecordStorage,
    index_records: ImageIndexRecordsSqlite,
    other_user_id: str,
) -> None:
    # A private board individually shared with other_user via shared_boards
    # must expose its images to them — mirroring the board-listing access
    # model. No service writes shared_boards yet, so insert the row directly.
    _save_image(image_records, "own-individually-shared.png", user_id=SYSTEM_USER_ID)
    index_records.upsert_embedding("own-individually-shared.png", MODEL_ID, _vec(1))
    board_id = board_records.save("Individually shared", SYSTEM_USER_ID).board_id
    board_image_records.add_image_to_board(board_id, "own-individually-shared.png")
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO shared_boards (board_id, user_id, can_edit) VALUES (?, ?, ?);",
            (board_id, other_user_id, False),
        )

    assert index_records.list_accessible_embedded_images(other_user_id, MODEL_ID) == ["own-individually-shared.png"]
    # A third party without the share still cannot see it.
    users = UserService(db=db)
    third_user = users.create(
        UserCreateRequest(email="third@example.com", display_name="Third", password="TestPass123", is_admin=False)
    )
    assert index_records.list_accessible_embedded_images(third_user.user_id, MODEL_ID) == []


def test_accessible_images_includes_boards_owned_by_user(
    image_records: SqliteImageRecordStorage,
    board_records: SqliteBoardRecordStorage,
    board_image_records: SqliteBoardImageRecordStorage,
    index_records: ImageIndexRecordsSqlite,
    other_user_id: str,
) -> None:
    # An image uploaded by another user onto a board the system user OWNS is
    # accessible to the board owner — matching the gallery "all" listing
    # (image_records_sqlite), which grants access via boards.user_id even
    # without shared/public visibility or a shared_boards row.
    _save_image(image_records, "theirs-on-my-board.png", user_id=other_user_id)
    index_records.upsert_embedding("theirs-on-my-board.png", MODEL_ID, _vec(1))
    my_board = board_records.save("Mine", SYSTEM_USER_ID).board_id
    board_image_records.add_image_to_board(my_board, "theirs-on-my-board.png")

    assert index_records.list_accessible_embedded_images(SYSTEM_USER_ID, MODEL_ID) == ["theirs-on-my-board.png"]
    # The uploader placed it on a private board they neither own nor were
    # granted; like the gallery "all" listing, they no longer see it.
    assert index_records.list_accessible_embedded_images(other_user_id, MODEL_ID) == []


def test_accessible_images_excludes_archived_boards(
    image_records: SqliteImageRecordStorage,
    board_records: SqliteBoardRecordStorage,
    board_image_records: SqliteBoardImageRecordStorage,
    index_records: ImageIndexRecordsSqlite,
    other_user_id: str,
) -> None:
    # Archived boards hide their images from every scope, mirroring the
    # gallery "all" listing: even the owner and the administrative scope.
    _save_image(image_records, "own-archived.png", user_id=SYSTEM_USER_ID)
    _save_image(image_records, "own-unboarded.png", user_id=SYSTEM_USER_ID)
    _save_image(image_records, "shared-archived.png", user_id=SYSTEM_USER_ID)
    for seed, name in enumerate(["own-archived.png", "own-unboarded.png", "shared-archived.png"]):
        index_records.upsert_embedding(name, MODEL_ID, _vec(seed))

    archived_board = board_records.save("Archived", SYSTEM_USER_ID).board_id
    archived_shared_board = board_records.save("Archived shared", SYSTEM_USER_ID).board_id
    board_records.update(archived_shared_board, BoardChanges(board_visibility=BoardVisibility.Shared))
    board_image_records.add_image_to_board(archived_board, "own-archived.png")
    board_image_records.add_image_to_board(archived_shared_board, "shared-archived.png")
    board_records.update(archived_board, BoardChanges(archived=True))
    board_records.update(archived_shared_board, BoardChanges(archived=True))

    # Owner: archived-board images are hidden; unboarded images are unaffected.
    assert index_records.list_accessible_embedded_images(SYSTEM_USER_ID, MODEL_ID) == ["own-unboarded.png"]
    # An archived shared board grants nothing to other users either.
    assert index_records.list_accessible_embedded_images(other_user_id, MODEL_ID) == []
    # The administrative scope also hides archived-board images.
    assert index_records.list_accessible_embedded_images(None, MODEL_ID) == ["own-unboarded.png"]


def test_accessible_images_returns_boarded_images_once(
    image_records: SqliteImageRecordStorage,
    board_records: SqliteBoardRecordStorage,
    board_image_records: SqliteBoardImageRecordStorage,
    index_records: ImageIndexRecordsSqlite,
) -> None:
    # `board_images` has PRIMARY KEY (image_name), so an image is on at most one board and the
    # board join cannot multiply rows. This pins that the join stays single-valued: if
    # board_images ever became many-to-many, the result would duplicate and the scope hash
    # (derived from this list) would change without the accessible set changing.
    _save_image(image_records, "a.png")
    index_records.upsert_embedding("a.png", MODEL_ID, _vec(1))
    board_a = board_records.save("A", SYSTEM_USER_ID).board_id
    board_image_records.add_image_to_board(board_a, "a.png")

    assert index_records.list_accessible_embedded_images(SYSTEM_USER_ID, MODEL_ID) == ["a.png"]


def test_accessible_images_are_filtered_by_model_id(
    image_records: SqliteImageRecordStorage,
    board_records: SqliteBoardRecordStorage,
    board_image_records: SqliteBoardImageRecordStorage,
    index_records: ImageIndexRecordsSqlite,
    other_user_id: str,
) -> None:
    # The scope hash is derived from this listing while the embedding matrix is fetched by
    # model_id. If the listing ignored model_id, a model switch would put images into the scope
    # whose embeddings get_embeddings cannot return, desynchronizing names from coordinates.
    _save_image(image_records, "current.png")
    _save_image(image_records, "stale.png")
    index_records.upsert_embedding("current.png", MODEL_ID, _vec(1))
    index_records.upsert_embedding("stale.png", OTHER_MODEL_ID, _vec(2))

    shared = board_records.save("Shared", SYSTEM_USER_ID).board_id
    board_records.update(shared, BoardChanges(board_visibility=BoardVisibility.Shared))
    board_image_records.add_image_to_board(shared, "current.png")
    board_image_records.add_image_to_board(shared, "stale.png")

    # Every scope must apply the filter, not just the owner's.
    assert index_records.list_accessible_embedded_images(SYSTEM_USER_ID, MODEL_ID) == ["current.png"]
    assert index_records.list_accessible_embedded_images(other_user_id, MODEL_ID) == ["current.png"]
    assert index_records.list_accessible_embedded_images(None, MODEL_ID) == ["current.png"]
    assert index_records.list_accessible_embedded_images(None, OTHER_MODEL_ID) == ["stale.png"]


def test_accessible_images_exclude_individually_shared_archived_board(
    image_records: SqliteImageRecordStorage,
    board_records: SqliteBoardRecordStorage,
    board_image_records: SqliteBoardImageRecordStorage,
    index_records: ImageIndexRecordsSqlite,
    db: SqliteDatabase,
    other_user_id: str,
) -> None:
    # Archiving is tested for visibility-shared boards; the shared_boards grant is a separate
    # branch of the access clause and must be archived-gated too.
    _save_image(image_records, "granted.png")
    index_records.upsert_embedding("granted.png", MODEL_ID, _vec(1))
    board_id = board_records.save("Private", SYSTEM_USER_ID).board_id
    board_image_records.add_image_to_board(board_id, "granted.png")
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO shared_boards (board_id, user_id) VALUES (?, ?);",
            (board_id, other_user_id),
        )

    assert index_records.list_accessible_embedded_images(other_user_id, MODEL_ID) == ["granted.png"]

    board_records.update(board_id, BoardChanges(archived=True))

    assert index_records.list_accessible_embedded_images(other_user_id, MODEL_ID) == []


# --- Projections ---


def test_projection_roundtrip(index_records: ImageIndexRecordsSqlite) -> None:
    coords = np.array([[0.5, -1.5], [2.0, 3.0]], dtype=np.float32)
    index_records.set_projection(SYSTEM_USER_ID, MODEL_ID, "hash-1", '{"n_neighbors": 15}', ["a.png", "b.png"], coords)

    record = index_records.get_projection(SYSTEM_USER_ID, MODEL_ID)

    assert record is not None
    assert record.scope_hash == "hash-1"
    assert record.params == '{"n_neighbors": 15}'
    assert record.point_count == 2
    assert record.image_names == ["a.png", "b.png"]
    assert np.array_equal(record.coords, coords)


def test_projection_upsert_replaces(index_records: ImageIndexRecordsSqlite) -> None:
    index_records.set_projection(
        SYSTEM_USER_ID, MODEL_ID, "hash-1", "{}", ["a.png"], np.zeros((1, 2), dtype=np.float32)
    )
    index_records.set_projection(
        SYSTEM_USER_ID, MODEL_ID, "hash-2", "{}", ["a.png", "b.png"], np.ones((2, 2), dtype=np.float32)
    )

    record = index_records.get_projection(SYSTEM_USER_ID, MODEL_ID)

    assert record is not None
    assert record.scope_hash == "hash-2"
    assert record.point_count == 2


def test_projection_missing_and_delete_idempotent(index_records: ImageIndexRecordsSqlite) -> None:
    assert index_records.get_projection(SYSTEM_USER_ID, MODEL_ID) is None
    index_records.delete_projection(SYSTEM_USER_ID, MODEL_ID)  # no-op

    index_records.set_projection(
        SYSTEM_USER_ID, MODEL_ID, "hash-1", "{}", ["a.png"], np.zeros((1, 2), dtype=np.float32)
    )
    index_records.delete_projection(SYSTEM_USER_ID, MODEL_ID)
    assert index_records.get_projection(SYSTEM_USER_ID, MODEL_ID) is None


def test_projection_rejects_mismatched_lengths(index_records: ImageIndexRecordsSqlite) -> None:
    with pytest.raises(ValueError):
        index_records.set_projection(
            SYSTEM_USER_ID, MODEL_ID, "hash-1", "{}", ["a.png"], np.zeros((2, 2), dtype=np.float32)
        )
