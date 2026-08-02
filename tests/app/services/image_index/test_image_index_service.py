"""Tests for the image index worker service, using an injected fake encoder (no models/GPU)."""

import time
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest
from PIL import Image

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.events.events_common import ImageIndexStatusEvent, ImageIndexUpdatedEvent
from invokeai.app.services.image_index.image_index_default import ImageIndexService
from invokeai.app.services.image_index.image_index_records_sqlite import ImageIndexRecordsSqlite
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage
from invokeai.app.services.images.images_common import image_record_to_dto
from invokeai.app.services.images.images_default import ImageService
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database
from tests.test_nodes import TestEventService

MODEL_ID = "test-model-hash"
DIM = 8


def _wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Condition not met within timeout")


def _fake_encode(images: list[Image.Image]) -> np.ndarray:
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((len(images), DIM)).astype(np.float32)
    return vectors


@pytest.fixture
def db() -> SqliteDatabase:
    config = InvokeAIAppConfig(use_memory_db=True)
    return create_mock_sqlite_database(config=config, logger=InvokeAILogger.get_logger())


@pytest.fixture
def image_records(db: SqliteDatabase) -> SqliteImageRecordStorage:
    return SqliteImageRecordStorage(db=db)


@pytest.fixture
def index_records(db: SqliteDatabase) -> ImageIndexRecordsSqlite:
    return ImageIndexRecordsSqlite(db=db)


@pytest.fixture
def images_service() -> ImageService:
    images = ImageService()
    # The worker only needs get_pil_image; bypass file storage entirely.
    images.get_pil_image = lambda image_name: Image.new("RGB", (16, 16), "purple")  # type: ignore[method-assign]
    return images


def _make_invoker(
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
    enabled: bool = True,
    image_records: SqliteImageRecordStorage | None = None,
) -> SimpleNamespace:
    config = InvokeAIAppConfig(
        use_memory_db=True,
        image_index_enabled=enabled,
        image_index_device="cpu",
        image_index_batch_size=4,
    )
    services = SimpleNamespace(
        configuration=config,
        logger=InvokeAILogger.get_logger(),
        images=images_service,
        image_records=image_records,
        image_index_records=index_records,
        events=TestEventService(),
        session_queue=None,
        model_manager=None,
    )
    return SimpleNamespace(services=services)


@pytest.fixture
def service() -> ImageIndexService:
    svc = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    yield svc
    svc.stop()


def _save_image(
    image_records: SqliteImageRecordStorage,
    image_name: str,
    is_intermediate: bool = False,
    image_category: ImageCategory = ImageCategory.GENERAL,
) -> None:
    image_records.save(
        image_name=image_name,
        image_origin=ResourceOrigin.INTERNAL,
        image_category=image_category,
        width=16,
        height=16,
        has_workflow=False,
        is_intermediate=is_intermediate,
        user_id="system",
    )


def _dto_for(image_records: SqliteImageRecordStorage, image_name: str):
    record = image_records.get(image_name)
    return image_record_to_dto(record, image_url="http://x/i.png", thumbnail_url="http://x/t.png", board_id=None)


def test_constructor_requires_matched_test_seams() -> None:
    with pytest.raises(ValueError):
        ImageIndexService(encode_fn=_fake_encode)
    with pytest.raises(ValueError):
        ImageIndexService(model_id=MODEL_ID)


def test_disabled_service_is_inert(
    images_service: ImageService, index_records: ImageIndexRecordsSqlite, service: ImageIndexService
) -> None:
    invoker = _make_invoker(images_service, index_records, enabled=False)
    service.start(invoker)

    assert service.model_id is None
    assert service.get_status() is None
    assert images_service._on_changed_callbacks == []
    assert images_service._on_deleted_callbacks == []


def test_backfill_indexes_preexisting_eligible_images(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
    service: ImageIndexService,
) -> None:
    for i in range(6):
        _save_image(image_records, f"img-{i}.png")
    _save_image(image_records, "intermediate.png", is_intermediate=True)
    _save_image(image_records, "mask.png", image_category=ImageCategory.MASK)

    service.start(_make_invoker(images_service, index_records))

    _wait_until(lambda: index_records.count_index_status(MODEL_ID).embedded == 6)
    status = index_records.count_index_status(MODEL_ID)
    assert status.total == 6
    assert status.pending == 0
    # Ineligible images have no rows.
    assert index_records.get_embeddings(["intermediate.png", "mask.png"], MODEL_ID)[0] == []
    # Stored embeddings are L2-normalized.
    _, matrix = index_records.get_embeddings([f"img-{i}.png" for i in range(6)], MODEL_ID)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)


def test_on_changed_indexes_new_eligible_image_and_skips_ineligible(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
    service: ImageIndexService,
) -> None:
    invoker = _make_invoker(images_service, index_records)
    service.start(invoker)
    _wait_until(lambda: not service._backfill_pending.is_set())

    _save_image(image_records, "new.png")
    _save_image(image_records, "new-intermediate.png", is_intermediate=True)
    # Fire the callbacks the way ImageService.create would.
    images_service._on_changed(_dto_for(image_records, "new.png"))
    images_service._on_changed(_dto_for(image_records, "new-intermediate.png"))

    _wait_until(lambda: index_records.get_embeddings(["new.png"], MODEL_ID)[0] == ["new.png"])
    assert index_records.get_embeddings(["new-intermediate.png"], MODEL_ID)[0] == []


def test_unloadable_image_is_skipped_and_backfill_completes(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
    service: ImageIndexService,
) -> None:
    _save_image(image_records, "good.png")
    _save_image(image_records, "bad.png")

    def get_pil_image(image_name: str) -> Image.Image:
        if image_name == "bad.png":
            raise FileNotFoundError(image_name)
        return Image.new("RGB", (16, 16), "purple")

    images_service.get_pil_image = get_pil_image  # type: ignore[method-assign]
    service.start(_make_invoker(images_service, index_records))

    _wait_until(lambda: not service._backfill_pending.is_set())
    assert index_records.get_embeddings(["good.png"], MODEL_ID)[0] == ["good.png"]
    assert index_records.get_embeddings(["bad.png"], MODEL_ID)[0] == []
    assert "bad.png" in service._failed


def test_transient_encode_failure_is_retried_to_success(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
) -> None:
    calls = {"n": 0}

    def flaky_encode(images: list[Image.Image]) -> np.ndarray:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("transient failure (e.g. OOM)")
        return _fake_encode(images)

    service = ImageIndexService(encode_fn=flaky_encode, model_id=MODEL_ID)
    try:
        _save_image(image_records, "a.png")
        service.start(_make_invoker(images_service, index_records))
        _wait_until(lambda: index_records.get_embeddings(["a.png"], MODEL_ID)[0] == ["a.png"], timeout=15)
        assert "a.png" not in service._failed
    finally:
        service.stop()


def test_processor_falls_back_to_defaults_when_config_missing(tmp_path) -> None:
    # InvokeAI-published CLIP Vision model dirs ship no preprocessor_config.json;
    # the processor must fall back to library defaults rather than fail every batch.
    from types import SimpleNamespace

    from transformers import CLIPImageProcessor

    from invokeai.backend.model_manager.taxonomy import ModelType

    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    service._invoker = SimpleNamespace(services=SimpleNamespace(configuration=SimpleNamespace(models_path=tmp_path)))  # type: ignore[assignment]
    service._model_config = SimpleNamespace(type=ModelType.CLIPVision, path=str(tmp_path))  # type: ignore[assignment]

    processor = service._get_processor()

    assert isinstance(processor, CLIPImageProcessor)
    assert service._get_processor() is processor  # cached


def test_model_not_installed_message_flags_same_name_wrong_type() -> None:
    # The starter catalog has a clip_embed text encoder under the same name as
    # the default image encoder; the warning must name the type mismatch.
    from types import SimpleNamespace

    from invokeai.backend.model_manager.taxonomy import ModelType

    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)

    def search_by_attr(model_name=None, model_type=None):
        return [] if model_type is not None else [SimpleNamespace(type=ModelType.CLIPEmbed)]

    service._invoker = SimpleNamespace(  # type: ignore[assignment]
        services=SimpleNamespace(model_manager=SimpleNamespace(store=SimpleNamespace(search_by_attr=search_by_attr)))
    )
    message = service._model_not_installed_message("clip-vit-large-patch14")
    assert "clip_embed" in message
    assert "InvokeAI/clip-vit-large-patch14" in message

    service._invoker.services.model_manager.store.search_by_attr = lambda model_name=None: []
    message = service._model_not_installed_message("clip-vit-large-patch14")
    assert "is not installed" in message


def test_bad_encoder_output_marks_batch_failed(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
) -> None:
    service = ImageIndexService(encode_fn=lambda images: np.zeros((1,), dtype=np.float32), model_id=MODEL_ID)
    try:
        _save_image(image_records, "a.png")
        service.start(_make_invoker(images_service, index_records))
        _wait_until(lambda: not service._backfill_pending.is_set())
        assert index_records.count_index_status(MODEL_ID).embedded == 0
        assert "a.png" in service._failed
    finally:
        service.stop()


def test_status_event_emitted(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
    service: ImageIndexService,
) -> None:
    _save_image(image_records, "a.png")
    invoker = _make_invoker(images_service, index_records)
    service.start(invoker)

    _wait_until(lambda: not service._backfill_pending.is_set())
    _wait_until(
        lambda: any(
            isinstance(e, ImageIndexStatusEvent) and e.embedded == 1 and e.total == 1
            for e in invoker.services.events.events
        )
    )


def _status_events(invoker) -> list[ImageIndexStatusEvent]:
    return [e for e in invoker.services.events.events if isinstance(e, ImageIndexStatusEvent)]


def test_on_changed_emits_pending_status_before_embedding(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
    service: ImageIndexService,
) -> None:
    invoker = _make_invoker(images_service, index_records)
    service.start(invoker)
    _wait_until(lambda: not service._backfill_pending.is_set())

    _save_image(image_records, "new.png")
    images_service._on_changed(_dto_for(image_records, "new.png"))

    # The callback flags status-dirty before enqueueing, and the worker
    # emits before it embeds, so a pending=1 snapshot is always observable.
    _wait_until(lambda: any(e.total == 1 and e.pending == 1 for e in _status_events(invoker)))
    _wait_until(lambda: index_records.count_index_status(MODEL_ID).embedded == 1)
    _wait_until(lambda: any(e.total == 1 and e.pending == 0 for e in _status_events(invoker)))


def test_on_deleted_emits_status(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
    service: ImageIndexService,
) -> None:
    _save_image(image_records, "a.png")
    invoker = _make_invoker(images_service, index_records)
    service.start(invoker)
    _wait_until(lambda: index_records.count_index_status(MODEL_ID).embedded == 1)

    image_records.delete("a.png")
    images_service._on_deleted("a.png")

    # Deletions give the worker nothing to embed; the dirty flag set by the
    # callback is the only path to this emit, within one poll interval.
    _wait_until(lambda: any(e.total == 0 and e.embedded == 0 and e.pending == 0 for e in _status_events(invoker)))


def test_permanently_failed_image_still_reaches_quiescence(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
    service: ImageIndexService,
) -> None:
    """A permanently-failing image must not wedge pending above zero forever."""
    _save_image(image_records, "good.png")
    _save_image(image_records, "bad.png")

    def get_pil_image(image_name: str) -> Image.Image:
        if image_name == "bad.png":
            raise FileNotFoundError(image_name)
        return Image.new("RGB", (16, 16), "purple")

    images_service.get_pil_image = get_pil_image  # type: ignore[method-assign]
    invoker = _make_invoker(images_service, index_records)
    service.start(invoker)

    _wait_until(lambda: "bad.png" in service._failed, timeout=15.0)
    # Failed images are excluded from pending, so the index settles and the
    # final emitted status reports quiescence over the embeddable remainder.
    _wait_until(lambda: any(e.total == 2 and e.embedded == 1 and e.pending == 0 for e in _status_events(invoker)))
    status = service.get_status()
    assert status is not None
    assert status.pending == 0
    assert status.failed == 1


def test_upsert_failure_routes_through_retry_to_success(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
    service: ImageIndexService,
) -> None:
    """A raise while storing embeddings must feed the retry path, not strand the image."""
    invoker = _make_invoker(images_service, index_records)
    service.start(invoker)
    _wait_until(lambda: not service._backfill_pending.is_set())

    real_upsert = index_records.upsert_embedding
    calls = {"count": 0}

    def flaky_upsert(name: str, model_id: str, embedding: np.ndarray) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("database is locked")
        real_upsert(name, model_id, embedding)

    index_records.upsert_embedding = flaky_upsert  # type: ignore[method-assign]

    _save_image(image_records, "flaky.png")
    images_service._on_changed(_dto_for(image_records, "flaky.png"))

    _wait_until(lambda: index_records.get_embeddings(["flaky.png"], MODEL_ID)[0] == ["flaky.png"], timeout=15.0)
    _wait_until(lambda: any(e.total == 1 and e.embedded == 1 and e.pending == 0 for e in _status_events(invoker)))


def test_ineligible_transition_clears_failure_bookkeeping(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
    service: ImageIndexService,
) -> None:
    """An image that leaves eligibility must stop counting against `failed`."""
    invoker = _make_invoker(images_service, index_records)
    service.start(invoker)
    _wait_until(lambda: not service._backfill_pending.is_set())

    service._failed.add("gone.png")
    service._attempts["gone.png"] = 3
    _save_image(image_records, "gone.png", image_category=ImageCategory.MASK)
    images_service._on_changed(_dto_for(image_records, "gone.png"))

    assert "gone.png" not in service._failed
    assert "gone.png" not in service._attempts
    status = service.get_status()
    assert status is not None
    assert status.failed == 0


def test_owner_poke_emitted_at_quiescence(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
    service: ImageIndexService,
) -> None:
    """Owners get a counts-free user-routed poke once their embeds settle."""
    invoker = _make_invoker(images_service, index_records, image_records=image_records)
    service.start(invoker)
    _wait_until(lambda: not service._backfill_pending.is_set())

    _save_image(image_records, "mine.png")
    images_service._on_changed(_dto_for(image_records, "mine.png"))

    _wait_until(
        lambda: any(
            isinstance(e, ImageIndexUpdatedEvent) and e.user_id == "system" for e in invoker.services.events.events
        )
    )


def test_stop_joins_worker(
    images_service: ImageService, index_records: ImageIndexRecordsSqlite, service: ImageIndexService
) -> None:
    service.start(_make_invoker(images_service, index_records))
    assert service._worker is not None and service._worker.is_alive()

    service.stop()

    assert not service._worker.is_alive()
