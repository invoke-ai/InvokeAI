"""Tests for the image index worker service, using an injected fake encoder (no models/GPU)."""

import time
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest
import torch
from PIL import Image

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.events.events_common import ImageIndexStatusEvent, ImageIndexUpdatedEvent
from invokeai.app.services.image_index.image_index_default import (
    _MAX_ATTEMPTS,
    _MAX_BACKOFF_SECONDS,
    _POLL_SECONDS,
    ImageIndexService,
)
from invokeai.app.services.image_index.image_index_records_sqlite import ImageIndexRecordsSqlite
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_sqlite import SqliteImageRecordStorage
from invokeai.app.services.images.images_common import image_record_to_dto
from invokeai.app.services.images.images_default import ImageService
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.backend.util.devices import TorchDevice
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


def _unit_vec() -> np.ndarray:
    """A storable embedding: float32, finite, non-zero."""
    v = np.ones(DIM, dtype=np.float32)
    return v / np.linalg.norm(v)


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
    device: str | None = "cpu",
    session_queue: object | None = None,
    model_manager: object | None = None,
) -> SimpleNamespace:
    config = InvokeAIAppConfig(
        use_memory_db=True,
        image_index_enabled=enabled,
        image_index_device=device,
        image_index_batch_size=4,
    )
    services = SimpleNamespace(
        configuration=config,
        logger=InvokeAILogger.get_logger(),
        images=images_service,
        image_records=image_records,
        image_index_records=index_records,
        events=TestEventService(),
        session_queue=session_queue,
        model_manager=model_manager,
    )
    return SimpleNamespace(services=services)


@pytest.fixture
def service() -> ImageIndexService:
    svc = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    yield svc
    svc.stop()


@pytest.fixture
def accelerator_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the host has a non-CPU device.

    With no `image_index_device` override, CPU mode is decided by autodetection, so on a
    CPU-only machine — every linux-cpu and windows-cpu CI runner — the generation wait is
    skipped entirely. Tests covering that wait have to pin the device or they assert nothing
    there while still passing.
    """
    monkeypatch.setattr(TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cuda")))


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


def test_broken_encoder_leaves_images_pending_rather_than_quarantined(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
) -> None:
    service = ImageIndexService(encode_fn=lambda images: np.zeros((1,), dtype=np.float32), model_id=MODEL_ID)
    try:
        _save_image(image_records, "a.png")
        service.start(_make_invoker(images_service, index_records))
        _wait_until(lambda: service._systemic_failures > 0)

        assert index_records.count_index_status(MODEL_ID).embedded == 0
        # NOT quarantined. A broken encoder is a fault of the machinery, and `_MAX_ATTEMPTS`
        # bounds per-image badness only. Retiring the image here would be a lie about the image
        # and — since nothing but a successful embed clears `_failed` — would survive the
        # encoder being fixed, leaving the index short until a restart.
        assert "a.png" not in service._failed
        assert service.get_status().pending == 1
    finally:
        service.stop()


def test_status_event_reports_failures_so_a_settled_index_is_not_mistaken_for_complete(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
) -> None:
    """`pending` excludes failures, so on its own it cannot express "gave up on some".

    Without `failed` in the event, a client sees pending == 0 with embedded < total and has no
    way to tell a finished index from one that quietly skipped images — it would render
    "complete" over a gallery with holes.
    """

    def get_pil_image(image_name: str) -> Image.Image:
        if image_name == "bad.png":
            raise FileNotFoundError(image_name)
        return Image.new("RGB", (16, 16), "purple")

    images_service.get_pil_image = get_pil_image  # type: ignore[method-assign]

    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    try:
        _save_image(image_records, "good.png")
        _save_image(image_records, "bad.png")
        invoker = _make_invoker(images_service, index_records)
        service.start(invoker)

        _wait_until(lambda: any(e.failed == 1 and e.pending == 0 for e in _status_events(invoker)), timeout=20.0)
        settled = [e for e in _status_events(invoker) if e.pending == 0][-1]
        # The three numbers together are what make the state legible.
        assert (settled.total, settled.embedded, settled.failed) == (2, 1, 1)
    finally:
        service.stop()


def test_index_recovers_from_an_encoder_outage_without_a_restart(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
) -> None:
    """The point of separating systemic from per-image failure.

    An outage lasting more than `_MAX_ATTEMPTS` sweeps used to retire every image it touched.
    Nothing but a successful embed clears `_failed`, so the images stayed dead after the model
    came back and only a process restart recovered them. They must now come back on their own.
    """
    outage = {"active": True}

    def encode(images: list[Image.Image]) -> np.ndarray:
        if outage["active"]:
            raise RuntimeError("model is not installed")
        return _fake_encode(images)

    service = ImageIndexService(encode_fn=encode, model_id=MODEL_ID)
    try:
        for i in range(6):
            _save_image(image_records, f"img-{i}.png")
        service.start(_make_invoker(images_service, index_records))

        # Outlast _MAX_ATTEMPTS sweeps, which is what used to retire the images.
        _wait_until(lambda: service._systemic_failures > _MAX_ATTEMPTS, timeout=15.0)
        assert service._failed == set()
        assert service._attempts == {}
        assert index_records.count_index_status(MODEL_ID).embedded == 0

        outage["active"] = False

        _wait_until(lambda: index_records.count_index_status(MODEL_ID).embedded == 6, timeout=30.0)
        assert service._failed == set()
        assert service.get_status().pending == 0
        # The backoff must unwind too, or the next transient blip would start at the ceiling.
        assert service._systemic_failures == 0
    finally:
        service.stop()


def test_sustained_storage_failure_does_not_quarantine_images(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
) -> None:
    """A database that is down is no more the images' fault than a missing model is.

    "database is locked" outlasting `_MAX_ATTEMPTS` sweeps must not retire the images, for the
    same reason an encoder outage must not: `_failed` would survive the database recovering.
    """
    outage = {"active": True}
    real_upsert = index_records.upsert_embedding

    def flaky_upsert(name: str, model_id: str, embedding: np.ndarray) -> None:
        if outage["active"]:
            raise RuntimeError("database is locked")
        real_upsert(name, model_id, embedding)

    index_records.upsert_embedding = flaky_upsert  # type: ignore[method-assign]

    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    try:
        for i in range(3):
            _save_image(image_records, f"img-{i}.png")
        service.start(_make_invoker(images_service, index_records))

        _wait_until(lambda: service._systemic_failures > _MAX_ATTEMPTS, timeout=15.0)
        assert service._failed == set()
        assert service._attempts == {}

        outage["active"] = False

        _wait_until(lambda: index_records.count_index_status(MODEL_ID).embedded == 3, timeout=30.0)
        assert service._failed == set()
    finally:
        service.stop()


def test_batch_failure_is_charged_to_the_images_when_the_encoder_is_healthy(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
) -> None:
    """The other half: a poisonous image must still be quarantined so the backlog can advance.

    Not charging the images would be just as wrong in the other direction — one image the
    encoder chokes on would block the sweep forever, since backfill always returns it first.
    The encoder is probed with a trivial image to tell the two situations apart.
    """

    def encode(images: list[Image.Image]) -> np.ndarray:
        # Healthy for the one-image probe, broken for any real batch.
        if len(images) == 1 and images[0].size == (16, 16):
            return _fake_encode(images)
        raise RuntimeError("cannot encode these images")

    service = ImageIndexService(encode_fn=encode, model_id=MODEL_ID)
    try:
        for i in range(2):
            _save_image(image_records, f"img-{i}.png")
        service.start(_make_invoker(images_service, index_records))

        _wait_until(lambda: len(service._failed) == 2, timeout=20.0)
        # Charged to the images, not to the machinery.
        assert service._systemic_failures == 0
        # And the index settles rather than retrying them forever.
        _wait_until(lambda: service.get_status().pending == 0, timeout=15.0)
    finally:
        service.stop()


def test_systemic_backoff_grows_and_is_capped(service: ImageIndexService) -> None:
    service._systemic_failures = 0
    assert service._backoff_seconds() == _POLL_SECONDS
    service._systemic_failures = 1
    assert service._backoff_seconds() == _POLL_SECONDS
    service._systemic_failures = 3
    assert service._backoff_seconds() == _POLL_SECONDS * 4
    # Capped, and no overflow for an outage that lasts a very long time.
    service._systemic_failures = 10_000
    assert service._backoff_seconds() == _MAX_BACKOFF_SECONDS


def test_zero_norm_embedding_fails_only_its_own_image(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
) -> None:
    """A degenerate encoder row must not cost the rest of its batch their embeddings.

    An all-zero vector cannot be L2-normalized and is rejected by the storage layer, since it
    yields NaN in every similarity it takes part in. Both images here are encoded in one batch.
    """

    def encode(images: list[Image.Image]) -> np.ndarray:
        vectors = np.ones((len(images), DIM), dtype=np.float32)
        vectors[0] = 0.0  # first image of the batch is degenerate
        return vectors

    service = ImageIndexService(encode_fn=encode, model_id=MODEL_ID)
    try:
        _save_image(image_records, "a-bad.png")
        _save_image(image_records, "b-good.png")
        service.start(_make_invoker(images_service, index_records))
        _wait_until(lambda: not service._backfill_pending.is_set())

        # The healthy image is embedded despite sharing a batch with the degenerate one.
        assert index_records.get_embeddings(["b-good.png"], MODEL_ID)[0] == ["b-good.png"]
        assert index_records.get_embeddings(["a-bad.png"], MODEL_ID)[0] == []
        _wait_until(lambda: "a-bad.png" in service._failed)
        assert "b-good.png" not in service._failed
    finally:
        service.stop()


def test_start_discards_only_other_models_embeddings(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
) -> None:
    """The one destructive operation in the service: prove what it does and does not delete.

    `start()` prunes embeddings computed by a previously-configured model. If it ever pruned
    the current model's rows the whole index would be silently rebuilt from scratch on every
    boot, and if it pruned nothing the index would accumulate dead rows forever.
    """
    _save_image(image_records, "a.png")
    index_records.upsert_embedding("a.png", MODEL_ID, _unit_vec())
    index_records.upsert_embedding("a.png", "stale-model-hash", _unit_vec())
    assert index_records.get_embeddings(["a.png"], "stale-model-hash")[0] == ["a.png"]

    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    try:
        service.start(_make_invoker(images_service, index_records))
        _wait_until(lambda: not service._backfill_pending.is_set())

        assert index_records.get_embeddings(["a.png"], "stale-model-hash")[0] == []
        assert index_records.get_embeddings(["a.png"], MODEL_ID)[0] == ["a.png"]
    finally:
        service.stop()


def test_disabled_service_does_not_discard_embeddings(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
) -> None:
    """Turning the feature off must not destroy an index built while it was on."""
    _save_image(image_records, "a.png")
    index_records.upsert_embedding("a.png", "stale-model-hash", _unit_vec())

    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    try:
        service.start(_make_invoker(images_service, index_records, enabled=False))
        assert index_records.get_embeddings(["a.png"], "stale-model-hash")[0] == ["a.png"]
    finally:
        service.stop()


def test_worker_waits_for_generation_to_finish_when_not_on_cpu(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
    accelerator_host: None,
) -> None:
    """The VRAM contract: off the CPU path, embedding must pause while a generation runs.

    Every other test sets device='cpu' and session_queue=None, so `_wait_for_idle_generation`
    returns at its first statement and this contract is never exercised.
    """
    queue_status = SimpleNamespace(in_progress=1)
    session_queue = SimpleNamespace(get_queue_status=lambda queue_id: queue_status)

    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    try:
        _save_image(image_records, "a.png")
        service.start(_make_invoker(images_service, index_records, device=None, session_queue=session_queue))

        # Generation in progress: the worker must hold off rather than embed.
        time.sleep(0.5)
        assert index_records.count_index_status(MODEL_ID).embedded == 0

        queue_status.in_progress = 0
        _wait_until(lambda: index_records.count_index_status(MODEL_ID).embedded == 1, timeout=15.0)
    finally:
        service.stop()


def test_generation_wait_does_not_block_shutdown(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
    accelerator_host: None,
) -> None:
    """A generation that never ends must not stop the worker from honouring stop()."""
    session_queue = SimpleNamespace(get_queue_status=lambda queue_id: SimpleNamespace(in_progress=1))

    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    _save_image(image_records, "a.png")
    service.start(_make_invoker(images_service, index_records, device=None, session_queue=session_queue))
    time.sleep(0.2)

    started = time.monotonic()
    service.stop()
    assert time.monotonic() - started < 5.0
    assert service._worker is not None and not service._worker.is_alive()


def test_unparseable_device_is_ignored_rather_than_wedging_the_worker(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
) -> None:
    """`image_index_device` is free-form config with no validator.

    A near-miss like 'CPU' used to be handed to torch.device(), which raises — inside the
    worker loop, before any batch was attempted, so the _MAX_ATTEMPTS bound never applied and
    the worker spun on the same exception forever with pending stuck above zero.
    """
    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    try:
        _save_image(image_records, "a.png")
        service.start(_make_invoker(images_service, index_records, device="CPU"))
        _wait_until(lambda: index_records.count_index_status(MODEL_ID).embedded == 1, timeout=15.0)
        assert service.get_status().pending == 0
    finally:
        service.stop()


def test_empty_model_name_does_not_resolve_to_an_arbitrary_model(
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
) -> None:
    """`search_by_attr` drops its name predicate for a falsy name.

    Without a guard an empty `image_index_model` adopts whichever model happens to sort first
    and then discards every embedding computed by the model the user actually configured.
    """
    installed = SimpleNamespace(key="some-key", name="clip-vit-large-patch14", hash="some-hash")
    store = SimpleNamespace(search_by_attr=lambda model_name, model_type: [installed])
    model_manager = SimpleNamespace(store=store)

    service = ImageIndexService()
    service._invoker = _make_invoker(images_service, index_records, model_manager=model_manager)

    assert service._resolve_model_config("") is None
    # Sanity: the same store does resolve a real name, so the None above is the guard talking
    # and not simply an empty store.
    assert service._resolve_model_config("clip-vit-large-patch14") is installed


def test_duplicate_model_names_resolve_deterministically(
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
) -> None:
    """Two models can share a name, and the pick must not move when one is reinstalled.

    `search_by_attr` breaks ties by insertion order, so reinstalling a duplicate flips which
    config wins. That changes the model hash, and `start()` then discards every embedding
    computed under the previous one — the whole index, silently.
    """
    a = SimpleNamespace(key="key-a", name="clip-vit-large-patch14", hash="hash-a")
    b = SimpleNamespace(key="key-b", name="clip-vit-large-patch14", hash="hash-b")

    def resolve(order: list[SimpleNamespace]) -> SimpleNamespace:
        store = SimpleNamespace(search_by_attr=lambda model_name, model_type: list(order))
        service = ImageIndexService()
        service._invoker = _make_invoker(images_service, index_records, model_manager=SimpleNamespace(store=store))
        return service._resolve_model_config("clip-vit-large-patch14")

    # Same set, either insertion order — the winner must not move.
    assert resolve([a, b]).key == resolve([b, a]).key == "key-a"


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


def test_upsert_value_error_also_routes_through_retry(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
    service: ImageIndexService,
) -> None:
    """ValueError must reach the retry path like any other raise.

    The storage layer uses ValueError for rejected embeddings, so it is tempting to catch it
    per-image and skip to the next one. That strands the image: the batch would still report
    full success, the backfill would never be re-armed, and `pending` would sit above zero
    forever with the image at one attempt and never retried.
    """
    invoker = _make_invoker(images_service, index_records)
    service.start(invoker)
    _wait_until(lambda: not service._backfill_pending.is_set())

    real_upsert = index_records.upsert_embedding
    calls = {"count": 0}

    def flaky_upsert(name: str, model_id: str, embedding: np.ndarray) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("rejected by a future storage-layer validation")
        real_upsert(name, model_id, embedding)

    index_records.upsert_embedding = flaky_upsert  # type: ignore[method-assign]

    _save_image(image_records, "flaky.png")
    images_service._on_changed(_dto_for(image_records, "flaky.png"))

    _wait_until(lambda: index_records.get_embeddings(["flaky.png"], MODEL_ID)[0] == ["flaky.png"], timeout=15.0)
    _wait_until(lambda: any(e.total == 1 and e.embedded == 1 and e.pending == 0 for e in _status_events(invoker)))


def test_normalizable_extreme_magnitudes_are_not_dropped(
    image_records: SqliteImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecordsSqlite,
) -> None:
    """Tiny-but-normalizable vectors must not be misread as degenerate.

    In float32 the sum of squares underflows long before the vector itself does, so computing
    the norm at the storage dtype would report 0.0 for a row that normalizes perfectly.
    """

    def encode(images: list[Image.Image]) -> np.ndarray:
        return np.full((len(images), DIM), 1e-25, dtype=np.float32)

    service = ImageIndexService(encode_fn=encode, model_id=MODEL_ID)
    try:
        _save_image(image_records, "tiny.png")
        service.start(_make_invoker(images_service, index_records))
        _wait_until(lambda: not service._backfill_pending.is_set())

        names, matrix = index_records.get_embeddings(["tiny.png"], MODEL_ID)
        assert names == ["tiny.png"]
        assert np.isclose(np.linalg.norm(matrix[0]), 1.0)
        assert "tiny.png" not in service._failed
    finally:
        service.stop()


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
