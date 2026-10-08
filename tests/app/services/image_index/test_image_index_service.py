"""Tests for the image index worker service, using an injected fake encoder (no models/GPU)."""

import inspect
import threading
import time
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.events.events_common import ImageIndexStatusEvent, ImageIndexUpdatedEvent
from invokeai.app.services.image_index import image_index_default
from invokeai.app.services.image_index.image_index_common import EMBEDDING_DTYPE, IndexedItem
from invokeai.app.services.image_index.image_index_default import (
    _ACTIVATION_RETRY_INTERVAL_S,
    _MAX_ATTEMPTS,
    _MAX_BACKOFF_SECONDS,
    _POLL_SECONDS,
    ImageIndexService,
)
from invokeai.app.services.image_index.image_index_records_default import ImageIndexRecords
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.images.images_common import image_record_to_dto
from invokeai.app.services.images.images_default import ImageService
from invokeai.app.services.model_load.model_load_default import ModelLoadService
from invokeai.app.services.model_records.model_records_sql import ModelRecordServiceSQL
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import LockTimeoutError
from invokeai.app.services.video_records.video_records_default import VideoRecordStorage
from invokeai.app.services.videos.videos_common import VideoDTO, video_record_to_dto
from invokeai.app.services.videos.videos_default import VideoService
from invokeai.backend.model_manager.configs.clip_vision import CLIPVision_Diffusers_Config
from invokeai.backend.model_manager.configs.siglip import SigLIP_Diffusers_Config
from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
from invokeai.backend.model_manager.taxonomy import ModelRepoVariant, ModelSourceType
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database
from tests.test_nodes import TestEventService

MODEL_ID = "test-model-hash"
DIM = 8
# An encoder swap evicts the retired model from every per-device cache; most tests have none.
_NO_MODEL_CACHES = SimpleNamespace(ram_caches={})


def _wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Condition not met within timeout")


def _wait_for_spent_retry(service: "ImageIndexService", user_id: str, scope: str) -> None:
    """Wait until the worker has actually charged a failed scope's retry.

    The budget is spent only once the failed result is durably cached, which is
    strictly after the fit runs and after the job leaves the request map — so
    waiting on the fit count or on an empty `_projection_requests` and then
    asserting the refusal races the worker, and loses on a slow runner.
    """
    _wait_until(lambda: service._failed_projection_scopes.get(user_id) == scope, timeout=15)


def imgs(*names: str) -> list[IndexedItem]:
    """The image-namespace items for these names."""
    return [IndexedItem("image", name) for name in names]


def vids(*names: str) -> list[IndexedItem]:
    """The video-namespace items for these names."""
    return [IndexedItem("video", name) for name in names]


def _unit_vec() -> np.ndarray:
    """A storable embedding: float32, finite, non-zero."""
    v = np.ones(DIM, dtype=np.float32)
    return v / np.linalg.norm(v)


def _fake_encode(images: list[Image.Image]) -> np.ndarray:
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((len(images), DIM)).astype(np.float32)
    return vectors


@pytest.fixture
def db() -> Database:
    config = InvokeAIAppConfig(use_memory_db=True)
    return create_mock_sqlite_database(config=config, logger=InvokeAILogger.get_logger())


@pytest.fixture
def image_records(db: Database) -> ImageRecordStorage:
    return ImageRecordStorage(db)


@pytest.fixture
def index_records(db: Database) -> ImageIndexRecords:
    return ImageIndexRecords(db)


@pytest.fixture(params=[CLIPVision_Diffusers_Config, SigLIP_Diffusers_Config], ids=["clip", "siglip"])
def encoder_config(
    request: pytest.FixtureRequest, tmp_path: Path
) -> CLIPVision_Diffusers_Config | SigLIP_Diffusers_Config:
    return request.param(
        key="encoder-key",
        hash=MODEL_ID,
        path=str(tmp_path / "encoder"),
        file_size=1,
        name="encoder",
        source=str(tmp_path / "encoder"),
        source_type=ModelSourceType.Path,
    )


@pytest.fixture
def images_service() -> ImageService:
    images = ImageService()
    # The worker only needs get_pil_image; bypass file storage entirely.
    images.get_pil_image = lambda image_name: Image.new("RGB", (16, 16), "purple")  # type: ignore[method-assign]
    return images


@pytest.fixture
def video_records(db: Database) -> VideoRecordStorage:
    return VideoRecordStorage(db)


@pytest.fixture
def videos_service(tmp_path: Path) -> VideoService:
    """A video service whose thumbnails are real files, as the worker reads them from disk."""
    videos = VideoService()
    thumbnails = tmp_path / "video-thumbnails"
    thumbnails.mkdir()

    def get_path(video_name: str, thumbnail: bool = False) -> str:
        assert thumbnail, "the index only ever reads a video's thumbnail"
        path = thumbnails / f"{video_name}.webp"
        if not path.exists():
            Image.new("RGB", (16, 16), "teal").save(path, "WEBP")
        return str(path)

    videos.get_path = get_path  # type: ignore[method-assign]
    return videos


def _make_invoker(
    images_service: ImageService,
    index_records: ImageIndexRecords,
    enabled: bool = True,
    image_records: ImageRecordStorage | None = None,
    device: str | None = "cpu",
    session_queue: object | None = None,
    model_manager: object | None = None,
    videos_service: VideoService | None = None,
    video_records: VideoRecordStorage | None = None,
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
        videos=videos_service if videos_service is not None else VideoService(),
        video_records=video_records,
        image_index_records=index_records,
        events=TestEventService(),
        session_queue=session_queue,
        model_manager=model_manager if model_manager is not None else SimpleNamespace(load=_NO_MODEL_CACHES),
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
    image_records: ImageRecordStorage,
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


def _save_video(
    video_records: VideoRecordStorage,
    video_name: str,
    is_intermediate: bool = False,
    video_category: ImageCategory = ImageCategory.GENERAL,
) -> None:
    video_records.save(
        video_name=video_name,
        video_origin=ResourceOrigin.INTERNAL,
        video_category=video_category,
        width=16,
        height=16,
        duration=2.0,
        fps=24.0,
        has_workflow=False,
        is_intermediate=is_intermediate,
        user_id="system",
    )


def _video_dto_for(video_records: VideoRecordStorage, video_name: str) -> VideoDTO:
    record = video_records.get(video_name)
    return video_record_to_dto(record, video_url="http://x/v.mp4", thumbnail_url="http://x/v.webp", board_id=None)


def _dto_for(image_records: ImageRecordStorage, image_name: str):
    record = image_records.get(image_name)
    return image_record_to_dto(record, image_url="http://x/i.png", thumbnail_url="http://x/t.png", board_id=None)


def test_constructor_requires_matched_test_seams() -> None:
    with pytest.raises(ValueError):
        ImageIndexService(encode_fn=_fake_encode)
    with pytest.raises(ValueError):
        ImageIndexService(model_id=MODEL_ID)


def test_disabled_service_is_inert(
    images_service: ImageService, index_records: ImageIndexRecords, service: ImageIndexService
) -> None:
    invoker = _make_invoker(images_service, index_records, enabled=False)
    service.start(invoker)

    assert service.model_id is None
    assert service.get_status() is None
    assert images_service._on_changed_callbacks == []
    assert images_service._on_deleted_callbacks == []


def test_backfill_indexes_preexisting_eligible_images(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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
    assert index_records.get_embeddings(imgs("intermediate.png", "mask.png"), MODEL_ID)[0] == []
    # Stored embeddings are L2-normalized.
    _, matrix = index_records.get_embeddings([IndexedItem("image", f"img-{i}.png") for i in range(6)], MODEL_ID)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)


def test_on_changed_indexes_new_eligible_image_and_skips_ineligible(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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

    _wait_until(lambda: index_records.get_embeddings(imgs("new.png"), MODEL_ID)[0] == imgs("new.png"))
    assert index_records.get_embeddings(imgs("new-intermediate.png"), MODEL_ID)[0] == []


def test_unloadable_image_is_skipped_and_backfill_completes(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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
    assert index_records.get_embeddings(imgs("good.png"), MODEL_ID)[0] == imgs("good.png")
    assert index_records.get_embeddings(imgs("bad.png"), MODEL_ID)[0] == []
    assert IndexedItem("image", "bad.png") in service._failed


def test_backfill_logs_what_it_indexed_and_that_it_finished(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
    caplog: pytest.LogCaptureFixture,
) -> None:
    for i in range(3):
        _save_image(image_records, f"img-{i}.png")

    with caplog.at_level("INFO", logger="InvokeAI"):
        service.start(_make_invoker(images_service, index_records))
        # Waited on the log line itself: the pass reports its outcome after clearing the flag
        # that says it is running, so waiting on the flag races the message.
        _wait_until(lambda: "Image index: indexing complete (3 of 3 items indexed)" in caplog.text)
        assert "Image index: indexing 3 item(s)" in caplog.text

        # An image arriving afterwards is embedded from the queue, not by a backfill pass,
        # so it does not re-announce one.
        caplog.clear()
        _save_image(image_records, "new.png")
        images_service._on_changed(_dto_for(image_records, "new.png"))
        _wait_until(lambda: index_records.get_embeddings(imgs("new.png"), MODEL_ID)[0] == imgs("new.png"))
        assert "Image index: indexing" not in caplog.text


def test_restart_over_an_indexed_gallery_logs_nothing(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The common case on every restart: a pass with nothing to do must not announce itself,
    # or the log gains a line per server start that says nothing happened.
    _save_image(image_records, "already.png")
    index_records.upsert_embedding(IndexedItem("image", "already.png"), MODEL_ID, _unit_vec())
    invoker = _make_invoker(images_service, index_records)

    with caplog.at_level("INFO", logger="InvokeAI"):
        service.start(invoker)
        # The sweep that found nothing emits the status event, so this is the observable that
        # the pass has been and gone.
        _wait_until(lambda: any(e.embedded == 1 and e.pending == 0 for e in _status_events(invoker)))
        assert "Image index: indexing" not in caplog.text


def test_backfill_reports_images_it_could_not_embed(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _save_image(image_records, "good.png")
    _save_image(image_records, "bad.png")

    def get_pil_image(image_name: str) -> Image.Image:
        if image_name == "bad.png":
            raise FileNotFoundError(image_name)
        return Image.new("RGB", (16, 16), "purple")

    images_service.get_pil_image = get_pil_image  # type: ignore[method-assign]

    with caplog.at_level("INFO", logger="InvokeAI"):
        service.start(_make_invoker(images_service, index_records))
        # The pass ends because the failure retired, so the outcome has to say so rather than
        # claiming a complete index.
        _wait_until(lambda: "Image index: indexing finished with 1 item(s) that could not be embedded" in caplog.text)
        assert "indexing complete" not in caplog.text

        # A later pass speaks for itself: the image that retired earlier is still in the
        # cumulative failure set, but nothing failed this time.
        caplog.clear()
        _save_image(image_records, "later.png")
        service._backfill_pending.set()
        _wait_until(lambda: "Image index: indexing complete (2 of 3 items indexed)" in caplog.text)
        assert "could not be embedded" not in caplog.text


def test_transient_encode_failure_is_retried_to_success(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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
        _wait_until(lambda: index_records.get_embeddings(imgs("a.png"), MODEL_ID)[0] == imgs("a.png"), timeout=15)
        assert IndexedItem("image", "a.png") not in service._failed
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
    assert "apple/DFN2B-CLIP-ViT-L-14-39B" in message

    service._invoker.services.model_manager.store.search_by_attr = lambda model_name=None: []
    message = service._model_not_installed_message("clip-vit-large-patch14")
    assert "is not installed" in message


def test_try_activate_picks_up_a_model_installed_after_startup(
    encoder_config: CLIPVision_Diffusers_Config | SigLIP_Diffusers_Config,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The encoder is usually installed from the image map itself, long after
    # the server came up; that must not need a restart.
    installed: list[object] = []
    resolutions = 0

    def resolve(self, model_name: str):
        nonlocal resolutions
        resolutions += 1
        return installed[0] if installed else None

    monkeypatch.setattr(ImageIndexService, "_resolve_model_config", resolve)
    store = SimpleNamespace(search_by_attr=lambda **kwargs: [])
    invoker = _make_invoker(
        images_service, index_records, model_manager=SimpleNamespace(store=store, load=_NO_MODEL_CACHES)
    )
    service = ImageIndexService()
    try:
        service.start(invoker)
        assert service.model_id is None

        # Throttled: start() has just resolved, so a poll cannot re-query the
        # model store on its heels.
        assert service.try_activate() is False
        assert resolutions == 1

        service._last_activation_attempt = time.monotonic() - _ACTIVATION_RETRY_INTERVAL_S - 1
        assert service.try_activate() is False
        assert resolutions == 2

        installed.append(encoder_config)
        service._last_activation_attempt = time.monotonic() - _ACTIVATION_RETRY_INTERVAL_S - 1

        assert service.try_activate() is True
        assert service.model_id == MODEL_ID
        _wait_until(lambda: service._worker is not None and service._worker.is_alive())
        # Running and missing states share the same throttled catalog check.
        assert service.try_activate() is True
        assert resolutions == 3
        service._last_activation_attempt = time.monotonic() - _ACTIVATION_RETRY_INTERVAL_S - 1
        assert service.try_activate() is True
        assert resolutions == 4
    finally:
        service.stop()

    # A request racing shutdown must not start a worker the invoker will never join.
    service._worker = None
    service._model_id = None
    service._last_activation_attempt = time.monotonic() - _ACTIVATION_RETRY_INTERVAL_S - 1
    assert service.try_activate() is False


def test_failed_late_activation_rolls_back_rather_than_wedging_the_service(
    encoder_config: CLIPVision_Diffusers_Config | SigLIP_Diffusers_Config,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Publishing the model before the worker exists would leave the service
    # claiming to run with nothing consuming the queue — and try_activate's
    # fast path would answer True forever, so no later request would retry.
    monkeypatch.setattr(ImageIndexService, "_resolve_model_config", lambda self, model_name: encoder_config)
    launches = 0

    def launch(self, invoker):
        nonlocal launches
        launches += 1
        if launches == 1:
            raise RuntimeError("sqlite write failed")
        return original_launch(self, invoker)

    original_launch = ImageIndexService._launch_worker
    store = SimpleNamespace(search_by_attr=lambda **kwargs: [])
    invoker = _make_invoker(
        images_service, index_records, model_manager=SimpleNamespace(store=store, load=_NO_MODEL_CACHES)
    )
    service = ImageIndexService()
    try:
        # Inert at start: the encoder was not installed yet.
        monkeypatch.setattr(ImageIndexService, "_resolve_model_config", lambda self, model_name: None)
        service.start(invoker)
        assert service.model_id is None

        monkeypatch.setattr(ImageIndexService, "_resolve_model_config", lambda self, model_name: encoder_config)
        monkeypatch.setattr(ImageIndexService, "_launch_worker", launch)
        service._last_activation_attempt = time.monotonic() - _ACTIVATION_RETRY_INTERVAL_S - 1

        assert service.try_activate() is False
        # Rolled back, so the state a request reads still says "not running".
        assert service.model_id is None
        assert service._encode_fn is None
        assert service.get_status() is None

        service._last_activation_attempt = time.monotonic() - _ACTIVATION_RETRY_INTERVAL_S - 1

        assert service.try_activate() is True
        assert service.model_id == MODEL_ID
        _wait_until(lambda: service._worker is not None and service._worker.is_alive())
    finally:
        service.stop()


@pytest.mark.parametrize("normalize_loaded_path", [False, True], ids=["catalog-path", "normalized-loaded-path"])
def test_encoder_metadata_edits_keep_worker_indexing_without_map_requests(
    normalize_loaded_path: bool,
    db: Database,
    encoder_config: CLIPVision_Diffusers_Config | SigLIP_Diffusers_Config,
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models_dir = Path(encoder_config.path).parent
    encoder_config.path = "encoder"
    store = ModelRecordServiceSQL(db, InvokeAILogger.get_logger())
    store.add_model(encoder_config)
    invoker = _make_invoker(
        images_service,
        index_records,
        image_records=image_records,
        model_manager=SimpleNamespace(store=store, load=_NO_MODEL_CACHES),
    )
    invoker.services.configuration.image_index_model = encoder_config.name
    invoker.services.configuration.models_dir = models_dir

    def encode(self, images):
        if normalize_loaded_path:
            # The shared model loader makes its supplied config path absolute.
            self._model_config.path = str((models_dir / self._model_config.path).resolve())
        return _fake_encode(images)

    monkeypatch.setattr(ImageIndexService, "_encode_with_model", encode)
    service = ImageIndexService()
    try:
        _save_image(image_records, "before-edit.png")
        service.start(invoker)
        _wait_until(lambda: index_records.count_index_status(MODEL_ID).embedded == 1)
        worker = service._worker

        # Edit the catalog between worker checks, retaining the same configured
        # encoder after its rename. No map request drives this reconciliation.
        with service._activation_lock:
            store.replace_model(
                encoder_config.key,
                encoder_config.model_copy(
                    update={"name": "renamed encoder", "description": "Edited description", "cover_image": "cover.png"}
                ),
            )
            invoker.services.configuration.image_index_model = "renamed encoder"
            expired = time.monotonic() - _ACTIVATION_RETRY_INTERVAL_S - 1
            service._last_activation_attempt = expired
        _wait_until(lambda: service._last_activation_attempt != expired)
        # The timestamp is written before the catalog read. Wait for the whole
        # check to finish before observing availability or adding new work.
        with service._activation_lock:
            assert service.model_id == MODEL_ID
        assert service._worker is worker
        assert worker is not None and worker.is_alive()

        _save_image(image_records, "after-edit.png")
        images_service._on_changed(_dto_for(image_records, "after-edit.png"))
        _wait_until(lambda: index_records.count_index_status(MODEL_ID).embedded == 2)
    finally:
        service.stop()


def test_deleted_encoder_becomes_unavailable_and_reinstall_resumes_without_duplicate_callbacks(
    encoder_config: CLIPVision_Diffusers_Config | SigLIP_Diffusers_Config,
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = [encoder_config]
    monkeypatch.setattr(
        ImageIndexService, "_resolve_model_config", lambda self, name: installed[0] if installed else None
    )
    monkeypatch.setattr(ImageIndexService, "_encode_with_model", lambda self, images: _fake_encode(images))
    invoker = _make_invoker(images_service, index_records, image_records=image_records)
    service = ImageIndexService()
    try:
        _save_image(image_records, "before.png")
        service.start(invoker)
        _wait_until(lambda: index_records.count_index_status(MODEL_ID).embedded == 1)
        service._processor = object()
        service._cpu_model = object()
        service._text_encoder_failure = "previous installation was incomplete"
        service._vocab_cache = (["old"], np.ones((1, DIM), dtype=np.float32))

        installed.clear()
        service._last_activation_attempt = time.monotonic() - _ACTIVATION_RETRY_INTERVAL_S - 1
        assert service.try_activate() is False
        assert service.model_id is None
        assert not service.replacing_model
        assert service.get_status() is None
        assert service.get_vocab_build_state() == ("unavailable", None)
        assert service.request_projection("system") is False
        # The exiting worker releases the deleted encoder's resources itself.
        _wait_until(lambda: not service._worker.is_alive())
        assert service._processor is None
        assert service._cpu_model is None
        assert service._vocab_cache is None

        _save_image(image_records, "during.png")
        images_service._on_changed(_dto_for(image_records, "during.png"))
        assert service._queue.empty()
        installed.append(encoder_config.model_copy(update={"path": encoder_config.path + "-reinstalled"}))
        service._last_activation_attempt = time.monotonic() - _ACTIVATION_RETRY_INTERVAL_S - 1
        assert service.try_activate() is True
        _wait_until(lambda: index_records.count_index_status(MODEL_ID).embedded == 2)
        assert len(images_service._on_changed_callbacks) == 1
        assert len(images_service._on_deleted_callbacks) == 1
        assert len(invoker.services.videos._on_changed_callbacks) == 1
        assert len(invoker.services.videos._on_deleted_callbacks) == 1
        assert service._processor is None
        assert service._cpu_model is None
        assert service._text_encoder_failure is None
        assert service._vocab_cache is None
    finally:
        service.stop()


@pytest.mark.parametrize("replacement_kind", ["reinstall", "key", "path", "cpu_only", "repo_variant", "type"])
@pytest.mark.parametrize("background_batch", [False, True], ids=["query", "worker"])
def test_encoder_replacement_waits_for_inflight_embedding_and_discards_retired_batch(
    replacement_kind: str,
    background_batch: bool,
    encoder_config: CLIPVision_Diffusers_Config | SigLIP_Diffusers_Config,
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = [encoder_config]
    entered = threading.Event()
    release = threading.Event()
    replacements = {
        "reinstall": {"hash": "replacement-model-hash", "path": encoder_config.path + "-replacement"},
        "key": {"key": "replacement-key"},
        "path": {"path": encoder_config.path + "-moved"},
        "cpu_only": {"cpu_only": True},
        "repo_variant": {"repo_variant": ModelRepoVariant.FP16},
    }
    if replacement_kind == "type":
        replacement_type = (
            SigLIP_Diffusers_Config
            if isinstance(encoder_config, CLIPVision_Diffusers_Config)
            else CLIPVision_Diffusers_Config
        )
        replacement = replacement_type(**encoder_config.model_dump(exclude={"type"}))
    else:
        replacement = encoder_config.model_copy(update=replacements[replacement_kind])
    seen_models: list[CLIPVision_Diffusers_Config | SigLIP_Diffusers_Config] = []

    def encode(self, images):
        seen_models.append(self._model_config)
        if len(seen_models) == 1:
            entered.set()
            assert release.wait(timeout=10)
            assert self._model_config is encoder_config
        return _fake_encode(images)

    monkeypatch.setattr(
        ImageIndexService, "_resolve_model_config", lambda self, name: installed[0] if installed else None
    )
    monkeypatch.setattr(ImageIndexService, "_encode_with_model", encode)
    service = ImageIndexService()
    query = None
    try:
        if background_batch:
            _save_image(image_records, "pending.png")
        service.start(_make_invoker(images_service, index_records, image_records=image_records))
        if not background_batch:
            query = threading.Thread(target=lambda: service.embed_image(Image.new("RGB", (16, 16))))
            query.start()
        assert entered.wait(timeout=10)

        if replacement_kind == "reinstall":
            installed.clear()
        else:
            installed[0] = replacement
        service._last_activation_attempt = time.monotonic() - _ACTIVATION_RETRY_INTERVAL_S - 1
        assert service.try_activate() is False
        assert service.model_id is None
        assert service.replacing_model is (replacement_kind != "reinstall")
        installed[:] = [replacement]
        service._last_activation_attempt = time.monotonic() - _ACTIVATION_RETRY_INTERVAL_S - 1
        assert service.try_activate() is False
        assert service.replacing_model
        assert service._model_config is encoder_config

        release.set()
        if query is not None:
            query.join(timeout=10)
            assert not query.is_alive()
        # No further request: the retiring worker completes the swap once its users drain.
        _wait_until(lambda: service.model_id == replacement.hash)
        assert not service.replacing_model
        if background_batch:
            # Re-encoded by the replacement: the retired encoder's batch was not stored.
            _wait_until(lambda: index_records.count_index_status(replacement.hash).embedded == 1)
        else:
            service.embed_image(Image.new("RGB", (16, 16)))
        assert seen_models == [encoder_config, replacement]
    finally:
        release.set()
        if query is not None:
            query.join(timeout=10)
        service.stop()


def _write_clip_vision_weights(path: Path, seed: int) -> None:
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection

    config = CLIPVisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=224,
        patch_size=112,
        projection_dim=DIM,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        CLIPVisionModelWithProjection(config).save_pretrained(path)


def test_reinstalled_encoder_keeping_its_key_embeds_with_the_new_weights(
    tmp_path: Path,
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The shared model cache is keyed by model key alone. Reinstalling the encoder in place changes
    # its hash but not its key, and embeddings computed by the cached, retired weights would be
    # stored under the replacement's hash and survive a restart.
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

    encoder_path = tmp_path / "encoder"
    _write_clip_vision_weights(encoder_path, seed=1)
    retired = CLIPVision_Diffusers_Config(
        key="encoder-key",
        hash="retired-hash",
        path=str(encoder_path),
        file_size=1,
        name="encoder",
        source=str(encoder_path),
        source_type=ModelSourceType.Path,
    )
    installed = [retired]
    monkeypatch.setattr(ImageIndexService, "_resolve_model_config", lambda self, name: installed[0])
    # Embed through the shared cache, as an accelerator host does, but on CPU tensors.
    monkeypatch.setattr(TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
    monkeypatch.setattr(ImageIndexService, "_cpu_mode", lambda self: False)
    invoker = _make_invoker(images_service, index_records, image_records=image_records, device=None)
    cache = ModelCache(
        execution_device_working_mem_gb=1,
        enable_partial_loading=False,
        keep_ram_copy_of_weights=True,
        execution_device=torch.device("cpu"),
        logger=invoker.services.logger,
        # The default store is process-global; keep this encoder's weights out of it.
        shared_cpu_weights=None,
    )
    invoker.services.model_manager = SimpleNamespace(
        load=ModelLoadService(app_config=invoker.services.configuration, ram_cache=cache)
    )
    item = IndexedItem("image", "a.png")
    service = ImageIndexService()
    try:
        _save_image(image_records, item.name)
        service.start(invoker)
        _wait_until(lambda: index_records.count_index_status(retired.hash).embedded == 1)
        retired_embedding = index_records.get_embeddings([item], retired.hash)[1][0]

        _write_clip_vision_weights(encoder_path, seed=2)
        replacement = retired.model_copy(update={"hash": "replacement-hash"})
        with service._activation_lock:
            installed[0] = replacement
            service._last_activation_attempt = time.monotonic() - _ACTIVATION_RETRY_INTERVAL_S - 1
        _wait_until(lambda: index_records.count_index_status(replacement.hash).embedded == 1)

        with torch.no_grad():
            pixel_values = CLIPImageProcessor()(images=images_service.get_pil_image(item.name), return_tensors="pt")
            expected = CLIPVisionModelWithProjection.from_pretrained(encoder_path)(**pixel_values).image_embeds[0]
        expected = (expected / expected.norm()).numpy()
        stored = index_records.get_embeddings([item], replacement.hash)[1][0]
        assert not np.allclose(expected, retired_embedding, atol=1e-3), "the two encoders must disagree"
        np.testing.assert_allclose(stored, expected, atol=1e-4)
        cosine = float(np.dot(stored.astype(np.float64), expected.astype(np.float64)))
        assert 1.0 - cosine <= 1e-6
    finally:
        service.stop()
        cache.shutdown()


def test_worker_detected_replacement_resumes_indexing_without_requests(
    encoder_config: CLIPVision_Diffusers_Config | SigLIP_Diffusers_Config,
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Deleting and re-adding the encoder gives it a new key. With the image map
    # closed, nothing but the worker notices, and gallery search and new-image
    # indexing must not stay down until someone opens the map.
    installed = [encoder_config]
    monkeypatch.setattr(ImageIndexService, "_resolve_model_config", lambda self, name: installed[0])
    monkeypatch.setattr(ImageIndexService, "_encode_with_model", lambda self, images: _fake_encode(images))
    service = ImageIndexService()
    try:
        _save_image(image_records, "before.png")
        service.start(_make_invoker(images_service, index_records, image_records=image_records))
        _wait_until(lambda: index_records.count_index_status(MODEL_ID).embedded == 1)
        retired = service._worker

        replacement = encoder_config.model_copy(update={"key": "re-added-key"})
        with service._activation_lock:
            installed[0] = replacement
            service._last_activation_attempt = time.monotonic() - _ACTIVATION_RETRY_INTERVAL_S - 1

        _wait_until(lambda: service._model_config is replacement and service.model_id == MODEL_ID)
        assert retired is not None
        _wait_until(lambda: not retired.is_alive())
        assert service._worker is not retired and service._worker.is_alive()

        _save_image(image_records, "after.png")
        images_service._on_changed(_dto_for(image_records, "after.png"))
        _wait_until(lambda: index_records.count_index_status(MODEL_ID).embedded == 2)
        assert len(images_service._on_changed_callbacks) == 1
    finally:
        service.stop()


def test_stop_releases_a_worker_waiting_for_request_users_to_drain(
    encoder_config: CLIPVision_Diffusers_Config | SigLIP_Diffusers_Config,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = [encoder_config]
    monkeypatch.setattr(ImageIndexService, "_resolve_model_config", lambda self, name: installed[0])
    monkeypatch.setattr(ImageIndexService, "_encode_with_model", lambda self, images: _fake_encode(images))
    service = ImageIndexService()
    service.start(_make_invoker(images_service, index_records))
    worker = service._worker
    assert worker is not None
    with service.use_model():
        installed[0] = encoder_config.model_copy(update={"key": "replacement-key"})
        service._last_activation_attempt = time.monotonic() - _ACTIVATION_RETRY_INTERVAL_S - 1
        assert service.try_activate() is False
        assert service.replacing_model

        # Shutdown must not wait out the request, nor start the replacement.
        service.stop()
        assert not worker.is_alive()
        assert service._worker is worker
        assert not service.replacing_model
        assert service.model_id is None


def test_projection_finishing_after_encoder_removal_does_not_publish(
    encoder_config: CLIPVision_Diffusers_Config | SigLIP_Diffusers_Config,
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = [encoder_config]
    entered = threading.Event()
    release = threading.Event()

    def project(matrix):
        entered.set()
        assert release.wait(timeout=10)
        return np.zeros((len(matrix), 2), dtype=np.float32)

    monkeypatch.setattr(
        ImageIndexService, "_resolve_model_config", lambda self, name: installed[0] if installed else None
    )
    monkeypatch.setattr(ImageIndexService, "_encode_with_model", lambda self, images: _fake_encode(images))
    monkeypatch.setattr(image_index_default, "compute_umap", project)
    service = ImageIndexService()
    try:
        _save_image(image_records, "projected.png")
        service.start(_make_invoker(images_service, index_records, image_records=image_records))
        _wait_until(lambda: index_records.count_index_status(MODEL_ID).embedded == 1)
        assert service.request_projection("system")
        assert entered.wait(timeout=10)
        installed.clear()
        service._last_activation_attempt = time.monotonic() - _ACTIVATION_RETRY_INTERVAL_S - 1
        assert service.try_activate() is False
        release.set()
        _wait_until(lambda: not service._worker.is_alive())
        assert index_records.get_projection("system", MODEL_ID) is None
    finally:
        release.set()
        service.stop()


def test_late_activation_survives_a_model_store_failure(
    images_service: ImageService,
    index_records: ImageIndexRecords,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # These endpoints reported `model_missing` before they resolved anything;
    # a model-store failure must not turn them into 500s.
    def explode(self, model_name):
        raise RuntimeError("model store unavailable")

    store = SimpleNamespace(search_by_attr=lambda **kwargs: [])
    invoker = _make_invoker(
        images_service, index_records, model_manager=SimpleNamespace(store=store, load=_NO_MODEL_CACHES)
    )
    service = ImageIndexService()
    monkeypatch.setattr(ImageIndexService, "_resolve_model_config", lambda self, model_name: None)
    service.start(invoker)
    monkeypatch.setattr(ImageIndexService, "_resolve_model_config", explode)
    service._last_activation_attempt = time.monotonic() - _ACTIVATION_RETRY_INTERVAL_S - 1

    assert service.try_activate() is False
    assert service.model_id is None


def test_broken_encoder_leaves_images_pending_rather_than_quarantined(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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
        assert IndexedItem("image", "a.png") not in service._failed
        assert service.get_status().pending == 1
    finally:
        service.stop()


def test_status_event_reports_failures_so_a_settled_index_is_not_mistaken_for_complete(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
) -> None:
    """A database that is down is no more the images' fault than a missing model is.

    "database is locked" outlasting `_MAX_ATTEMPTS` sweeps must not retire the images, for the
    same reason an encoder outage must not: `_failed` would survive the database recovering.
    """
    outage = {"active": True}
    real_upsert = index_records.upsert_embedding

    def flaky_upsert(item: IndexedItem, model_id: str, embedding: np.ndarray) -> None:
        if outage["active"]:
            raise RuntimeError("database is locked")
        real_upsert(item, model_id, embedding)

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
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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
        assert index_records.get_embeddings(imgs("b-good.png"), MODEL_ID)[0] == imgs("b-good.png")
        assert index_records.get_embeddings(imgs("a-bad.png"), MODEL_ID)[0] == []
        _wait_until(lambda: IndexedItem("image", "a-bad.png") in service._failed)
        assert IndexedItem("image", "b-good.png") not in service._failed
    finally:
        service.stop()


def test_start_discards_only_other_models_embeddings(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
) -> None:
    """The one destructive operation in the service: prove what it does and does not delete.

    `start()` prunes embeddings computed by a previously-configured model. If it ever pruned
    the current model's rows the whole index would be silently rebuilt from scratch on every
    boot, and if it pruned nothing the index would accumulate dead rows forever.
    """
    _save_image(image_records, "a.png")
    index_records.upsert_embedding(IndexedItem("image", "a.png"), MODEL_ID, _unit_vec())
    index_records.upsert_embedding(IndexedItem("image", "a.png"), "stale-model-hash", _unit_vec())
    assert index_records.get_embeddings(imgs("a.png"), "stale-model-hash")[0] == imgs("a.png")

    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    try:
        service.start(_make_invoker(images_service, index_records))
        _wait_until(lambda: not service._backfill_pending.is_set())

        assert index_records.get_embeddings(imgs("a.png"), "stale-model-hash")[0] == []
        assert index_records.get_embeddings(imgs("a.png"), MODEL_ID)[0] == imgs("a.png")
    finally:
        service.stop()


def test_disabled_service_does_not_discard_embeddings(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
) -> None:
    """Turning the feature off must not destroy an index built while it was on."""
    _save_image(image_records, "a.png")
    index_records.upsert_embedding(IndexedItem("image", "a.png"), "stale-model-hash", _unit_vec())

    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    try:
        service.start(_make_invoker(images_service, index_records, enabled=False))
        assert index_records.get_embeddings(imgs("a.png"), "stale-model-hash")[0] == imgs("a.png")
    finally:
        service.stop()


def test_worker_waits_for_generation_to_finish_when_not_on_cpu(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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


def test_projection_does_not_wait_for_an_in_progress_generation(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    accelerator_host: None,
) -> None:
    """A projection reads stored embeddings only — no encoder, no GPU — so it has no
    reason to queue behind a generation the way an embed does.

    The worker parks in _wait_for_idle_generation as soon as ONE image is pending, and
    that wait is unbounded, so ordering the projection after it made /points report
    "computing" for the entire length of a run. Every other projection test builds its
    invoker with device='cpu'/session_queue=None, where the wait returns immediately —
    which is why this was invisible to the suite.
    """
    session_queue = SimpleNamespace(get_queue_status=lambda queue_id: SimpleNamespace(in_progress=1))

    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    try:
        # One image already embedded (so the projection has input) and one that cannot
        # be embedded while the generation holds the GPU (so the worker parks).
        _save_image(image_records, "done.png")
        index_records.upsert_embedding(IndexedItem("image", "done.png"), MODEL_ID, _unit_vec())
        _save_image(image_records, "waiting.png")

        service.start(_make_invoker(images_service, index_records, device=None, session_queue=session_queue))
        assert service.request_projection("system") is True

        # The generation never ends; the projection must land anyway.
        _wait_until(lambda: index_records.get_projection("system", MODEL_ID) is not None, timeout=20.0)
        record = index_records.get_projection("system", MODEL_ID)
        assert record is not None
        assert record.items == imgs("done.png")
        # And the embed really is still parked behind the generation.
        assert index_records.get_embeddings(imgs("waiting.png"), MODEL_ID)[0] == []
    finally:
        service.stop()


def test_a_partially_stored_batch_does_not_escalate_the_backoff(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
) -> None:
    """A batch that stores ANYTHING resets the systemic-failure counter, even though the
    batch also failed.

    The counter exists to stop a hot retry loop when NO progress is possible. A batch that
    stored an image is making progress: the backlog drains and quiescence arrives on its
    own, so escalating is wrong. Counting these instead — reachable by moving the reset off
    the `finally` — leaves no reset path at all while every batch partially fails, which
    walks the wait up to its 60s ceiling while the index is still working. That is worse
    under mild write contention than the flat 1Hz retry it would be correcting.
    """
    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    try:
        for name in ("stored.png", "locked.png"):
            _save_image(image_records, name)

        real_upsert = index_records.upsert_embedding

        def flaky_upsert(item, model_id, embedding):
            if item.name == "stored.png":
                return real_upsert(item, model_id, embedding)
            raise RuntimeError("database is locked")

        index_records.upsert_embedding = flaky_upsert  # type: ignore[method-assign]
        service._invoker = _make_invoker(images_service, index_records)
        service._model_id = MODEL_ID
        service._encode_fn = _fake_encode

        # Several rounds: the escalation this guards against is cumulative.
        for _ in range(8):
            assert service._process_batch(imgs("stored.png", "locked.png")) is False

        assert service._systemic_failures == 0, "progress must clear the outage counter"
        assert service._backoff_seconds() == _POLL_SECONDS, "a draining index must not back off"
        # The half that stored is stored, and no image was charged an attempt.
        assert index_records.get_embeddings(imgs("stored.png"), MODEL_ID)[0] == imgs("stored.png")
        assert service._failed == set()
        assert service._attempts == {}
    finally:
        service.stop()


def test_unparseable_device_is_ignored_rather_than_wedging_the_worker(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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
    index_records: ImageIndexRecords,
) -> None:
    """`search_by_attr` drops its name predicate for a falsy name.

    Without a guard an empty `image_index_model` adopts whichever model happens to sort first
    and then discards every embedding computed by the model the user actually configured.
    """
    installed = SimpleNamespace(key="some-key", name="clip-vit-large-patch14", hash="some-hash")
    store = SimpleNamespace(search_by_attr=lambda model_name, model_type: [installed])
    model_manager = SimpleNamespace(store=store, load=_NO_MODEL_CACHES)

    service = ImageIndexService()
    service._invoker = _make_invoker(images_service, index_records, model_manager=model_manager)

    assert service._resolve_model_config("") is None
    # Sanity: the same store does resolve a real name, so the None above is the guard talking
    # and not simply an empty store.
    assert service._resolve_model_config("clip-vit-large-patch14") is installed


def test_duplicate_model_names_resolve_deterministically(
    images_service: ImageService,
    index_records: ImageIndexRecords,
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
        service._invoker = _make_invoker(
            images_service, index_records, model_manager=SimpleNamespace(store=store, load=_NO_MODEL_CACHES)
        )
        return service._resolve_model_config("clip-vit-large-patch14")

    # Same set, either insertion order — the winner must not move.
    assert resolve([a, b]).key == resolve([b, a]).key == "key-a"


def test_status_event_emitted(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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

    _wait_until(lambda: IndexedItem("image", "bad.png") in service._failed, timeout=15.0)
    # Failed images are excluded from pending, so the index settles and the
    # final emitted status reports quiescence over the embeddable remainder.
    _wait_until(lambda: any(e.total == 2 and e.embedded == 1 and e.pending == 0 for e in _status_events(invoker)))
    status = service.get_status()
    assert status is not None
    assert status.pending == 0
    assert status.failed == 1


def test_upsert_failure_routes_through_retry_to_success(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
) -> None:
    """A raise while storing embeddings must feed the retry path, not strand the image."""
    invoker = _make_invoker(images_service, index_records)
    service.start(invoker)
    _wait_until(lambda: not service._backfill_pending.is_set())

    real_upsert = index_records.upsert_embedding
    calls = {"count": 0}

    def flaky_upsert(item: IndexedItem, model_id: str, embedding: np.ndarray) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("database is locked")
        real_upsert(item, model_id, embedding)

    index_records.upsert_embedding = flaky_upsert  # type: ignore[method-assign]

    _save_image(image_records, "flaky.png")
    images_service._on_changed(_dto_for(image_records, "flaky.png"))

    _wait_until(lambda: index_records.get_embeddings(imgs("flaky.png"), MODEL_ID)[0] == imgs("flaky.png"), timeout=15.0)
    _wait_until(lambda: any(e.total == 1 and e.embedded == 1 and e.pending == 0 for e in _status_events(invoker)))


def test_upsert_value_error_also_routes_through_retry(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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

    def flaky_upsert(item: IndexedItem, model_id: str, embedding: np.ndarray) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("rejected by a future storage-layer validation")
        real_upsert(item, model_id, embedding)

    index_records.upsert_embedding = flaky_upsert  # type: ignore[method-assign]

    _save_image(image_records, "flaky.png")
    images_service._on_changed(_dto_for(image_records, "flaky.png"))

    _wait_until(lambda: index_records.get_embeddings(imgs("flaky.png"), MODEL_ID)[0] == imgs("flaky.png"), timeout=15.0)
    _wait_until(lambda: any(e.total == 1 and e.embedded == 1 and e.pending == 0 for e in _status_events(invoker)))


def test_normalizable_extreme_magnitudes_are_not_dropped(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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

        names, matrix = index_records.get_embeddings(imgs("tiny.png"), MODEL_ID)
        assert names == imgs("tiny.png")
        assert np.isclose(np.linalg.norm(matrix[0]), 1.0)
        assert IndexedItem("image", "tiny.png") not in service._failed
    finally:
        service.stop()


def test_ineligible_transition_clears_failure_bookkeeping(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
) -> None:
    """An image that leaves eligibility must stop counting against `failed`."""
    invoker = _make_invoker(images_service, index_records)
    service.start(invoker)
    _wait_until(lambda: not service._backfill_pending.is_set())

    service._failed.add(IndexedItem("image", "gone.png"))
    service._attempts[IndexedItem("image", "gone.png")] = 3
    _save_image(image_records, "gone.png", image_category=ImageCategory.MASK)
    images_service._on_changed(_dto_for(image_records, "gone.png"))

    assert IndexedItem("image", "gone.png") not in service._failed
    assert IndexedItem("image", "gone.png") not in service._attempts
    status = service.get_status()
    assert status is not None
    assert status.failed == 0


def test_owner_poke_emitted_at_quiescence(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
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
    images_service: ImageService, index_records: ImageIndexRecords, service: ImageIndexService
) -> None:
    service.start(_make_invoker(images_service, index_records))
    assert service._worker is not None and service._worker.is_alive()

    service.stop()

    assert not service._worker.is_alive()


# --- Projection jobs ---


def test_projection_job_computes_and_caches(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
) -> None:
    from invokeai.app.services.events.events_common import ImageMapProjectionReadyEvent

    # Three images stay on compute_umap's deterministic PCA fallback: the
    # first real UMAP fit JIT-compiles numba, which blows CI timeouts on slow
    # (Windows/macOS) runners. The worker pipeline under test is identical.
    for i in range(3):
        _save_image(image_records, f"img-{i}.png")
    invoker = _make_invoker(images_service, index_records)
    service.start(invoker)
    _wait_until(lambda: not service._backfill_pending.is_set())

    assert service.request_projection("system") is True

    _wait_until(lambda: index_records.get_projection("system", MODEL_ID) is not None, timeout=30)
    record = index_records.get_projection("system", MODEL_ID)
    assert record is not None
    assert record.point_count == 3
    assert sorted(record.items) == [IndexedItem("image", f"img-{i}.png") for i in range(3)]
    assert record.coords.shape == (3, 2)
    _wait_until(
        lambda: any(
            isinstance(e, ImageMapProjectionReadyEvent) and e.point_count == 3 for e in invoker.services.events.events
        )
    )


def test_projection_failure_caches_empty_result_instead_of_looping(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
    monkeypatch,
) -> None:
    from invokeai.app.services.image_index.projection import scope_hash

    def broken_umap(embeddings, seed=42):
        raise RuntimeError("synthetic UMAP failure")

    monkeypatch.setattr(image_index_default, "compute_umap", broken_umap)
    _save_image(image_records, "a.png")
    service.start(_make_invoker(images_service, index_records))
    _wait_until(lambda: not service._backfill_pending.is_set())

    service.request_projection("system")

    _wait_until(lambda: index_records.get_projection("system", MODEL_ID) is not None, timeout=15)
    record = index_records.get_projection("system", MODEL_ID)
    assert record is not None
    assert record.point_count == 0
    # The empty cache claims the scope it failed against, so it is NOT stale —
    # clients see "empty" rather than re-enqueueing a doomed recompute forever.
    accessible = index_records.list_accessible_embedded_items(None, MODEL_ID)
    assert record.scope_hash == scope_hash(MODEL_ID, accessible)

    # ...but "not stale" must not mean "never again". Asserting only the state
    # above is what let the failure become terminal: the stamped hash plus the
    # unchanged-scope short-circuit meant no later request could ever displace
    # the empty row, so one transient fit failure blanked the map until the
    # gallery changed — across restarts, since the row is in SQLite.
    monkeypatch.setattr(image_index_default, "compute_umap", lambda matrix, seed=42: np.zeros((matrix.shape[0], 2)))
    service.request_projection("system")

    _wait_until(lambda: (r := index_records.get_projection("system", MODEL_ID)) is not None and r.point_count == 1)


def test_a_permanently_failing_projection_is_retried_once_not_every_request(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
    monkeypatch,
) -> None:
    """The other half of the bargain: recovering from a transient failure must not
    turn a permanent one into a fit per request, which is what the empty-cache
    stamp was protecting against in the first place."""
    fits = {"n": 0}

    def broken_umap(embeddings, seed=42):
        fits["n"] += 1
        raise RuntimeError("synthetic UMAP failure")

    monkeypatch.setattr(image_index_default, "compute_umap", broken_umap)
    _save_image(image_records, "a.png")
    service.start(_make_invoker(images_service, index_records))
    _wait_until(lambda: not service._backfill_pending.is_set())

    service.request_projection("system")
    _wait_until(lambda: index_records.get_projection("system", MODEL_ID) is not None, timeout=15)
    assert fits["n"] == 1

    # The retry is spent on the second request; every request after it must
    # short-circuit rather than re-enter the doomed fit.
    for _ in range(4):
        service.request_projection("system")
        _wait_until(lambda: not service._projection_requests, timeout=15)

    _wait_until(lambda: fits["n"] == 2, timeout=15)
    time.sleep(0.5)
    assert fits["n"] == 2, "a permanently failing scope must be retried once per process, not per request"


def test_a_cached_row_with_no_finite_points_is_a_failed_fit_not_a_result(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
    monkeypatch,
) -> None:
    """A row with point_count > 0 and every coordinate non-finite.

    /points drops non-finite rows before serving, so this row is empty to every
    client while looking populated to the service. Deciding "failed" on the
    cached count meant /points asked for a retry on this row's behalf, the
    service granted the request without ever entering the retry branch, the
    worker short-circuited and emitted projection_ready anyway, and the client —
    which refetches on that event — asked again. The budget could never be spent,
    so the refusal that is supposed to break the cycle never fired: a permanent
    request/emit loop at the worker's poll rate, and a permanent spinner.

    The router's fake service cannot show this: it decides the refusal itself,
    from the argument alone, with no view of the cached row.
    """
    from invokeai.app.services.image_index.projection import projection_params, scope_hash

    fits = {"n": 0}

    def broken_umap(embeddings, seed=42):
        fits["n"] += 1
        raise RuntimeError("synthetic UMAP failure")

    monkeypatch.setattr(image_index_default, "compute_umap", broken_umap)
    _save_image(image_records, "a.png")
    service.start(_make_invoker(images_service, index_records))
    _wait_until(lambda: not service._backfill_pending.is_set())

    names = index_records.list_accessible_embedded_items(None, MODEL_ID)
    current_hash = scope_hash(MODEL_ID, names)
    index_records.set_projection(
        "system",
        MODEL_ID,
        current_hash,
        projection_params(n_points=len(names)),
        names,
        np.full((len(names), 2), np.nan, dtype=np.float32),
    )

    # The first request on this row's behalf is granted and spends the budget.
    assert service.request_projection("system", failed_scope=current_hash) is True
    _wait_until(lambda: fits["n"] == 1, timeout=15)
    _wait_for_spent_retry(service, "system", current_hash)

    # And every one after it is refused, so /points settles into "empty".
    for _ in range(5):
        assert service.request_projection("system", failed_scope=current_hash) is False
    time.sleep(0.5)
    assert fits["n"] == 1, "a row with nothing servable must be retried once, not on every poll"


def test_a_lost_projection_write_does_not_burn_the_retry(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
    monkeypatch,
) -> None:
    """The budget bounds failed fits, so only a failed fit that reached the cache may spend it.

    Spending it before the fit looked safe — nothing between the spend and the
    fit can return — but it ignored the write. A fit that SUCCEEDS and then loses
    its set_projection to a locked database re-queues, and the re-queued job finds
    the old empty row with the budget already gone: minutes of correct work
    discarded and the map blank for good, without a single failed fit anywhere.
    """
    from invokeai.app.services.image_index.projection import projection_params, scope_hash

    monkeypatch.setattr(image_index_default, "compute_umap", lambda matrix, seed=42: np.zeros((matrix.shape[0], 2)))
    _save_image(image_records, "a.png")
    service.start(_make_invoker(images_service, index_records))
    _wait_until(lambda: not service._backfill_pending.is_set())

    # The empty row a failed fit leaves behind, stamped with the current scope:
    # what the retry is granted against.
    names = index_records.list_accessible_embedded_items(None, MODEL_ID)
    current_hash = scope_hash(MODEL_ID, names)
    index_records.set_projection(
        "system",
        MODEL_ID,
        current_hash,
        projection_params(n_points=0),
        [],
        np.empty((0, 2), dtype=np.float32),
    )

    writes = {"n": 0}
    real_set_projection = index_records.set_projection

    def failing_set_projection(*args, **kwargs):
        writes["n"] += 1
        if writes["n"] == 1:
            raise LockTimeoutError("database is locked")
        return real_set_projection(*args, **kwargs)

    monkeypatch.setattr(index_records, "set_projection", failing_set_projection)

    # The fit succeeds; its write is lost. The re-queued job must still be
    # allowed to run, which means the budget must not have moved.
    assert service.request_projection("system", failed_scope=current_hash) is True
    _wait_until(
        lambda: (r := index_records.get_projection("system", MODEL_ID)) is not None and r.point_count == 1,
        timeout=20,
    )
    assert writes["n"] == 2, "the lost write must be retried, not dropped"


def test_an_explicit_refresh_restores_a_spent_retry(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
    monkeypatch,
) -> None:
    """Otherwise a spent budget is unrecoverable while the server runs.

    /refresh answered `enqueued: true` while the worker was guaranteed to
    short-circuit — the API reporting that it had accepted work it could not do,
    with no way back short of a restart.
    """
    from invokeai.app.services.image_index.projection import scope_hash

    fits = {"n": 0}

    def broken_umap(embeddings, seed=42):
        fits["n"] += 1
        raise RuntimeError("synthetic UMAP failure")

    monkeypatch.setattr(image_index_default, "compute_umap", broken_umap)
    _save_image(image_records, "a.png")
    service.start(_make_invoker(images_service, index_records))
    _wait_until(lambda: not service._backfill_pending.is_set())

    service.request_projection("system")
    _wait_until(lambda: fits["n"] == 1, timeout=15)
    current_hash = scope_hash(MODEL_ID, index_records.list_accessible_embedded_items(None, MODEL_ID))
    assert service.request_projection("system", failed_scope=current_hash) is True
    _wait_until(lambda: fits["n"] == 2, timeout=15)
    _wait_for_spent_retry(service, "system", current_hash)
    assert service.request_projection("system", failed_scope=current_hash) is False, "the budget is spent"

    # A person pressing Refresh gets a real fit, not a short-circuit...
    assert service.request_projection("system", user_initiated=True) is True
    _wait_until(lambda: fits["n"] == 3, timeout=15)

    # ...while a poller still cannot, so the loop stays closed.
    _wait_for_spent_retry(service, "system", current_hash)
    assert service.request_projection("system", failed_scope=current_hash) is False


def test_projection_request_dedup_is_last_writer_wins(service: ImageIndexService) -> None:
    # Not started: requests are refused outright.
    assert service.request_projection("system") is False

    # Simulate a running worker to exercise the dedup map directly.
    service._model_id = MODEL_ID
    service._worker = threading.Thread(target=lambda: time.sleep(0.2), daemon=True)
    service._worker.start()

    assert service.request_projection("system", all_images=True) is True
    assert service.request_projection("system", all_images=False) is True
    assert service._projection_requests == {"system": False}


def test_systemic_embedding_outage_does_not_starve_projections(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
) -> None:
    """The emergent interaction between "never retire on a systemic failure" and
    "projections only at quiescence".

    A systemic failure charges no image, by design, so the same batch is returned on every
    pass and quiescence never arrives. If projections only ran in the quiescent branch, an
    outage would make the image map report "computing" forever over images that ARE embedded —
    and the projection needs no encoder, so there is no reason for it to wait.
    """
    embedded_ok = _unit_vec()

    def broken_encode(images: list[Image.Image]) -> np.ndarray:
        raise RuntimeError("model is gone")

    service = ImageIndexService(encode_fn=broken_encode, model_id=MODEL_ID)
    try:
        # One image already embedded (the projection has something to work with) and one that
        # can never embed while the encoder is down.
        _save_image(image_records, "done.png")
        _save_image(image_records, "stuck.png")
        index_records.upsert_embedding(IndexedItem("image", "done.png"), MODEL_ID, embedded_ok)

        service.start(_make_invoker(images_service, index_records))
        _wait_until(lambda: service._systemic_failures >= 1, timeout=20.0)

        assert service.request_projection("system") is True

        # The projection must land despite embedding being permanently stalled.
        _wait_until(lambda: index_records.get_projection("system", MODEL_ID) is not None, timeout=30.0)
        record = index_records.get_projection("system", MODEL_ID)
        assert record is not None
        assert record.items == imgs("done.png")
        # And the outage is still an outage: no image was retired to make this happen.
        assert service._failed == set()
        assert service._systemic_failures >= 1
    finally:
        service.stop()


# --- Semantic search ---


def test_search_similar_ranks_by_cosine_and_respects_scope(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
) -> None:
    service.start(_make_invoker(images_service, index_records))
    _wait_until(lambda: not service._backfill_pending.is_set())

    # Overwrite with hand-built vectors so the ranking is deterministic.
    def unit(index: int, mix: float = 0.0) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        v[index] = 1.0
        v[0] += mix
        return v / np.linalg.norm(v)

    for name, vec in [("a.png", unit(0)), ("close.png", unit(1, mix=0.9)), ("far.png", unit(2))]:
        _save_image(image_records, name)
        index_records.upsert_embedding(IndexedItem("image", name), MODEL_ID, vec)

    results = service.search_similar(None, unit(0), limit=2)

    assert [item for item, _ in results] == imgs("a.png", "close.png")
    assert results[0][1] > results[1][1] > 0.0

    # limit caps the result count; scores are descending.
    assert len(service.search_similar(None, unit(0), limit=1)) == 1

    # A name restriction applies before the limit, so a narrow scope still gets its best matches.
    assert [item for item, _ in service.search_similar(None, unit(0), limit=1, within={"far.png"})] == imgs("far.png")
    assert service.search_similar(None, unit(0), limit=5, within=set()) == []
    # Kind and name restrictions both apply.
    assert service.search_similar(None, unit(0), limit=5, kinds=("video",), within={"a.png"}) == []


def test_embed_image_normalizes_and_requires_running_service(
    images_service: ImageService, index_records: ImageIndexRecords, service: ImageIndexService
) -> None:
    probe = Image.new("RGB", (4, 4))

    with pytest.raises(RuntimeError):
        service.embed_image(probe)  # not started yet

    service.start(_make_invoker(images_service, index_records))
    vector = service.embed_image(probe)

    assert vector.shape == (DIM,)
    assert np.isclose(float(np.linalg.norm(vector)), 1.0)


def test_embed_image_retries_once_after_a_failed_encode(
    images_service: ImageService, index_records: ImageIndexRecords
) -> None:
    # A failed load evicts the model from the RAM cache so the next attempt
    # rebuilds it from disk; embed_image must make that second attempt itself.
    calls = {"count": 0}

    def flaky_encode(images):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("model cache entry was left in a bad state")
        return _fake_encode(images)

    service = ImageIndexService(encode_fn=flaky_encode, model_id=MODEL_ID)
    service.start(_make_invoker(images_service, index_records))

    vector = service.embed_image(Image.new("RGB", (4, 4)))

    try:
        assert calls["count"] == 2
        assert vector.shape == (DIM,)

        # A second consecutive failure propagates.
        def always_failing(images):
            raise RuntimeError("still broken")

        service._encode_fn = always_failing
        with pytest.raises(RuntimeError, match="still broken"):
            service.embed_image(Image.new("RGB", (4, 4)))
    finally:
        service.stop()


def test_a_projection_whose_database_work_fails_for_good_is_given_up_once(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
) -> None:
    """A failure that is not the database being busy (a statement the server refuses, say) fails the same way on
    every retry: re-queueing it would spin the worker forever. The waiting client still hears that nothing newer is
    coming."""
    from invokeai.app.services.events.events_common import ImageMapProjectionReadyEvent

    calls = {"n": 0}

    def refused(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("the server refused the statement")

    index_records.list_accessible_embedded_items = refused  # type: ignore[method-assign]

    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    invoker = _make_invoker(images_service, index_records)
    try:
        _save_image(image_records, "img-0.png")
        service.start(invoker)
        _wait_until(lambda: not service._backfill_pending.is_set())

        service.request_projection("system")

        _wait_until(
            lambda: any(isinstance(e, ImageMapProjectionReadyEvent) for e in invoker.services.events.events),
            timeout=30.0,
        )
        time.sleep(0.5)
        assert calls["n"] == 1
        ready = [e for e in invoker.services.events.events if isinstance(e, ImageMapProjectionReadyEvent)]
        assert [(e.user_id, e.point_count) for e in ready] == [("system", 0)]
    finally:
        service.stop()


def test_projection_request_is_requeued_when_the_database_read_fails(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
) -> None:
    """The job is popped from the dedup map before the work runs.

    A raise outside the fit's own try unwinds to the generic worker handler, which knows
    nothing about projections — so the request would be dropped after /refresh had already
    answered `enqueued: true`, and an event-driven client would wait forever.
    """
    calls = {"n": 0}
    real_list = index_records.list_accessible_embedded_items

    def flaky_list(user_id, model_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise LockTimeoutError("database is locked")
        return real_list(user_id, model_id)

    index_records.list_accessible_embedded_items = flaky_list  # type: ignore[method-assign]

    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    try:
        for i in range(3):
            _save_image(image_records, f"img-{i}.png")
        service.start(_make_invoker(images_service, index_records))
        _wait_until(lambda: not service._backfill_pending.is_set())

        service.request_projection("system")

        # Retried rather than dropped: the projection still lands.
        _wait_until(lambda: index_records.get_projection("system", MODEL_ID) is not None, timeout=30.0)
        assert calls["n"] >= 2
    finally:
        service.stop()


def test_unchanged_scope_does_not_recompute_the_projection(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
) -> None:
    """Repeat requests over an unchanged gallery must not re-run the fit.

    The fit is seeded, so recomputing burns minutes of single-threaded worker CPU to produce
    identical coordinates — and a client that refetches on `projection_ready` would drive it
    in a loop.
    """
    fits = {"n": 0}

    def counting_umap(matrix: np.ndarray) -> np.ndarray:
        # A stub, not the real fit: phase two runs at 4 points, past compute_umap's
        # small-N PCA fallback, and the first real UMAP fit JIT-compiles numba —
        # which can outlive stop()'s 10s join. The abandoned worker then fits
        # concurrently with a later test's own fit, which aborts the process
        # (SIGABRT on macOS). This test's claim is about WHETHER the fit runs,
        # never about its output.
        fits["n"] += 1
        return np.zeros((matrix.shape[0], 2), dtype=np.float32)

    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    try:
        for i in range(3):
            _save_image(image_records, f"img-{i}.png")
        service.start(_make_invoker(images_service, index_records))
        _wait_until(lambda: not service._backfill_pending.is_set())

        with patch.object(image_index_default, "compute_umap", counting_umap):
            service.request_projection("system")
            _wait_until(lambda: index_records.get_projection("system", MODEL_ID) is not None, timeout=30.0)
            assert fits["n"] == 1

            for _ in range(3):
                service.request_projection("system")
                _wait_until(lambda: not service._projection_requests, timeout=30.0)

            assert fits["n"] == 1, "unchanged scope must reuse the cached projection"

        # A real scope change still recomputes. The callback is what enqueues work — writing
        # the row alone leaves the backfill unarmed, so the worker would never see it.
        _save_image(image_records, "new.png")
        images_service._on_changed(_dto_for(image_records, "new.png"))
        _wait_until(lambda: index_records.get_embeddings(imgs("new.png"), MODEL_ID)[0] == imgs("new.png"), timeout=30.0)
        with patch.object(image_index_default, "compute_umap", counting_umap):
            service.request_projection("system")
            _wait_until(lambda: fits["n"] == 2, timeout=30.0)
            # The fit-entry count races the store; wait for the stored row so
            # stop() joins an idle worker instead of abandoning a live one.
            _wait_until(
                lambda: (r := index_records.get_projection("system", MODEL_ID)) is not None and r.point_count == 4,
                timeout=30.0,
            )
    finally:
        service.stop()


def test_failed_batch_uses_the_escalating_backoff(service: ImageIndexService) -> None:
    """Pin the CALL SITE, not just the helper.

    `_backoff_seconds()` is unit-tested on its own, but reverting the worker's failed-batch
    wait to a fixed `_POLL_SECONDS` — the single most plausible way to lose this in a
    hand-resolved rebase conflict — was previously invisible to the suite.
    """
    source = inspect.getsource(ImageIndexService._worker_loop)
    assert "self._stop_event.wait(self._backoff_seconds())" in source
    assert "self._stop_event.wait(_POLL_SECONDS)" not in source.split("except Exception")[0]


def test_projection_job_is_popped_before_running(service: ImageIndexService) -> None:
    """A job left in the dedup map turns the worker into an infinite recompute loop.

    Nothing else stops it: with the scope-hash short-circuit the fit is skipped, but the
    `projection_ready` emit would still fire on every pass.
    """
    service._model_id = MODEL_ID
    with service._projection_lock:
        service._projection_requests["u1"] = False

    job = service._next_projection_job()

    assert job == ("u1", False)
    assert service._projection_requests == {}, "the job must be removed when it is taken"
    assert service._next_projection_job() is None


def test_embed_text_unavailable_without_model_config(
    images_service: ImageService, index_records: ImageIndexRecords, service: ImageIndexService
) -> None:
    from invokeai.app.services.image_index.image_index_base import TextSearchUnavailableError

    service.start(_make_invoker(images_service, index_records))

    # Test mode injects encode_fn without a real model config: the text tower
    # cannot exist, and the error must be the typed one the router maps to 409.
    with pytest.raises(TextSearchUnavailableError):
        service.embed_text("a query")


def test_embed_text_unavailable_when_tokenizer_files_missing(tmp_path) -> None:
    # The InvokeAI-published CLIP model dir ships a full-CLIP config.json but no
    # tokenizer files: AutoTokenizer resolves a tokenizer class from the config
    # and then fails with TypeError (not OSError) on the absent vocab file. The
    # failure must still surface as the typed error the router maps to 409.
    import json

    from invokeai.app.services.image_index.image_index_base import TextSearchUnavailableError
    from invokeai.backend.model_manager.taxonomy import ModelType

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "clip", "architectures": ["CLIPModel"], "text_config": {}, "vision_config": {}})
    )
    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    service._invoker = SimpleNamespace(services=SimpleNamespace(configuration=SimpleNamespace(models_path=tmp_path)))  # type: ignore[assignment]
    service._model_config = SimpleNamespace(type=ModelType.CLIPVision, path=str(tmp_path))  # type: ignore[assignment]

    with pytest.raises(TextSearchUnavailableError):
        service.embed_text("a query")


class _FakeTokenizer:
    """Stands in for a transformers tokenizer: only the call shape matters here."""

    def __init__(self, distinct: bool) -> None:
        self._distinct = distinct

    def __call__(self, texts, padding=True, return_tensors="pt", truncation=True):
        # A vocabulary-less tokenizer emits the same unknown token for every
        # word, so two unrelated phrases differ in nothing but length.
        rows = [[0, 2, 2, 2] if not self._distinct else [0, index + 3, index + 9, 1] for index in range(len(texts))]
        return {"input_ids": torch.tensor(rows, dtype=torch.long)}


class _FakeTextModel:
    """Returns rows straight from a fixture matrix, keyed by batch size."""

    def __init__(self, matrix: torch.Tensor) -> None:
        self._matrix = matrix

    def eval(self) -> "_FakeTextModel":
        return self

    def __call__(self, **inputs):
        count = inputs["input_ids"].shape[0]
        return SimpleNamespace(text_embeds=self._matrix[:count], pooler_output=self._matrix[:count])


def test_a_text_encoder_that_discriminates_reports_no_defect() -> None:
    from invokeai.app.services.image_index.image_index_default import _text_encoder_defect

    matrix = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

    assert _text_encoder_defect(_FakeTokenizer(distinct=True), _FakeTextModel(matrix), False) is None


@pytest.mark.parametrize(
    "matrix, expected",
    [
        (torch.tensor([[0.5, 0.25, 0.0, 0.0], [0.5, 0.25, 0.0, 0.0]]), "same embedding"),
        (torch.zeros((2, 4)), "all-zero"),
        (torch.tensor([[float("nan"), 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]), "non-finite"),
    ],
)
def test_a_degenerate_text_encoder_is_detected(matrix: torch.Tensor, expected: str) -> None:
    # The failure this exists for: a model directory with no tokenizer files
    # still loads, because transformers builds a tokenizer with an empty
    # vocabulary that maps every word to the same unknown token. Every phrase
    # then embeds identically, and both cluster labels and text search go on
    # returning confident nonsense — labels chosen by float noise among equal
    # scores — with nothing downstream able to tell.
    from invokeai.app.services.image_index.image_index_default import _text_encoder_defect

    defect = _text_encoder_defect(_FakeTokenizer(distinct=False), _FakeTextModel(matrix), False)

    assert defect is not None
    assert expected in defect


def test_a_degenerate_text_encoder_is_detected_even_when_its_norms_overflow_float32() -> None:
    # Identical rows are the failure this guard exists for, but in float32 a
    # large enough pair overflows the sum of squares: the norms come back inf,
    # the cosine NaN, and `NaN >= threshold` is False — so the guard used to wave
    # this through as healthy. The matrix itself is finite, so the non-finite
    # branch above does not catch it either. Reachable whenever the tower's
    # weights are garbage rather than absent (see skip_torch_weight_init in
    # _get_text_encoder, whose monkey-patch can leak process-wide).
    from invokeai.app.services.image_index.image_index_default import _text_encoder_defect

    matrix = torch.full((2, 4), 1e20)
    assert torch.equal(matrix[0], matrix[1])
    assert torch.isfinite(matrix).all()

    defect = _text_encoder_defect(_FakeTokenizer(distinct=False), _FakeTextModel(matrix), False)

    assert defect is not None
    assert "same embedding" in defect


def test_a_probe_that_raises_is_reported_as_a_defect_rather_than_escaping() -> None:
    # A probe that cannot complete has not cleared the encoder. Left to escape it
    # reaches the router untyped and becomes a 500 instead of the 409 this path
    # exists to produce. The trigger is a real broken install: a tokenizer whose
    # ids overrun the tower's vocabulary raises IndexError inside nn.Embedding.
    from invokeai.app.services.image_index.image_index_default import _text_encoder_defect

    class _ExplodingModel:
        def eval(self) -> "_ExplodingModel":
            return self

        def __call__(self, **inputs):
            raise IndexError("index out of range in self")

    defect = _text_encoder_defect(_FakeTokenizer(distinct=True), _ExplodingModel(), False)

    assert defect is not None
    assert "could not be exercised" in defect
    assert "IndexError" in defect


def test_embed_text_refuses_a_text_encoder_that_cannot_tell_phrases_apart(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # End to end through the lazy loader: the typed error is what the router
    # maps to a 409, so labels and search report unavailable rather than
    # serving results computed from identical vectors.
    import gc
    import json

    import transformers

    from invokeai.app.services.image_index.image_index_base import TextSearchUnavailableError
    from invokeai.backend.model_manager.taxonomy import ModelType

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "clip", "architectures": ["CLIPModel"]}))
    constant = torch.ones((2, DIM))
    loads: list[weakref.ReferenceType] = []

    def _load_model(cls, *args, **kwargs):
        model = _FakeTextModel(constant)
        loads.append(weakref.ref(model))
        return model

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: _FakeTokenizer(distinct=False))
    )
    monkeypatch.setattr(transformers.CLIPTextModelWithProjection, "from_pretrained", classmethod(_load_model))

    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    service._invoker = SimpleNamespace(services=SimpleNamespace(configuration=SimpleNamespace(models_path=tmp_path)))  # type: ignore[assignment]
    service._model_config = SimpleNamespace(type=ModelType.CLIPVision, path=str(tmp_path))  # type: ignore[assignment]
    service._model_id = MODEL_ID

    for _ in range(3):
        with pytest.raises(TextSearchUnavailableError, match="same embedding"):
            service.embed_text("a query")

    # The rejected encoder is not served...
    assert service._text_encoder is None
    # ...and not reloaded either. Reloading cannot change the verdict, and every
    # attempt drags the whole text tower through MODEL_LOAD_LOCK, contending with
    # generation's model loads — once per search keystroke on a broken install.
    assert len(loads) == 1
    # What is remembered is the message. A remembered *exception* would keep the
    # rejected tower alive through the frames its traceback captured — the leak
    # test below covers that end; here it is enough that nothing holds a tower.
    assert isinstance(service._text_encoder_failure, str)
    gc.collect()
    assert loads[0]() is None


def test_a_failed_vocabulary_build_does_not_pin_the_rejected_text_encoder(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The vocabulary build remembers its failure so a broken install answers the
    # next request from memory. It must not remember the traceback with it: the
    # probe rejects the encoder *after* loading it, so those frames hold the
    # loaded tower — hundreds of MB kept alive for the life of the process.
    import gc
    import json

    import transformers

    from invokeai.app.services.image_index.image_index_base import TextSearchUnavailableError
    from invokeai.backend.model_manager.taxonomy import ModelType

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "clip", "architectures": ["CLIPModel"]}))
    towers: list[weakref.ReferenceType] = []

    def _load_model(cls, *args, **kwargs):
        model = _FakeTextModel(torch.ones((2, DIM)))
        towers.append(weakref.ref(model))
        return model

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: _FakeTokenizer(distinct=False))
    )
    monkeypatch.setattr(transformers.CLIPTextModelWithProjection, "from_pretrained", classmethod(_load_model))

    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    service._invoker = SimpleNamespace(  # type: ignore[assignment]
        services=SimpleNamespace(
            configuration=SimpleNamespace(models_path=tmp_path, db_path=tmp_path / "db.sqlite"),
            image_index_records=SimpleNamespace(get_custom_vocab_terms=lambda: []),
            # Not a real Logger: pytest's logging plugin retains every LogRecord
            # for the test, and a record carrying exc_info holds the very
            # traceback whose release is being asserted.
            logger=SimpleNamespace(warning=lambda *args, **kwargs: None),
        )
    )
    service._model_config = SimpleNamespace(type=ModelType.CLIPVision, path=str(tmp_path))  # type: ignore[assignment]
    service._model_id = MODEL_ID

    service._build_vocab_embeddings()

    assert service._vocab_failure is not None
    assert service._vocab_failure.__traceback__ is None
    assert service._vocab_failure.__cause__ is None
    gc.collect()
    assert towers and towers[0]() is None

    # And it does not regrow. Re-raising a persistent exception re-attaches a
    # traceback to it, so a stored failure cleared only once accumulates a frame
    # set per labels request, each holding that request's caller frame.
    holders: list[weakref.ReferenceType] = []

    def one_labels_request() -> None:
        # Stands in for the router frame, whose `record` is the whole projection.
        record = np.zeros((50_000, 2), dtype=np.float32)
        holders.append(weakref.ref(record))
        with pytest.raises(TextSearchUnavailableError):
            service.get_vocab_embeddings()

    def frame_count() -> int:
        count, traceback = 0, service._vocab_failure.__traceback__
        while traceback is not None:
            count, traceback = count + 1, traceback.tb_next
        return count

    one_labels_request()
    after_one = frame_count()
    for _ in range(9):
        one_labels_request()

    # Bounded at the one propagation in flight rather than growing with traffic,
    # so every caller frame but the current one is released.
    assert frame_count() == after_one
    gc.collect()
    assert [index for index, ref in enumerate(holders) if ref() is not None] == [len(holders) - 1]


def test_search_similar_returns_empty_when_not_running(service: ImageIndexService) -> None:
    assert service.search_similar(None, np.ones(DIM, dtype=np.float32), limit=5) == []


def test_search_similar_scopes_to_the_requesting_user(
    db: Database,
    image_records: ImageRecordStorage,
    images_service: ImageService,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
) -> None:
    from invokeai.app.services.users.users_common import UserCreateRequest
    from invokeai.app.services.users.users_default import UserService

    other_user = UserService(db).create(
        UserCreateRequest(email="scoped@example.com", display_name="Scoped", password="TestPass123", is_admin=False)
    )
    service.start(_make_invoker(images_service, index_records))
    _wait_until(lambda: not service._backfill_pending.is_set())

    def unit(index: int) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        v[index] = 1.0
        return v

    # system owns mine.png; the other user owns theirs.png (both unboarded).
    _save_image(image_records, "mine.png")
    index_records.upsert_embedding(IndexedItem("image", "mine.png"), MODEL_ID, unit(0))
    image_records.save(
        image_name="theirs.png",
        image_origin=ResourceOrigin.INTERNAL,
        image_category=ImageCategory.GENERAL,
        width=16,
        height=16,
        has_workflow=False,
        user_id=other_user.user_id,
    )
    index_records.upsert_embedding(IndexedItem("image", "theirs.png"), MODEL_ID, unit(0))

    # The other user's scope must exclude the system user's private image
    # even though it scores identically.
    names = [name for name, _ in service.search_similar(other_user.user_id, unit(0), limit=10)]
    assert names == imgs("theirs.png")

    # Admin scope (None) sees both.
    admin_items = {item for item, _ in service.search_similar(None, unit(0), limit=10)}
    assert admin_items == set(imgs("mine.png", "theirs.png"))


def test_query_vectors_reject_a_non_finite_embedding(service: ImageIndexService) -> None:
    # The indexer drops rows whose norm is non-finite because they poison every
    # similarity they take part in. A query has nothing to drop: dividing anyway
    # gives an all-NaN vector, then all-NaN scores, arbitrary argpartition
    # results, and a response body containing bare `NaN` — which is not valid
    # JSON, so the browser fails to parse it rather than showing no matches.
    from invokeai.app.services.image_index.image_index_default import _normalize_query_vector

    for bad in (np.inf, np.nan):
        vector = np.ones(DIM, dtype=np.float32)
        vector[0] = bad

        with pytest.raises(RuntimeError, match="degenerate"):
            _normalize_query_vector(vector)

    with pytest.raises(RuntimeError, match="degenerate"):
        _normalize_query_vector(np.zeros(DIM, dtype=np.float32))


def test_query_vectors_normalize_in_float64(service: ImageIndexService) -> None:
    from invokeai.app.services.image_index.image_index_default import _normalize_query_vector

    # A float32 sum of squares overflows to inf well inside the range these
    # encoders produce; float64 carries it, so this must normalize rather than
    # trip the guard above.
    vector = np.full(DIM, 3.0e19, dtype=np.float32)
    normalized = _normalize_query_vector(vector)

    assert np.isfinite(normalized).all()
    assert float(np.linalg.norm(normalized.astype(np.float64))) == pytest.approx(1.0, rel=1e-3)


def test_lazy_model_construction_takes_the_process_global_load_lock() -> None:
    # skip_torch_weight_init monkey-patches torch.nn.*.reset_parameters process-wide
    # and restores whatever it saw on entry. Two threads inside it at once means the
    # second saves the no-op, and whoever leaves last restores the no-op forever —
    # every layer built afterwards silently skips weight init. MODEL_LOAD_LOCK is
    # what makes it safe, so both lazy loaders must hold it.
    import inspect

    from invokeai.app.services.image_index import image_index_default

    loaders = (
        image_index_default.ImageIndexService._get_text_encoder,
        image_index_default.ImageIndexService._encode_with_model,
    )

    for fn in loaders:
        source = inspect.getsource(fn)

        if "skip_torch_weight_init" in source:
            assert "MODEL_LOAD_LOCK.write_lock()" in source, (
                f"{fn.__qualname__} patches torch globally without the process-global load lock"
            )


def test_vocab_embeddings_are_never_built_on_the_caller(service: ImageIndexService) -> None:
    # The build is minutes of encoder work and callers reach it through
    # asyncio.to_thread on the loop's shared executor. Blocking there — worse,
    # blocking there under _vocab_lock — starves every other to_thread in the
    # app, /points and image search included. The request only asks the worker.
    from invokeai.app.services.image_index.image_index_base import TextSearchUnavailableError

    service._invoker = SimpleNamespace()  # type: ignore[assignment]
    service._model_id = MODEL_ID

    with pytest.raises(TextSearchUnavailableError, match="still being prepared"):
        service.get_vocab_embeddings()

    assert service._vocab_build_requested.is_set()


def test_a_failed_vocabulary_build_is_not_retried_on_every_request(service: ImageIndexService) -> None:
    # A vision-only install has no text tower. Without remembering the failure,
    # every points refresh retries from_pretrained inside MODEL_LOAD_LOCK,
    # contending with generation's model loads for as long as the app runs.
    from invokeai.app.services.image_index.image_index_base import TextSearchUnavailableError

    service._vocab_failure = TextSearchUnavailableError("no text encoder")  # type: ignore[assignment]
    # As the worker's except branch always does — a fresh failure, inside the
    # retry window.
    service._vocab_failed_at = time.monotonic()

    with pytest.raises(TextSearchUnavailableError, match="no text encoder"):
        service.get_vocab_embeddings()

    # Not even queued: there is nothing for the worker to retry.
    assert not service._vocab_build_requested.is_set()


def test_an_aged_vocabulary_failure_requeues_the_build(service: ImageIndexService) -> None:
    # The memo protects a vision-only install from per-refresh text-tower
    # retries, but the failure itself can be transient — an OOM while the GPU
    # was busy generating, a load that lost a race. Answering from memory
    # forever pins one bad minute as a permanent labels outage; after the
    # retry window, the next request must drop the memo and ask the worker
    # to build again.
    from invokeai.app.services.image_index.image_index_base import TextSearchUnavailableError

    service._invoker = SimpleNamespace()  # type: ignore[assignment]
    service._model_id = MODEL_ID
    service._vocab_failure = TextSearchUnavailableError("encoder OOM")  # type: ignore[assignment]
    service._vocab_failed_at = time.monotonic() - image_index_default._VOCAB_FAILURE_RETRY_SECONDS - 1

    with pytest.raises(TextSearchUnavailableError, match="still being prepared"):
        service.get_vocab_embeddings()

    assert service._vocab_build_requested.is_set()
    assert service._vocab_failure is None
    assert service._vocab_failed_at is None


def test_a_failed_build_records_when_it_failed(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The retry window is measured from the build's failure, so the stamp must
    # land with the memo.
    from invokeai.app.services.image_index import cluster_labels

    def _oom(embed_fn, vocabulary):
        raise RuntimeError("encoder OOM")

    matrix = np.arange(3 * DIM, dtype=EMBEDDING_DTYPE).reshape(3, DIM)
    service = _vocab_build_service(tmp_path, monkeypatch, matrix)
    monkeypatch.setattr(cluster_labels, "ensemble_phrase_embeddings", _oom)

    service._build_vocab_embeddings()

    assert service._vocab_failure is not None
    assert service._vocab_failed_at is not None
    # Two-sided: a stamp written with the wrong clock (time.time against a
    # monotonic read) is a huge negative difference that a one-sided bound
    # passes vacuously — and the memo would then never age out.
    assert abs(time.monotonic() - service._vocab_failed_at) < 5


def test_an_aged_vocabulary_failure_does_not_requeue_against_a_stopped_worker(
    service: ImageIndexService,
) -> None:
    # The decay assumes the index worker will consume the build request. After
    # stop() nothing will, so a request arriving in the shutdown window must
    # answer with the memoized failure — the real cause — rather than queue a
    # rebuild that 409s "still being prepared" for the rest of the process's
    # life.
    from invokeai.app.services.image_index.image_index_base import TextSearchUnavailableError

    service._vocab_failure = TextSearchUnavailableError("encoder OOM")  # type: ignore[assignment]
    service._vocab_failed_at = time.monotonic() - image_index_default._VOCAB_FAILURE_RETRY_SECONDS - 1
    service._stop_event.set()

    with pytest.raises(TextSearchUnavailableError, match="encoder OOM"):
        service.get_vocab_embeddings()

    assert not service._vocab_build_requested.is_set()
    assert service._vocab_failure is not None


def test_the_worker_clears_the_failure_stamp_with_the_memo(service: ImageIndexService) -> None:
    """Pin the CALL SITE, not just the fields.

    The worker's invalidation block drops the memo and its stamp together; the
    stamp is only ever read under `if self._vocab_failure is not None`, so
    leaving it behind is invisible today — but any future read outside that
    guard turns the stale value live. Like the backoff call-site test above,
    this is the plausible hand-resolved-rebase loss the suite must see.
    """
    source = inspect.getsource(ImageIndexService._worker_loop)
    invalidate_block = source.split("_vocab_invalidate_requested.clear()")[1].split("try:")[0]

    assert "self._vocab_failure = None" in invalidate_block
    assert "self._vocab_failed_at = None" in invalidate_block


def _vocab_build_service(tmp_path, monkeypatch: pytest.MonkeyPatch, matrix: np.ndarray) -> ImageIndexService:
    """A service wired for `_build_vocab_embeddings` over a three-phrase vocabulary."""
    from invokeai.app.services.image_index import cluster_labels

    monkeypatch.setattr(cluster_labels, "load_vocabulary", lambda: ["a cat", "a dog", "a car"])
    monkeypatch.setattr(cluster_labels, "ensemble_phrase_embeddings", lambda embed_fn, vocabulary: matrix)

    svc = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    svc._invoker = SimpleNamespace(  # type: ignore[assignment]
        services=SimpleNamespace(
            configuration=SimpleNamespace(db_path=tmp_path / "invokeai.db"),
            logger=InvokeAILogger.get_logger(),
            image_index_records=SimpleNamespace(get_custom_vocab_terms=lambda: []),
        )
    )
    svc._model_id = MODEL_ID

    return svc


def test_the_vocabulary_cache_reaches_disk_under_its_final_name(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # np.savez appends `.npz` to a path that lacks it, so a staging name ending
    # in `.tmp` produced `.tmp.npz` on disk and the rename that followed looked
    # for a file that was never written. The handler swallowed the error, which
    # left the in-memory cache working and hid the fact that nothing persisted:
    # every restart re-embedded ~1700 phrases (minutes, with no cluster labels
    # until it finished) and leaked one staging file per run.
    matrix = np.arange(3 * DIM, dtype=EMBEDDING_DTYPE).reshape(3, DIM)
    service = _vocab_build_service(tmp_path, monkeypatch, matrix)

    service._build_vocab_embeddings()

    cache_path = tmp_path / f"cluster_vocab_{MODEL_ID.replace(':', '_')[:24]}.npz"
    assert cache_path.exists()
    # Nothing else: a leftover staging file means the rename did not happen.
    assert [path.name for path in sorted(tmp_path.iterdir())] == [cache_path.name]


def test_a_cached_vocabulary_is_reused_instead_of_re_embedded(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The point of persisting it: the next process reads the file rather than
    # spending minutes in the text tower before it can serve a single label.
    matrix = np.arange(3 * DIM, dtype=EMBEDDING_DTYPE).reshape(3, DIM)
    _vocab_build_service(tmp_path, monkeypatch, matrix)._build_vocab_embeddings()

    from invokeai.app.services.image_index import cluster_labels

    def _refuse(embed_fn: Callable[[list[str]], np.ndarray], vocabulary: list[str]) -> np.ndarray:
        raise AssertionError("re-embedded a vocabulary that was already cached")

    restarted = _vocab_build_service(tmp_path, monkeypatch, matrix)
    monkeypatch.setattr(cluster_labels, "ensemble_phrase_embeddings", _refuse)

    restarted._build_vocab_embeddings()

    assert restarted._vocab_cache is not None
    vocabulary, cached = restarted._vocab_cache
    assert vocabulary == ["a cat", "a dog", "a car"]
    assert np.array_equal(cached, matrix)


def test_a_failed_cache_write_says_what_went_wrong(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Nothing else reports this failure: the in-memory cache below the handler
    # is assigned either way, so labelling keeps working and the only symptom
    # is a slow startup nobody attributes to it. The `.tmp` staging-name bug
    # survived for exactly that reason — the warning named the cache path but
    # never the FileNotFoundError that would have identified it outright.
    matrix = np.arange(3 * DIM, dtype=EMBEDDING_DTYPE).reshape(3, DIM)
    service = _vocab_build_service(tmp_path, monkeypatch, matrix)
    recorded: list[tuple[str, object]] = []
    service._invoker.services.logger = SimpleNamespace(  # type: ignore[union-attr]
        warning=lambda message, exc_info=False: recorded.append((str(message), exc_info))
    )

    def _refuse(src: object, dst: object) -> None:
        raise PermissionError("simulated: another instance holds the cache open")

    monkeypatch.setattr(image_index_default.os, "replace", _refuse)

    service._build_vocab_embeddings()

    assert len(recorded) == 1
    message, exc_info = recorded[0]
    assert "Could not write cluster vocabulary cache" in message
    assert exc_info is True
    # The run itself is unaffected — which is why the warning has to carry it.
    assert service._vocab_cache is not None


def test_batch_normalization_zeroes_a_degenerate_row_instead_of_failing(service: ImageIndexService) -> None:
    # One bad row out of ~11,700 must not discard the whole vocabulary build:
    # that raises past the endpoint's handling as a 500 and, nothing having been
    # cached, repeats the full embed on the next request. A zeroed row scores 0
    # against everything, so the phrase simply never wins a label.
    from invokeai.app.services.image_index.image_index_default import _normalize_batch

    matrix = np.ones((3, DIM), dtype=np.float32)
    matrix[1] = 0.0
    matrix[2, 0] = np.inf

    normalized = _normalize_batch(matrix)

    assert np.isfinite(normalized).all()
    assert float(np.linalg.norm(normalized[0])) == pytest.approx(1.0, rel=1e-5)
    assert not normalized[1].any()
    assert not normalized[2].any()


# --- Custom (supplementary) vocabulary ---


def _phrase_matrix(phrases: list[str]) -> np.ndarray:
    """Deterministic, phrase-distinguishable rows: every value is the phrase's length."""
    return np.stack([np.full(DIM, float(len(phrase)), dtype=EMBEDDING_DTYPE) for phrase in phrases])


def _vocab_service_with_custom_terms(tmp_path, monkeypatch: pytest.MonkeyPatch, custom: list[str]) -> ImageIndexService:
    """A service wired for `_build_vocab_embeddings` over a three-phrase bundled vocabulary."""
    from invokeai.app.services.image_index import cluster_labels

    monkeypatch.setattr(cluster_labels, "load_vocabulary", lambda: ["a cat", "a dog", "a car"])
    monkeypatch.setattr(cluster_labels, "ensemble_phrase_embeddings", lambda embed_fn, phrases: _phrase_matrix(phrases))

    svc = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    svc._invoker = SimpleNamespace(  # type: ignore[assignment]
        services=SimpleNamespace(
            configuration=SimpleNamespace(db_path=tmp_path / "invokeai.db"),
            logger=InvokeAILogger.get_logger(),
            image_index_records=SimpleNamespace(get_custom_vocab_terms=lambda: list(custom)),
        )
    )
    svc._model_id = MODEL_ID

    return svc


def test_custom_terms_are_appended_and_bundled_phrases_win_collisions(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "a dog" collides with the bundled vocabulary and is dropped (the label a
    # user would get is the same phrase either way); "zebra crossing" extends it.
    service = _vocab_service_with_custom_terms(tmp_path, monkeypatch, ["zebra crossing", "a dog"])

    service._build_vocab_embeddings()

    assert service._vocab_cache is not None
    vocabulary, matrix = service._vocab_cache
    assert vocabulary == ["a cat", "a dog", "a car", "zebra crossing"]
    assert matrix.shape == (4, DIM)
    # Rows stay aligned with the merged phrase list across the concatenation.
    assert np.array_equal(matrix[3], np.full(DIM, float(len("zebra crossing")), dtype=EMBEDDING_DTYPE))
    # Both tiers persisted, separately.
    tag = MODEL_ID.replace(":", "_")[:24]
    assert (tmp_path / f"cluster_vocab_{tag}.npz").exists()
    assert (tmp_path / f"cluster_vocab_custom_{tag}.npz").exists()


def test_editing_custom_terms_re_embeds_only_the_custom_tier(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The bundled tier is ~1700 phrases and minutes of encoder work; an edit to
    # the custom list must be answered from the bundled tier's disk cache.
    from invokeai.app.services.image_index import cluster_labels

    _vocab_service_with_custom_terms(tmp_path, monkeypatch, ["zebra"])._build_vocab_embeddings()

    embedded: list[list[str]] = []

    def _record(embed_fn: Callable[[list[str]], np.ndarray], phrases: list[str]) -> np.ndarray:
        embedded.append(list(phrases))
        return _phrase_matrix(phrases)

    second = _vocab_service_with_custom_terms(tmp_path, monkeypatch, ["zebra", "okapi"])
    monkeypatch.setattr(cluster_labels, "ensemble_phrase_embeddings", _record)

    second._build_vocab_embeddings()

    assert embedded == [["zebra", "okapi"]]
    assert second._vocab_cache is not None
    assert second._vocab_cache[0] == ["a cat", "a dog", "a car", "zebra", "okapi"]
    assert second._vocab_cache[1].shape == (5, DIM)


def test_an_unchanged_custom_tier_is_loaded_from_disk(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from invokeai.app.services.image_index import cluster_labels

    _vocab_service_with_custom_terms(tmp_path, monkeypatch, ["zebra"])._build_vocab_embeddings()

    def _refuse(embed_fn: Callable[[list[str]], np.ndarray], phrases: list[str]) -> np.ndarray:
        raise AssertionError("re-embedded a tier that was already cached")

    second = _vocab_service_with_custom_terms(tmp_path, monkeypatch, ["zebra"])
    monkeypatch.setattr(cluster_labels, "ensemble_phrase_embeddings", _refuse)

    second._build_vocab_embeddings()

    assert second._vocab_cache is not None
    assert second._vocab_cache[0] == ["a cat", "a dog", "a car", "zebra"]


def test_vocab_build_state_reports_each_phase(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    fresh = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    assert fresh.get_vocab_build_state() == ("unavailable", None)

    service = _vocab_service_with_custom_terms(tmp_path, monkeypatch, [])
    assert service.get_vocab_build_state() == ("idle", None)

    service.invalidate_vocab()
    assert service.get_vocab_build_state() == ("building", None)

    # The worker's pass: clear the flags, then build.
    service._vocab_build_requested.clear()
    service._vocab_invalidate_requested.clear()
    service._build_vocab_embeddings()
    assert service.get_vocab_build_state() == ("ready", None)


def test_vocab_build_state_reports_error_with_the_failure_message(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from invokeai.app.services.image_index import cluster_labels

    service = _vocab_service_with_custom_terms(tmp_path, monkeypatch, [])

    def _raise(embed_fn: Callable[[list[str]], np.ndarray], phrases: list[str]) -> np.ndarray:
        raise RuntimeError("simulated: no text tower")

    monkeypatch.setattr(cluster_labels, "ensemble_phrase_embeddings", _raise)
    service._build_vocab_embeddings()

    state, message = service.get_vocab_build_state()
    assert state == "error"
    assert message is not None and "no text tower" in message


def test_invalidate_vocab_never_blocks_on_an_in_flight_build(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The worker holds _vocab_lock for the whole of a minutes-long build;
    # invalidation runs on request threads and must return without it.
    service = _vocab_service_with_custom_terms(tmp_path, monkeypatch, [])
    assert service._vocab_lock.acquire(blocking=False)
    try:
        done = threading.Event()

        def _invalidate() -> None:
            service.invalidate_vocab()
            done.set()

        thread = threading.Thread(target=_invalidate)
        thread.start()
        thread.join(timeout=2)
        assert done.is_set(), "invalidate_vocab blocked behind the vocabulary lock"
        # And an in-flight build reads as building.
        assert service.get_vocab_build_state() == ("building", None)
    finally:
        service._vocab_lock.release()


def test_invalidate_vocab_rebuilds_with_the_new_terms_on_the_worker(
    images_service: ImageService,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from invokeai.app.services.image_index import cluster_labels
    from invokeai.app.services.image_index.image_index_base import TextSearchUnavailableError

    monkeypatch.setattr(cluster_labels, "load_vocabulary", lambda: ["a cat"])
    monkeypatch.setattr(cluster_labels, "ensemble_phrase_embeddings", lambda embed_fn, phrases: _phrase_matrix(phrases))
    monkeypatch.setattr(InvokeAIAppConfig, "db_path", property(lambda self: tmp_path / "invokeai.db"))
    index_records.set_custom_vocab_terms(["dog"])

    service.start(_make_invoker(images_service, index_records))
    with pytest.raises(TextSearchUnavailableError, match="still being prepared"):
        service.get_vocab_embeddings()
    _wait_until(lambda: service.get_vocab_build_state() == ("ready", None))
    assert service.get_vocab_embeddings()[0] == ["a cat", "dog"]

    index_records.set_custom_vocab_terms(["dog", "zebra"])
    service.invalidate_vocab()

    _wait_until(
        lambda: service.get_vocab_build_state() == ("ready", None)
        and service._vocab_cache is not None
        and "zebra" in service._vocab_cache[0]
    )
    vocabulary, matrix = service.get_vocab_embeddings()
    assert vocabulary == ["a cat", "dog", "zebra"]
    assert matrix.shape == (3, DIM)


def test_invalidate_vocab_retries_a_failed_build(
    images_service: ImageService,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A memoized build failure is deliberately never retried on its own (see
    # test_a_failed_vocabulary_build_is_not_retried_on_every_request);
    # invalidation is the one path that clears it.
    from invokeai.app.services.image_index import cluster_labels
    from invokeai.app.services.image_index.image_index_base import TextSearchUnavailableError

    flaky = {"fail": True}

    def _ensemble(embed_fn: Callable[[list[str]], np.ndarray], phrases: list[str]) -> np.ndarray:
        if flaky["fail"]:
            raise RuntimeError("simulated: no text tower")
        return _phrase_matrix(phrases)

    monkeypatch.setattr(cluster_labels, "load_vocabulary", lambda: ["a cat"])
    monkeypatch.setattr(cluster_labels, "ensemble_phrase_embeddings", _ensemble)
    monkeypatch.setattr(InvokeAIAppConfig, "db_path", property(lambda self: tmp_path / "invokeai.db"))

    service.start(_make_invoker(images_service, index_records))
    with pytest.raises(TextSearchUnavailableError, match="still being prepared"):
        service.get_vocab_embeddings()
    _wait_until(lambda: service.get_vocab_build_state()[0] == "error")

    flaky["fail"] = False
    service.invalidate_vocab()

    _wait_until(lambda: service.get_vocab_build_state() == ("ready", None))
    assert service.get_vocab_embeddings()[0] == ["a cat"]


def test_a_cache_with_a_mismatched_row_count_is_discarded_and_re_embedded(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The fingerprint alone cannot be trusted to prove the file describes these
    # phrases (corruption, hash collision): a wrong row count would misalign
    # the merged matrix and break every labels request.
    from invokeai.app.services.image_index.cluster_labels import vocab_fingerprint

    service = _vocab_service_with_custom_terms(tmp_path, monkeypatch, ["okapi"])
    tag = MODEL_ID.replace(":", "_")[:24]
    custom_cache = tmp_path / f"cluster_vocab_custom_{tag}.npz"
    np.savez(
        custom_cache,
        embeddings=np.zeros((2, DIM), dtype=EMBEDDING_DTYPE),
        fingerprint=np.str_(vocab_fingerprint(["okapi"])),
    )

    service._build_vocab_embeddings()

    assert service._vocab_cache is not None
    vocabulary, matrix = service._vocab_cache
    assert vocabulary == ["a cat", "a dog", "a car", "okapi"]
    assert matrix.shape == (4, DIM)
    # Re-embedded, not served from the bogus file.
    assert np.array_equal(matrix[3], np.full(DIM, float(len("okapi")), dtype=EMBEDDING_DTYPE))


def test_a_transient_custom_terms_read_failure_keeps_the_rebuild_queued(
    images_service: ImageService,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The worker clears the request flag and drops the caches before the build
    # reads its state; a transient DB failure there must not strand the
    # rebuild in 'idle' — the flag is re-set so the next pass retries.
    from invokeai.app.services.image_index import cluster_labels

    monkeypatch.setattr(cluster_labels, "load_vocabulary", lambda: ["a cat"])
    monkeypatch.setattr(cluster_labels, "ensemble_phrase_embeddings", lambda embed_fn, phrases: _phrase_matrix(phrases))
    monkeypatch.setattr(InvokeAIAppConfig, "db_path", property(lambda self: tmp_path / "invokeai.db"))
    index_records.set_custom_vocab_terms(["dog"])

    calls = {"count": 0}
    original_read = index_records.get_custom_vocab_terms

    def _flaky() -> list[str]:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("simulated: database is locked")
        return original_read()

    monkeypatch.setattr(index_records, "get_custom_vocab_terms", _flaky)

    service.start(_make_invoker(images_service, index_records))
    service.invalidate_vocab()

    _wait_until(lambda: service.get_vocab_build_state() == ("ready", None), timeout=15)
    assert calls["count"] >= 2
    assert service.get_vocab_embeddings()[0] == ["a cat", "dog"]


# --- Videos ---


def test_backfill_indexes_preexisting_videos(
    image_records: ImageRecordStorage,
    images_service: ImageService,
    videos_service: VideoService,
    video_records: VideoRecordStorage,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
) -> None:
    _save_image(image_records, "img.png")
    _save_video(video_records, "clip.mp4")
    _save_video(video_records, "intermediate.mp4", is_intermediate=True)
    _save_video(video_records, "mask.mp4", video_category=ImageCategory.MASK)

    service.start(
        _make_invoker(images_service, index_records, videos_service=videos_service, video_records=video_records)
    )

    _wait_until(lambda: index_records.count_index_status(MODEL_ID).embedded == 2)
    assert index_records.get_embeddings(vids("clip.mp4"), MODEL_ID)[0] == vids("clip.mp4")
    assert index_records.get_embeddings(vids("intermediate.mp4", "mask.mp4"), MODEL_ID)[0] == []


def test_new_video_is_indexed_from_its_thumbnail(
    images_service: ImageService,
    videos_service: VideoService,
    video_records: VideoRecordStorage,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
) -> None:
    invoker = _make_invoker(images_service, index_records, videos_service=videos_service, video_records=video_records)
    service.start(invoker)
    _wait_until(lambda: not service._backfill_pending.is_set())

    _save_video(video_records, "new.mp4")
    _save_video(video_records, "new-intermediate.mp4", is_intermediate=True)
    # Fire the callbacks the way VideoService.create would, i.e. after the thumbnail is written.
    videos_service._on_changed(_video_dto_for(video_records, "new.mp4"))
    videos_service._on_changed(_video_dto_for(video_records, "new-intermediate.mp4"))

    _wait_until(lambda: index_records.get_embeddings(vids("new.mp4"), MODEL_ID)[0] == vids("new.mp4"))
    assert index_records.get_embeddings(vids("new-intermediate.mp4"), MODEL_ID)[0] == []


def test_video_thumbnail_is_what_gets_embedded(
    images_service: ImageService,
    videos_service: VideoService,
    video_records: VideoRecordStorage,
    index_records: ImageIndexRecords,
) -> None:
    # The embedding must come from the video's own thumbnail, not from the image service (which
    # here returns a different colour) — a wrong source would still produce a plausible vector.
    embedded: list[Image.Image] = []

    def capture(images: list[Image.Image]) -> np.ndarray:
        embedded.extend(images)
        return np.stack([_unit_vec() for _ in images])

    service = ImageIndexService(encode_fn=capture, model_id=MODEL_ID)
    _save_video(video_records, "clip.mp4")
    try:
        service.start(
            _make_invoker(images_service, index_records, videos_service=videos_service, video_records=video_records)
        )
        _wait_until(lambda: index_records.get_embeddings(vids("clip.mp4"), MODEL_ID)[0] == vids("clip.mp4"))
    finally:
        service.stop()

    assert embedded, "the encoder was never called"
    assert embedded[0].size == (16, 16)
    # Nearest-colour rather than equality: the thumbnail is a lossy WEBP, so its pixels are
    # near the colour it was written with, not identical to it.
    pixel = embedded[0].getpixel((0, 0))
    thumbnail_colour = Image.new("RGB", (1, 1), "teal").getpixel((0, 0))
    image_service_colour = Image.new("RGB", (1, 1), "purple").getpixel((0, 0))

    def distance(a: tuple[int, ...], b: tuple[int, ...]) -> int:
        return sum(abs(x - y) for x, y in zip(a, b, strict=True))

    assert distance(pixel, thumbnail_colour) < distance(pixel, image_service_colour)


def test_unreadable_video_thumbnail_is_charged_to_the_video(
    images_service: ImageService,
    videos_service: VideoService,
    video_records: VideoRecordStorage,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
) -> None:
    # A video whose thumbnail never got written must not stall the backfill; it retires after
    # _MAX_ATTEMPTS exactly as an unreadable image does.
    _save_video(video_records, "no-thumbnail.mp4")
    videos_service.get_path = lambda video_name, thumbnail=False: "/nonexistent/thumb.webp"  # type: ignore[method-assign]

    service.start(
        _make_invoker(images_service, index_records, videos_service=videos_service, video_records=video_records)
    )

    _wait_until(lambda: IndexedItem("video", "no-thumbnail.mp4") in service._failed)
    assert index_records.get_embeddings(vids("no-thumbnail.mp4"), MODEL_ID)[0] == []


def test_deleted_video_is_forgotten(
    images_service: ImageService,
    videos_service: VideoService,
    video_records: VideoRecordStorage,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
) -> None:
    invoker = _make_invoker(images_service, index_records, videos_service=videos_service, video_records=video_records)
    service.start(invoker)
    _save_video(video_records, "clip.mp4")
    videos_service._on_changed(_video_dto_for(video_records, "clip.mp4"))
    _wait_until(lambda: index_records.get_embeddings(vids("clip.mp4"), MODEL_ID)[0] == vids("clip.mp4"))

    # Seeded so the assertions below depend on the delete callback rather than on the FK
    # cascade: bookkeeping a deleted video leaves behind inflates `failed` for the life of the
    # process, and `pending` can then never drain to zero.
    service._failed.add(IndexedItem("video", "clip.mp4"))
    service._attempts[IndexedItem("video", "clip.mp4")] = 3

    video_records.delete("clip.mp4")
    videos_service._on_deleted("clip.mp4")

    _wait_until(lambda: index_records.count_index_status(MODEL_ID).total == 0)
    assert IndexedItem("video", "clip.mp4") not in service._pending
    assert IndexedItem("video", "clip.mp4") not in service._failed
    assert IndexedItem("video", "clip.mp4") not in service._attempts


def test_video_owner_is_poked_when_its_embedding_lands(
    images_service: ImageService,
    videos_service: VideoService,
    video_records: VideoRecordStorage,
    image_records: ImageRecordStorage,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
) -> None:
    # The status event is admin-only, so this per-user poke is the only signal a non-admin
    # gets that their generation reached the index. The owner has to be looked up in the
    # videos table: asking the images table for a video name finds nothing, and the lookup is
    # deliberately swallowed, so the map would simply never refresh for them.
    invoker = _make_invoker(
        images_service,
        index_records,
        image_records=image_records,
        videos_service=videos_service,
        video_records=video_records,
    )
    service.start(invoker)
    _wait_until(lambda: not service._backfill_pending.is_set())

    _save_video(video_records, "mine.mp4")
    videos_service._on_changed(_video_dto_for(video_records, "mine.mp4"))

    _wait_until(
        lambda: any(
            isinstance(e, ImageIndexUpdatedEvent) and e.user_id == "system" for e in invoker.services.events.events
        )
    )


def test_video_leaving_eligibility_clears_its_failure_bookkeeping(
    images_service: ImageService,
    videos_service: VideoService,
    video_records: VideoRecordStorage,
    index_records: ImageIndexRecords,
    service: ImageIndexService,
) -> None:
    # A video that stops being a gallery item must stop counting against `failed`, or it skews
    # `pending` for the rest of the process. The category half of the predicate is the part
    # that is genuinely video-specific — a different field on a different DTO.
    invoker = _make_invoker(images_service, index_records, videos_service=videos_service, video_records=video_records)
    service.start(invoker)
    _wait_until(lambda: not service._backfill_pending.is_set())

    service._failed.add(IndexedItem("video", "gone.mp4"))
    service._attempts[IndexedItem("video", "gone.mp4")] = 3
    _save_video(video_records, "gone.mp4", video_category=ImageCategory.MASK)
    videos_service._on_changed(_video_dto_for(video_records, "gone.mp4"))

    assert IndexedItem("video", "gone.mp4") not in service._failed
    assert IndexedItem("video", "gone.mp4") not in service._attempts
    status = service.get_status()
    assert status is not None
    assert status.failed == 0


def test_gpu_embedding_runs_on_an_idle_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Indexing runs outside the session queue, so a batch borrows the idle GPU instead of the busy one."""
    from unittest.mock import MagicMock

    from tests.fixtures.device_pool import GPU0, GPU1, two_gpu_pool

    seen: list[object] = []
    model = torch.nn.Linear(1, 1)
    loaded = MagicMock()
    loaded.model_on_device.return_value.__enter__.return_value = (None, model)

    def load_model(config: object) -> MagicMock:
        seen.append(TorchDevice.get_session_device())
        return loaded

    service = ImageIndexService(encode_fn=_fake_encode, model_id=MODEL_ID)
    service._invoker = SimpleNamespace(  # type: ignore[assignment]
        services=SimpleNamespace(
            configuration=SimpleNamespace(image_index_device=None),
            model_manager=SimpleNamespace(load=SimpleNamespace(load_model=load_model)),
        )
    )
    service._model_config = SimpleNamespace()  # type: ignore[assignment]
    monkeypatch.setattr(
        service, "_embed", lambda model, images, device: seen.append(TorchDevice.get_session_device()) or np.zeros(1)
    )

    with two_gpu_pool(busy=(GPU0,)):
        service._encode_with_model([Image.new("RGB", (8, 8))])
        assert TorchDevice.get_session_device() is None

    assert seen == [GPU1, GPU1]
