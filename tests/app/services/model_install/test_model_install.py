"""
Test the model installer
"""

import gc
import json
import platform
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from pydantic_core import Url

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.download import DownloadQueueService
from invokeai.app.services.events.events_base import EventServiceBase
from invokeai.app.services.events.events_common import (
    ModelInstallCompleteEvent,
    ModelInstallDownloadProgressEvent,
    ModelInstallDownloadsCompleteEvent,
    ModelInstallDownloadStartedEvent,
    ModelInstallErrorEvent,
    ModelInstallStartedEvent,
)
from invokeai.app.services.model_install import (
    HFModelSource,
    ModelInstallService,
    ModelInstallServiceBase,
    model_install_default,
)
from invokeai.app.services.model_install.model_install_common import (
    InstallStatus,
    InvalidModelConfigException,
    LocalModelSource,
    ModelInstallJob,
    URLModelSource,
)
from invokeai.app.services.model_install.model_install_default import (
    INSTALL_MARKER_FILENAME,
    INSTALL_MARKER_VERSION,
    TMPDIR_PREFIX,
)
from invokeai.app.services.model_records import ModelRecordChanges, UnknownModelException
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig
from invokeai.backend.model_manager.metadata import RemoteModelFile
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    ModelFormat,
    ModelRepoVariant,
    ModelSourceType,
    ModelType,
)
from tests.backend.model_manager.model_manager_fixtures import *  # noqa F403
from tests.test_nodes import TestEventService

OS = platform.uname().system


def test_registration(mm2_installer: ModelInstallServiceBase, embedding_file: Path) -> None:
    store = mm2_installer.record_store
    matches = store.search_by_attr(model_name="test_embedding")
    assert len(matches) == 0
    key = mm2_installer.register_path(embedding_file)
    # Not raising here is sufficient - key should be UUIDv4
    uuid.UUID(key, version=4)


def test_registration_meta(mm2_installer: ModelInstallServiceBase, embedding_file: Path) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.register_path(embedding_file)
    model_record = store.get_model(key)
    assert model_record is not None
    assert model_record.name == "test_embedding"
    assert model_record.type == ModelType.TextualInversion
    assert Path(model_record.path) == embedding_file
    assert Path(model_record.path).exists()
    assert model_record.base == BaseModelType("sd-1")
    assert model_record.description is None
    assert model_record.source is not None
    assert Path(model_record.source) == embedding_file


def test_registration_meta_override_fail(mm2_installer: ModelInstallServiceBase, embedding_file: Path) -> None:
    with pytest.raises(InvalidModelConfigException):
        mm2_installer.register_path(embedding_file, ModelRecordChanges(name="banana_sushi", type=ModelType("lora")))


def test_registration_meta_override_succeed(mm2_installer: ModelInstallServiceBase, embedding_file: Path) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.register_path(
        embedding_file, ModelRecordChanges(name="banana_sushi", source="fake/repo_id", key="xyzzy")
    )
    model_record = store.get_model(key)
    assert model_record.name == "banana_sushi"
    assert model_record.source == "fake/repo_id"
    assert model_record.key == "xyzzy"


def test_install(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.install_path(embedding_file)
    model_record = store.get_model(key)
    assert model_record.path.endswith(f"{key}/test_embedding.safetensors")
    assert (mm2_app_config.models_path / model_record.path).exists()
    assert model_record.source == embedding_file.as_posix()


def test_rename(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.install_path(embedding_file)
    model_record = store.get_model(key)
    assert model_record.path.endswith(f"{key}/test_embedding.safetensors")
    new_model_record = store.update_model(
        key,
        ModelRecordChanges(name="new model name", base=BaseModelType.StableDiffusion2),
        allow_class_change=True,
    )
    # Renaming the model record shouldn't rename the file
    assert new_model_record.name == "new model name"
    assert model_record.path.endswith(f"{key}/test_embedding.safetensors")


@pytest.mark.parametrize(
    "fixture_name,size,key,destination",
    [
        ("embedding_file", 15440, "foo", "foo/test_embedding.safetensors"),
        ("diffusers_dir", 8241 if OS == "Windows" else 7907, "bar", "bar"),  # EOL chars
    ],
)
def test_background_install(
    mm2_installer: ModelInstallServiceBase,
    fixture_name: str,
    key: str,
    size: int,
    destination: str,
    mm2_app_config: InvokeAIAppConfig,
    request: pytest.FixtureRequest,
) -> None:
    """Note: may want to break this down into several smaller unit tests."""
    path: Path = request.getfixturevalue(fixture_name)
    description = "Test of metadata assignment"
    source = LocalModelSource(path=path, inplace=False)
    job = mm2_installer.import_model(source, config=ModelRecordChanges(key=key, description=description))
    assert job is not None
    assert isinstance(job, ModelInstallJob)

    # See if job is registered properly
    assert job in mm2_installer.get_job_by_source(source)

    # test that the job object tracked installation correctly
    jobs = mm2_installer.wait_for_installs()
    assert len(jobs) > 0
    my_job = [x for x in jobs if x.source == source]
    assert len(my_job) == 1
    assert job == my_job[0]
    assert job.status == InstallStatus.COMPLETED
    assert job.total_bytes == size

    # test that the expected events were issued
    bus: TestEventService = mm2_installer.event_bus
    assert bus
    assert hasattr(bus, "events")

    assert len(bus.events) == 2
    assert isinstance(bus.events[0], ModelInstallStartedEvent)
    assert isinstance(bus.events[1], ModelInstallCompleteEvent)
    assert Path(bus.events[0].source.path) == source
    assert Path(bus.events[1].source.path) == source
    key = bus.events[1].key
    assert key is not None

    # see if the thing actually got installed at the expected location
    model_record = mm2_installer.record_store.get_model(key)
    assert model_record is not None
    assert model_record.path.endswith(destination)
    assert (mm2_app_config.models_path / model_record.path).exists()

    # see if metadata was properly passed through
    assert model_record.description == description

    # see if job filtering works
    assert mm2_installer.get_job_by_source(source)[0] == job

    # see if prune works properly
    mm2_installer.prune_jobs()
    assert not mm2_installer.get_job_by_source(source)


def test_not_inplace_install(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    # An non in-place install will/should call `register_path()` internally
    source = LocalModelSource(path=embedding_file, inplace=False)
    job = mm2_installer.import_model(source)
    mm2_installer.wait_for_installs()
    assert job is not None
    assert job.config_out is not None
    # Non in-place install should _move_ the model from the original location to the models path
    # The model config's path should be different from the original file
    assert Path(job.config_out.path) != embedding_file
    # Original file should _not_ exist after install
    assert not embedding_file.exists()
    assert (mm2_app_config.models_path / job.config_out.path).exists()


def test_inplace_install(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    # An in-place install will/should call `install_path()` internally
    source = LocalModelSource(path=embedding_file, inplace=True)
    job = mm2_installer.import_model(source)
    mm2_installer.wait_for_installs()
    assert job is not None
    assert job.config_out is not None
    # In-place install should not touch the model file, just register it
    # The model config's path should be the same as the original file
    assert Path(job.config_out.path) == embedding_file
    # Model file should still exist after install
    assert embedding_file.exists()
    assert Path(job.config_out.path).exists()


def test_external_install(mm2_installer: ModelInstallServiceBase) -> None:
    config = ModelRecordChanges(name="ChatGPT Image", description="External model", key="chatgpt_image")
    job = mm2_installer.heuristic_import("external://openai/gpt-image-1", config=config)

    mm2_installer.wait_for_installs()

    assert job.status == InstallStatus.COMPLETED
    assert job.config_out is not None
    assert isinstance(job.config_out, ExternalApiModelConfig)
    assert job.config_out.provider_id == "openai"
    assert job.config_out.provider_model_id == "gpt-image-1"
    assert job.config_out.base == BaseModelType.External
    assert job.config_out.type == ModelType.ExternalImageGenerator
    assert job.config_out.source_type == ModelSourceType.External


def test_external_install_is_idempotent(mm2_installer: ModelInstallServiceBase) -> None:
    first_job = mm2_installer.heuristic_import(
        "external://openai/gpt-image-1",
        config=ModelRecordChanges(name="Initial name"),
    )
    mm2_installer.wait_for_installs()

    second_job = mm2_installer.heuristic_import(
        "external://openai/gpt-image-1",
        config=ModelRecordChanges(name="Updated name"),
    )
    mm2_installer.wait_for_installs()

    assert first_job.status == InstallStatus.COMPLETED
    assert second_job.status == InstallStatus.COMPLETED
    assert first_job.config_out is not None
    assert second_job.config_out is not None
    assert first_job.config_out.key == second_job.config_out.key

    external_models = mm2_installer.record_store.search_by_attr(
        base_model=BaseModelType.External,
        model_type=ModelType.ExternalImageGenerator,
    )
    assert len(external_models) == 1
    assert isinstance(external_models[0], ExternalApiModelConfig)
    assert external_models[0].name == "Updated name"


def test_delete_install(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.install_path(embedding_file)  # non in-place install
    model_record = store.get_model(key)
    assert (mm2_app_config.models_path / model_record.path).exists()
    assert not embedding_file.exists()
    # ensure file handles are released on Windows
    gc.collect()
    mm2_installer.delete(key)
    # after deletion, installed copy should not exist
    assert not (mm2_app_config.models_path / model_record.path).exists()
    with pytest.raises(UnknownModelException):
        store.get_model(key)


def test_delete_register(
    mm2_installer: ModelInstallServiceBase, embedding_file: Path, mm2_app_config: InvokeAIAppConfig
) -> None:
    store = mm2_installer.record_store
    key = mm2_installer.register_path(embedding_file)  # in-place install
    model_record = store.get_model(key)
    assert Path(model_record.path).exists()
    assert embedding_file.exists()
    mm2_installer.delete(key)
    assert Path(model_record.path).exists()
    with pytest.raises(UnknownModelException):
        store.get_model(key)


@pytest.mark.timeout(timeout=10, method="thread")
def test_simple_download(mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig) -> None:
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))

    bus: TestEventService = mm2_installer.event_bus
    store = mm2_installer.record_store
    assert store is not None
    assert bus is not None
    assert hasattr(bus, "events")  # the dummy event service has this

    job = mm2_installer.import_model(source)
    assert job.source == source
    job_list = mm2_installer.wait_for_installs(timeout=10)
    assert len(job_list) == 1
    assert job.complete
    assert job.config_out

    key = job.config_out.key
    model_record = store.get_model(key)
    assert (mm2_app_config.models_path / model_record.path).exists()

    assert len(bus.events) == 5
    assert isinstance(bus.events[0], ModelInstallDownloadStartedEvent)  # download starts
    assert isinstance(bus.events[1], ModelInstallDownloadProgressEvent)  # download progresses
    assert isinstance(bus.events[2], ModelInstallDownloadsCompleteEvent)  # download completed
    assert isinstance(bus.events[3], ModelInstallStartedEvent)  # install started
    assert isinstance(bus.events[4], ModelInstallCompleteEvent)  # install completed


def test_import_waits_for_startup_restore(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    embedding_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    restore_started = threading.Event()
    release_restore = threading.Event()
    imported = threading.Event()

    def _blocked_restore() -> None:
        restore_started.set()
        assert release_restore.wait(timeout=5)

    monkeypatch.setattr(installer, "_restore_incomplete_installs", _blocked_restore)

    try:
        installer.start()
        assert restore_started.wait(timeout=5)

        import_thread = threading.Thread(
            target=lambda: (
                installer.import_model(LocalModelSource(path=embedding_file)),
                imported.set(),
            )
        )
        import_thread.start()

        time.sleep(0.1)
        assert not imported.is_set()

        release_restore.set()
        import_thread.join(timeout=5)
        assert imported.is_set()
        installer.wait_for_installs(timeout=5)
    finally:
        release_restore.set()
        installer.stop()


def _write_test_install_marker(tmpdir: Path, source_str: str) -> None:
    """Create a tmp install dir containing an active (downloading) install marker."""
    tmpdir.mkdir(parents=True)
    marker = {
        "version": INSTALL_MARKER_VERSION,
        "source": source_str,
        "access_token": None,
        "config_in": {},
        "status": InstallStatus.DOWNLOADING.value,
        "updated_at": "",
        "files": [],
    }
    with open(tmpdir / INSTALL_MARKER_FILENAME, "wt", encoding="utf-8") as f:
        json.dump(marker, f)


@pytest.mark.timeout(timeout=20, method="thread")
def test_restore_skips_source_queued_during_restore(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for #9141: if a foreground thread queues a job for a source after
    restore has parsed that source's install marker but before restore appends its own job,
    restore must notice the active job and skip the source instead of enqueuing it twice."""
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))

    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}race"
    _write_test_install_marker(tmpdir, str(source))

    marker_observed = threading.Event()
    release_restore = threading.Event()
    real_guess_source = installer._guess_source

    def _pausing_guess_source(source_str: str):
        result = real_guess_source(source_str)
        marker_observed.set()
        assert release_restore.wait(timeout=10)
        return result

    resumed: list[ModelInstallJob] = []
    monkeypatch.setattr(installer, "_guess_source", _pausing_guess_source)
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: resumed.append(job))

    try:
        installer.start()
        assert marker_observed.wait(timeout=10)

        # Restore is now paused between parsing the marker and its locked duplicate
        # check. Queue an active job for the same source, as a concurrent
        # import_model call would.
        foreground_job = ModelInstallJob(
            id=installer._next_id(),
            source=source,
            config_in=ModelRecordChanges(),
            local_path=tmpdir,
        )
        with installer._lock:
            installer._install_jobs.append(foreground_job)

        release_restore.set()
        installer._wait_for_restore_complete()

        jobs_for_source = [job for job in installer._install_jobs if str(job.source) == str(source)]
        assert jobs_for_source == [foreground_job]
        assert resumed == []
    finally:
        release_restore.set()
        installer.stop()


@pytest.mark.timeout(timeout=20, method="thread")
def test_restore_preserves_active_jobs_tmpdir(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When two tmpdirs hold markers for the same source and an active job owns the
    later-visited one, restore must not delete the active job's tmpdir: seeing the
    stale dir first must not cause the active dir to be treated as a duplicate."""
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))

    stale_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}0_stale"
    active_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}1_active"
    _write_test_install_marker(stale_dir, str(source))
    _write_test_install_marker(active_dir, str(source))

    # Force restore to visit the stale dir before the active one.
    real_glob = Path.glob
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter(sorted(real_glob(self, pattern))))

    active_job = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=active_dir,
    )
    active_job._install_tmpdir = active_dir
    active_job.status = InstallStatus.DOWNLOADING
    installer._install_jobs.append(active_job)

    resumed: list[ModelInstallJob] = []
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: resumed.append(job))

    try:
        installer.start()
        installer._wait_for_restore_complete()

        assert active_dir.exists()
        jobs_for_source = [job for job in installer._install_jobs if str(job.source) == str(source)]
        assert jobs_for_source == [active_job]
        assert resumed == []
    finally:
        installer.stop()


@pytest.mark.timeout(timeout=20, method="thread")
def test_restore_preserves_tmpdir_of_job_in_download_cache(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Like test_restore_preserves_active_jobs_tmpdir, but the active job is tracked
    only in _download_cache - the shape of a real remote import mid-download, where
    _enqueue_remote_download registers the job before import_model appends it to
    _install_jobs."""
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))

    stale_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}0_stale"
    active_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}1_active"
    _write_test_install_marker(stale_dir, str(source))
    _write_test_install_marker(active_dir, str(source))

    real_glob = Path.glob
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter(sorted(real_glob(self, pattern))))

    active_job = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=active_dir,
    )
    active_job._install_tmpdir = active_dir
    active_job.status = InstallStatus.DOWNLOADING
    installer._download_cache[999] = active_job

    resumed: list[ModelInstallJob] = []
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: resumed.append(job))

    try:
        installer.start()
        installer._wait_for_restore_complete()

        assert active_dir.exists()
        assert [job for job in installer._install_jobs if str(job.source) == str(source)] == []
        assert resumed == []
    finally:
        installer.stop()


@pytest.mark.timeout(timeout=30, method="thread")
def test_concurrent_import_and_restore_register_single_job(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real import_model() call that passes _wait_for_restore_complete() before
    start() clears the event must not race _restore_incomplete_installs into
    registering the same source twice. Unlike
    test_restore_skips_source_queued_during_restore, this drives the production
    import path instead of manually appending a job under the lock."""
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    _write_test_install_marker(mm2_app_config.models_path / f"{TMPDIR_PREFIX}race", str(source))

    import_reached = threading.Event()
    release_import = threading.Event()
    real_import_from_url = installer._import_from_url

    def _pausing_import_from_url(src, config=None):
        import_reached.set()
        assert release_import.wait(timeout=10)
        return real_import_from_url(src, config)

    monkeypatch.setattr(installer, "_import_from_url", _pausing_import_from_url)

    restore_observed_marker = threading.Event()
    real_guess_source = installer._guess_source

    def _observing_guess_source(source_str: str):
        result = real_guess_source(source_str)
        restore_observed_marker.set()
        return result

    monkeypatch.setattr(installer, "_guess_source", _observing_guess_source)

    resumed: list[ModelInstallJob] = []
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: resumed.append(job))

    # Buffer the install queue so the import job cannot reach a terminal state
    # (and delete its marker dir) before restore performs its active-source
    # check - otherwise the assertions below race the install thread.
    queued: list[ModelInstallJob] = []
    real_put_in_queue = installer._put_in_queue
    monkeypatch.setattr(installer, "_put_in_queue", lambda job: queued.append(job))

    import_thread = threading.Thread(target=lambda: installer.import_model(source))
    start_thread = threading.Thread(target=installer.start)
    try:
        # The import passes _wait_for_restore_complete() (the event starts out
        # set) and its duplicate check, then pauses before registering anything.
        import_thread.start()
        assert import_reached.wait(timeout=10)

        # Start the service so restoration processes the marker for the same
        # source. With the fix, restore sees the source reserved in
        # _pending_sources, defers the marker, and its deferred recheck waits
        # for the reservation to resolve - so restore cannot complete until the
        # import is released. Without the fix, restore registers a duplicate
        # job now.
        start_thread.start()
        assert restore_observed_marker.wait(timeout=10)

        release_import.set()
        import_thread.join(timeout=10)
        start_thread.join(timeout=10)
        installer._wait_for_restore_complete()

        jobs_for_source = [job for job in installer._install_jobs if str(job.source) == str(source)]
        assert len(jobs_for_source) == 1
        assert resumed == []

        # Wait for the download-complete callback to hand the job to the
        # (buffered) install queue, then un-buffer and let it finish end-to-end.
        # The test's single download produces exactly one such hand-off, so once
        # it has arrived no more calls hit the buffering lambda.
        deadline = time.time() + 10
        while not queued and time.time() < deadline:
            time.sleep(0.05)
        assert queued
        monkeypatch.setattr(installer, "_put_in_queue", real_put_in_queue)
        for queued_job in queued:
            real_put_in_queue(queued_job)
        installer.wait_for_installs(timeout=10)
        assert jobs_for_source[0].complete
    finally:
        release_import.set()
        installer.stop()


@pytest.mark.timeout(timeout=30, method="thread")
def test_import_during_paused_download_callback_does_not_deadlock(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a lock-order inversion: download-queue callbacks run on
    queue threads that hold the download queue's lock while acquiring the installer
    lock, so import_model must never call into the download queue while holding the
    installer lock. Here an import for a second source requests a download job ID
    while a callback for the first source is paused inside the queue lock; if
    import_model holds the installer lock across its download enqueue, the two
    threads deadlock.

    Uses a private download queue rather than the mm2_download_queue fixture: if
    the deadlock regresses, the fixture's teardown would join the wedged worker
    threads and hang the whole test run instead of failing this one test."""
    download_queue = DownloadQueueService(requests_session=mm2_session)
    download_queue.start()
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source_a = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    source_b = URLModelSource(
        url=Url(
            "https://huggingface.co/InvokeAI-test/textual_inversion_tests/resolve/main/learned_embeds-steps-1000.safetensors"
        )
    )

    callback_entered = threading.Event()
    release_callback = threading.Event()
    real_started_callback = installer._download_started_callback

    def _pausing_started_callback(download_job) -> None:
        # Runs on a download-queue thread that holds the queue's lock and has
        # not yet acquired the installer lock.
        callback_entered.set()
        assert release_callback.wait(timeout=20)
        real_started_callback(download_job)

    monkeypatch.setattr(installer, "_download_started_callback", _pausing_started_callback)

    import_b_jobs: list[ModelInstallJob] = []
    import_b_at_queue = threading.Event()
    import_b_thread: Optional[threading.Thread] = None

    try:
        installer.start()
        installer._wait_for_restore_complete()

        job_a = installer.import_model(source_a)
        # The download worker for source A is now paused inside its on_start
        # callback, holding the download queue's lock.
        assert callback_entered.wait(timeout=10)

        real_multifile_download = installer._multifile_download

        def _signaling_multifile_download(*args, **kwargs):
            # Import B is about to request a download job ID from the queue.
            import_b_at_queue.set()
            return real_multifile_download(*args, **kwargs)

        monkeypatch.setattr(installer, "_multifile_download", _signaling_multifile_download)

        import_b_thread = threading.Thread(
            target=lambda: import_b_jobs.append(installer.import_model(source_b)),
            daemon=True,  # must not block interpreter exit if the deadlock regresses
        )
        import_b_thread.start()
        assert import_b_at_queue.wait(timeout=10)

        release_callback.set()
        import_b_thread.join(timeout=15)
        assert not import_b_thread.is_alive(), "import_model deadlocked against a download callback"

        installer.wait_for_installs(timeout=15)
        assert job_a.complete
        assert import_b_jobs and import_b_jobs[0].complete
    finally:
        release_callback.set()
        if import_b_thread is not None and import_b_thread.is_alive():
            # The threads are deadlocked (the bug): stopping the services would
            # block forever on the wedged download-queue lock. All the involved
            # threads are daemons, so leak them and let the test report its
            # failure.
            pass
        else:
            installer.stop()
            download_queue.stop()


@pytest.mark.timeout(timeout=20, method="thread")
def test_restore_skips_marker_of_job_completing_mid_scan(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source whose job is active when restoration begins must stay owned for the
    whole scan. If the owner reaches a terminal state after restore has read its
    marker but before the locked active check, restore must not enqueue a duplicate
    job for a directory the owner is about to clean up."""
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))

    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}owned"
    _write_test_install_marker(tmpdir, str(source))

    owner_job = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
    )
    owner_job._install_tmpdir = tmpdir
    owner_job.status = InstallStatus.DOWNLOADING
    installer._install_jobs.append(owner_job)

    marker_observed = threading.Event()
    release_restore = threading.Event()
    real_guess_source = installer._guess_source

    def _pausing_guess_source(source_str: str):
        result = real_guess_source(source_str)
        marker_observed.set()
        assert release_restore.wait(timeout=10)
        return result

    resumed: list[ModelInstallJob] = []
    monkeypatch.setattr(installer, "_guess_source", _pausing_guess_source)
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: resumed.append(job))

    try:
        installer.start()
        assert marker_observed.wait(timeout=10)

        # While restore is paused between reading the marker and its locked
        # active check, the owner finishes. Its marker and directory are still
        # on disk for a moment (or linger indefinitely if cleanup fails).
        owner_job.status = InstallStatus.COMPLETED

        release_restore.set()
        installer._wait_for_restore_complete()

        jobs_for_source = [job for job in installer._install_jobs if str(job.source) == str(source)]
        assert jobs_for_source == [owner_job]
        assert resumed == []
        assert tmpdir.exists()
    finally:
        release_restore.set()
        installer.stop()


@pytest.mark.timeout(timeout=20, method="thread")
def test_restore_recovers_marker_after_pending_import_fails(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A _pending_sources reservation is not an owner: the reserving import may
    fail before registering a job. If restore treated the reservation as a
    permanent owner, a failed import would leave its marker neither restored nor
    owned until the next restart. Restore must defer the marker and, once the
    reservation resolves without registering a job, restore it."""
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))

    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}orphan"
    _write_test_install_marker(tmpdir, str(source))

    import_reserved = threading.Event()
    release_import = threading.Event()

    def _failing_import_from_url(src, config=None):
        import_reserved.set()
        assert release_import.wait(timeout=10)
        raise RuntimeError("simulated import failure before job registration")

    monkeypatch.setattr(installer, "_import_from_url", _failing_import_from_url)

    marker_scanned = threading.Event()
    real_guess_source = installer._guess_source

    def _observing_guess_source(source_str: str):
        result = real_guess_source(source_str)
        marker_scanned.set()
        return result

    monkeypatch.setattr(installer, "_guess_source", _observing_guess_source)

    resumed: list[ModelInstallJob] = []
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: resumed.append(job))

    import_errors: list[Exception] = []

    def _run_import() -> None:
        try:
            installer.import_model(source)
        except Exception as e:  # noqa: BLE001 - the simulated failure is the point
            import_errors.append(e)

    import_thread = threading.Thread(target=_run_import)
    start_thread = threading.Thread(target=installer.start)
    try:
        # The import passes _wait_for_restore_complete() (the event starts out
        # set), reserves the source in _pending_sources, then pauses inside its
        # helper - before any job is registered.
        import_thread.start()
        assert import_reserved.wait(timeout=10)

        # Restoration scans the marker while the reservation is pending. Its
        # main pass must defer the marker rather than restore or discard it.
        start_thread.start()
        assert marker_scanned.wait(timeout=10)

        # Fail the import before it registers a job. The reservation is
        # discarded, so the marker has no owner; restore's deferred recheck
        # must now restore it.
        release_import.set()
        import_thread.join(timeout=10)
        assert import_errors, "the paused import was expected to fail"

        start_thread.join(timeout=10)
        installer._wait_for_restore_complete()

        jobs_for_source = [job for job in installer._install_jobs if str(job.source) == str(source)]
        assert len(jobs_for_source) == 1, "restore did not recover the marker of the failed import"
        assert jobs_for_source[0]._install_tmpdir == tmpdir
        assert resumed == jobs_for_source
        assert tmpdir.exists()
    finally:
        release_import.set()
        installer.stop()


@pytest.mark.timeout(timeout=20, method="thread")
def test_deferred_restore_skips_marker_of_import_that_completed(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pending import whose job reaches a terminal state before restore's deferred
    recheck still resolved by REGISTERING a job, so it owned the source. The recheck
    must not mistake the terminal job for a failed reservation and register a
    duplicate job for the deferred marker."""
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))

    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}stale"
    _write_test_install_marker(tmpdir, str(source))

    import_reserved = threading.Event()
    release_import = threading.Event()

    def _instantly_completing_import(src, config=None):
        # Stand-in for an import whose download and install finish before
        # restore's deferred recheck runs: the job it registers is already
        # terminal, and nothing for the source remains in _download_cache.
        import_reserved.set()
        assert release_import.wait(timeout=10)
        job = ModelInstallJob(
            id=installer._next_id(),
            source=src,
            config_in=config or ModelRecordChanges(),
            local_path=mm2_app_config.models_path,
        )
        job.status = InstallStatus.COMPLETED
        return job

    monkeypatch.setattr(installer, "_import_from_url", _instantly_completing_import)

    marker_scanned = threading.Event()
    real_guess_source = installer._guess_source

    def _observing_guess_source(source_str: str):
        result = real_guess_source(source_str)
        marker_scanned.set()
        return result

    monkeypatch.setattr(installer, "_guess_source", _observing_guess_source)

    resumed: list[ModelInstallJob] = []
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: resumed.append(job))

    import_jobs: list[ModelInstallJob] = []
    import_thread = threading.Thread(target=lambda: import_jobs.append(installer.import_model(source)))
    start_thread = threading.Thread(target=installer.start)
    try:
        import_thread.start()
        assert import_reserved.wait(timeout=10)

        # Restoration scans the marker while the reservation is pending and
        # defers it.
        start_thread.start()
        assert marker_scanned.wait(timeout=10)

        # The import registers an already-terminal job and clears the
        # reservation. The deferred recheck runs only after that, and must
        # recognize the newly registered job as the reservation's outcome.
        release_import.set()
        import_thread.join(timeout=10)
        assert import_jobs and import_jobs[0].status == InstallStatus.COMPLETED

        start_thread.join(timeout=10)
        installer._wait_for_restore_complete()

        jobs_for_source = [job for job in installer._install_jobs if str(job.source) == str(source)]
        assert jobs_for_source == import_jobs, "restore registered a duplicate job for a completed import"
        assert resumed == []
        assert tmpdir.exists()
    finally:
        release_import.set()
        installer.stop()


@pytest.mark.timeout(timeout=20, method="thread")
def test_restore_completes_when_pending_import_hangs(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    embedding_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An import whose helpers hang (e.g. a metadata request that never returns)
    must not hold up restoration forever: every import_model call waits on the
    startup barrier restore sets, so an unbounded deferred wait would wedge all
    installs for all sources. Restore must time out, leave the hung source's
    marker for a later startup, and let unrelated imports proceed."""
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    monkeypatch.setattr(model_install_default, "DEFERRED_RESTORE_TIMEOUT", 0.25, raising=False)
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))

    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}hung"
    _write_test_install_marker(tmpdir, str(source))

    import_reserved = threading.Event()
    release_import = threading.Event()

    def _hanging_import_from_url(src, config=None):
        import_reserved.set()
        # Simulates a metadata fetch with no timeout: blocks until the test
        # tears down, far beyond the deferred-restore timeout.
        assert release_import.wait(timeout=15)
        raise RuntimeError("simulated hung import aborted by test teardown")

    monkeypatch.setattr(installer, "_import_from_url", _hanging_import_from_url)

    resumed: list[ModelInstallJob] = []
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: resumed.append(job))

    def _run_import() -> None:
        try:
            installer.import_model(source)
        except Exception:  # noqa: BLE001 - the simulated hang ends in a teardown error
            pass

    import_thread = threading.Thread(target=_run_import)
    try:
        import_thread.start()
        assert import_reserved.wait(timeout=10)

        # Restore defers the marker behind the hung reservation. It must give
        # up after the (shortened) timeout instead of blocking forever.
        installer.start()
        assert installer._restore_completed_event.wait(timeout=10), "restore never completed while an import was hung"

        # The hung source's marker is left alone for a later startup.
        jobs_for_source = [job for job in installer._install_jobs if str(job.source) == str(source)]
        assert jobs_for_source == []
        assert resumed == []
        assert tmpdir.exists()

        # The startup barrier lifted, so an unrelated import proceeds normally.
        unrelated_job = installer.import_model(LocalModelSource(path=embedding_file))
        installer.wait_for_installs(timeout=10)
        assert unrelated_job.complete
    finally:
        release_import.set()
        import_thread.join(timeout=10)
        installer.stop()


def test_deferred_restore_timeout_is_shared_across_markers_for_one_source(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate markers must not multiply the startup restoration timeout."""
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    for index in range(3):
        _write_test_install_marker(mm2_app_config.models_path / f"{TMPDIR_PREFIX}hung_{index}", str(source))

    with installer._lock:
        installer._pending_sources.add(str(source))

    clock = 0.0
    waits: list[float] = []

    def _monotonic() -> float:
        return clock

    def _wait(timeout: Optional[float] = None) -> None:
        nonlocal clock
        assert timeout is not None
        waits.append(timeout)
        clock += timeout

    monkeypatch.setattr(model_install_default.time, "monotonic", _monotonic)
    monkeypatch.setattr(installer._install_cond, "wait", _wait)

    installer._restore_incomplete_installs()

    assert waits == [model_install_default.DEFERRED_RESTORE_TIMEOUT]
    assert installer._install_jobs == []


def test_deferred_restore_timeout_is_global_across_sources(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source_a = URLModelSource(url=Url("https://www.test.foo/download/a.safetensors"))
    source_b = URLModelSource(url=Url("https://www.test.foo/download/b.safetensors"))
    _write_test_install_marker(mm2_app_config.models_path / f"{TMPDIR_PREFIX}a", str(source_a))
    _write_test_install_marker(mm2_app_config.models_path / f"{TMPDIR_PREFIX}b", str(source_b))

    with installer._lock:
        installer._pending_sources.update({str(source_a), str(source_b)})

    clock = 0.0
    waits: list[float] = []

    def _monotonic() -> float:
        return clock

    def _wait(timeout: Optional[float] = None) -> None:
        nonlocal clock
        assert timeout is not None
        waits.append(timeout)
        clock += timeout

    monkeypatch.setattr(model_install_default.time, "monotonic", _monotonic)
    monkeypatch.setattr(installer._install_cond, "wait", _wait)
    real_glob = Path.glob
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter(sorted(real_glob(self, pattern))))
    installer._restore_incomplete_installs()

    assert sum(waits) == model_install_default.DEFERRED_RESTORE_TIMEOUT
    assert installer._install_jobs == []


def test_deferred_restore_remembers_registered_job_after_prune(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pruning a terminal import job must not make its deferred marker look unowned."""
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}stale"
    _write_test_install_marker(tmpdir, str(source))

    imported_job = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=mm2_app_config.models_path,
    )
    imported_job.status = InstallStatus.COMPLETED
    installer._restore_completed_event.clear()
    with installer._lock:
        installer._pending_sources.add(str(source))

    def _finish_and_prune_import(timeout: Optional[float] = None) -> None:
        installer._append_install_job(imported_job, from_import=True)
        installer._pending_sources.discard(str(source))
        installer._install_jobs = [job for job in installer._install_jobs if not job.in_terminal_state]

    resumed: list[ModelInstallJob] = []
    monkeypatch.setattr(installer._install_cond, "wait", _finish_and_prune_import)
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: resumed.append(job))

    installer._restore_incomplete_installs()

    assert installer._install_jobs == []
    assert resumed == []
    assert tmpdir.exists()


def test_deferred_restore_ignores_non_import_job_generation(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the pending import's registration may satisfy its deferred marker."""
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}deferred"
    _write_test_install_marker(tmpdir, str(source))

    other_restore_job = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=mm2_app_config.models_path,
    )
    other_restore_job.status = InstallStatus.ERROR
    with installer._lock:
        installer._pending_sources.add(str(source))

    def _finish_failed_import(timeout: Optional[float] = None) -> None:
        installer._append_install_job(other_restore_job)
        installer._pending_sources.discard(str(source))
        installer._install_jobs = [job for job in installer._install_jobs if not job.in_terminal_state]

    resumed: list[ModelInstallJob] = []
    monkeypatch.setattr(installer._install_cond, "wait", _finish_failed_import)
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: resumed.append(job))

    installer._restore_incomplete_installs()

    assert len(installer._install_jobs) == 1
    assert installer._install_jobs[0]._install_tmpdir == tmpdir
    assert resumed == installer._install_jobs


@pytest.mark.timeout(timeout=20, method="thread")
def test_restore_removes_stale_marker_when_active_source_has_multiple_markers(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    stale_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}0_stale"
    active_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}1_active"
    _write_test_install_marker(stale_dir, str(source))
    _write_test_install_marker(active_dir, str(source))

    active_job = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=active_dir,
    )
    active_job._install_tmpdir = active_dir
    active_job.status = InstallStatus.DOWNLOADING
    installer._install_jobs.append(active_job)
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: None)

    installer._restore_incomplete_installs()

    assert active_dir.exists()
    assert not stale_dir.exists()
    assert installer._install_jobs == [active_job]


def test_restore_does_not_delete_tmpdir_claimed_after_stale_check(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/claimed-during-delete.safetensors"))
    stale_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}0_stale"
    active_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}1_active"
    _write_test_install_marker(stale_dir, str(source))
    active_dir.mkdir()
    owner = ModelInstallJob(
        id=installer._next_id(), source=source, config_in=ModelRecordChanges(), local_path=active_dir
    )
    owner._install_tmpdir = active_dir
    owner.status = InstallStatus.DOWNLOADING
    installer._install_jobs.append(owner)

    deleting = threading.Event()
    release_delete = threading.Event()
    real_safe_rmtree = installer._safe_rmtree

    def _blocked_safe_rmtree(path: Path, logger: Any) -> None:
        if path == stale_dir:
            deleting.set()
            assert release_delete.wait(timeout=5)
        real_safe_rmtree(path, logger)

    monkeypatch.setattr(installer, "_safe_rmtree", _blocked_safe_rmtree)
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: None)
    restore_thread = threading.Thread(target=installer._restore_incomplete_installs)
    restore_thread.start()
    assert deleting.wait(timeout=5)

    owner.status = InstallStatus.COMPLETED
    installer.prune_jobs()
    imported_job = ModelInstallJob(
        id=installer._next_id(), source=source, config_in=ModelRecordChanges(), local_path=active_dir
    )
    replacement_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}2_replacement"

    def _import_from_url(*args, **kwargs) -> ModelInstallJob:
        assert installer._find_reusable_tmpdir(source) is None
        replacement_dir.mkdir()
        imported_job.local_path = replacement_dir
        imported_job._install_tmpdir = replacement_dir
        return imported_job

    monkeypatch.setattr(installer, "_import_from_url", _import_from_url)
    installer.import_model(source)
    release_delete.set()
    restore_thread.join(timeout=5)

    assert not restore_thread.is_alive()
    assert not stale_dir.exists()
    assert replacement_dir.exists()


def test_import_generation_tracking_is_bounded_to_active_restore(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )

    for index in range(100):
        source = URLModelSource(url=Url(f"https://www.test.foo/download/{index}.safetensors"))
        job = ModelInstallJob(
            id=installer._next_id(),
            source=source,
            config_in=ModelRecordChanges(),
            local_path=mm2_app_config.models_path,
        )
        job.status = InstallStatus.COMPLETED
        with installer._lock:
            installer._append_install_job(job, from_import=True)

    assert installer._source_import_generations == {}

    installer._restore_completed_event.clear()
    source = URLModelSource(url=Url("https://www.test.foo/download/during-restore.safetensors"))
    job = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=mm2_app_config.models_path,
    )
    with installer._lock:
        installer._pending_sources.add(str(source))
        installer._append_install_job(job, from_import=True)

    assert installer._source_import_generations == {str(source): 1}


def test_deferred_restore_removes_stale_marker_after_import_registration(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    stale_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}0_stale"
    active_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}1_active"
    _write_test_install_marker(stale_dir, str(source))
    _write_test_install_marker(active_dir, str(source))

    imported_job = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=active_dir,
    )
    imported_job._install_tmpdir = active_dir
    imported_job.status = InstallStatus.DOWNLOADING
    installer._restore_completed_event.clear()
    with installer._lock:
        installer._pending_sources.add(str(source))

    def _finish_import(timeout: Optional[float] = None) -> None:
        installer._append_install_job(imported_job, from_import=True)
        installer._pending_sources.discard(str(source))

    real_glob = Path.glob
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter(sorted(real_glob(self, pattern))))
    monkeypatch.setattr(installer._install_cond, "wait", _finish_import)
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: None)

    installer._restore_incomplete_installs()

    assert active_dir.exists()
    assert not stale_dir.exists()
    assert installer._install_jobs == [imported_job]


def test_prune_jobs_keeps_concurrent_import_registration(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    terminal_job = ModelInstallJob(
        id=installer._next_id(),
        source=URLModelSource(url=Url("https://www.test.foo/download/terminal.safetensors")),
        config_in=ModelRecordChanges(),
        local_path=mm2_app_config.models_path,
    )
    terminal_job.status = InstallStatus.COMPLETED
    imported_job = ModelInstallJob(
        id=installer._next_id(),
        source=URLModelSource(url=Url("https://www.test.foo/download/imported.safetensors")),
        config_in=ModelRecordChanges(),
        local_path=mm2_app_config.models_path,
    )

    iteration_started = threading.Event()
    release_iteration = threading.Event()

    class BlockingJobs(list[ModelInstallJob]):
        def __iter__(self):
            snapshot = iter(list(list.__iter__(self)))
            iteration_started.set()
            assert release_iteration.wait(timeout=5)
            return snapshot

    installer._install_jobs = BlockingJobs([terminal_job])
    prune_thread = threading.Thread(target=installer.prune_jobs)

    def _append_import() -> None:
        with installer._lock:
            installer._append_install_job(imported_job)

    append_thread = threading.Thread(target=_append_import)
    prune_thread.start()
    assert iteration_started.wait(timeout=5)
    append_thread.start()
    release_iteration.set()
    prune_thread.join(timeout=5)
    append_thread.join(timeout=5)

    assert not prune_thread.is_alive()
    assert not append_thread.is_alive()
    assert installer._install_jobs == [imported_job]


def test_stop_waits_for_inflight_restore_launch(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/launch.safetensors"))
    job = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=mm2_app_config.models_path,
    )
    entered = threading.Event()
    release = threading.Event()

    def _blocked_resume(restore_job: ModelInstallJob) -> None:
        entered.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr(installer, "_resume_remote_download", _blocked_resume)
    installer.start()
    installer._wait_for_restore_complete()
    launch_thread = threading.Thread(target=lambda: installer._launch_restored_job(job))
    launch_thread.start()
    assert entered.wait(timeout=5)

    stop_done = threading.Event()
    stop_thread = threading.Thread(target=lambda: (installer.stop(), stop_done.set()))
    stop_thread.start()
    assert not stop_done.wait(timeout=0.25)
    release.set()
    launch_thread.join(timeout=5)
    stop_thread.join(timeout=5)

    assert not launch_thread.is_alive()
    assert not stop_thread.is_alive()
    assert stop_done.is_set()


def test_stop_waits_for_restore_thread(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    restore_started = threading.Event()
    release_restore = threading.Event()

    def _blocked_restore() -> None:
        restore_started.set()
        assert release_restore.wait(timeout=5)

    monkeypatch.setattr(installer, "_restore_incomplete_installs", _blocked_restore)
    installer.start()
    assert restore_started.wait(timeout=5)

    stop_done = threading.Event()
    stop_thread = threading.Thread(target=lambda: (installer.stop(), stop_done.set()))
    stop_thread.start()
    assert not stop_done.wait(timeout=0.25)
    release_restore.set()
    stop_thread.join(timeout=5)

    assert not stop_thread.is_alive()
    assert stop_done.is_set()
    assert installer._restore_completed_event.is_set()


def test_stop_waits_for_startup_before_joining_restore_thread(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    startup_started = threading.Event()
    release_startup = threading.Event()

    def _blocked_restore_start() -> None:
        startup_started.set()
        assert release_startup.wait(timeout=5)

    monkeypatch.setattr(installer, "_restore_incomplete_installs_async", _blocked_restore_start)
    start_thread = threading.Thread(target=installer.start)
    start_thread.start()
    assert startup_started.wait(timeout=5)

    stop_done = threading.Event()
    stop_thread = threading.Thread(target=lambda: (installer.stop(), stop_done.set()))
    stop_thread.start()
    assert not stop_done.wait(timeout=0.25)
    release_startup.set()
    start_thread.join(timeout=5)
    stop_thread.join(timeout=5)

    assert not start_thread.is_alive()
    assert not stop_thread.is_alive()
    assert stop_done.is_set()
    assert installer._running is False


def test_restore_uses_snapshot_tmpdir_when_owner_finishes_mid_scan(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/snapshot.safetensors"))
    stale_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}0_stale"
    active_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}1_active"
    _write_test_install_marker(stale_dir, str(source))
    _write_test_install_marker(active_dir, str(source))
    owner = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=active_dir,
    )
    owner._install_tmpdir = active_dir
    owner.status = InstallStatus.DOWNLOADING
    installer._install_jobs.append(owner)

    scan_started = threading.Event()
    release_scan = threading.Event()
    real_guess_source = installer._guess_source

    def _pause_scan(source_str: str):
        result = real_guess_source(source_str)
        scan_started.set()
        assert release_scan.wait(timeout=5)
        return result

    monkeypatch.setattr(installer, "_guess_source", _pause_scan)
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: None)
    restore_thread = threading.Thread(target=installer._restore_incomplete_installs)
    restore_thread.start()
    assert scan_started.wait(timeout=5)

    owner.status = InstallStatus.COMPLETED
    installer._delete_install_marker(active_dir)
    installer.prune_jobs()
    release_scan.set()
    restore_thread.join(timeout=5)

    assert not restore_thread.is_alive()
    assert not stale_dir.exists()


def test_import_waiter_rechecks_shutdown_after_reservation_clears(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/wakeup.safetensors"))
    installer.start()
    installer._wait_for_restore_complete()
    with installer._lock:
        installer._pending_sources.add(str(source))

    list_jobs_entered = threading.Event()
    release_list_jobs = threading.Event()
    real_list_jobs = installer.list_jobs
    list_jobs_calls = 0

    def _clear_reservation(timeout: Optional[float] = None) -> bool:
        installer._pending_sources.discard(str(source))
        return True

    def _pause_list_jobs():
        nonlocal list_jobs_calls
        list_jobs_calls += 1
        if list_jobs_calls > 1:
            return real_list_jobs()
        installer._stop_event.set()
        list_jobs_entered.set()
        assert release_list_jobs.wait(timeout=5)
        return real_list_jobs()

    monkeypatch.setattr(installer._install_cond, "wait", _clear_reservation)
    monkeypatch.setattr(installer, "list_jobs", _pause_list_jobs)
    helper_called = threading.Event()
    monkeypatch.setattr(installer, "_import_from_url", lambda *args, **kwargs: (helper_called.set(), None)[1])
    errors: list[Exception] = []

    def _run_import() -> None:
        try:
            installer.import_model(source)
        except Exception as exc:
            errors.append(exc)

    import_thread = threading.Thread(target=_run_import)
    import_thread.start()
    assert list_jobs_entered.wait(timeout=5)
    assert installer._stop_event.wait(timeout=5)
    release_list_jobs.set()
    import_thread.join(timeout=5)
    installer.stop()

    assert not import_thread.is_alive()
    assert not helper_called.is_set()
    assert len(errors) == 1
    assert str(errors[0]) == "Model install service stopped"


def test_stop_cancels_deferred_restore_and_prevents_late_launch(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deferred restoration must not launch work after the installer stops."""
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}shutdown"
    _write_test_install_marker(tmpdir, str(source))
    with installer._lock:
        installer._pending_sources.add(str(source))

    restore_waiting = threading.Event()
    real_wait = installer._install_cond.wait

    def _observing_wait(timeout: Optional[float] = None) -> bool:
        restore_waiting.set()
        return real_wait(timeout)

    resumed: list[ModelInstallJob] = []
    monkeypatch.setattr(installer._install_cond, "wait", _observing_wait)
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: resumed.append(job))

    try:
        installer.start()
        assert restore_waiting.wait(timeout=10)
        installer.stop()

        with installer._install_cond:
            installer._pending_sources.discard(str(source))
            installer._install_cond.notify_all()
        assert installer._restore_completed_event.wait(timeout=5)

        late_job = ModelInstallJob(
            id=installer._next_id(),
            source=source,
            config_in=ModelRecordChanges(),
            local_path=tmpdir,
        )
        late_job._install_tmpdir = tmpdir
        installer._launch_restored_job(late_job)

        assert resumed == []
        assert installer._install_jobs == []
        assert tmpdir.exists()
    finally:
        with installer._install_cond:
            installer._pending_sources.discard(str(source))
            installer._install_cond.notify_all()
        installer.stop()


def test_import_waiter_returns_registered_job(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    existing_job = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=mm2_app_config.models_path,
    )
    existing_job.status = InstallStatus.DOWNLOADING
    with installer._lock:
        installer._install_jobs.append(existing_job)
        installer._pending_sources.add(str(source))

    waiter_entered = threading.Event()
    real_wait = installer._install_cond.wait

    def _observing_wait(timeout: Optional[float] = None) -> bool:
        waiter_entered.set()
        return real_wait(timeout)

    monkeypatch.setattr(installer._install_cond, "wait", _observing_wait)
    result: list[ModelInstallJob] = []
    errors: list[Exception] = []

    def _wait_for_import() -> None:
        try:
            result.append(installer.import_model(source))
        except Exception as exc:  # noqa: BLE001 - assertion target
            errors.append(exc)

    thread = threading.Thread(target=_wait_for_import)
    thread.start()
    assert waiter_entered.wait(timeout=5)
    with installer._install_cond:
        installer._pending_sources.discard(str(source))
        installer._install_cond.notify_all()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert errors == []
    assert result == [existing_job]


@pytest.mark.timeout(timeout=20, method="thread")
def test_import_waiter_aborts_when_service_stops(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    installer.start()
    installer._wait_for_restore_complete()
    with installer._lock:
        installer._pending_sources.add(str(source))

    waiter_entered = threading.Event()
    real_wait = installer._install_cond.wait

    def _observing_wait(timeout: Optional[float] = None) -> bool:
        waiter_entered.set()
        return real_wait(timeout)

    monkeypatch.setattr(installer._install_cond, "wait", _observing_wait)
    monkeypatch.setattr(installer, "_import_from_url", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError()))
    errors: list[Exception] = []

    def _wait_for_import() -> None:
        try:
            installer.import_model(source)
        except Exception as exc:  # noqa: BLE001 - assertion target
            errors.append(exc)

    thread = threading.Thread(target=_wait_for_import, daemon=True)
    thread.start()
    assert waiter_entered.wait(timeout=5)
    installer.stop()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert str(errors[0]) == "Model install service stopped"


def test_stop_does_not_wait_forever_for_restore_metadata(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/hung-restore.safetensors"))
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}hung_restore"
    job = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
    )
    job._install_tmpdir = tmpdir
    metadata_started = threading.Event()
    release_metadata = threading.Event()

    def _hung_metadata(model_source):
        metadata_started.set()
        assert release_metadata.wait(timeout=5)
        raise RuntimeError("released metadata request")

    monkeypatch.setattr(installer, "_remote_files_from_source", _hung_metadata)
    monkeypatch.setattr(model_install_default, "RESTORE_SHUTDOWN_TIMEOUT", 0.1, raising=False)
    installer.start()
    installer._wait_for_restore_complete()
    _write_test_install_marker(tmpdir, str(source))
    with installer._lock:
        installer._append_install_job(job)
    launch_thread = threading.Thread(target=lambda: installer._launch_restored_job(job))
    launch_thread.start()
    assert metadata_started.wait(timeout=5)

    stop_done = threading.Event()
    stop_thread = threading.Thread(target=lambda: (installer.stop(), stop_done.set()))
    stop_thread.start()
    try:
        assert stop_done.wait(timeout=3)
    finally:
        release_metadata.set()
        launch_thread.join(timeout=5)
        stop_thread.join(timeout=5)

    assert tmpdir.exists()


def test_import_helper_cannot_register_after_stop(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/stopped-import.safetensors"))
    helper_started = threading.Event()
    release_helper = threading.Event()
    returned_job = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=mm2_app_config.models_path,
    )

    def _blocked_helper(model_source, config=None):
        helper_started.set()
        assert release_helper.wait(timeout=5)
        return returned_job

    monkeypatch.setattr(installer, "_import_from_url", _blocked_helper)
    installer.start()
    installer._wait_for_restore_complete()
    errors: list[Exception] = []

    def _run_import() -> None:
        try:
            installer.import_model(source)
        except Exception as exc:
            errors.append(exc)

    import_thread = threading.Thread(target=_run_import)
    import_thread.start()
    assert helper_started.wait(timeout=5)
    installer.stop()
    release_helper.set()
    import_thread.join(timeout=5)

    assert not import_thread.is_alive()
    assert installer._install_jobs == []
    assert installer._source_import_generations == {}
    assert len(errors) == 1
    assert str(errors[0]) == "Model install service stopped"


def test_stop_waits_for_remote_enqueue_before_stopping(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/stopped-enqueue.safetensors"))
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}stopped_enqueue"
    tmpdir.mkdir()
    job = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
    )
    multifile_started = threading.Event()
    release_multifile = threading.Event()
    real_multifile_download = installer._multifile_download
    remote_files = [
        RemoteModelFile(
            url=source.url,
            path=Path("stopped-enqueue.safetensors"),
            size=1,
        )
    ]

    def _blocked_multifile(*args, **kwargs):
        multifile_started.set()
        assert release_multifile.wait(timeout=5)
        return real_multifile_download(*args, **kwargs)

    monkeypatch.setattr(installer, "_multifile_download", _blocked_multifile)
    monkeypatch.setattr(mm2_download_queue, "submit_download_job", lambda *args, **kwargs: None)
    installer.start()
    installer._wait_for_restore_complete()
    errors: list[Exception] = []

    def _enqueue() -> None:
        try:
            installer._enqueue_remote_download(job, source, remote_files, None, tmpdir)
        except Exception as exc:
            errors.append(exc)

    enqueue_thread = threading.Thread(target=_enqueue)
    enqueue_thread.start()
    assert multifile_started.wait(timeout=5)
    stop_done = threading.Event()
    stop_thread = threading.Thread(target=lambda: (installer.stop(), stop_done.set()))
    stop_thread.start()
    assert not stop_done.wait(timeout=0.25)
    release_multifile.set()
    enqueue_thread.join(timeout=5)
    stop_thread.join(timeout=5)

    assert not enqueue_thread.is_alive()
    assert not stop_thread.is_alive()
    assert installer._download_cache == {}
    assert errors == []
    assert job._multifile_job is not None
    assert installer._marker_path(tmpdir).exists()


@pytest.mark.parametrize("reuse_existing", [False, True])
def test_stopped_remote_import_cleans_only_new_tmpdir(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
    reuse_existing: bool,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/stopped-import-dir.safetensors"))
    reusable_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}reusable"
    remote_files = [RemoteModelFile(url=source.url, path=Path("stopped-import-dir.safetensors"), size=1)]
    enqueue_started = threading.Event()
    release_enqueue = threading.Event()
    real_enqueue = installer._enqueue_remote_download

    def _blocked_enqueue(*args, **kwargs):
        enqueue_started.set()
        assert release_enqueue.wait(timeout=5)
        return real_enqueue(*args, **kwargs)

    monkeypatch.setattr(installer, "_remote_files_from_source", lambda model_source: (remote_files, None))
    monkeypatch.setattr(installer, "_enqueue_remote_download", _blocked_enqueue)
    installer.start()
    installer._wait_for_restore_complete()
    if reuse_existing:
        _write_test_install_marker(reusable_dir, str(source))
    before = set(mm2_app_config.models_path.glob(f"{TMPDIR_PREFIX}*"))
    errors: list[Exception] = []

    def _import() -> None:
        try:
            installer.import_model(source)
        except Exception as exc:
            errors.append(exc)

    import_thread = threading.Thread(target=_import)
    import_thread.start()
    assert enqueue_started.wait(timeout=5)
    installer.stop()
    release_enqueue.set()
    import_thread.join(timeout=5)

    assert not import_thread.is_alive()
    assert len(errors) == 1
    assert str(errors[0]) == "Model install service stopped"
    assert set(mm2_app_config.models_path.glob(f"{TMPDIR_PREFIX}*")) == before
    if reuse_existing:
        assert reusable_dir.exists()


def test_import_generations_do_not_accumulate_after_restore(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )

    def _completed_import(source, config=None):
        job = ModelInstallJob(
            id=installer._next_id(),
            source=source,
            config_in=ModelRecordChanges(),
            local_path=mm2_app_config.models_path,
        )
        job.status = InstallStatus.COMPLETED
        return job

    monkeypatch.setattr(installer, "_import_from_url", _completed_import)
    installer.start()
    installer._wait_for_restore_complete()
    try:
        for index in range(10):
            source = URLModelSource(url=Url(f"https://www.test.foo/download/post-restore-{index}.safetensors"))
            installer.import_model(source)
            installer.prune_jobs()

        assert installer._source_import_generations == {}
    finally:
        installer.stop()


def test_pending_import_preserves_tmpdir_rejected_by_owner_snapshot(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/pending-owner.safetensors"))
    owner_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}0_owner"
    pending_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}1_pending"
    _write_test_install_marker(owner_dir, str(source))
    _write_test_install_marker(pending_dir, str(source))
    owner = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=owner_dir,
    )
    owner._install_tmpdir = owner_dir
    owner.status = InstallStatus.DOWNLOADING
    installer._install_jobs.append(owner)
    imported_job = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=pending_dir,
    )
    imported_job._install_tmpdir = pending_dir
    imported_job.status = InstallStatus.DOWNLOADING
    scan_started = threading.Event()
    release_scan = threading.Event()
    real_guess_source = installer._guess_source

    def _pause_first_marker(source_str: str):
        result = real_guess_source(source_str)
        if not scan_started.is_set():
            scan_started.set()
            assert release_scan.wait(timeout=5)
        return result

    def _finish_import(timeout: Optional[float] = None) -> None:
        installer._append_install_job(imported_job, from_import=True)
        installer._pending_sources.discard(str(source))

    real_glob = Path.glob
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter(sorted(real_glob(self, pattern))))
    monkeypatch.setattr(installer, "_guess_source", _pause_first_marker)
    monkeypatch.setattr(installer._install_cond, "wait", _finish_import)
    restore_thread = threading.Thread(target=installer._restore_incomplete_installs)
    restore_thread.start()
    assert scan_started.wait(timeout=5)
    owner.status = InstallStatus.COMPLETED
    installer.prune_jobs()
    with installer._lock:
        installer._pending_sources.add(str(source))
    installer._restore_completed_event.clear()
    release_scan.set()
    restore_thread.join(timeout=5)

    assert not restore_thread.is_alive()
    assert pending_dir.exists()
    assert installer._install_jobs == [imported_job]


def test_duplicate_deferred_markers_share_timeout_resolution(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/timeout-resolution.safetensors"))
    first_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}0_timeout"
    second_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}1_timeout"
    _write_test_install_marker(first_dir, str(source))
    _write_test_install_marker(second_dir, str(source))
    with installer._lock:
        installer._pending_sources.add(str(source))

    clock = 0.0
    real_warning = installer._logger.warning

    def _monotonic() -> float:
        return clock

    def _wait(timeout: Optional[float] = None) -> None:
        nonlocal clock
        assert timeout is not None
        clock += timeout

    def _warning(message: str) -> None:
        real_warning(message)
        installer._pending_sources.discard(str(source))

    monkeypatch.setattr(model_install_default.time, "monotonic", _monotonic)
    monkeypatch.setattr(installer._install_cond, "wait", _wait)
    monkeypatch.setattr(installer._logger, "warning", _warning)
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: None)

    installer._restore_incomplete_installs()

    assert installer._install_jobs == []
    assert first_dir.exists()
    assert second_dir.exists()


def test_late_import_after_restore_timeout_removes_duplicate_marker(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = URLModelSource(url=Url("https://www.test.foo/download/late-timeout.safetensors"))
    first_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}0_timeout"
    second_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}1_timeout"
    _write_test_install_marker(first_dir, str(source))
    _write_test_install_marker(second_dir, str(source))
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    helper_started = threading.Event()
    release_helper = threading.Event()
    selected_dirs: list[Path] = []

    def _late_import(*args, **kwargs) -> ModelInstallJob:
        helper_started.set()
        assert release_helper.wait(timeout=5)
        selected_dir = installer._find_reusable_tmpdir(source)
        assert selected_dir in {first_dir, second_dir}
        selected_dirs.append(selected_dir)
        job = ModelInstallJob(
            id=installer._next_id(), source=source, config_in=ModelRecordChanges(), local_path=selected_dir
        )
        job._install_tmpdir = selected_dir
        return job

    monkeypatch.setattr(installer, "_import_from_url", _late_import)
    import_thread = threading.Thread(target=lambda: installer.import_model(source))
    import_thread.start()
    assert helper_started.wait(timeout=5)
    monkeypatch.setattr(model_install_default, "DEFERRED_RESTORE_TIMEOUT", 0.0)
    installer._restore_completed_event.clear()
    installer._restore_incomplete_installs()
    installer._restore_completed_event.set()
    release_helper.set()
    import_thread.join(timeout=5)

    assert not import_thread.is_alive()
    assert len(selected_dirs) == 1
    selected_dir = selected_dirs[0]
    installer._safe_rmtree(selected_dir, installer._logger)

    restarted = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    resumed: list[ModelInstallJob] = []
    monkeypatch.setattr(restarted, "_resume_remote_download", lambda job: resumed.append(job))
    restarted._restore_incomplete_installs()

    assert resumed == []
    assert restarted._install_jobs == []


def test_deferred_restore_ignores_historical_terminal_tmpdirs(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )
    source = URLModelSource(url=Url("https://www.test.foo/download/historical.safetensors"))
    stale_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}0_historical"
    active_dir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}1_historical"
    _write_test_install_marker(stale_dir, str(source))
    _write_test_install_marker(active_dir, str(source))
    historical_job = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=stale_dir,
    )
    historical_job._install_tmpdir = stale_dir
    historical_job.status = InstallStatus.COMPLETED
    imported_job = ModelInstallJob(
        id=installer._next_id(),
        source=source,
        config_in=ModelRecordChanges(),
        local_path=active_dir,
    )
    imported_job._install_tmpdir = active_dir
    imported_job.status = InstallStatus.DOWNLOADING
    installer._install_jobs.append(historical_job)
    installer._restore_completed_event.clear()
    with installer._lock:
        installer._pending_sources.add(str(source))

    def _finish_import(timeout: Optional[float] = None) -> None:
        installer._append_install_job(imported_job, from_import=True)
        installer._pending_sources.discard(str(source))

    real_glob = Path.glob
    monkeypatch.setattr(Path, "glob", lambda self, pattern: iter(sorted(real_glob(self, pattern))))
    monkeypatch.setattr(installer._install_cond, "wait", _finish_import)
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: None)

    installer._restore_incomplete_installs()

    assert not stale_dir.exists()
    assert active_dir.exists()
    assert installer._install_jobs == [historical_job, imported_job]


def test_huggingface_blob_url_uses_resolve_download_url(mm2_installer: ModelInstallServiceBase) -> None:
    source = URLModelSource(
        url=Url("https://huggingface.co/h94/IP-Adapter/blob/main/sdxl_models/ip-adapter.safetensors")
    )

    assert isinstance(mm2_installer, ModelInstallService)
    files, metadata = mm2_installer._remote_files_from_source(source)

    assert metadata is None
    assert len(files) == 1
    assert str(files[0].url) == "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter.safetensors"


@pytest.mark.timeout(timeout=10, method="thread")
def test_huggingface_install(mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig) -> None:
    source = URLModelSource(url=Url("https://huggingface.co/stabilityai/sdxl-turbo"))

    bus: TestEventService = mm2_installer.event_bus
    store = mm2_installer.record_store
    assert isinstance(bus, EventServiceBase)
    assert store is not None

    job = mm2_installer.import_model(source)
    job_list = mm2_installer.wait_for_installs(timeout=10)
    assert len(job_list) == 1
    assert job.complete
    assert job.config_out

    key = job.config_out.key
    model_record = store.get_model(key)
    assert (mm2_app_config.models_path / model_record.path).exists()
    assert model_record.type == ModelType.Main
    assert model_record.format == ModelFormat.Diffusers

    assert any(isinstance(x, ModelInstallStartedEvent) for x in bus.events)
    assert any(isinstance(x, ModelInstallDownloadProgressEvent) for x in bus.events)
    assert any(isinstance(x, ModelInstallCompleteEvent) for x in bus.events)
    assert len(bus.events) >= 3


@pytest.mark.timeout(timeout=10, method="thread")
def test_huggingface_repo_id(mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig) -> None:
    source = HFModelSource(repo_id="stabilityai/sdxl-turbo", variant=ModelRepoVariant.Default)

    bus = mm2_installer.event_bus
    store = mm2_installer.record_store
    assert isinstance(bus, EventServiceBase)
    assert store is not None

    job = mm2_installer.import_model(source)
    job_list = mm2_installer.wait_for_installs(timeout=10)
    assert len(job_list) == 1
    assert job.complete
    assert job.config_out

    key = job.config_out.key
    model_record = store.get_model(key)
    assert (mm2_app_config.models_path / model_record.path).exists()
    assert model_record.type == ModelType.Main
    assert model_record.format == ModelFormat.Diffusers

    assert hasattr(bus, "events")  # the dummyeventservice has this
    assert len(bus.events) >= 3
    event_types = [type(x) for x in bus.events]
    assert all(
        x in event_types
        for x in [
            ModelInstallDownloadProgressEvent,
            ModelInstallDownloadsCompleteEvent,
            ModelInstallStartedEvent,
            ModelInstallCompleteEvent,
        ]
    )

    completed_events = [x for x in bus.events if isinstance(x, ModelInstallCompleteEvent)]
    downloading_events = [x for x in bus.events if isinstance(x, ModelInstallDownloadProgressEvent)]
    assert completed_events[0].total_bytes == downloading_events[-1].bytes
    assert job.total_bytes == completed_events[0].total_bytes
    print(downloading_events[-1])
    print(job.download_parts)
    assert job.total_bytes == sum(x["total_bytes"] for x in downloading_events[-1].parts)


def test_restore_paused_hf_install_preserves_access_token(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    mm2_download_queue,
    mm2_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert isinstance(mm2_installer, ModelInstallService)

    access_token = "hf_test_access_token"
    tmpdir = mm2_app_config.models_path / f"tmpinstall_resume_token_{uuid.uuid4().hex}"
    tmpdir.mkdir(parents=True, exist_ok=True)

    try:
        paused_job = ModelInstallJob(
            id=99999,
            source=HFModelSource(
                repo_id="stabilityai/sdxl-turbo",
                variant=ModelRepoVariant.Default,
                access_token=access_token,
            ),
            config_in=ModelRecordChanges(),
            local_path=tmpdir,
        )
        paused_job._install_tmpdir = tmpdir
        paused_job.status = InstallStatus.PAUSED

        mm2_installer._write_install_marker(paused_job, status=InstallStatus.PAUSED)

        marker = mm2_installer._read_install_marker(tmpdir)
        assert marker is not None
        assert marker["access_token"] == access_token

        restored_installer = ModelInstallService(
            app_config=mm2_app_config,
            record_store=mm2_installer.record_store,
            download_queue=mm2_download_queue,
            session=mm2_session,
        )
        restored_installer._restore_incomplete_installs()
        restored_jobs = restored_installer.list_jobs()
        assert len(restored_jobs) == 1

        restored_job = restored_jobs[0]
        assert restored_job.paused
        assert isinstance(restored_job.source, HFModelSource)
        assert restored_job.source.access_token == access_token

        captured: dict[str, str | None] = {}

        def _capture_resume(job: ModelInstallJob) -> None:
            assert isinstance(job.source, HFModelSource)
            captured["access_token"] = job.source.access_token

        monkeypatch.setattr(restored_installer, "_resume_remote_download", _capture_resume)
        restored_installer.resume_job(restored_job)
        assert captured["access_token"] == access_token
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_404_download(mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig) -> None:
    source = URLModelSource(url=Url("https://test.com/missing_model.safetensors"))
    job = mm2_installer.import_model(source)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.status == InstallStatus.ERROR
    assert job.errored
    assert job.error_type == "HTTPError"
    assert job.error
    assert "NOT FOUND" in job.error
    assert job.error_traceback is not None
    assert job.error_traceback.startswith("Traceback")
    bus = mm2_installer.event_bus
    assert bus is not None
    assert hasattr(bus, "events")  # the dummyeventservice has this
    event_types = [type(x) for x in bus.events]
    assert ModelInstallErrorEvent in event_types


def test_other_error_during_install(
    monkeypatch: pytest.MonkeyPatch, mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("Test error")

    monkeypatch.setattr(
        "invokeai.app.services.model_install.model_install_default.ModelInstallService._register_or_install",
        raise_runtime_error,
    )
    source = LocalModelSource(path=Path("tests/data/embedding/test_embedding.safetensors"))
    job = mm2_installer.import_model(source)
    mm2_installer.wait_for_installs(timeout=10)
    assert job.status == InstallStatus.ERROR
    assert job.errored
    assert job.error_type == "RuntimeError"
    assert job.error == "Test error"


@pytest.mark.parametrize(
    "model_params",
    [
        # SDXL, Lora
        {
            "repo_id": "InvokeAI-test/textual_inversion_tests::learned_embeds-steps-1000.safetensors",
            "name": "test_lora",
            "type": "embedding",
        },
        # SDXL, Lora - incorrect type
        {
            "repo_id": "InvokeAI-test/textual_inversion_tests::learned_embeds-steps-1000.safetensors",
            "name": "test_lora",
            "type": "lora",
        },
    ],
)
@pytest.mark.timeout(timeout=10, method="thread")
def test_heuristic_import_with_type(mm2_installer: ModelInstallServiceBase, model_params: Dict[str, str]):
    """Test whether or not type is respected on configs when passed to heuristic import."""
    assert "name" in model_params and "type" in model_params
    config1: Dict[str, Any] = {
        "name": f"{model_params['name']}_1",
        "type": model_params["type"],
        "hash": "placeholder1",
    }
    config2: Dict[str, Any] = {
        "name": f"{model_params['name']}_2",
        "type": ModelType(model_params["type"]),
        "hash": "placeholder2",
    }
    assert "repo_id" in model_params
    install_job1 = mm2_installer.heuristic_import(source=model_params["repo_id"], config=config1)
    mm2_installer.wait_for_job(install_job1, timeout=10)
    if model_params["type"] != "embedding":
        assert install_job1.errored
        assert install_job1.error_type == "InvalidModelConfigException"
        return
    assert install_job1.complete
    assert install_job1.config_out if model_params["type"] == "embedding" else not install_job1.config_out

    install_job2 = mm2_installer.heuristic_import(source=model_params["repo_id"], config=config2)
    mm2_installer.wait_for_job(install_job2, timeout=10)
    assert install_job2.complete
    assert install_job2.config_out if model_params["type"] == "embedding" else not install_job2.config_out
