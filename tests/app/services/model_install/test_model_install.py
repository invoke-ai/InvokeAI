"""
Test the model installer
"""

import gc
import platform
import shutil
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest
from pydantic_core import Url

from invokeai.app.services.config import InvokeAIAppConfig
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
)
from invokeai.app.services.model_install.model_install_common import (
    InstallStatus,
    InvalidModelConfigException,
    LocalModelSource,
    ModelInstallJob,
    URLModelSource,
)
from invokeai.app.services.model_install.model_install_default import TMPDIR_PREFIX
from invokeai.app.services.model_records import ModelRecordChanges, UnknownModelException
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig
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


@pytest.mark.timeout(timeout=10, method="thread")
def test_wait_for_installs_waits_for_download_completion_to_enqueue_install(
    mm2_installer: ModelInstallServiceBase, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not report all installs done between download-cache removal and install enqueue."""
    assert isinstance(mm2_installer, ModelInstallService)
    job = ModelInstallJob(
        id=123,
        source=URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors")),
        local_path=tmp_path,
    )
    with mm2_installer._lock:
        mm2_installer._download_cache[job.id] = job
    monkeypatch.setattr(mm2_installer, "_signal_job_downloads_done", lambda install_job: None)
    assert mm2_installer._wait_for_restore_complete(timeout=2)

    wait_finished = threading.Event()

    def wait_for_installs() -> None:
        mm2_installer.wait_for_installs(timeout=2)
        wait_finished.set()

    wait_thread = threading.Thread(target=wait_for_installs)
    wait_thread.start()

    enqueue_started = threading.Event()
    release_enqueue = threading.Event()

    def blocked_enqueue(install_job: ModelInstallJob) -> None:
        assert install_job is job
        enqueue_started.set()
        assert release_enqueue.wait(timeout=2)

    monkeypatch.setattr(mm2_installer, "_put_in_queue", blocked_enqueue)
    callback_thread = threading.Thread(
        target=mm2_installer._download_complete_callback,
        args=(SimpleNamespace(id=job.id),),
    )
    callback_thread.start()
    assert enqueue_started.wait(timeout=2)
    assert not wait_finished.wait(timeout=0.5)

    release_enqueue.set()
    callback_thread.join(timeout=2)
    wait_thread.join(timeout=2)
    assert not callback_thread.is_alive()
    assert not wait_thread.is_alive()
    assert wait_finished.is_set()


@pytest.mark.timeout(timeout=10, method="thread")
def test_import_waits_for_startup_restore(
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
    import_waiting = threading.Event()
    imported = threading.Event()
    imported_jobs: list[ModelInstallJob] = []
    source = URLModelSource(url=Url("https://www.test.foo/download/interrupted.safetensors"))
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}interrupted"
    tmpdir.mkdir()
    interrupted_job = ModelInstallJob(
        id=99999,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
    )
    interrupted_job._install_tmpdir = tmpdir
    installer._write_install_marker(interrupted_job, status=InstallStatus.DOWNLOADING)
    restore = installer._restore_incomplete_installs
    wait_for_restore = installer._wait_for_restore_complete

    def _blocked_restore() -> None:
        restore_started.set()
        assert release_restore.wait(timeout=5)
        restore()

    def _import() -> None:
        imported_jobs.append(installer.import_model(source))
        imported.set()

    def _observed_wait_for_restore() -> bool:
        import_waiting.set()
        return wait_for_restore()

    monkeypatch.setattr(installer, "_restore_incomplete_installs", _blocked_restore)
    monkeypatch.setattr(installer, "_resume_remote_download", lambda job: None)

    try:
        assert not installer._restore_completed_event.is_set()
        installer.start()
        assert restore_started.wait(timeout=5)
        with pytest.raises(TimeoutError):
            installer.wait_for_installs(timeout=0.1)

        monkeypatch.setattr(installer, "_wait_for_restore_complete", _observed_wait_for_restore)
        import_thread = threading.Thread(target=_import)
        import_thread.start()
        assert import_waiting.wait(timeout=5)
        assert not imported.is_set()

        release_restore.set()
        import_thread.join(timeout=5)
        assert imported.is_set()
        jobs = installer.get_job_by_source(source)
        assert len(jobs) == 1
        assert imported_jobs == jobs
    finally:
        release_restore.set()
        installer.stop()


@pytest.mark.timeout(timeout=30, method="thread")
def test_concurrent_imports_of_same_source_return_one_job(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    assert mm2_installer._restore_completed_event.wait(timeout=5)

    # Both imports can reuse this interrupted install directory. Hold the first helper after its duplicate check so the
    # second import can reach the same post-restore race window.
    tmpdir = mm2_app_config.models_path / f"{TMPDIR_PREFIX}reusable"
    tmpdir.mkdir()
    interrupted_job = ModelInstallJob(
        id=99998,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=tmpdir,
    )
    interrupted_job._install_tmpdir = tmpdir
    mm2_installer._write_install_marker(interrupted_job, status=InstallStatus.DOWNLOADING)

    first_import_ready = threading.Event()
    second_import_waiting = threading.Event()
    second_helper_entered = threading.Event()
    release_first_import = threading.Event()
    helper_calls = 0
    helper_calls_lock = threading.Lock()
    import_from_url = mm2_installer._import_from_url
    imported_jobs: list[ModelInstallJob] = []
    import_errors: list[BaseException] = []
    condition_wait = mm2_installer._install_condition.wait

    def _observed_condition_wait(timeout: float | None = None) -> bool:
        second_import_waiting.set()
        return condition_wait(timeout)

    def _synchronized_import_from_url(
        import_source: URLModelSource, config: ModelRecordChanges | None = None
    ) -> ModelInstallJob:
        nonlocal helper_calls
        with helper_calls_lock:
            helper_calls += 1
            is_first_import = helper_calls == 1
        if is_first_import:
            first_import_ready.set()
            assert release_first_import.wait(timeout=5)
        else:
            second_helper_entered.set()
        return import_from_url(import_source, config)

    def _import() -> None:
        try:
            imported_jobs.append(mm2_installer.import_model(source))
        except BaseException as error:
            import_errors.append(error)

    monkeypatch.setattr(mm2_installer._install_condition, "wait", _observed_condition_wait)
    monkeypatch.setattr(mm2_installer, "_import_from_url", _synchronized_import_from_url)

    first_import_thread = threading.Thread(target=_import)
    second_import_thread = threading.Thread(target=_import)
    first_import_thread.start()
    assert first_import_ready.wait(timeout=5)
    second_import_thread.start()
    assert second_import_waiting.wait(timeout=5)
    assert not second_helper_entered.is_set()
    release_first_import.set()

    import_threads = [first_import_thread, second_import_thread]
    for import_thread in import_threads:
        import_thread.join(timeout=20)

    assert all(not import_thread.is_alive() for import_thread in import_threads)
    assert not import_errors
    jobs = mm2_installer.get_job_by_source(source)
    assert len(jobs) == 1
    assert len(imported_jobs) == 2
    assert imported_jobs[0] is imported_jobs[1] is jobs[0]


@pytest.mark.timeout(timeout=10, method="thread")
def test_failed_import_releases_source_reservation(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    import_attempts = 0

    def _import_from_url(import_source: URLModelSource, config: ModelRecordChanges | None = None) -> ModelInstallJob:
        nonlocal import_attempts
        import_attempts += 1
        if import_attempts == 1:
            raise RuntimeError("metadata request failed")
        return ModelInstallJob(
            id=99997,
            source=import_source,
            config_in=config or ModelRecordChanges(),
            local_path=mm2_app_config.models_path,
        )

    monkeypatch.setattr(mm2_installer, "_import_from_url", _import_from_url)

    with pytest.raises(RuntimeError, match="metadata request failed"):
        mm2_installer.import_model(source)

    job = mm2_installer.import_model(source)
    assert job.source == source
    assert mm2_installer.get_job_by_source(source) == [job]


@pytest.mark.timeout(timeout=30, method="thread")
def test_waiting_import_prefers_a_live_job_over_a_terminal_one(
    mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    """A waiter can be released into a state where the source has BOTH a terminal job registered while it
    waited and a live one. It must return the live job, not the dead one."""
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    source_key = str(source)
    condition = mm2_installer._install_condition
    assert mm2_installer._restore_completed_event.wait(timeout=10)

    terminal_job = ModelInstallJob(
        id=88001, source=source, config_in=ModelRecordChanges(), local_path=mm2_app_config.models_path
    )
    terminal_job.status = InstallStatus.ERROR
    live_job = ModelInstallJob(
        id=88002, source=source, config_in=ModelRecordChanges(), local_path=mm2_app_config.models_path
    )
    live_job.status = InstallStatus.DOWNLOADING

    # Stand in for an owner that has reserved the source but not yet registered its job.
    with condition:
        mm2_installer._pending_sources.add(source_key)

    waiter_result: list[ModelInstallJob] = []
    waiter = threading.Thread(target=lambda: waiter_result.append(mm2_installer.import_model(source)))
    waiter.start()

    # Wait until the importer is actually parked on the condition rather than sleeping a fixed interval.
    deadline = time.time() + 10
    while not condition._waiters and time.time() < deadline:
        time.sleep(0.01)
    assert condition._waiters, "importer never parked on the install condition"

    # Everything the waiter can observe happens in one critical section: an earlier attempt that ended in ERROR
    # and a later one that is still downloading. The waiter must not see an intermediate state.
    with condition:
        mm2_installer._install_jobs.append(terminal_job)
        mm2_installer._install_jobs.append(live_job)
        mm2_installer._pending_sources.discard(source_key)
        condition.notify_all()

    waiter.join(timeout=10)
    assert not waiter.is_alive()
    assert waiter_result and waiter_result[0] is live_job


@pytest.mark.timeout(timeout=30, method="thread")
def test_prune_jobs_cannot_drop_a_concurrent_registration(
    mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    """prune_jobs() filters and reassigns _install_jobs. Unlocked, an import_model() registration landing between
    the two is dropped, leaving a live install invisible to the duplicate check."""
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    assert mm2_installer._restore_completed_event.wait(timeout=10)

    entered = threading.Event()
    release = threading.Event()

    class _BlockingJob:
        """prune_jobs() only reads .in_terminal_state; block there to suspend it mid-prune."""

        id = -1
        source = "blocking"

        @property
        def in_terminal_state(self) -> bool:
            entered.set()
            assert release.wait(timeout=20)
            return True

    blocking_job = _BlockingJob()
    mm2_installer._install_jobs.append(blocking_job)  # type: ignore[arg-type]
    importer_done = threading.Event()
    imported: list[ModelInstallJob] = []

    def _import() -> None:
        imported.append(mm2_installer.import_model(source))
        importer_done.set()

    pruner = threading.Thread(target=mm2_installer.prune_jobs)
    importer = threading.Thread(target=_import)

    try:
        pruner.start()
        assert entered.wait(timeout=10)

        # prune_jobs() must hold the installer lock across the whole filter-and-reassign, so no registration can
        # land in between. Assert that directly rather than inferring it from the importer's timing.
        lock_was_free = mm2_installer._lock.acquire(timeout=0.5)
        if lock_was_free:
            mm2_installer._lock.release()
        assert not lock_was_free, "prune_jobs() ran its filter without holding the lock"

        importer.start()
        assert not importer_done.wait(timeout=1), "import_model registered a job while prune_jobs was mid-prune"
    finally:
        # Always release, or _BlockingJob survives in _install_jobs and blocks fixture teardown too.
        release.set()
        for thread in (pruner, importer):
            if thread.ident is not None:  # an early assertion may have fired before importer.start()
                thread.join(timeout=10)
        if blocking_job in mm2_installer._install_jobs:
            mm2_installer._install_jobs.remove(blocking_job)  # type: ignore[arg-type]

    assert not pruner.is_alive() and not importer.is_alive()
    assert imported and mm2_installer.get_job_by_source(source) == imported


@pytest.mark.timeout(timeout=10, method="thread")
def test_waiting_import_returns_its_new_terminal_job_when_no_live_job_exists(
    mm2_installer: ModelInstallServiceBase,
    mm2_app_config: InvokeAIAppConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = URLModelSource(url=Url("https://www.test.foo/download/test_embedding.safetensors"))
    source_key = str(source)
    condition = mm2_installer._install_condition
    assert mm2_installer._restore_completed_event.wait(timeout=5)

    with condition:
        mm2_installer._pending_sources.add(source_key)

    importer_waiting = threading.Event()
    condition_wait = condition.wait

    def _observed_condition_wait(timeout: float | None = None) -> bool:
        importer_waiting.set()
        return condition_wait(timeout)

    monkeypatch.setattr(condition, "wait", _observed_condition_wait)
    imported_jobs: list[ModelInstallJob] = []
    importer = threading.Thread(target=lambda: imported_jobs.append(mm2_installer.import_model(source)))
    importer.start()
    assert importer_waiting.wait(timeout=5)

    terminal_job = ModelInstallJob(
        id=99001,
        source=source,
        config_in=ModelRecordChanges(),
        local_path=mm2_app_config.models_path,
    )
    terminal_job.status = InstallStatus.ERROR
    with condition:
        mm2_installer._install_jobs.append(terminal_job)
        mm2_installer._pending_sources.remove(source_key)
        condition.notify_all()

    importer.join(timeout=5)
    assert not importer.is_alive()
    assert imported_jobs == [terminal_job]


def test_prune_jobs_rebind_preserves_existing_iterators(
    mm2_installer: ModelInstallServiceBase, mm2_app_config: InvokeAIAppConfig
) -> None:
    jobs = [
        ModelInstallJob(
            id=99002 + index,
            source=URLModelSource(url=Url(f"https://www.test.foo/download/model-{index}.safetensors")),
            config_in=ModelRecordChanges(),
            local_path=mm2_app_config.models_path,
            status=InstallStatus.COMPLETED,
        )
        for index in range(3)
    ]
    mm2_installer._install_jobs.extend(jobs)
    existing_iterator = iter(mm2_installer.list_jobs())
    assert next(existing_iterator) is jobs[0]

    mm2_installer.prune_jobs()

    assert list(existing_iterator) == jobs[1:]
    assert all(job not in mm2_installer.list_jobs() for job in jobs)


@pytest.mark.timeout(timeout=30, method="thread")
def test_prune_jobs_waits_for_the_installer_lock(mm2_installer: ModelInstallServiceBase) -> None:
    lock_held = threading.Event()
    release_lock = threading.Event()
    prune_completed = threading.Event()

    def _hold_lock() -> None:
        with mm2_installer._lock:
            lock_held.set()
            assert release_lock.wait(timeout=3)

    lock_holder = threading.Thread(target=_hold_lock)
    pruner = threading.Thread(target=lambda: (mm2_installer.prune_jobs(), prune_completed.set()))
    try:
        lock_holder.start()
        assert lock_held.wait(timeout=3)
        pruner.start()
        assert not prune_completed.wait(timeout=0.25)
    finally:
        release_lock.set()
        for thread in (lock_holder, pruner):
            if thread.ident is not None:
                thread.join(timeout=3)

    assert not lock_holder.is_alive() and not pruner.is_alive()
    assert prune_completed.is_set()


@pytest.mark.timeout(timeout=30, method="thread")
def test_import_and_wait_for_installs_fail_before_start(
    mm2_app_config: InvokeAIAppConfig,
    mm2_record_store,
    mm2_download_queue,
    mm2_session,
    embedding_file: Path,
) -> None:
    installer = ModelInstallService(
        app_config=mm2_app_config,
        record_store=mm2_record_store,
        download_queue=mm2_download_queue,
        event_bus=TestEventService(),
        session=mm2_session,
    )

    with pytest.raises(RuntimeError, match="not running"):
        installer.import_model(LocalModelSource(path=embedding_file))
    with pytest.raises(RuntimeError, match="not running"):
        installer.wait_for_installs(timeout=0.1)


@pytest.mark.timeout(timeout=30, method="thread")
def test_import_fails_after_startup_failure(
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

    def _fail_startup() -> None:
        raise RuntimeError("startup failed")

    monkeypatch.setattr(installer, "_migrate_yaml", _fail_startup)

    try:
        with pytest.raises(RuntimeError, match="startup failed"):
            installer.start()

        with pytest.raises(RuntimeError, match="Model install service failed to start"):
            installer.import_model(LocalModelSource(path=embedding_file))
    finally:
        installer.stop()


@pytest.mark.timeout(timeout=30, method="thread")
def test_base_exception_during_startup_releases_import_waiters(
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

    def _interrupt_startup() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(installer, "_migrate_yaml", _interrupt_startup)

    try:
        with pytest.raises(KeyboardInterrupt):
            installer.start()

        assert installer._restore_completed_event.is_set()
        with pytest.raises(RuntimeError, match="Model install service failed to start"):
            installer.import_model(LocalModelSource(path=embedding_file))
    finally:
        installer.stop()


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
