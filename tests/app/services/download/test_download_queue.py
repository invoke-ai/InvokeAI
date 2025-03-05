"""Test the queued download facility"""

import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, Optional

import pytest
from pydantic.networks import AnyHttpUrl
from requests.sessions import Session
from requests_testadapter import TestAdapter

from invokeai.app.services.config import get_config
from invokeai.app.services.config.config_default import URLRegexTokenPair
from invokeai.app.services.download import DownloadJob, DownloadJobStatus, DownloadQueueService, MultiFileDownloadJob
from invokeai.app.services.events.events_common import (
    DownloadCancelledEvent,
    DownloadCompleteEvent,
    DownloadErrorEvent,
    DownloadProgressEvent,
    DownloadStartedEvent,
)
from invokeai.backend.model_manager.metadata import HuggingFaceMetadataFetch, ModelMetadataWithFiles, RemoteModelFile
from tests.backend.model_manager.model_manager_fixtures import *  # noqa F403
from tests.test_nodes import TestEventService

# Prevent pytest deprecation warnings
TestAdapter.__test__ = False


@pytest.mark.timeout(timeout=10, method="thread")
def test_basic_queue_download(tmp_path: Path, mm2_session: Session) -> None:
    events = set()

    def event_handler(job: DownloadJob, excp: Optional[Exception] = None) -> None:
        events.add(job.status)

    queue = DownloadQueueService(
        requests_session=mm2_session,
    )
    queue.start()
    job = queue.download(
        source=AnyHttpUrl("http://www.civitai.com/models/12345"),
        dest=tmp_path,
        on_start=event_handler,
        on_progress=event_handler,
        on_complete=event_handler,
        on_error=event_handler,
    )
    assert isinstance(job, DownloadJob), "expected the job to be of type DownloadJobBase"
    assert isinstance(job.id, int), "expected the job id to be numeric"
    queue.join()

    assert job.status == DownloadJobStatus("completed"), "expected job status to be completed"
    assert job.download_path == tmp_path / "mock12345.safetensors"
    assert Path(tmp_path, "mock12345.safetensors").exists(), f"expected {tmp_path}/mock12345.safetensors to exist"

    assert events == {DownloadJobStatus.RUNNING, DownloadJobStatus.COMPLETED}
    queue.stop()


@pytest.mark.timeout(timeout=10, method="thread")
def test_errors(tmp_path: Path, mm2_session: Session) -> None:
    queue = DownloadQueueService(
        requests_session=mm2_session,
    )
    queue.start()

    for bad_url in ["http://www.civitai.com/models/broken", "http://www.civitai.com/models/missing"]:
        queue.download(AnyHttpUrl(bad_url), dest=tmp_path)

    queue.join()
    jobs = queue.list_jobs()
    print(jobs)
    assert len(jobs) == 2
    jobs_dict = {str(x.source): x for x in jobs}
    assert jobs_dict["http://www.civitai.com/models/broken"].status == DownloadJobStatus.ERROR
    assert jobs_dict["http://www.civitai.com/models/broken"].error_type == "HTTPError(NOT FOUND)"
    assert jobs_dict["http://www.civitai.com/models/missing"].status == DownloadJobStatus.COMPLETED
    assert jobs_dict["http://www.civitai.com/models/missing"].total_bytes == 0
    queue.stop()


@pytest.mark.timeout(timeout=10, method="thread")
def test_event_bus(tmp_path: Path, mm2_session: Session) -> None:
    event_bus = TestEventService()

    queue = DownloadQueueService(requests_session=mm2_session, event_bus=event_bus)
    queue.start()
    queue.download(
        source=AnyHttpUrl("http://www.civitai.com/models/12345"),
        dest=tmp_path,
    )
    queue.join()
    events = event_bus.events
    assert len(events) == 3
    assert isinstance(events[0], DownloadStartedEvent)
    assert isinstance(events[1], DownloadProgressEvent)
    assert isinstance(events[2], DownloadCompleteEvent)
    assert events[0].timestamp <= events[1].timestamp
    assert events[1].timestamp <= events[2].timestamp
    assert events[1].total_bytes > 0
    assert events[1].current_bytes <= events[1].total_bytes
    assert events[2].total_bytes == 32029

    # test a failure
    event_bus.events = []  # reset our accumulator
    queue.download(source=AnyHttpUrl("http://www.civitai.com/models/broken"), dest=tmp_path)
    queue.join()
    events = event_bus.events
    print("\n".join([x.model_dump_json() for x in events]))
    assert len(events) == 1
    assert isinstance(events[0], DownloadErrorEvent)
    assert events[0].error_type == "HTTPError(NOT FOUND)"
    assert events[0].error is not None
    assert re.search(r"requests.exceptions.HTTPError: NOT FOUND", events[0].error)
    queue.stop()


@pytest.mark.timeout(timeout=10, method="thread")
def test_broken_callbacks(tmp_path: Path, mm2_session: Session, capsys) -> None:
    queue = DownloadQueueService(
        requests_session=mm2_session,
    )
    queue.start()

    callback_ran = False

    def broken_callback(job: DownloadJob) -> None:
        nonlocal callback_ran
        callback_ran = True
        print(1 / 0)  # deliberate error here

    job = queue.download(
        source=AnyHttpUrl("http://www.civitai.com/models/12345"),
        dest=tmp_path,
        on_progress=broken_callback,
    )

    queue.join()
    assert job.status == DownloadJobStatus.COMPLETED  # should complete even though the callback is borked
    assert Path(tmp_path, "mock12345.safetensors").exists()
    assert callback_ran
    # LS: The pytest capsys fixture does not seem to be working. I can see the
    # correct stderr message in the pytest log, but it is not appearing in
    # capsys.readouterr().
    # captured = capsys.readouterr()
    # assert re.search("division by zero", captured.err)
    queue.stop()


@pytest.mark.timeout(timeout=10, method="thread")
def test_cancel(tmp_path: Path, mm2_session: Session) -> None:
    event_bus = TestEventService()

    queue = DownloadQueueService(requests_session=mm2_session, event_bus=event_bus)
    queue.start()

    cancelled = False

    def slow_callback(job: DownloadJob) -> None:
        time.sleep(2)

    def cancelled_callback(job: DownloadJob) -> None:
        nonlocal cancelled
        cancelled = True

    job = queue.download(
        source=AnyHttpUrl("http://www.civitai.com/models/12345"),
        dest=tmp_path,
        on_start=slow_callback,
        on_cancelled=cancelled_callback,
    )
    queue.cancel_job(job)
    queue.join()

    assert job.status == DownloadJobStatus.CANCELLED
    assert cancelled
    events = event_bus.events
    assert isinstance(events[-1], DownloadCancelledEvent)
    assert events[-1].source == "http://www.civitai.com/models/12345"
    queue.stop()


@pytest.mark.timeout(timeout=10, method="thread")
def test_multifile_download(tmp_path: Path, mm2_session: Session) -> None:
    fetcher = HuggingFaceMetadataFetch(mm2_session)
    metadata = fetcher.from_id("stabilityai/sdxl-turbo")
    assert isinstance(metadata, ModelMetadataWithFiles)
    events = set()

    def event_handler(job: DownloadJob | MultiFileDownloadJob, excp: Optional[Exception] = None) -> None:
        events.add(job.status)

    queue = DownloadQueueService(
        requests_session=mm2_session,
    )
    queue.start()
    job = queue.multifile_download(
        parts=metadata.download_urls(session=mm2_session),
        dest=tmp_path,
        on_start=event_handler,
        on_progress=event_handler,
        on_complete=event_handler,
        on_error=event_handler,
    )
    assert isinstance(job, MultiFileDownloadJob), "expected the job to be of type MultiFileDownloadJobBase"
    queue.join()

    assert job.status == DownloadJobStatus("completed"), "expected job status to be completed"
    assert job.bytes > 0, "expected download bytes to be positive"
    assert job.bytes == job.total_bytes, "expected download bytes to equal total bytes"
    assert job.download_path == tmp_path / "sdxl-turbo"
    assert Path(
        tmp_path, "sdxl-turbo/model_index.json"
    ).exists(), f"expected {tmp_path}/sdxl-turbo/model_inded.json to exist"
    assert Path(
        tmp_path, "sdxl-turbo/text_encoder/config.json"
    ).exists(), f"expected {tmp_path}/sdxl-turbo/text_encoder/config.json to exist"

    assert events == {DownloadJobStatus.RUNNING, DownloadJobStatus.COMPLETED}
    queue.stop()


@pytest.mark.timeout(timeout=10, method="thread")
def test_multifile_download_error(tmp_path: Path, mm2_session: Session) -> None:
    fetcher = HuggingFaceMetadataFetch(mm2_session)
    metadata = fetcher.from_id("stabilityai/sdxl-turbo")
    assert isinstance(metadata, ModelMetadataWithFiles)
    events = set()

    def event_handler(job: DownloadJob | MultiFileDownloadJob, excp: Optional[Exception] = None) -> None:
        events.add(job.status)

    queue = DownloadQueueService(
        requests_session=mm2_session,
    )
    queue.start()
    files = metadata.download_urls(session=mm2_session)
    # this will give a 404 error
    files.append(RemoteModelFile(url="https://test.com/missing_model.safetensors", path=Path("sdxl-turbo/broken")))
    job = queue.multifile_download(
        parts=files,
        dest=tmp_path,
        on_start=event_handler,
        on_progress=event_handler,
        on_complete=event_handler,
        on_error=event_handler,
    )
    queue.join()

    assert job.status == DownloadJobStatus("error"), "expected job status to be errored"
    assert job.error_type is not None
    assert "HTTPError(NOT FOUND)" in job.error_type
    assert DownloadJobStatus.ERROR in events
    queue.stop()


@pytest.mark.timeout(timeout=10, method="thread")
def test_multifile_cancel(tmp_path: Path, mm2_session: Session, monkeypatch: Any) -> None:
    event_bus = TestEventService()

    queue = DownloadQueueService(requests_session=mm2_session, event_bus=event_bus)
    queue.start()

    cancelled = False

    def cancelled_callback(job: DownloadJob) -> None:
        nonlocal cancelled
        cancelled = True

    fetcher = HuggingFaceMetadataFetch(mm2_session)
    metadata = fetcher.from_id("stabilityai/sdxl-turbo")
    assert isinstance(metadata, ModelMetadataWithFiles)

    job = queue.multifile_download(
        parts=metadata.download_urls(session=mm2_session),
        dest=tmp_path,
        on_cancelled=cancelled_callback,
    )
    queue.cancel_job(job)
    queue.join()

    assert job.status == DownloadJobStatus.CANCELLED
    assert cancelled
    events = event_bus.events
    assert DownloadCancelledEvent in [type(x) for x in events]
    queue.stop()


def test_multifile_onefile(tmp_path: Path, mm2_session: Session) -> None:
    queue = DownloadQueueService(
        requests_session=mm2_session,
    )
    queue.start()
    job = queue.multifile_download(
        parts=[
            RemoteModelFile(url=AnyHttpUrl("http://www.civitai.com/models/12345"), path=Path("mock12345.safetensors"))
        ],
        dest=tmp_path,
    )
    assert isinstance(job, MultiFileDownloadJob), "expected the job to be of type MultiFileDownloadJobBase"
    queue.join()

    assert job.status == DownloadJobStatus("completed"), "expected job status to be completed"
    assert job.bytes > 0, "expected download bytes to be positive"
    assert job.bytes == job.total_bytes, "expected download bytes to equal total bytes"
    assert job.download_path == tmp_path / "mock12345.safetensors"
    assert Path(tmp_path, "mock12345.safetensors").exists(), f"expected {tmp_path}/mock12345.safetensors to exist"
    queue.stop()


def test_multifile_no_rel_paths(tmp_path: Path, mm2_session: Session) -> None:
    queue = DownloadQueueService(
        requests_session=mm2_session,
    )

    with pytest.raises(AssertionError) as error:
        queue.multifile_download(
            parts=[RemoteModelFile(url=AnyHttpUrl("http://www.civitai.com/models/12345"), path=Path("/etc/passwd"))],
            dest=tmp_path,
        )
    assert str(error.value) == "only relative download paths accepted"


@contextmanager
def clear_config() -> Generator[None, None, None]:
    try:
        yield None
    finally:
        get_config.cache_clear()


def test_tokens(tmp_path: Path, mm2_session: Session):
    with clear_config():
        config = get_config()
        config.remote_api_tokens = [URLRegexTokenPair(url_regex="civitai", token="cv_12345")]
        queue = DownloadQueueService(requests_session=mm2_session)
        queue.start()
        # this one has an access token assigned
        job1 = queue.download(
            source=AnyHttpUrl("http://www.civitai.com/models/12345"),
            dest=tmp_path,
        )
        # this one doesn't
        job2 = queue.download(
            source=AnyHttpUrl(
                "http://www.huggingface.co/foo.txt",
            ),
            dest=tmp_path,
        )
        queue.join()
        # this token is defined in the temporary root invokeai.yaml
        # see tests/backend/model_manager/data/invokeai_root/invokeai.yaml
        assert job1.access_token == "cv_12345"
        assert job2.access_token is None
        queue.stop()
