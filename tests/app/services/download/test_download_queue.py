"""Test the queued download facility"""

import re
import time
from pathlib import Path

import pytest
from pydantic.networks import AnyHttpUrl
from requests.sessions import Session
from requests_testadapter import TestAdapter, TestSession

from invokeai.app.services.download import DownloadJob, DownloadJobStatus, DownloadQueueService
from tests.test_nodes import TestEventService

# Prevent pytest deprecation warnings
TestAdapter.__test__ = False  # type: ignore


@pytest.fixture
def session() -> Session:
    sess = TestSession()
    for i in ["12345", "9999", "54321"]:
        content = (
            b"I am a safetensors file " + bytearray(i, "utf-8") + bytearray(32_000)
        )  # for pause tests, must make content large
        sess.mount(
            f"http://www.civitai.com/models/{i}",
            TestAdapter(
                content,
                headers={
                    "Content-Length": len(content),
                    "Content-Disposition": f'filename="mock{i}.safetensors"',
                },
            ),
        )

    # here are some malformed URLs to test
    # missing the content length
    sess.mount(
        "http://www.civitai.com/models/missing",
        TestAdapter(
            b"Missing content length",
            headers={
                "Content-Disposition": 'filename="missing.txt"',
            },
        ),
    )
    # not found test
    sess.mount("http://www.civitai.com/models/broken", TestAdapter(b"Not found", status=404))

    return sess


def test_basic_queue_download(tmp_path: Path, session: Session) -> None:
    events = set()

    def event_handler(job: DownloadJob) -> None:
        events.add(job.status)

    queue = DownloadQueueService(
        requests_session=session,
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
    assert Path(tmp_path, "mock12345.safetensors").exists(), f"expected {tmp_path}/mock12345.safetensors to exist"

    assert events == {DownloadJobStatus.RUNNING, DownloadJobStatus.COMPLETED}
    queue.stop()


def test_errors(tmp_path: Path, session: Session) -> None:
    queue = DownloadQueueService(
        requests_session=session,
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


def test_event_bus(tmp_path: Path, session: Session) -> None:
    event_bus = TestEventService()

    queue = DownloadQueueService(requests_session=session, event_bus=event_bus)
    queue.start()
    queue.download(
        source=AnyHttpUrl("http://www.civitai.com/models/12345"),
        dest=tmp_path,
    )
    queue.join()
    events = event_bus.events
    assert len(events) == 3
    assert events[0].payload["timestamp"] <= events[1].payload["timestamp"]
    assert events[1].payload["timestamp"] <= events[2].payload["timestamp"]
    assert events[0].event_name == "download_started"
    assert events[1].event_name == "download_progress"
    assert events[1].payload["total_bytes"] > 0
    assert events[1].payload["current_bytes"] <= events[1].payload["total_bytes"]
    assert events[2].event_name == "download_complete"
    assert events[2].payload["total_bytes"] == 32029

    # test a failure
    event_bus.events = []  # reset our accumulator
    queue.download(source=AnyHttpUrl("http://www.civitai.com/models/broken"), dest=tmp_path)
    queue.join()
    events = event_bus.events
    print("\n".join([x.model_dump_json() for x in events]))
    assert len(events) == 1
    assert events[0].event_name == "download_error"
    assert events[0].payload["error_type"] == "HTTPError(NOT FOUND)"
    assert events[0].payload["error"] is not None
    assert re.search(r"requests.exceptions.HTTPError: NOT FOUND", events[0].payload["error"])
    queue.stop()


def test_broken_callbacks(tmp_path: Path, session: Session, capsys) -> None:
    queue = DownloadQueueService(
        requests_session=session,
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


def test_cancel(tmp_path: Path, session: Session) -> None:
    event_bus = TestEventService()

    queue = DownloadQueueService(requests_session=session, event_bus=event_bus)
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
    assert events[-1].event_name == "download_cancelled"
    assert events[-1].payload["source"] == "http://www.civitai.com/models/12345"
    queue.stop()
