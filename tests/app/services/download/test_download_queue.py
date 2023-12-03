"""Test the queued download facility"""
from pathlib import Path

import pytest
import requests
from requests_testadapter import TestAdapter

from invokeai.app.services.download import DownloadJob, DownloadJobStatus, DownloadQueueService

# Prevent pytest deprecation warnings
TestAdapter.__test__ = False


@pytest.fixture
def session() -> requests.sessions.Session:
    sess = requests.Session()
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


def test_basic_queue_download(datadir, session):
    events = set()

    def event_handler(job: DownloadJob):
        print(job, "\n")
        events.add(job.status)

    queue = DownloadQueueService(
        requests_session=session,
    )
    job = queue.download(
        source="http://www.civitai.com/models/12345",
        dest=datadir,
        on_start=event_handler,
        on_progress=event_handler,
        on_complete=event_handler,
        on_error=event_handler,
    )
    assert isinstance(job, DownloadJob), "expected the job to be of type DownloadJobBase"
    assert isinstance(job.id, int), "expected the job id to be numeric"
    queue.join()

    assert job.status == DownloadJobStatus("completed"), "expected job status to be completed"
    assert Path(datadir, "mock12345.safetensors").exists(), f"expected {datadir}/mock12345.safetensors to exist"

    assert events == {DownloadJobStatus.RUNNING, DownloadJobStatus.COMPLETED}


def test_errors(datadir, session):
    queue = DownloadQueueService(
        requests_session=session,
    )

    for bad_url in ["http://www.civitai.com/models/broken", "http://www.civitai.com/models/missing"]:
        queue.download(bad_url, dest=datadir)

    queue.join()
    jobs = queue.list_jobs()
    print(jobs)
    assert len(jobs) == 2
    jobs_dict = {str(x.source): x for x in jobs}
    assert jobs_dict["http://www.civitai.com/models/broken"].status == DownloadJobStatus.ERROR
    assert jobs_dict["http://www.civitai.com/models/broken"].error_type == "HTTPError"
    assert jobs_dict["http://www.civitai.com/models/missing"].status == DownloadJobStatus.COMPLETED
    assert jobs_dict["http://www.civitai.com/models/missing"].total_bytes == 0
