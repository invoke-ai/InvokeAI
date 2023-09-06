"""
Test the queued download facility
"""

import pytest
import tempfile
import requests
from requests_testadapter import TestAdapter
from pathlib import Path

TestAdapter.__test__ = False

from invokeai.backend.model_manager.download import (
    DownloadJobStatus,
    DownloadQueue,
    DownloadJobBase,
)


SAFETENSORS1_CONTENT = b'I am a safetensors file (1)'
SAFETENSORS1_HEADER = {
    'Content-Length' : len(SAFETENSORS1_CONTENT),
    'Content-Disposition': 'filename="mock1.safetensors"'
}
SAFETENSORS2_CONTENT = b'I am a safetensors file (2)'
SAFETENSORS2_HEADER = {
    'Content-Length' : len(SAFETENSORS2_CONTENT),
    'Content-Disposition': 'filename="mock2.safetensors"'
}
session = requests.Session()
session.mount('http://www.civitai.com/models/12345',
              TestAdapter(SAFETENSORS1_CONTENT,
                          status=200,
                          headers=SAFETENSORS1_HEADER)
              )
session.mount('http://www.civitai.com/models/9999',
              TestAdapter(SAFETENSORS2_CONTENT,
                          status=200,
                          headers=SAFETENSORS2_HEADER)
              )

session.mount('http://www.civitai.com/models/broken',
              TestAdapter(b'Not found',
                          status=404)
              )

def test_basic_queue():

    events = list()

    def event_handler(job: DownloadJobBase):
        events.append(job.status)

    queue = DownloadQueue(
        requests_session=session,
        event_handlers=[event_handler],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        id1 = queue.create_download_job(
            source='http://www.civitai.com/models/12345',
            destdir=tmpdir,
            start=False
        )
        assert type(id1) == int, "expected first job id to be numeric"

        job = queue.id_to_job(id1)
        assert isinstance(job, DownloadJobBase), "expected job to be a DownloadJobBase"
        assert job.status == 'idle', "expected job status to be idle"
        assert job.status == DownloadJobStatus.IDLE

        queue.start_job(id1)
        queue.join()
        assert events[0] == DownloadJobStatus.ENQUEUED
        assert events[-1] == DownloadJobStatus.COMPLETED
        assert DownloadJobStatus.RUNNING in events
        assert Path(tmpdir, 'mock1.safetensors').exists(), f"expected {tmpdir}/mock1.safetensors to exist"

def test_queue_priority():
    queue = DownloadQueue(
        requests_session=session,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        id1 = queue.create_download_job(
            source='http://www.civitai.com/models/12345',
            destdir=tmpdir,
            start=False
        )

        id2 = queue.create_download_job(
            source='http://www.civitai.com/models/9999',
            destdir=tmpdir,
            start=False
        )

        queue.change_priority(id1, -10)  #  make id1 run first
        job1 = queue.id_to_job(id1)
        job2 = queue.id_to_job(id2)
        assert job1 < job2

        id_list = list()
        queue.start_all_jobs()
        queue.join()
        assert job1.job_sequence < job2.job_sequence

        id1 = queue.create_download_job(
            source='http://www.civitai.com/models/12345',
            destdir=tmpdir,
            start=False
        )

        id2 = queue.create_download_job(
            source='http://www.civitai.com/models/9999',
            destdir=tmpdir,
            start=False
        )

        queue.change_priority(id2, -10)  #  make id2 run first
        job1 = queue.id_to_job(id1)
        job2 = queue.id_to_job(id2)
        assert job2 < job1

        queue.start_all_jobs()
        queue.join()
        assert job2.job_sequence < job1.job_sequence

        assert Path(tmpdir, 'mock1.safetensors').exists(), f"expected {tmpdir}/mock1.safetensors to exist"
        assert Path(tmpdir, 'mock2.safetensors').exists(), f"expected {tmpdir}/mock1.safetensors to exist"
