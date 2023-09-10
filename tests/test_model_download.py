"""Test the queued download facility"""

import tempfile
import time
import requests
from requests_testadapter import TestAdapter
from requests import HTTPError
from pathlib import Path

from invokeai.backend.model_manager.download import (
    DownloadJobStatus,
    DownloadQueue,
    DownloadJobBase,
    UnknownJobIDException,
)

TestAdapter.__test__ = False

INTERNET_AVAILABLE = requests.get("http://www.google.com/").status_code == 200

########################################################################################
# Lots of dummy content here to test model download without using lots of bandwidth
# The repo_id tests are not self-contained because they still need to use the HF API
# to retrieve metainformation about the files to retrieve. However, the big weights files
# are not downloaded.

# If the internet is not available, then the repo_id tests are skipped, but the single
# URL tests are still run.

session = requests.Session()
for i in ["12345", "9999", "54321"]:
    content = (
        b"I am a safetensors file " + bytearray(i, "utf-8") + bytearray(32_000)
    )  # for pause tests, must make content large
    session.mount(
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
session.mount(
    "http://www.civitai.com/models/missing",
    TestAdapter(
        b"Missing content length",
        headers={
            "Content-Disposition": 'filename="missing.txt"',
        },
    ),
)
# not found test
session.mount("http://www.civitai.com/models/broken", TestAdapter(b"Not found", status=404))

# prevent us from going to civitai to get metadata
session.mount("https://civitai.com/api/download/models/", TestAdapter(b"Not found", status=404))
session.mount("https://civitai.com/api/v1/models/", TestAdapter(b"Not found", status=404))
session.mount("https://civitai.com/api/v1/model-versions/", TestAdapter(b"Not found", status=404))

# specifies a content disposition that may overwrite files in the parent directory
session.mount(
    "http://www.civitai.com/models/malicious",
    TestAdapter(
        b"Malicious URL",
        headers={
            "Content-Disposition": 'filename="../badness.txt"',
        },
    ),
)
# Would create a path that is too long
session.mount(
    "http://www.civitai.com/models/long",
    TestAdapter(
        b"Malicious URL",
        headers={
            "Content-Disposition": f'filename="{"i"*1000}"',
        },
    ),
)

# mock HuggingFace URLs
hf_sd2_paths = [
    "feature_extractor/preprocessor_config.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "text_encoder/model.safetensors",
    "text_encoder/pytorch_model.fp16.bin",
    "text_encoder/pytorch_model.bin",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]
for path in hf_sd2_paths:
    url = f"https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/{path}"
    path = Path(path)
    filename = path.name
    content = b"This is the content for path " + bytearray(path.as_posix(), "utf-8")
    session.mount(
        url,
        TestAdapter(
            content,
            status=200,
            headers={"Content-Length": len(content), "Content-Disposition": f'filename="{filename}"'},
        ),
    )

# This is the content of `model_index.json` for stable-diffusion-2-1
model_index_content = b'{"_class_name": "StableDiffusionPipeline", "_diffusers_version": "0.8.0", "feature_extractor": ["transformers", "CLIPImageProcessor"], "requires_safety_checker": false, "safety_checker": [null, null], "scheduler": ["diffusers", "DDIMScheduler"], "text_encoder": ["transformers", "CLIPTextModel"], "tokenizer": ["transformers", "CLIPTokenizer"], "unet": ["diffusers", "UNet2DConditionModel"], "vae": ["diffusers", "AutoencoderKL"]}'

session.mount(
    "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/model_index.json",
    TestAdapter(
        model_index_content,
        status=200,
        headers={"Content-Length": len(model_index_content), "Content-Disposition": 'filename="model_index.json"'},
    ),
)

# ================================================================================================================== #


def test_basic_queue_download():
    events = list()

    def event_handler(job: DownloadJobBase):
        events.append(job.status)

    queue = DownloadQueue(
        requests_session=session,
        event_handlers=[event_handler],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        job = queue.create_download_job(source="http://www.civitai.com/models/12345", destdir=tmpdir, start=False)
        assert isinstance(job, DownloadJobBase), "expected the job to be of type DownloadJobBase"
        assert isinstance(job.id, int), "expected the job id to be numeric"
        assert job.status == "idle", "expected job status to be idle"
        assert job.status == DownloadJobStatus.IDLE

        queue.start_job(job)
        queue.join()
        assert events[0] == DownloadJobStatus.ENQUEUED
        assert events[-1] == DownloadJobStatus.COMPLETED
        assert DownloadJobStatus.RUNNING in events
        assert Path(tmpdir, "mock12345.safetensors").exists(), f"expected {tmpdir}/mock12345.safetensors to exist"


def test_queue_priority():
    queue = DownloadQueue(
        requests_session=session,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        job1 = queue.create_download_job(source="http://www.civitai.com/models/12345", destdir=tmpdir, start=False)
        job2 = queue.create_download_job(source="http://www.civitai.com/models/9999", destdir=tmpdir, start=False)

        queue.change_priority(job1, -10)  # make id1 run first
        assert job1 < job2

        queue.start_all_jobs()
        queue.join()
        assert job1.job_sequence < job2.job_sequence

        job1 = queue.create_download_job(source="http://www.civitai.com/models/12345", destdir=tmpdir, start=False)
        job2 = queue.create_download_job(source="http://www.civitai.com/models/9999", destdir=tmpdir, start=False)

        queue.change_priority(job2, -10)  # make id2 run first
        assert job2 < job1

        queue.start_all_jobs()
        queue.join()
        assert job2.job_sequence < job1.job_sequence

        assert Path(tmpdir, "mock12345.safetensors").exists(), f"expected {tmpdir}/mock12345.safetensors to exist"
        assert Path(tmpdir, "mock9999.safetensors").exists(), f"expected {tmpdir}/mock9999.safetensors to exist"


def test_repo_id_download():
    if not INTERNET_AVAILABLE:
        return
    repo_id = "stabilityai/stable-diffusion-2-1"
    queue = DownloadQueue(
        requests_session=session,
    )

    # first with fp16 variant
    with tempfile.TemporaryDirectory() as tmpdir:
        queue.create_download_job(source=repo_id, destdir=tmpdir, variant="fp16", start=True)
        queue.join()
        repo_root = Path(tmpdir, "stable-diffusion-2-1")
        assert repo_root.exists()
        assert Path(repo_root, "model_index.json").exists()
        assert Path(repo_root, "text_encoder", "config.json").exists()
        assert Path(repo_root, "text_encoder", "model.fp16.safetensors").exists()

    # then without fp16
    with tempfile.TemporaryDirectory() as tmpdir:
        queue.create_download_job(source=repo_id, destdir=tmpdir, start=True)
        queue.join()
        repo_root = Path(tmpdir, "stable-diffusion-2-1")
        assert Path(repo_root, "text_encoder", "model.safetensors").exists()
        assert not Path(repo_root, "text_encoder", "model.fp16.safetensors").exists()


def test_bad_urls():
    queue = DownloadQueue(
        requests_session=session,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # do we handle 404 and other HTTP errors?
        job = queue.create_download_job(source="http://www.civitai.com/models/broken", destdir=tmpdir)
        queue.join()
        assert job.status == "error"
        assert isinstance(job.error, HTTPError)
        assert str(job.error) == "NOT FOUND"

        # Do we handle missing content length field?
        job = queue.create_download_job(source="http://www.civitai.com/models/missing", destdir=tmpdir)
        queue.join()
        assert job.status == "completed"
        assert job.total_bytes == 0
        assert job.bytes > 0
        assert job.bytes == Path(tmpdir, "missing.txt").stat().st_size

        # Don't let the URL specify a filename with slashes or double dots... (e.g. '../../etc/passwd')
        job = queue.create_download_job(source="http://www.civitai.com/models/malicious", destdir=tmpdir)
        queue.join()
        assert job.status == "completed"
        assert job.destination == Path(tmpdir, "malicious")
        assert Path(tmpdir, "malicious").exists()

        # Nor a destination that would exceed the maximum filename or path length
        job = queue.create_download_job(source="http://www.civitai.com/models/long", destdir=tmpdir)
        queue.join()
        assert job.status == "completed"
        assert job.destination == Path(tmpdir, "long")
        assert Path(tmpdir, "long").exists()

    # create a foreign job which will be invalid for the queue
    bad_job = DownloadJobBase(id=999, source="mock", destination="mock")
    try:
        queue.start_job(bad_job)  # this should fail
        succeeded = True
    except UnknownJobIDException:
        succeeded = False
    assert not succeeded


def test_pause_cancel_url():  # this one is tricky because of potential race conditions
    def event_handler(job: DownloadJobBase):
        time.sleep(0.5)  # slow down the thread by blocking it just a bit at every step

    queue = DownloadQueue(requests_session=session, event_handlers=[event_handler])

    with tempfile.TemporaryDirectory() as tmpdir:
        job1 = queue.create_download_job(source="http://www.civitai.com/models/12345", destdir=tmpdir, start=False)
        job2 = queue.create_download_job(source="http://www.civitai.com/models/9999", destdir=tmpdir, start=False)
        job3 = queue.create_download_job(source="http://www.civitai.com/models/54321", destdir=tmpdir, start=False)

        assert job1.status == "idle"
        queue.start_job(job1)
        queue.start_job(job3)
        time.sleep(0.1)  # wait for enqueueing
        assert job1.status in ["enqueued", "running"]

        # check pause and restart
        queue.pause_job(job1)
        time.sleep(0.1)  # wait to be paused
        assert job1.status == "paused"

        queue.start_job(job1)
        time.sleep(0.1)
        assert job1.status == "running"

        # check cancel
        queue.start_job(job2)
        time.sleep(0.1)
        assert job2.status == "running"
        queue.cancel_job(job2)
        time.sleep(0.1)
        assert job2.status == "cancelled"

        queue.join()
        assert job1.status == "completed"
        assert job2.status == "cancelled"
        assert job3.status == "completed"

        assert Path(tmpdir, "mock12345.safetensors").exists()
        assert Path(tmpdir, "mock9999.safetensors").exists() is False, "cancelled file should be deleted"
        assert Path(tmpdir, "mock54321.safetensors").exists()

        assert len(queue.list_jobs()) == 0

    def test_pause_cancel_repo_id():  # this one is tricky because of potential race conditions
        def event_handler(job: DownloadJobBase):
            time.sleep(0.5)  # slow down the thread by blocking it just a bit at every step

        if not INTERNET_AVAILABLE:
            return

        repo_id = "stabilityai/stable-diffusion-2-1"
        queue = DownloadQueue(requests_session=session, event_handlers=[event_handler])

        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            job1 = queue.create_download_job(source=repo_id, destdir=tmpdir1, variant="fp16", start=False)
            job2 = queue.create_download_job(source=repo_id, destdir=tmpdir2, variant="fp16", start=False)
            assert job1.status == "idle"
            queue.start_job(job1)
            time.sleep(0.1)  # wait for enqueueing
            assert job1.status in ["enqueued", "running"]

            # check pause and restart
            queue.pause_job(job1)
            time.sleep(0.1)  # wait to be paused
            assert job1.status == "paused"

            queue.start_job(job1)
            time.sleep(0.1)
            assert job1.status == "running"

            # check cancel
            queue.start_job(job2)
            time.sleep(0.1)
            assert job2.status == "running"
            queue.cancel_job(job2)
            time.sleep(0.1)
            assert job2.status == "cancelled"

            queue.join()
            assert job1.status == "completed"
            assert job2.status == "cancelled"

            assert Path(tmpdir1, "stable-diffusion-2-1", "model_index.json").exists()
            assert not Path(
                tmpdir2, "stable-diffusion-2-1", "model_index.json"
            ).exists(), "cancelled file should be deleted"

            assert len(queue.list_jobs()) == 0
