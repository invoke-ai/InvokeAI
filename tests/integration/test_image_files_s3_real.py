"""Real object-store integration tests for ``S3CompatibleImageFileStorage``.

These run the full save -> read -> delete pipeline against an ACTUAL S3-compatible
bucket (AWS S3 or Backblaze B2). They are disabled by default (marked ``slow``)
and skipped unless the ``CI_TEST_S3_*`` env vars are set, so normal/CI test runs
never touch a real bucket.

Run against Backblaze B2::

    CI_TEST_S3_BUCKET=my-test-bucket \
    CI_TEST_S3_ENDPOINT_URL=https://s3.us-west-004.backblazeb2.com \
    CI_TEST_S3_REGION=us-west-004 \
    CI_TEST_S3_ACCESS_KEY_ID=<B2 keyID> \
    CI_TEST_S3_SECRET_ACCESS_KEY=<B2 applicationKey> \
    pytest -m s3_integration tests/integration/test_image_files_s3_real.py -v

Run against AWS S3: omit ``CI_TEST_S3_ENDPOINT_URL`` and set ``CI_TEST_S3_REGION``
to the bucket's region. The credentials are the standard access key / secret.

Each test uses a unique object key and deletes it in a ``finally`` so the bucket
is left clean even if an assertion fails.
"""

import os
import uuid

import pytest
from PIL import Image

from invokeai.app.services.image_files.image_files_common import ImageFileNotFoundException
from invokeai.app.services.image_files.image_files_s3 import S3CompatibleImageFileStorage
from invokeai.app.util.thumbnails import get_thumbnail_name

_BUCKET = os.environ.get("CI_TEST_S3_BUCKET")
_ACCESS = os.environ.get("CI_TEST_S3_ACCESS_KEY_ID")
_SECRET = os.environ.get("CI_TEST_S3_SECRET_ACCESS_KEY")

pytestmark = [
    pytest.mark.s3_integration,
    pytest.mark.slow,
    pytest.mark.skipif(
        not (_BUCKET and _ACCESS and _SECRET),
        reason="Set CI_TEST_S3_BUCKET / CI_TEST_S3_ACCESS_KEY_ID / CI_TEST_S3_SECRET_ACCESS_KEY to run real S3/B2 tests",
    ),
]

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _make_png() -> Image.Image:
    return Image.new("RGB", (16, 16), color=(0, 128, 255))


@pytest.fixture(params=["aws_env", "b2_env"])
def storage(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> S3CompatibleImageFileStorage:
    """Build a storage instance, supplying credentials via the AWS_* or B2_* env path.

    Parametrizing over both proves the backend works whether credentials are
    given as standard AWS vars or via the Backblaze B2 convenience names. The
    B2-credential case is only meaningful against a B2 endpoint, so it is skipped
    in an AWS-only setup (no CI_TEST_S3_ENDPOINT_URL).
    """
    if request.param == "b2_env" and not os.environ.get("CI_TEST_S3_ENDPOINT_URL"):
        pytest.skip("b2_env targets a Backblaze B2 endpoint; set CI_TEST_S3_ENDPOINT_URL to run it")
    for var in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_SECURITY_TOKEN",
        "B2_APPLICATION_KEY_ID",
        "B2_APPLICATION_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    if request.param == "aws_env":
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", _ACCESS or "")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", _SECRET or "")
    else:
        monkeypatch.setenv("B2_APPLICATION_KEY_ID", _ACCESS or "")
        monkeypatch.setenv("B2_APPLICATION_KEY", _SECRET or "")
    return S3CompatibleImageFileStorage(
        bucket=_BUCKET,
        endpoint_url=os.environ.get("CI_TEST_S3_ENDPOINT_URL") or None,
        region_name=os.environ.get("CI_TEST_S3_REGION") or None,
    )


def test_full_pipeline_round_trip(storage: S3CompatibleImageFileStorage) -> None:
    image_name = f"ci-test-{uuid.uuid4().hex}.png"
    image_key = f"images/{image_name}"
    thumb_key = f"thumbnails/{get_thumbnail_name(image_name)}"
    # Non-ASCII workflow exercises the hex-encoding path through real S3 user-metadata;
    # the plain graph exercises the verbatim path.
    workflow = "wörkflöw — 日本語 ✓"
    graph = "GRAPH-DATA"

    try:
        storage.save(_make_png(), image_name, metadata="META", workflow=workflow, graph=graph)

        # get() decodes a PIL image
        assert storage.get(image_name).size == (16, 16)

        # get_bytes(): PNG for the image, WEBP for the thumbnail
        assert storage.get_bytes(image_name).startswith(_PNG_MAGIC)
        thumb_bytes = storage.get_bytes(image_name, thumbnail=True)
        assert thumb_bytes[:4] == b"RIFF" and thumb_bytes[8:12] == b"WEBP"

        # open_stream() yields a readable stream (the caller closes it)
        stream = storage.open_stream(image_name)
        try:
            assert stream.read(8) == _PNG_MAGIC
        finally:
            stream.close()

        # metadata round-trips through real S3 user-metadata (hex for non-ASCII)
        assert storage.get_workflow(image_name) == workflow
        assert storage.get_graph(image_name) == graph

        # validate_path() confirms both objects exist
        assert storage.validate_path(image_key) is True
        assert storage.validate_path(thumb_key) is True

        # delete() removes both the image and its thumbnail
        storage.delete(image_name)
        assert storage.validate_path(image_key) is False
        assert storage.validate_path(thumb_key) is False
        with pytest.raises(ImageFileNotFoundException):
            storage.get_bytes(image_name)
    finally:
        # Best-effort cleanup (idempotent); a delete error here must not mask a
        # failure raised above.
        try:
            storage.delete(image_name)
        except Exception:
            pass
