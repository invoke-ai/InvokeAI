from __future__ import annotations

import io
from typing import Any
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError
from PIL import Image

from invokeai.app.services.image_files.image_files_common import (
    ImageFileNotFoundException,
    ImageFileSaveException,
)
from invokeai.app.services.image_files.image_files_s3 import S3CompatibleImageFileStorage
from invokeai.app.util.thumbnails import get_thumbnail_name


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "TestOperation")


class _FakeS3Client:
    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}
        self.bodies: list[_BodyStream] = []

    def put_object(self, Bucket: str, Key: str, Body: Any, **kwargs: Any) -> dict:
        # Mirror boto3: a file-like Body is read to bytes for storage/upload.
        data = Body.read() if hasattr(Body, "read") else Body
        self.store[Key] = {
            "Body": data,
            "Metadata": kwargs.get("Metadata", {}) or {},
            "ContentType": kwargs.get("ContentType"),
        }
        return {}

    def get_object(self, Bucket: str, Key: str) -> dict:
        if Key not in self.store:
            raise _client_error("NoSuchKey")
        record = self.store[Key]
        body = _BodyStream(record["Body"])
        self.bodies.append(body)
        return {"Body": body, "Metadata": record["Metadata"]}

    def head_object(self, Bucket: str, Key: str) -> dict:
        if Key not in self.store:
            raise _client_error("404")
        return {"Metadata": self.store[Key]["Metadata"]}

    def delete_object(self, Bucket: str, Key: str) -> dict:
        self.store.pop(Key, None)
        return {}


class _BodyStream:
    def __init__(self, data: bytes) -> None:
        self._buffer = io.BytesIO(data)
        self.closed = False

    def read(self, amt: int | None = None) -> bytes:
        # Mirror boto3 StreamingBody.read(amt=None): full read or a sized chunk.
        return self._buffer.read() if amt is None else self._buffer.read(amt)

    def close(self) -> None:
        self.closed = True
        self._buffer.close()


def _make_storage() -> tuple[S3CompatibleImageFileStorage, _FakeS3Client]:
    fake = _FakeS3Client()
    storage = S3CompatibleImageFileStorage(bucket="test-bucket", client=fake)
    return storage, fake


def _solid_png() -> Image.Image:
    return Image.new("RGB", (8, 8), color=(255, 0, 0))


def test_save_then_get_round_trips_image_bytes() -> None:
    storage, fake = _make_storage()
    storage.save(_solid_png(), "round-trip.png")

    assert "images/round-trip.png" in fake.store
    assert f"thumbnails/{get_thumbnail_name('round-trip.png')}" in fake.store

    fetched = storage.get("round-trip.png")
    assert isinstance(fetched, Image.Image)
    assert fetched.size == (8, 8)
    assert fetched.getpixel((0, 0))[:3] == (255, 0, 0)


def test_get_missing_raises_image_file_not_found() -> None:
    storage, _ = _make_storage()
    with pytest.raises(ImageFileNotFoundException):
        storage.get("does-not-exist.png")


def test_delete_removes_object_and_thumbnail() -> None:
    storage, fake = _make_storage()
    storage.save(_solid_png(), "to-delete.png")
    assert "images/to-delete.png" in fake.store

    storage.delete("to-delete.png")
    assert "images/to-delete.png" not in fake.store
    assert f"thumbnails/{get_thumbnail_name('to-delete.png')}" not in fake.store


def test_validate_path_for_present_and_missing_keys() -> None:
    storage, _ = _make_storage()
    storage.save(_solid_png(), "exists.png")

    assert storage.validate_path("images/exists.png") is True
    assert storage.validate_path("s3://test-bucket/images/exists.png") is True
    assert storage.validate_path("images/missing.png") is False


def test_validate_path_accepts_get_path_round_trip() -> None:
    # recall_parameters.py validates ``str(get_path(name))`` before loading.
    # pathlib collapses ``s3://bucket/k`` to ``s3:/bucket/k``, which previously
    # failed validation and broke that route for the S3 backend (per review).
    storage, _ = _make_storage()
    storage.save(_solid_png(), "rt.png")
    assert storage.validate_path(str(storage.get_path("rt.png"))) is True


def test_validate_path_accepts_windows_backslash_form() -> None:
    # On Windows, ``str(Path("s3://bucket/k"))`` yields ``s3:\\bucket\\k``;
    # validate_path must normalize separators before stripping the prefix
    # (per review), otherwise an existing object fails validation there.
    storage, _ = _make_storage()
    storage.save(_solid_png(), "win.png")
    assert storage.validate_path("s3:\\test-bucket\\images\\win.png") is True


def test_validate_path_rejects_unmanaged_key_without_oracle() -> None:
    # A key outside images/ or thumbnails/ must NOT validate, even if the object
    # exists -- validate_path must not become an existence oracle (per review).
    storage, fake = _make_storage()
    fake.store["secret/key"] = {"Body": b"x", "Metadata": {}, "ContentType": None}
    assert storage.validate_path("secret/key") is False
    assert storage.validate_path("s3://test-bucket/secret/key") is False


def test_save_rolls_back_image_when_thumbnail_upload_fails() -> None:
    # If the thumbnail upload fails, save() must not leave the image orphaned:
    # it best-effort deletes the already-uploaded image, then raises (per review).
    class _FailingThumbClient(_FakeS3Client):
        def put_object(self, Bucket: str, Key: str, Body: Any, **kwargs: Any) -> dict:
            if Key.startswith("thumbnails/"):
                raise RuntimeError("thumbnail upload failed")
            return super().put_object(Bucket, Key, Body, **kwargs)

    fake = _FailingThumbClient()
    storage = S3CompatibleImageFileStorage(bucket="test-bucket", client=fake)
    with pytest.raises(ImageFileSaveException):
        storage.save(_solid_png(), "rollback.png")
    assert "images/rollback.png" not in fake.store  # rolled back


def test_body_stream_supports_chunked_read() -> None:
    # The fake StreamingBody must accept read(amt) like boto3, so streaming
    # consumers (e.g. shutil.copyfileobj) are exercised faithfully.
    storage, fake = _make_storage()
    storage.save(_solid_png(), "chunk.png")
    body = fake.get_object(Bucket="test-bucket", Key="images/chunk.png")["Body"]
    first = body.read(4)
    rest = body.read()
    assert len(first) == 4
    assert first + rest == fake.store["images/chunk.png"]["Body"]


def test_get_path_returns_synthetic_s3_uri() -> None:
    storage, _ = _make_storage()
    p = str(storage.get_path("foo.png", thumbnail=True))
    assert "test-bucket" in p and "thumbnails/foo.webp" in p


def test_get_bytes_returns_raw_image_payload() -> None:
    """`get_bytes` is what the image-serving routes call, so this is the
    end-to-end shape they depend on (per Copilot review feedback)."""
    storage, fake = _make_storage()
    storage.save(_solid_png(), "raw.png")

    image_bytes = storage.get_bytes("raw.png", thumbnail=False)
    thumb_bytes = storage.get_bytes("raw.png", thumbnail=True)

    assert image_bytes == fake.store["images/raw.png"]["Body"]
    assert thumb_bytes == fake.store[f"thumbnails/{get_thumbnail_name('raw.png')}"]["Body"]


def test_get_bytes_missing_raises_image_file_not_found() -> None:
    storage, _ = _make_storage()
    with pytest.raises(ImageFileNotFoundException):
        storage.get_bytes("missing.png")


def test_get_bytes_closes_response_body() -> None:
    """Real boto3 returns a StreamingBody that must be closed after reading to
    release the HTTP connection back to the pool (per Copilot review)."""
    storage, fake = _make_storage()
    storage.save(_solid_png(), "close.png")
    storage.get_bytes("close.png")
    assert fake.bodies and all(b.closed for b in fake.bodies)


def test_open_stream_returns_readable_body() -> None:
    """`open_stream` exposes the raw stream so callers (e.g. bulk download) can
    read in chunks without buffering the whole object."""
    storage, fake = _make_storage()
    storage.save(_solid_png(), "stream.png")
    body = storage.open_stream("stream.png")
    try:
        assert body.read() == fake.store["images/stream.png"]["Body"]
    finally:
        body.close()


def test_open_stream_missing_raises_image_file_not_found() -> None:
    storage, _ = _make_storage()
    with pytest.raises(ImageFileNotFoundException):
        storage.open_stream("missing.png")


def test_pil_compress_level_is_passed_to_image_save() -> None:
    """`pil_compress_level` must be threaded through so disk and S3
    backends produce comparably-sized PNGs (per Copilot review)."""
    captured: dict[str, Any] = {}

    real_save = Image.Image.save

    def _spy_save(self: Image.Image, fp: Any, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("format") == "PNG":
            captured["compress_level"] = kwargs.get("compress_level")
        real_save(self, fp, *args, **kwargs)

    fake = _FakeS3Client()
    storage = S3CompatibleImageFileStorage(bucket="b", client=fake, pil_compress_level=7)
    with patch.object(Image.Image, "save", _spy_save):
        storage.save(_solid_png(), "compress.png")

    assert captured.get("compress_level") == 7


def test_get_workflow_and_graph_via_object_metadata() -> None:
    storage, _ = _make_storage()
    storage.save(_solid_png(), "meta.png", workflow="WF-DATA", graph="GRAPH-DATA")

    assert storage.get_workflow("meta.png") == "WF-DATA"
    assert storage.get_graph("meta.png") == "GRAPH-DATA"


def test_non_ascii_metadata_round_trips_via_hex() -> None:
    # Non-ASCII metadata can't go into S3 user-metadata verbatim, so it is
    # hex-encoded under the ``__hex__`` marker and decoded on read.
    storage, fake = _make_storage()
    workflow = "wörkflöw — 日本語 ✓"
    storage.save(_solid_png(), "uni.png", workflow=workflow)

    stored = fake.store["images/uni.png"]["Metadata"]["invokeai-workflow"]
    assert "__hex__" in stored and "__b64__" not in stored
    assert storage.get_workflow("uni.png") == workflow


def test_control_char_metadata_is_encoded_for_header_safety() -> None:
    # ASCII is not enough: x-amz-meta-* HTTP headers can't carry control chars, so
    # newline-bearing metadata (e.g. pretty-printed JSON) is hex-encoded as well.
    storage, fake = _make_storage()
    workflow = '{\n  "nodes": [],\n  "note": "a\r\nb"\n}'
    storage.save(_solid_png(), "ctl.png", workflow=workflow)

    stored = fake.store["images/ctl.png"]["Metadata"]["invokeai-workflow"]
    assert "__hex__" in stored
    assert "\n" not in stored and "\r" not in stored  # header-safe
    assert storage.get_workflow("ctl.png") == workflow


def test_metadata_resembling_hex_wrapper_round_trips() -> None:
    # A user value that itself looks like the encoded __hex__ wrapper must round-trip
    # unchanged, not be silently decoded on read.
    storage, fake = _make_storage()
    workflow = '{"__hex__": true, "v": "68656c6c6f"}'  # "v" is valid hex ("hello")
    storage.save(_solid_png(), "collide.png", workflow=workflow)
    assert storage.get_workflow("collide.png") == workflow


def test_image_subfolder_round_trips_under_prefixed_keys() -> None:
    # storage_backend="s3" must support the image_subfolder feature: keys live
    # under images/<subfolder>/<name> and thumbnails/<subfolder>/<name>,
    # mirroring the disk backend layout.
    storage, fake = _make_storage()
    subfolder = "2026/06/02"
    storage.save(_solid_png(), "sub.png", workflow="WF", image_subfolder=subfolder)

    thumb_key = f"thumbnails/{subfolder}/{get_thumbnail_name('sub.png')}"
    assert f"images/{subfolder}/sub.png" in fake.store
    assert thumb_key in fake.store

    assert storage.get_bytes("sub.png", image_subfolder=subfolder) == fake.store[f"images/{subfolder}/sub.png"]["Body"]
    assert storage.get("sub.png", image_subfolder=subfolder).size == (8, 8)
    assert storage.get_workflow("sub.png", image_subfolder=subfolder) == "WF"

    storage.delete("sub.png", image_subfolder=subfolder)
    assert f"images/{subfolder}/sub.png" not in fake.store
    assert thumb_key not in fake.store


@pytest.mark.parametrize("bad", ["../escape", "/abs/path", "a//b", "a\\b", "..", ".", "a/./b", "C:/x"])
def test_invalid_subfolder_is_rejected(bad: str) -> None:
    from invokeai.app.services.image_files.image_files_common import validate_subfolder

    with pytest.raises(ValueError):
        validate_subfolder(bad)


@pytest.mark.parametrize("bad_name", ["a/b.png", "a\\b.png", "../x.png"])
def test_image_name_with_separator_rejected(bad_name: str) -> None:
    # Path separators (incl. backslash, which Path(...).name ignores on POSIX)
    # must be rejected so they can't produce unexpected/escaping object keys.
    storage, _ = _make_storage()
    with pytest.raises(ValueError):
        storage.get_path(bad_name)


def test_get_workflow_falls_back_to_png_pnginfo(monkeypatch: pytest.MonkeyPatch) -> None:
    storage, fake = _make_storage()
    storage.save(_solid_png(), "fallback.png", workflow="FROM-PNG")
    fake.store["images/fallback.png"]["Metadata"] = {}

    assert storage.get_workflow("fallback.png") == "FROM-PNG"


def test_constructor_reads_invokeai_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INVOKEAI_S3_BUCKET", "env-bucket")
    monkeypatch.setenv("INVOKEAI_S3_ENDPOINT_URL", "https://example.invalid")
    monkeypatch.setenv("INVOKEAI_S3_REGION", "us-west-2")

    captured: dict[str, Any] = {}

    def _fake_boto_client(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _FakeS3Client()

    with patch("boto3.client", side_effect=_fake_boto_client):
        storage = S3CompatibleImageFileStorage()

    assert storage._bucket == "env-bucket"
    assert captured["endpoint_url"] == "https://example.invalid"
    assert captured["region_name"] == "us-west-2"


def test_constructor_omits_region_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # With no explicit region, region_name must NOT be forced (e.g. to us-east-1);
    # it is left unset so boto3 resolves it from AWS_REGION/AWS_DEFAULT_REGION/config
    # (per Copilot review: forcing a default breaks other-region buckets).
    for var in ("INVOKEAI_S3_REGION", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(var, raising=False)

    captured: dict[str, Any] = {}

    def _fake_boto_client(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _FakeS3Client()

    with patch("boto3.client", side_effect=_fake_boto_client):
        S3CompatibleImageFileStorage(bucket="b")

    assert "region_name" not in captured


def test_constructor_maps_b2_credentials_onto_aws(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.setenv("B2_APPLICATION_KEY_ID", "b2-id")
    monkeypatch.setenv("B2_APPLICATION_KEY", "b2-secret")

    captured: dict[str, Any] = {}

    def _fake_boto_client(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _FakeS3Client()

    with patch("boto3.client", side_effect=_fake_boto_client):
        S3CompatibleImageFileStorage(bucket="b2-bucket")

    assert captured["aws_access_key_id"] == "b2-id"
    assert captured["aws_secret_access_key"] == "b2-secret"


def test_constructor_defers_to_boto3_for_aws_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    # With AWS_* creds present, credentials must NOT be passed explicitly, so boto3's
    # own resolution (AWS_SESSION_TOKEN, shared profiles, instance roles) is preserved.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "aws-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "aws-token")
    # B2 convenience names present too: they must not override real AWS creds.
    monkeypatch.setenv("B2_APPLICATION_KEY_ID", "b2-id")
    monkeypatch.setenv("B2_APPLICATION_KEY", "b2-secret")

    captured: dict[str, Any] = {}

    def _fake_boto_client(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _FakeS3Client()

    with patch("boto3.client", side_effect=_fake_boto_client):
        S3CompatibleImageFileStorage(bucket="b")

    assert "aws_access_key_id" not in captured
    assert "aws_secret_access_key" not in captured
    assert "aws_session_token" not in captured


def test_constructor_requires_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INVOKEAI_S3_BUCKET", raising=False)
    with pytest.raises(ValueError, match="bucket"):
        S3CompatibleImageFileStorage(client=_FakeS3Client())
