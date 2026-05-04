from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from botocore.exceptions import ClientError
from PIL import Image

from invokeai.app.services.image_files.image_files_common import (
    ImageFileNotFoundException,
)
from invokeai.app.services.image_files.image_files_s3 import S3CompatibleImageFileStorage


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "TestOperation")


class _FakeS3Client:
    def __init__(self) -> None:
        self.store: dict[str, dict[str, Any]] = {}

    def put_object(self, Bucket: str, Key: str, Body: bytes, **kwargs: Any) -> dict:
        self.store[Key] = {
            "Body": Body,
            "Metadata": kwargs.get("Metadata", {}) or {},
            "ContentType": kwargs.get("ContentType"),
        }
        return {}

    def get_object(self, Bucket: str, Key: str) -> dict:
        if Key not in self.store:
            raise _client_error("NoSuchKey")
        record = self.store[Key]
        return {"Body": _BodyStream(record["Body"]), "Metadata": record["Metadata"]}

    def head_object(self, Bucket: str, Key: str) -> dict:
        if Key not in self.store:
            raise _client_error("404")
        return {"Metadata": self.store[Key]["Metadata"]}

    def delete_object(self, Bucket: str, Key: str) -> dict:
        self.store.pop(Key, None)
        return {}


class _BodyStream:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


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
    assert "thumbnails/round-trip.webp" in fake.store

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
    assert "thumbnails/to-delete.webp" not in fake.store


def test_validate_path_for_present_and_missing_keys() -> None:
    storage, _ = _make_storage()
    storage.save(_solid_png(), "exists.png")

    assert storage.validate_path("images/exists.png") is True
    assert storage.validate_path("s3://test-bucket/images/exists.png") is True
    assert storage.validate_path("images/missing.png") is False


def test_get_path_returns_synthetic_s3_uri() -> None:
    storage, _ = _make_storage()
    p = str(storage.get_path("foo.png", thumbnail=True))
    assert "test-bucket" in p and "thumbnails/foo.webp" in p


def test_get_workflow_and_graph_via_object_metadata() -> None:
    storage, _ = _make_storage()
    storage.save(_solid_png(), "meta.png", workflow="WF-DATA", graph="GRAPH-DATA")

    assert storage.get_workflow("meta.png") == "WF-DATA"
    assert storage.get_graph("meta.png") == "GRAPH-DATA"


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

    def _fake_boto_client(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _FakeS3Client()

    with patch("boto3.client", side_effect=_fake_boto_client):
        storage = S3CompatibleImageFileStorage()

    assert storage._bucket == "env-bucket"
    assert captured["endpoint_url"] == "https://example.invalid"
    assert captured["region_name"] == "us-west-2"


def test_constructor_maps_b2_credentials_onto_aws(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setenv("B2_APPLICATION_KEY_ID", "b2-id")
    monkeypatch.setenv("B2_APPLICATION_KEY", "b2-secret")

    captured: dict[str, Any] = {}

    def _fake_boto_client(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return _FakeS3Client()

    with patch("boto3.client", side_effect=_fake_boto_client):
        S3CompatibleImageFileStorage(bucket="b2-bucket")

    assert captured["aws_access_key_id"] == "b2-id"
    assert captured["aws_secret_access_key"] == "b2-secret"


def test_constructor_requires_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INVOKEAI_S3_BUCKET", raising=False)
    with pytest.raises(ValueError, match="bucket"):
        S3CompatibleImageFileStorage(client=_FakeS3Client())
