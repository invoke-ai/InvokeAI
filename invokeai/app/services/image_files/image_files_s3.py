# Copyright (c) 2026 The InvokeAI Team
import io
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from PIL import Image, PngImagePlugin
from PIL.Image import Image as PILImageType

from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.image_files.image_files_common import (
    ImageFileDeleteException,
    ImageFileNotFoundException,
    ImageFileSaveException,
)
from invokeai.app.util.thumbnails import get_thumbnail_name, make_thumbnail

if TYPE_CHECKING:
    from invokeai.app.services.invoker import Invoker


_IMAGES_PREFIX = "images/"
_THUMBNAILS_PREFIX = "thumbnails/"
_USER_AGENT_SUFFIX = "invokeai"
_META_INVOKEAI_METADATA = "invokeai-metadata"
_META_INVOKEAI_WORKFLOW = "invokeai-workflow"
_META_INVOKEAI_GRAPH = "invokeai-graph"


class S3CompatibleImageFileStorage(ImageFileStorageBase):
    """Stores images in any S3-compatible bucket"""

    def __init__(
        self,
        bucket: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        client: Optional[Any] = None,
    ) -> None:
        try:
            import boto3  # noqa: F401
            from botocore.config import Config as BotoConfig
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "boto3 is required for S3CompatibleImageFileStorage. Install it with `pip install boto3`."
            ) from e

        self._bucket = bucket or os.environ.get("INVOKEAI_S3_BUCKET")
        if not self._bucket:
            raise ValueError(
                "S3CompatibleImageFileStorage requires a bucket name (pass `bucket=` or set INVOKEAI_S3_BUCKET)."
            )

        self._endpoint_url = endpoint_url or os.environ.get("INVOKEAI_S3_ENDPOINT_URL")
        self._region_name = region_name or os.environ.get("INVOKEAI_S3_REGION") or "us-east-1"
        self._client = client if client is not None else self._build_client(BotoConfig)

    def _build_client(self, BotoConfig: Any) -> Any:
        import boto3

        # Map B2_* env vars onto AWS_* names without mutating os.environ globally.
        access_key = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("B2_APPLICATION_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("B2_APPLICATION_KEY")

        boto_config = BotoConfig(
            user_agent_extra=_USER_AGENT_SUFFIX,
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
        )

        kwargs: dict[str, Any] = {
            "service_name": "s3",
            "region_name": self._region_name,
            "config": boto_config,
        }
        if self._endpoint_url:
            kwargs["endpoint_url"] = self._endpoint_url
        if access_key and secret_key:
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key

        return boto3.client(**kwargs)

    @staticmethod
    def _object_key(image_name: str, thumbnail: bool = False) -> str:
        basename = Path(image_name).name
        if basename != image_name:
            raise ValueError("Invalid image name, potential directory traversal detected")
        if thumbnail:
            return f"{_THUMBNAILS_PREFIX}{get_thumbnail_name(basename)}"
        return f"{_IMAGES_PREFIX}{basename}"

    def start(self, invoker: "Invoker") -> None:
        self.__invoker = invoker

    def get(self, image_name: str) -> PILImageType:
        from botocore.exceptions import ClientError

        key = self._object_key(image_name)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NotFound"):
                raise ImageFileNotFoundException from e
            raise

        body = response["Body"].read()
        image = Image.open(io.BytesIO(body))
        image.load()
        return image

    def save(
        self,
        image: PILImageType,
        image_name: str,
        metadata: Optional[str] = None,
        workflow: Optional[str] = None,
        graph: Optional[str] = None,
        thumbnail_size: int = 256,
    ) -> None:
        try:
            key = self._object_key(image_name)
            thumb_key = self._object_key(image_name, thumbnail=True)

            pnginfo = PngImagePlugin.PngInfo()
            info_dict: dict[str, str] = {}
            obj_metadata: dict[str, str] = {}

            if metadata is not None:
                info_dict["invokeai_metadata"] = metadata
                pnginfo.add_text("invokeai_metadata", metadata)
                obj_metadata[_META_INVOKEAI_METADATA] = _safe_meta(metadata)
            if workflow is not None:
                info_dict["invokeai_workflow"] = workflow
                pnginfo.add_text("invokeai_workflow", workflow)
                obj_metadata[_META_INVOKEAI_WORKFLOW] = _safe_meta(workflow)
            if graph is not None:
                info_dict["invokeai_graph"] = graph
                pnginfo.add_text("invokeai_graph", graph)
                obj_metadata[_META_INVOKEAI_GRAPH] = _safe_meta(graph)

            image.info = info_dict

            png_buffer = io.BytesIO()
            image.save(png_buffer, format="PNG", pnginfo=pnginfo)
            png_buffer.seek(0)

            put_kwargs: dict[str, Any] = {
                "Bucket": self._bucket,
                "Key": key,
                "Body": png_buffer.getvalue(),
                "ContentType": "image/png",
            }
            if obj_metadata:
                put_kwargs["Metadata"] = obj_metadata
            self._client.put_object(**put_kwargs)

            thumbnail_image = make_thumbnail(image, thumbnail_size)
            thumb_buffer = io.BytesIO()
            thumbnail_image.save(thumb_buffer, format="WEBP")
            thumb_buffer.seek(0)
            self._client.put_object(
                Bucket=self._bucket,
                Key=thumb_key,
                Body=thumb_buffer.getvalue(),
                ContentType="image/webp",
            )
        except Exception as e:
            raise ImageFileSaveException from e

    def delete(self, image_name: str) -> None:
        try:
            key = self._object_key(image_name)
            thumb_key = self._object_key(image_name, thumbnail=True)
            self._client.delete_object(Bucket=self._bucket, Key=key)
            self._client.delete_object(Bucket=self._bucket, Key=thumb_key)
        except Exception as e:
            raise ImageFileDeleteException from e

    def get_path(self, image_name: str, thumbnail: bool = False) -> Path:
        # Synthetic s3:// path; callers needing a real filesystem path should
        # migrate to a presigned-URL service.
        key = self._object_key(image_name, thumbnail=thumbnail)
        return Path(f"s3://{self._bucket}/{key}")

    def validate_path(self, path: str) -> bool:
        """Validates the path given for an image or thumbnail."""
        from botocore.exceptions import ClientError

        if isinstance(path, Path):
            path = str(path)
        prefix = f"s3://{self._bucket}/"
        key = path[len(prefix) :] if path.startswith(prefix) else path

        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def get_workflow(self, image_name: str) -> str | None:
        return self._get_text_metadata(image_name, _META_INVOKEAI_WORKFLOW, "invokeai_workflow")

    def get_graph(self, image_name: str) -> str | None:
        return self._get_text_metadata(image_name, _META_INVOKEAI_GRAPH, "invokeai_graph")

    def _get_text_metadata(self, image_name: str, s3_meta_key: str, png_info_key: str) -> str | None:
        from botocore.exceptions import ClientError

        key = self._object_key(image_name)
        try:
            head = self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                raise ImageFileNotFoundException from e
            raise

        # boto3 lower-cases user-metadata keys.
        meta = head.get("Metadata", {}) or {}
        value = meta.get(s3_meta_key)
        if isinstance(value, str) and value:
            return _unsafe_meta(value)

        image = self.get(image_name)
        png_value = image.info.get(png_info_key, None)
        if isinstance(png_value, str):
            return png_value
        return None


def _safe_meta(value: str) -> str:
    """Encode arbitrary unicode metadata for safe transport as S3 user-metadata."""
    try:
        value.encode("ascii")
        return value
    except UnicodeEncodeError:
        return json.dumps({"__b64__": True, "v": value.encode("utf-8").hex()})


def _unsafe_meta(value: str) -> str:
    """Inverse of `_safe_meta`."""
    if value.startswith("{") and "__b64__" in value:
        try:
            payload = json.loads(value)
            if isinstance(payload, dict) and payload.get("__b64__"):
                return bytes.fromhex(payload["v"]).decode("utf-8")
        except (ValueError, KeyError):
            pass
    return value
