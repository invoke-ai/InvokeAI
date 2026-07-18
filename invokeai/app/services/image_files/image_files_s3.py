# Copyright (c) 2026 The InvokeAI Team
import io
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Optional, cast

from PIL import Image, PngImagePlugin
from PIL.Image import Image as PILImageType

from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.image_files.image_files_common import (
    ImageFileDeleteException,
    ImageFileNotFoundException,
    ImageFileSaveException,
    validate_subfolder,
)
from invokeai.app.util.thumbnails import get_thumbnail_name, make_thumbnail

if TYPE_CHECKING:
    from botocore.exceptions import ClientError

    from invokeai.app.services.invoker import Invoker


_IMAGES_PREFIX = "images/"
_THUMBNAILS_PREFIX = "thumbnails/"
_USER_AGENT_SUFFIX = "invokeai"
_META_INVOKEAI_METADATA = "invokeai-metadata"
_META_INVOKEAI_WORKFLOW = "invokeai-workflow"
_META_INVOKEAI_GRAPH = "invokeai-graph"

# S3 user-metadata is capped by providers (AWS: ~2 KB total across all keys and
# values). Budget under 2048 so key names and per-header overhead still fit beneath
# the cap; oversized values are skipped for S3 metadata but still embedded in the
# PNG text chunks (and read back from there).
_MAX_S3_METADATA_BYTES = 1800

# S3/B2 use several codes for a missing object/key; treat them uniformly so
# not-found detection can't drift across call sites.
_S3_NOT_FOUND_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


def _is_not_found_error(error: "ClientError") -> bool:
    """True when a botocore ClientError denotes a missing object/key."""
    return error.response.get("Error", {}).get("Code", "") in _S3_NOT_FOUND_CODES


class S3CompatibleImageFileStorage(ImageFileStorageBase):
    """Stores images in any S3-compatible bucket"""

    def __init__(
        self,
        bucket: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        pil_compress_level: int = 1,
        client: Optional[Any] = None,
    ) -> None:
        self._bucket = bucket or os.environ.get("INVOKEAI_S3_BUCKET")
        if not self._bucket:
            raise ValueError(
                "S3CompatibleImageFileStorage requires a bucket name (pass `bucket=` or set INVOKEAI_S3_BUCKET)."
            )

        self._endpoint_url = endpoint_url or os.environ.get("INVOKEAI_S3_ENDPOINT_URL")
        # Only an explicit region (kwarg or INVOKEAI_S3_REGION) is honored here.
        # If unset, leave it to boto3's normal resolution (AWS_REGION /
        # AWS_DEFAULT_REGION / AWS config) rather than forcing a single default
        # that would break auth/signing for buckets in other regions.
        self._region_name = region_name or os.environ.get("INVOKEAI_S3_REGION")
        self._pil_compress_level = pil_compress_level
        # Build the boto3 client only when one isn't injected, so constructing with an
        # injected client (tests, alternative implementations) skips the boto3 import
        # here. (botocore.exceptions.ClientError is still used by the read paths.)
        self._client = client if client is not None else self._build_client()

    def _build_client(self) -> Any:
        try:
            import boto3
            from botocore.config import Config as BotoConfig
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "boto3 is required for S3CompatibleImageFileStorage. Install the project's "
                '`s3` extra (`pip install "invokeai[s3]"`), or install boto3 directly '
                "(`pip install boto3`)."
            ) from e

        boto_config = BotoConfig(
            user_agent_extra=_USER_AGENT_SUFFIX,
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
        )

        # Only set region explicitly when provided; otherwise boto3 resolves it.
        client_kwargs: dict[str, Any] = {}
        if self._region_name:
            client_kwargs["region_name"] = self._region_name
        if self._endpoint_url:
            client_kwargs["endpoint_url"] = self._endpoint_url

        # Map B2_* creds onto AWS_* names only when NO standard AWS credential env var
        # is set. If any (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN)
        # is present, defer to boto3 so it resolves them (and fails fast on a partial
        # set) rather than silently switching to B2 or mixing credentials.
        _aws_cred_vars = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")
        if not any(os.environ.get(v) for v in _aws_cred_vars):
            b2_key_id = os.environ.get("B2_APPLICATION_KEY_ID")
            b2_key = os.environ.get("B2_APPLICATION_KEY")
            if b2_key_id and b2_key:
                client_kwargs["aws_access_key_id"] = b2_key_id
                client_kwargs["aws_secret_access_key"] = b2_key

        return boto3.client("s3", config=boto_config, **client_kwargs)

    @staticmethod
    def _object_key(image_name: str, thumbnail: bool = False, image_subfolder: str = "") -> str:
        # Forbid any path separators. ``Path(...).name`` does not treat "\\" as a
        # separator on POSIX, so check both explicitly to prevent traversal.
        if "/" in image_name or "\\" in image_name or Path(image_name).name != image_name:
            raise ValueError("Invalid image name, potential directory traversal detected")
        basename = image_name
        prefix = _THUMBNAILS_PREFIX if thumbnail else _IMAGES_PREFIX
        filename = get_thumbnail_name(basename) if thumbnail else basename
        # Mirror the disk backend's ``<base>/<subfolder>/<name>`` layout.
        if image_subfolder:
            validate_subfolder(image_subfolder)
            return f"{prefix}{image_subfolder}/{filename}"
        return f"{prefix}{filename}"

    def start(self, invoker: "Invoker") -> None:
        # No-op: this backend is fully configured at construction (bucket, creds,
        # region, pil_compress_level), so it does not retain the invoker.
        pass

    @property
    def image_root(self) -> Path:
        """Synthetic ``s3://`` root for full-size images (identification only).

        There is no local filesystem root for this backend; this mirrors the
        prefix used by :meth:`get_path` so callers that reason about the root
        (e.g. directory cleanup in the image-move service) get a consistent,
        non-filesystem URI rather than a real path.
        """
        return Path(f"s3://{self._bucket}/{_IMAGES_PREFIX.rstrip('/')}")

    @property
    def thumbnail_root(self) -> Path:
        """Synthetic ``s3://`` root for thumbnails (identification only). See :meth:`image_root`."""
        return Path(f"s3://{self._bucket}/{_THUMBNAILS_PREFIX.rstrip('/')}")

    def evict_cache_paths(self, paths: list[Path]) -> None:
        """No-op: this backend keeps no in-memory image cache to evict.

        Unlike the disk backend, S3 reads are not cached locally, so there is
        nothing to invalidate when objects are moved or deleted.
        """
        pass

    def get(self, image_name: str, image_subfolder: str = "") -> PILImageType:
        body = self.get_bytes(image_name, image_subfolder=image_subfolder)
        image = Image.open(io.BytesIO(body))
        image.load()
        return image

    def open_stream(self, image_name: str, thumbnail: bool = False, image_subfolder: str = "") -> BinaryIO:
        """Return the boto3 ``StreamingBody`` for the object.

        Lets callers read in chunks without buffering the whole file in memory
        (e.g. streaming into a zip for bulk downloads). The caller must close it.
        """
        from botocore.exceptions import ClientError

        key = self._object_key(image_name, thumbnail=thumbnail, image_subfolder=image_subfolder)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as e:
            if _is_not_found_error(e):
                raise ImageFileNotFoundException from e
            raise
        # boto3's StreamingBody is a readable binary stream; the cast aligns it
        # with the BinaryIO contract declared on the base class.
        return cast(BinaryIO, response["Body"])

    def get_bytes(self, image_name: str, thumbnail: bool = False, image_subfolder: str = "") -> bytes:
        """Return the raw object bytes for the given image (or its thumbnail).

        Used by the image-serving HTTP routes so they do not need to call
        ``open()`` on the result of ``get_path``, which is a synthetic
        ``s3://`` path for this backend.
        """
        # boto3 returns a StreamingBody; close it after reading so the HTTP
        # connection is released back to the pool instead of leaking.
        body = self.open_stream(image_name, thumbnail=thumbnail, image_subfolder=image_subfolder)
        try:
            return body.read()
        finally:
            body.close()

    def save(
        self,
        image: PILImageType,
        image_name: str,
        metadata: Optional[str] = None,
        workflow: Optional[str] = None,
        graph: Optional[str] = None,
        thumbnail_size: int = 256,
        image_subfolder: str = "",
    ) -> None:
        key: Optional[str] = None
        thumb_key: Optional[str] = None
        try:
            key = self._object_key(image_name, image_subfolder=image_subfolder)
            thumb_key = self._object_key(image_name, thumbnail=True, image_subfolder=image_subfolder)

            pnginfo = PngImagePlugin.PngInfo()
            obj_metadata: dict[str, str] = {}
            # Always embed metadata in the PNG text chunks (effectively
            # unbounded), but only mirror entries into S3 user-metadata while we
            # stay under a conservative size budget. Oversized values are read
            # back from the PNG by `_get_text_metadata`, so nothing is lost. We
            # deliberately do not mutate `image.info`, which would clobber
            # existing fields such as the ICC profile.
            remaining = _MAX_S3_METADATA_BYTES
            for png_key, s3_key, value in (
                ("invokeai_metadata", _META_INVOKEAI_METADATA, metadata),
                ("invokeai_workflow", _META_INVOKEAI_WORKFLOW, workflow),
                ("invokeai_graph", _META_INVOKEAI_GRAPH, graph),
            ):
                if value is None:
                    continue
                pnginfo.add_text(png_key, value)
                encoded = _encode_meta(value)
                cost = len(s3_key.encode("utf-8")) + len(encoded.encode("utf-8"))
                if cost <= remaining:
                    obj_metadata[s3_key] = encoded
                    remaining -= cost

            # ``with`` closes the buffer even if save/upload raises.
            with io.BytesIO() as png_buffer:
                image.save(
                    png_buffer,
                    format="PNG",
                    pnginfo=pnginfo,
                    compress_level=self._pil_compress_level,
                )
                png_buffer.seek(0)
                put_kwargs: dict[str, Any] = {
                    "Bucket": self._bucket,
                    "Key": key,
                    # Pass the file-like buffer (seeked to 0) so boto3 streams it
                    # rather than copying the whole buffer to bytes.
                    "Body": png_buffer,
                    "ContentType": "image/png",
                }
                if obj_metadata:
                    put_kwargs["Metadata"] = obj_metadata
                self._client.put_object(**put_kwargs)

            thumbnail_image = make_thumbnail(image, thumbnail_size)
            with io.BytesIO() as thumb_buffer:
                thumbnail_image.save(thumb_buffer, format="WEBP")
                thumb_buffer.seek(0)
                self._client.put_object(
                    Bucket=self._bucket,
                    Key=thumb_key,
                    Body=thumb_buffer,
                    ContentType="image/webp",
                )
        except Exception as e:
            # Best-effort rollback so a partial failure (e.g. the image uploaded
            # but the thumbnail did not) doesn't leave an orphaned object.
            self._delete_quietly(key, thumb_key)
            raise ImageFileSaveException from e

    def _delete_quietly(self, *keys: Optional[str]) -> None:
        """Best-effort delete of object keys; never raises (used to roll back a partial save)."""
        for key in keys:
            if not key:
                continue
            try:
                self._client.delete_object(Bucket=self._bucket, Key=key)
            except Exception:
                pass

    def delete(self, image_name: str, image_subfolder: str = "") -> None:
        try:
            key = self._object_key(image_name, image_subfolder=image_subfolder)
            thumb_key = self._object_key(image_name, thumbnail=True, image_subfolder=image_subfolder)
            self._client.delete_object(Bucket=self._bucket, Key=key)
            self._client.delete_object(Bucket=self._bucket, Key=thumb_key)
        except Exception as e:
            raise ImageFileDeleteException from e

    def get_path(self, image_name: str, thumbnail: bool = False, image_subfolder: str = "") -> Path:
        """Return a synthetic path identifying the object (an ``s3://``-style URI).

        For identification only, not filesystem I/O. The ``ImageFileStorageBase``
        contract returns ``Path``, but ``Path`` cannot faithfully round-trip a URI:
        ``str()`` collapses ``//`` (and uses backslashes on Windows), so callers
        must not rely on the exact ``s3://`` textual form. :meth:`validate_path`
        accounts for that normalization; use :meth:`get_bytes` to read contents and
        :meth:`validate_path` to check existence, rather than opening this path.
        """
        key = self._object_key(image_name, thumbnail=thumbnail, image_subfolder=image_subfolder)
        return Path(f"s3://{self._bucket}/{key}")

    def validate_path(self, path: Path | str) -> bool:
        """Validates the path given for an image or thumbnail."""
        from botocore.exceptions import ClientError

        # Normalize to a forward-slash string. Callers round-trip get_path()
        # (a ``Path``) through ``str()``; pathlib collapses ``s3://bucket/k`` to
        # ``s3:/bucket/k`` on POSIX and ``s3:\bucket\k`` on Windows, so we
        # canonicalize separators before stripping the scheme/bucket prefix.
        path = str(path).replace("\\", "/")
        key = path
        for prefix in (f"s3://{self._bucket}/", f"s3:/{self._bucket}/"):
            if path.startswith(prefix):
                key = path[len(prefix) :]
                break

        # Only keys under our managed prefixes are valid; never issue a HEAD for
        # an arbitrary key (that would be an object-existence oracle for the bucket).
        if not (key.startswith(_IMAGES_PREFIX) or key.startswith(_THUMBNAILS_PREFIX)):
            return False

        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError as e:
            if _is_not_found_error(e):
                return False
            raise

    def get_workflow(self, image_name: str, image_subfolder: str = "") -> str | None:
        return self._get_text_metadata(
            image_name, _META_INVOKEAI_WORKFLOW, "invokeai_workflow", image_subfolder=image_subfolder
        )

    def get_graph(self, image_name: str, image_subfolder: str = "") -> str | None:
        return self._get_text_metadata(
            image_name, _META_INVOKEAI_GRAPH, "invokeai_graph", image_subfolder=image_subfolder
        )

    def _get_text_metadata(
        self, image_name: str, s3_meta_key: str, png_info_key: str, image_subfolder: str = ""
    ) -> str | None:
        from botocore.exceptions import ClientError

        key = self._object_key(image_name, image_subfolder=image_subfolder)
        try:
            head = self._client.head_object(Bucket=self._bucket, Key=key)
        except ClientError as e:
            if _is_not_found_error(e):
                raise ImageFileNotFoundException from e
            raise

        # boto3 lower-cases user-metadata keys.
        meta = head.get("Metadata", {}) or {}
        value = meta.get(s3_meta_key)
        if isinstance(value, str) and value:
            return _decode_meta(value)

        image = self.get(image_name, image_subfolder=image_subfolder)
        png_value = image.info.get(png_info_key, None)
        if isinstance(png_value, str):
            return png_value
        return None


def _encode_meta(value: str) -> str:
    """Encode arbitrary unicode metadata for safe transport as S3 user-metadata.

    S3 user-metadata rides in ``x-amz-meta-*`` HTTP headers, which cannot carry
    non-ASCII or control characters (CR/LF break header framing). Clean printable
    ASCII passes through; anything else is hex-encoded under the ``__hex__`` marker
    (hex is itself header-safe, so the same marker covers both cases). A value that
    would itself decode as a ``__hex__`` payload is also encoded, so user metadata
    resembling the wrapper round-trips instead of being silently decoded on read.
    """
    if value.isascii() and value.isprintable() and _decode_meta(value) == value:
        return value
    return json.dumps({"__hex__": True, "v": value.encode("utf-8").hex()}, separators=(",", ":"))


def _decode_meta(value: str) -> str:
    """Inverse of `_encode_meta`."""
    if value.startswith("{") and "__hex__" in value:
        try:
            payload = json.loads(value)
            if isinstance(payload, dict) and payload.get("__hex__"):
                encoded = payload.get("v")
                # Guard corrupted/hand-edited metadata where "v" is missing or
                # not a hex string; fall through to returning the raw value.
                if isinstance(encoded, str):
                    return bytes.fromhex(encoded).decode("utf-8")
        # UnicodeDecodeError is a ValueError subclass (so already covered), but
        # list it explicitly to document that bad hex / invalid UTF-8 from
        # corrupted metadata falls back to the raw value instead of raising.
        except (ValueError, KeyError, UnicodeDecodeError):
            pass
    return value
