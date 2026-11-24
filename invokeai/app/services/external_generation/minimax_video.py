"""MiniMax H3 hosted video generation V2. A submission creates one billable remote task."""

from __future__ import annotations

import base64
import io
import json
import re
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal, get_args
from urllib.parse import urlsplit

import requests
from PIL.Image import Image
from urllib3.exceptions import HTTPError

from invokeai.app.services.external_generation.errors import (
    ExternalProviderNotConfiguredError,
    ExternalProviderRequestError,
)
from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.app.util.ssrf import build_guarded_session, validate_download_url

MiniMaxModel = Literal[
    "MiniMax-H3",
    "MiniMax-H3-Highspeed",
    "MiniMax-H3-Lite",
    "MiniMax-H3-Max",
    "MiniMax-H3-Max-Turbo",
    "MiniMax-H3-MG",
    "MiniMax-H3-NC",
]
MiniMaxResolution = Literal["480P", "768P", "2K"]
MiniMaxRatio = Literal["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]
MiniMaxRegion = Literal["global", "cn"]

_API_ORIGINS: dict[MiniMaxRegion, str] = {"global": "https://api.minimax.io", "cn": "https://api.minimax.cn"}
_GLOBAL_ONLY = {"MiniMax-H3-Max-Turbo", "MiniMax-H3-MG", "MiniMax-H3-NC"}
_POLL_INTERVAL = 5
_POLL_TIMEOUT = 30 * 60
_REQUEST_TIMEOUT = 30
_DOWNLOAD_TIMEOUT = 5 * 60
MAX_VIDEO_BYTES = 256 * 1024 * 1024
_MAX_IMAGE_BYTES = 30_000_000
_MAX_REQUEST_BYTES = 64_000_000


def minimax_api_base_url(base_url: str | None) -> str:
    """Resolve the default endpoint and ignore a trailing slash, including for custom image endpoints."""
    return _API_ORIGINS["global"] if base_url is None else base_url.rstrip("/")


def minimax_video_region(base_url: str | None) -> MiniMaxRegion | None:
    origin = minimax_api_base_url(base_url)
    return next((region for region, url in _API_ORIGINS.items() if url == origin), None)


@dataclass(frozen=True)
class MiniMaxVideoRequest:
    model: MiniMaxModel
    prompt: str
    resolution: MiniMaxResolution
    duration: int
    ratio: MiniMaxRatio
    generate_audio: bool = True
    first_frame: Callable[[], Image] | None = None
    last_frame: Callable[[], Image] | None = None
    reference_images: Sequence[Callable[[], Image]] = ()

    def validate(self, region: MiniMaxRegion) -> None:
        if self.model not in get_args(MiniMaxModel) or (region == "cn" and self.model in _GLOBAL_ONLY):
            raise ValueError("The selected MiniMax H3 model is unavailable in the configured region.")
        if self.model == "MiniMax-H3":
            resolutions = ("480P", "768P", "2K") if region == "global" else ("768P", "2K")
        elif self.model == "MiniMax-H3-MG":
            resolutions = ("768P", "2K")
        else:
            resolutions = ("480P", "768P")
        if self.resolution not in resolutions:
            raise ValueError(f"{self.model} supports {', '.join(resolutions)} in the configured region.")
        min_duration = 5 if self.model in ("MiniMax-H3-Max", "MiniMax-H3-Max-Turbo") else 4
        if type(self.duration) is not int or not min_duration <= self.duration <= 15:
            raise ValueError(f"{self.model} requires an integer duration from {min_duration} to 15 seconds.")
        if not self.prompt.strip() or len(self.prompt) > 7000:
            raise ValueError("MiniMax requires a non-empty prompt of at most 7000 characters.")
        if self.ratio not in get_args(MiniMaxRatio):
            raise ValueError("Unsupported MiniMax aspect ratio.")
        if len(self.reference_images) > 9:
            raise ValueError("MiniMax accepts up to nine reference images.")
        if self.first_frame is not None or self.last_frame is not None:
            if self.reference_images:
                raise ValueError("MiniMax frame images and reference images cannot be combined.")
        elif self.ratio == "adaptive" and (
            not self.reference_images or self.model in ("MiniMax-H3-Lite", "MiniMax-H3-NC")
        ):
            raise ValueError("A specific aspect ratio is required for this MiniMax input and model.")

    def encode(self, region: MiniMaxRegion, is_canceled: Callable[[], bool]) -> bytes:
        self.validate(region)
        content: list[dict] = [{"type": "text", "text": self.prompt}]
        images = [("first_frame", self.first_frame), ("last_frame", self.last_frame)]
        images.extend(("reference_image", load_image) for load_image in self.reference_images)
        total_encoded = 0
        for role, load_image in images:
            if load_image is None:
                continue
            _check_canceled(is_canceled)
            # get_pil supplies an owned copy. Release each one before loading the next reference.
            with closing(load_image()) as image:
                encoded = _encode_image(image, _MAX_REQUEST_BYTES - total_encoded)
            total_encoded += len(encoded)
            content.append(
                {"type": "image_url", "role": role, "image_url": {"url": f"data:image/png;base64,{encoded}"}}
            )
        payload = {
            "model": self.model,
            "content": content,
            "resolution": self.resolution,
            "duration": self.duration,
            "ratio": "adaptive" if self.first_frame is not None or self.last_frame is not None else self.ratio,
            "generate_audio": self.generate_audio,
        }
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(body) > _MAX_REQUEST_BYTES:
            raise ValueError("The MiniMax request exceeds 64 MB with the encoded images.")
        return body


def validate_minimax_image_size(width: int, height: int) -> None:
    if not (256 <= width <= 5760 and 256 <= height <= 5760 and 0.4 <= width / height <= 2.5):
        raise ValueError("MiniMax images must be 256–5760 pixels per side with an aspect ratio of 0.4–2.5.")


def _encode_image(image: Image, remaining: int) -> str:
    validate_minimax_image_size(*image.size)
    # Only send pixels; gallery PNGs can carry private workflows, prompts or EXIF.
    with closing(image.convert("RGBA" if image.has_transparency_data else "RGB")) as pixels, io.BytesIO() as png:
        pixels.info.clear()
        pixels.save(png, format="PNG")
        size = png.tell()
        if size > _MAX_IMAGE_BYTES:
            raise ValueError("A MiniMax input image exceeds 30 MB as a PNG.")
        if 4 * ((size + 2) // 3) > remaining:
            raise ValueError("The MiniMax request exceeds 64 MB with the encoded images.")
        return base64.b64encode(png.getbuffer()).decode("ascii")


def _check_canceled(is_canceled: Callable[[], bool]) -> None:
    if is_canceled():
        # DELETE can erase a completed record if a queued task finishes between requests.
        raise CanceledException()


def _reject_redirect(response: requests.Response, **kwargs: Any) -> None:
    # Requests otherwise consumes a redirect body to prepare Response.next even when redirects are disabled.
    if response.is_redirect:
        response.close()
        raise ExternalProviderRequestError(f"MiniMax returned a redirect (HTTP {response.status_code}).")


def _json_object(response: requests.Response, deadline: float, is_canceled: Callable[[], bool]) -> dict:
    body = bytearray()
    while True:
        _check_canceled(is_canceled)
        if time.monotonic() >= deadline:
            raise ExternalProviderRequestError("MiniMax task response timed out.")
        chunk = response.raw.read1(16384, decode_content=True)
        if not chunk:
            break
        if len(body) + len(chunk) > 128 * 1024:
            raise ExternalProviderRequestError("MiniMax returned an oversized task response.")
        body.extend(chunk)
    try:
        value = json.loads(body)
        if isinstance(value, dict):
            return value
    except (ValueError, RecursionError):
        pass
    raise ExternalProviderRequestError("MiniMax returned an invalid task response.")


def _submit(api: requests.Session, origin: str, body: bytes, is_canceled: Callable[[], bool]) -> str:
    deadline = time.monotonic() + _REQUEST_TIMEOUT
    # An interrupted POST can have succeeded. Never retry it or follow a redirect.
    try:
        with api.post(
            f"{origin}/v2/video_generation", data=body, stream=True, allow_redirects=False, timeout=_REQUEST_TIMEOUT
        ) as response:
            if response.status_code != 200:
                hint = " Check your MiniMax task history before retrying." if response.status_code >= 500 else ""
                raise ExternalProviderRequestError(f"MiniMax submission returned HTTP {response.status_code}.{hint}")
            try:
                task_id = _json_object(response, deadline, is_canceled).get("task_id")
            except ExternalProviderRequestError:
                task_id = None
    except (requests.RequestException, HTTPError):
        raise ExternalProviderRequestError(
            "MiniMax submission was interrupted. Check your MiniMax task history before retrying."
        ) from None
    if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", task_id):
        raise ExternalProviderRequestError(
            "MiniMax returned no valid task ID. Check your MiniMax task history before retrying."
        )
    return task_id


def _poll(
    api: requests.Session,
    origin: str,
    task_id: str,
    is_canceled: Callable[[], bool],
    on_progress: Callable[[str], None] | None,
) -> str:
    deadline = time.monotonic() + _POLL_TIMEOUT
    previous_status = None
    consecutive_failures = 0
    while (remaining := deadline - time.monotonic()) > 0:
        _check_canceled(is_canceled)
        task = None
        try:
            with api.get(
                f"{origin}/v2/query/video_generation/{task_id}",
                timeout=min(remaining, _REQUEST_TIMEOUT),
                allow_redirects=False,
                stream=True,
            ) as response:
                if response.status_code == 200:
                    try:
                        task = _json_object(response, deadline, is_canceled).get("task")
                    except ExternalProviderRequestError as error:
                        raise ExternalProviderRequestError(f"MiniMax query for task {task_id}: {error}") from None
                    if not isinstance(task, dict) or task.get("id") != task_id:
                        raise ExternalProviderRequestError(f"MiniMax returned an invalid response for task {task_id}.")
                elif response.status_code != 429 and not 500 <= response.status_code < 600:
                    raise ExternalProviderRequestError(
                        f"MiniMax query for task {task_id} returned HTTP {response.status_code}."
                    )
        except (requests.ConnectionError, requests.Timeout, HTTPError):
            # Querying the known task ID can be retried without creating another paid task.
            pass
        except requests.RequestException:
            raise ExternalProviderRequestError(f"MiniMax query for task {task_id} was interrupted.") from None
        if task is None:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                raise ExternalProviderRequestError(
                    f"Unable to query MiniMax task {task_id}. Check your MiniMax task history."
                )
        else:
            consecutive_failures = 0
            status = task.get("status")
            if status in ("failed", "cancelled"):
                raise ExternalProviderRequestError(f"MiniMax task {task_id} {status}.")
            if status not in ("queued", "running", "succeeded"):
                raise ExternalProviderRequestError(f"MiniMax task {task_id} returned an unknown status.")
            if on_progress and status != previous_status:
                on_progress(status)
            previous_status = status
            if status == "succeeded":
                content = task.get("content")
                if not isinstance(content, dict) or not isinstance(content.get("url"), str):
                    raise ExternalProviderRequestError(f"MiniMax task {task_id} returned no video URL.")
                return content["url"]
        next_poll = min(deadline, time.monotonic() + _POLL_INTERVAL)
        while (delay := next_poll - time.monotonic()) > 0:
            _check_canceled(is_canceled)
            time.sleep(min(delay, 0.5))
    raise ExternalProviderRequestError(
        f"Timed out waiting for MiniMax task {task_id}. Check your MiniMax task history."
    )


def _download(url: str, path: Path, is_canceled: Callable[[], bool]) -> None:
    deadline = time.monotonic() + _DOWNLOAD_TIMEOUT
    _check_canceled(is_canceled)
    try:
        parts = urlsplit(url)
        if parts.scheme != "https" or not parts.hostname or parts.username is not None or parts.password is not None:
            raise ValueError
        validate_download_url(url)
        # Use a fresh session so the API key, cookies and ambient netrc credentials stay off CDN requests.
        with build_guarded_session() as cdn:
            cdn.trust_env = False
            cdn.hooks["response"].append(_reject_redirect)
            with cdn.get(
                url,
                headers={"Accept-Encoding": "identity"},
                allow_redirects=False,
                timeout=_REQUEST_TIMEOUT,
                stream=True,
            ) as response:
                if (
                    response.status_code != 200
                    or response.headers.get("Content-Encoding", "identity").lower() != "identity"
                ):
                    raise ExternalProviderRequestError(
                        f"MiniMax video download returned an invalid response (HTTP {response.status_code})."
                    )
                length = response.headers.get("Content-Length")
                expected = int(length) if length is not None else None
                if expected is not None and not 0 < expected <= MAX_VIDEO_BYTES:
                    raise ExternalProviderRequestError("MiniMax video has an invalid or excessive size.")
                size = 0
                with path.open("xb") as output:
                    while True:
                        _check_canceled(is_canceled)
                        if time.monotonic() >= deadline:
                            raise ExternalProviderRequestError("MiniMax video download timed out.")
                        # read1 returns available data so a slow, trickling body cannot hold us until a full chunk arrives.
                        chunk = response.raw.read1(64 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > MAX_VIDEO_BYTES:
                            raise ExternalProviderRequestError("MiniMax video exceeds the download size limit.")
                        output.write(chunk)
                if size == 0 or (expected is not None and size != expected):
                    raise ExternalProviderRequestError("MiniMax returned an empty or incomplete video.")
    except (ValueError, requests.RequestException, HTTPError):
        # The URL can contain a signed token; do not include it or the underlying exception in queue errors.
        raise ExternalProviderRequestError("MiniMax video download failed or returned an unsafe URL.") from None


@contextmanager
def generate_minimax_video(
    *,
    api_key: str,
    base_url: str | None,
    request: MiniMaxVideoRequest,
    is_canceled: Callable[[], bool],
    on_progress: Callable[[str], None] | None = None,
) -> Iterator[Path]:
    """Yield a temporary downloaded video until the caller completes ingest or raises."""
    region = minimax_video_region(base_url)
    if region is None:
        raise ExternalProviderNotConfiguredError(
            "MiniMax hosted video requires https://api.minimax.io or https://api.minimax.cn."
        )
    api_key = (api_key or "").strip()
    if not api_key or not api_key.isascii() or not api_key.isprintable() or " " in api_key:
        raise ExternalProviderNotConfiguredError("A valid MiniMax API key is required for hosted video.")
    origin = _API_ORIGINS[region]
    _check_canceled(is_canceled)
    body = request.encode(region, is_canceled)
    with requests.Session() as api:
        api.trust_env = False
        api.hooks["response"].append(_reject_redirect)
        api.headers.update({"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        _check_canceled(is_canceled)
        task_id = _submit(api, origin, body, is_canceled)
        del body
        url = _poll(api, origin, task_id, is_canceled, on_progress)
    with TemporaryDirectory(prefix="invokeai-minimax-") as directory:
        path = Path(directory) / "video.mp4"
        try:
            _download(url, path, is_canceled)
        except ExternalProviderRequestError as error:
            raise ExternalProviderRequestError(f"MiniMax task {task_id}: {error}") from None
        _check_canceled(is_canceled)
        yield path
