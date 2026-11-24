"""Submit a MiniMax H3 Context-IR prompt task and wait for its text result."""

import json
import re
import time
from collections.abc import Callable
from typing import Any, Literal

import requests
from urllib3.exceptions import HTTPError, InvalidHeader
from urllib3.util.retry import Retry

from invokeai.app.services.external_generation.errors import (
    ExternalProviderNotConfiguredError,
    ExternalProviderRequestError,
)
from invokeai.app.services.session_processor.session_processor_common import CanceledException

MiniMaxTextRatio = Literal["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]

_GLOBAL = "https://api.minimax.io"
_CHINA = "https://api.minimax.cn"
_REQUEST_TIMEOUT = 15
_TASK_TIMEOUT = 10 * 60
_POLL_INTERVAL = 3
_MAX_RESPONSE_BYTES = 256 * 1024


def _check_canceled(is_canceled: Callable[[], bool]) -> None:
    if is_canceled():
        raise CanceledException()


def _reject_redirect(response: requests.Response, **kwargs: Any) -> None:
    if response.is_redirect:
        # Even with allow_redirects=False, Requests normally drains the body to prepare Response.next.
        response.close()
        raise ExternalProviderRequestError(
            "MiniMax Context-IR returned a redirect. Check your MiniMax task history before submitting again."
        )


def _read_object(response: requests.Response, deadline: float, is_canceled: Callable[[], bool]) -> dict[str, Any]:
    body = bytearray()
    while True:
        _check_canceled(is_canceled)
        if time.monotonic() >= deadline:
            raise ExternalProviderRequestError("MiniMax Context-IR response timed out.")
        chunk = response.raw.read1(16384, decode_content=True)
        if not chunk:
            break
        body.extend(chunk)
        if len(body) > _MAX_RESPONSE_BYTES:
            raise ExternalProviderRequestError("MiniMax Context-IR response exceeds the size limit.")
    try:
        value = json.loads(body)
    except (ValueError, RecursionError):
        value = None
    if not isinstance(value, dict):
        raise ExternalProviderRequestError("MiniMax Context-IR returned an invalid response.")
    return value


def expand_minimax_prompt(
    *,
    prompt: str,
    duration: int,
    ratio: MiniMaxTextRatio,
    api_key: str | None,
    base_url: str | None,
    is_canceled: Callable[[], bool],
) -> str:
    origin = _GLOBAL if base_url is None else base_url.rstrip("/")
    if origin not in (_GLOBAL, _CHINA):
        raise ExternalProviderNotConfiguredError(
            "MiniMax Context-IR requires an official China or global API endpoint."
        )
    key = (api_key or "").strip()
    if not key or not key.isascii() or not key.isprintable() or " " in key:
        raise ExternalProviderNotConfiguredError("Configure a valid MiniMax API key to use Context-IR.")
    if not prompt.strip() or len(prompt) > 7000:
        raise ValueError("MiniMax Context-IR requires a non-blank prompt of at most 7000 characters.")
    payload = {
        "model": "MiniMax-H3",
        "content": [{"type": "text", "text": prompt}],
        "duration": duration,
        "ratio": ratio,
    }
    with requests.Session() as client:
        client.trust_env = False
        client.hooks["response"].append(_reject_redirect)
        client.headers.update({"Authorization": f"Bearer {key}", "Accept-Encoding": "identity"})
        _check_canceled(is_canceled)
        deadline = time.monotonic() + _REQUEST_TIMEOUT
        # A lost response can still mean a billed task. Only a GET for a known task is safe to retry.
        try:
            with client.post(
                f"{origin}/v2/h3_context_ir", json=payload, allow_redirects=False, stream=True, timeout=_REQUEST_TIMEOUT
            ) as response:
                if response.status_code != 200:
                    raise ExternalProviderRequestError(
                        f"MiniMax Context-IR submission returned HTTP {response.status_code}. "
                        "Check your MiniMax task history before submitting again."
                    )
                try:
                    task_id = _read_object(response, deadline, is_canceled).get("task_id")
                except ExternalProviderRequestError:
                    task_id = None
        except (requests.RequestException, HTTPError):
            raise ExternalProviderRequestError(
                "MiniMax Context-IR submission was interrupted. Check your MiniMax task history before submitting again."
            ) from None
        if not isinstance(task_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", task_id):
            raise ExternalProviderRequestError(
                "MiniMax Context-IR returned no valid task ID. Check your MiniMax task history before submitting again."
            )
        return _wait_for_prompt(client, origin, task_id, is_canceled)


def _wait_for_prompt(client: requests.Session, origin: str, task_id: str, is_canceled: Callable[[], bool]) -> str:
    deadline = time.monotonic() + _TASK_TIMEOUT
    failures = 0
    while (remaining := deadline - time.monotonic()) > 0:
        _check_canceled(is_canceled)
        task = None
        interval = _POLL_INTERVAL
        try:
            with client.get(
                f"{origin}/v2/query/video_generation/{task_id}",
                allow_redirects=False,
                stream=True,
                timeout=min(_REQUEST_TIMEOUT, remaining),
            ) as response:
                if response.status_code == 200:
                    task = _read_object(response, deadline, is_canceled).get("task")
                    if (
                        not isinstance(task, dict)
                        or task.get("id") != task_id
                        or task.get("task_type") != "h3_context_ir"
                    ):
                        raise ExternalProviderRequestError("MiniMax returned an unexpected task ID or type.")
                elif response.status_code in (429, 503):
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            interval = max(interval, Retry.DEFAULT.parse_retry_after(retry_after))
                        except (InvalidHeader, ValueError, OverflowError, OSError):
                            pass  # An invalid delay falls back to the regular poll interval.
                elif not 500 <= response.status_code < 600:
                    raise ExternalProviderRequestError(f"MiniMax returned HTTP {response.status_code}.")
        except ExternalProviderRequestError as error:
            # Only our sanitized protocol errors belong in the queue. Preserve the paid task's receipt.
            raise ExternalProviderRequestError(f"MiniMax Context-IR query for task {task_id}: {error}") from None
        except (requests.ConnectionError, requests.Timeout, HTTPError):
            pass
        except requests.RequestException:
            raise ExternalProviderRequestError(
                f"MiniMax Context-IR query for task {task_id} was interrupted."
            ) from None
        if task is None:
            failures += 1
            if failures == 3:
                raise ExternalProviderRequestError(
                    f"Unable to query MiniMax Context-IR task {task_id}. Check your MiniMax task history."
                )
        else:
            failures = 0
            status = task.get("status")
            if status == "succeeded":
                content = task.get("content")
                prompt = content.get("prompt") if isinstance(content, dict) else None
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ExternalProviderRequestError(f"MiniMax Context-IR task {task_id} returned no prompt.")
                return prompt
            if status in ("failed", "cancelled"):
                raise ExternalProviderRequestError(f"MiniMax Context-IR task {task_id} {status}.")
            if status not in ("queued", "running"):
                raise ExternalProviderRequestError(f"MiniMax Context-IR task {task_id} returned an unknown status.")
        next_poll = min(deadline, time.monotonic() + interval)
        while (delay := next_poll - time.monotonic()) > 0:
            _check_canceled(is_canceled)
            time.sleep(min(delay, 0.25))
    raise ExternalProviderRequestError(
        f"Timed out waiting for MiniMax Context-IR task {task_id}. Check your MiniMax task history."
    )
