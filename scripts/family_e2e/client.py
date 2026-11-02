"""HTTP and socket.io client for the server under test, and the run timing built from its events.

This process must never import torch: a CUDA context here would land on the GPU being measured. Socket events are
stamped on arrival with `time.perf_counter()`, the clock of the memory sampler; the server's own event timestamps
have whole-second resolution, and `time.monotonic()` ticks only every 15.6 ms on Windows.
"""

from __future__ import annotations

import mimetypes
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import socketio

from scripts.family_e2e.records import ModelLoad, NodeTiming, Timing

QUEUE_ID = "default"
SOCKETIO_PATH = "/ws/socket.io"
REQUEST_TIMEOUT_S = 120.0
# While waiting for a queue item, its status is also polled, in case a socket event was lost.
POLL_S = 2.0
EVENT_GRACE_S = 1.0
TERMINAL_STATUSES = ("completed", "failed", "canceled")
EVENT_NAMES = (
    "invocation_started",
    "invocation_progress",
    "invocation_complete",
    "invocation_error",
    "queue_item_status_changed",
    "model_load_started",
    "model_load_complete",
)
# Upper bound for the step count inferred from progress percentages.
MAX_INFERRED_STEPS = 1000
# The server drops an image-bearing progress frame sent sooner than this after the previous one
# (invokeai/app/services/progress_previews/progress_previews_default.py).
PREVIEW_INTERVAL_S = 0.1


class ApiError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        """The HTTP status of an error response; None for an accepted response the client cannot use."""


class InvokeClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._http = requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        response = self._http.request(method, self.base_url + path, timeout=REQUEST_TIMEOUT_S, **kwargs)
        if response.status_code >= 400:
            raise ApiError(f"{method} {path} -> {response.status_code}: {response.text[:2000]}", response.status_code)
        return response

    def get(self, path: str, **params: Any) -> Any:
        return self._request("GET", path, params=params or None).json()

    def get_bytes(self, path: str, **params: Any) -> bytes:
        return self._request("GET", path, params=params or None).content

    def post(self, path: str, body: Any = None) -> Any:
        return self._request("POST", path, json=body).json()

    def image_bytes(self, image_name: str) -> bytes:
        return self._request("GET", f"/api/v1/images/i/{image_name}/full").content

    def upload_image(self, path: Path) -> str:
        content_type = mimetypes.guess_type(path.name)[0] or "image/png"
        with path.open("rb") as f:
            dto = self._request(
                "POST",
                "/api/v1/images/upload",
                params={"image_category": "user", "is_intermediate": "true"},
                files={"file": (path.name, f, content_type)},
            ).json()
        return dto["image_name"]

    def enqueue(self, batch: dict[str, Any]) -> int:
        result = self.post(f"/api/v1/queue/{QUEUE_ID}/enqueue_batch", {"batch": batch, "prepend": False})
        if len(result["item_ids"]) != 1:
            raise ApiError(f"expected one queue item from the batch, got {len(result['item_ids'])}")
        return result["item_ids"][0]

    def queue_item(self, item_id: int) -> dict[str, Any]:
        return self.get(f"/api/v1/queue/{QUEUE_ID}/i/{item_id}")

    def queue_busy(self) -> bool:
        queue = self.get(f"/api/v1/queue/{QUEUE_ID}/status")["queue"]
        return queue["pending"] + queue["in_progress"] > 0

    def cancel(self, item_id: int) -> None:
        self._request("PUT", f"/api/v1/queue/{QUEUE_ID}/i/{item_id}/cancel")


@dataclass(frozen=True)
class Event:
    name: str
    t: float
    """`time.perf_counter()` at arrival."""
    data: dict[str, Any]


class EventRecorder:
    """Records the queue and model-load events of a server from connect to close."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url
        # Polling only: websocket-client is not installed, and polling delivers within a few ms on localhost.
        self._sio = socketio.Client(reconnection=True)
        self._events: list[Event] = []
        self._changed = threading.Condition()
        self._sio.on("connect", self._subscribe)
        for name in EVENT_NAMES:
            self._sio.on(name, self._handler(name))

    def _subscribe(self) -> None:
        # Also runs after a reconnect, which starts a new socket without room memberships.
        self._sio.emit("subscribe_queue", {"queue_id": QUEUE_ID})

    def _handler(self, name: str) -> Callable[[dict[str, Any]], None]:
        def handle(data: dict[str, Any]) -> None:
            t = time.perf_counter()
            if name == "invocation_progress":
                # Preview images are large and only their presence matters (it marks denoise steps).
                data = {k: v for k, v in data.items() if k != "image"} | {"has_image": data.get("image") is not None}
            with self._changed:
                self._events.append(Event(name, t, data))
                self._changed.notify_all()

        return handle

    def connect(self) -> None:
        self._sio.connect(self._base_url, socketio_path=SOCKETIO_PATH, transports=["polling"], wait_timeout=30)
        # The connect handler's emit is not acknowledged; this call returns once the server joined the rooms.
        self._sio.call("subscribe_queue", {"queue_id": QUEUE_ID}, timeout=30)

    def close(self) -> None:
        self._sio.disconnect()

    def mark(self) -> int:
        with self._changed:
            return len(self._events)

    def since(self, mark: int) -> list[Event]:
        with self._changed:
            return self._events[mark:]

    def wait(self, timeout: float) -> None:
        with self._changed:
            self._changed.wait(timeout)


def wait_for_item(
    client: InvokeClient,
    recorder: EventRecorder,
    mark: int,
    item_id: int,
    timeout: float,
    tick: Callable[[], None],
) -> tuple[dict[str, Any], float]:
    """Waits for the item to finish. Returns the item and the `perf_counter` time it finished.

    The finish time is the last arrival among the item's events once its terminal status event is in. A poll that
    sees the item finished first waits EVENT_GRACE_S for that event, because the poll can overtake the events still
    in transit; without it, the poll time stands in. `tick` runs before every check, at least every POLL_S seconds,
    and may raise to abort the wait. Raises TimeoutError.
    """
    deadline = time.perf_counter() + timeout
    next_poll = time.perf_counter() + POLL_S
    polled: tuple[dict[str, Any], float] | None = None
    while True:
        tick()
        own = _item_events(recorder.since(mark), item_id)
        latest = max((e.t for e in own), default=0.0)
        if any(e.name == "queue_item_status_changed" and e.data.get("status") in TERMINAL_STATUSES for e in own):
            return client.queue_item(item_id), latest
        now = time.perf_counter()
        if polled is not None and now - polled[1] >= EVENT_GRACE_S:
            return polled[0], max(polled[1], latest)
        if polled is None and now >= next_poll:
            item = client.queue_item(item_id)
            if item["status"] in TERMINAL_STATUSES:
                polled = (item, now)
            next_poll = now + POLL_S
        if now >= deadline:
            raise TimeoutError(f"queue item {item_id} did not finish within {timeout:.0f}s")
        recorder.wait(min(POLL_S, EVENT_GRACE_S, max(0.0, deadline - now)))


def wait_for_idle_queue(client: InvokeClient, timeout: float, tick: Callable[[], None]) -> bool:
    """Waits until the queue has nothing pending or in progress; False if it is still busy after `timeout`."""
    deadline = time.perf_counter() + timeout
    while True:
        tick()
        if not client.queue_busy():
            return True
        if time.perf_counter() >= deadline:
            return False
        time.sleep(POLL_S)


def _item_events(events: list[Event], item_id: int) -> list[Event]:
    return [e for e in events if e.data.get("item_id") == item_id]


def per_step_seconds(progress: list[tuple[float, float]], expected_steps: int | None = None) -> float | None:
    """Seconds per denoise step from (arrival time, percentage) pairs of one denoise node.

    Image-bearing progress frames are throttled server-side and the throttled steps emit nothing, so each frame's
    step comes from its percentage (k / N); the slope over the first and last frame then needs no frame in between.
    N is `expected_steps` (a run over the full schedule) when every percentage fits it. Otherwise it is the smallest
    count every percentage fits, which may be a divisor of the real one when only every other step left a frame; a
    step slower than two throttle intervals rules that out, and a faster one gives None.
    """
    if len(progress) < 2:
        return None

    def fits(n: int) -> bool:
        return all(abs(p * n - round(p * n)) < 1e-6 for _, p in progress)

    trusted = expected_steps is not None and fits(expected_steps)
    steps = expected_steps if trusted else next((n for n in range(1, MAX_INFERRED_STEPS + 1) if fits(n)), None)
    if steps is None:
        return None
    (t_first, p_first), (t_last, p_last) = progress[0], progress[-1]
    k_first, k_last = round(p_first * steps), round(p_last * steps)
    if k_last <= k_first:
        return None
    seconds = (t_last - t_first) / (k_last - k_first)
    if not trusted and seconds < 2 * PREVIEW_INTERVAL_S:
        return None
    return seconds


def build_timing(
    events: list[Event],
    item_id: int,
    t_enqueue: float,
    t_done: float,
    item: dict[str, Any],
    step_counts: Mapping[str, int] | None = None,
) -> Timing:
    """Run timing from the recorded events; every `t_*` is relative to the enqueue instant.

    `step_counts` maps a graph node id to the denoise step count the node runs over its whole schedule
    (`graphs.denoise_step_counts`); the time per step trusts the count of the node that sent the preview frames.
    """
    # The socket client runs every handler on its own thread, so the list order can differ from the arrival order.
    events = sorted(events, key=lambda e: e.t)
    own = _item_events(events, item_id)
    starts: dict[str, tuple[str, str, float]] = {}
    ends: dict[str, float] = {}
    progress: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for event in own:
        invocation = event.data.get("invocation") or {}
        prepared_id = invocation.get("id")
        if prepared_id is None:
            continue
        if event.name == "invocation_started":
            starts[prepared_id] = (event.data["invocation_source_id"], invocation.get("type", ""), event.t)
        elif event.name in ("invocation_complete", "invocation_error"):
            ends[prepared_id] = event.t
        elif event.name == "invocation_progress" and event.data.get("has_image"):
            percentage = event.data.get("percentage")
            if percentage is not None:
                progress[prepared_id].append((event.t, percentage))

    nodes = [
        NodeTiming(
            node_id=source_id,
            type=node_type,
            t_start=t - t_enqueue,
            t_end=ends[prepared_id] - t_enqueue if prepared_id in ends else None,
        )
        for prepared_id, (source_id, node_type, t) in sorted(starts.items(), key=lambda kv: kv[1][2])
    ]

    # Model load events carry no queue item; within the run's window they belong to it.
    pending: dict[tuple[str, str | None], deque[int]] = defaultdict(deque)
    loads: list[dict[str, Any]] = []
    for event in events:
        if event.name not in ("model_load_started", "model_load_complete") or not t_enqueue <= event.t <= t_done:
            continue
        key = ((event.data.get("config") or {}).get("key", ""), event.data.get("submodel_type"))
        if event.name == "model_load_started":
            pending[key].append(len(loads))
            loads.append({"model": key[0], "submodel": key[1], "t_start": event.t - t_enqueue, "t_end": None})
        elif pending[key]:
            loads[pending[key].popleft()]["t_end"] = event.t - t_enqueue

    # The denoise node is the one with the most preview frames.
    denoise: list[tuple[float, float]] = []
    expected_steps = None
    if progress:
        denoise_id = max(progress, key=lambda prepared_id: len(progress[prepared_id]))
        denoise = progress[denoise_id]
        if denoise_id in starts:
            expected_steps = (step_counts or {}).get(starts[denoise_id][0])
    return Timing(
        total_s=t_done - t_enqueue,
        nodes=nodes,
        model_loads=[ModelLoad(**load) for load in loads],
        per_step_s=per_step_seconds(denoise, expected_steps),
        queue_started_at=item.get("started_at"),
        queue_completed_at=item.get("completed_at"),
    )
