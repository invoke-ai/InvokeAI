"""Guards that slow gallery queries do not stall the whole server.

The gallery list/name routes run synchronous SQLite work. If such a route is declared
`async def`, that work executes *on the event loop*, so for its whole duration the process
serves nothing else - no other HTTP request, no socket.io progress event. On a large
library a single search can take minutes, which users experience as the backend being
dead rather than as a slow search.

Declaring these routes `def` hands them to Starlette's threadpool instead, leaving the
loop free. These tests pin that property down deterministically: the service layer blocks
on an Event the TEST controls (released only after the probe completes), so the assertion
is causal rather than a wall-clock latency threshold. An earlier version asserted
"probe elapsed < BLOCKING_SECONDS / 2", which a descheduled CI runner could fail with a
free loop (observed: a 1.09s probe on a loaded macOS runner) - machine noise and a blocked
loop produced the same timing signature. Here, machine noise cannot fail the test: the
handler cannot finish before the probe unless loop dispatch is broken, because only the
test releases it.

The failure mode is equally deterministic: an `async def` regression blocks the loop
inside the handler's wait, the probe cannot be answered until that wait times out
(FAILURE_TIMEOUT_SECONDS, paid only when the guard actually catches a regression), and by
then the handler has finished - the finished-flag assertion fires.
"""

import asyncio
import threading
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.gallery.gallery_common import GalleryItem, GalleryItemNames, GalleryItemNamesResult
from invokeai.app.services.image_records.image_records_common import ImageNamesResult
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItemSummary
from invokeai.app.services.shared.pagination import OffsetPaginatedResults

# Upper bound on the blocked-handler wait. The healthy path never waits on it (the test
# releases the handler as soon as the probe returns); it only bounds how long a genuine
# `async def` regression takes to surface as a failure.
FAILURE_TIMEOUT_SECONDS = 30.0

# A trivial route with no auth dependency and no database access. If the loop is free, this
# answers no matter what else the server is doing.
PROBE_ROUTE = "/api/v1/app/version"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass
class BlockingServices:
    """The patched invoker plus the test's handles on its blocking behavior.

    - `started` is set the moment a slow handler begins executing (wherever that is: the
      threadpool if the route is a healthy `def`, the event loop if it regressed).
    - `release` is set by the test to let the handler return.
    - `finished` is set when the handler's wait ends - via `release`, or via the
      FAILURE_TIMEOUT_SECONDS backstop when a blocked loop prevented the release.
    """

    invoker: MagicMock
    started: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)

    def block(self) -> None:
        self.started.set()
        try:
            self.release.wait(timeout=FAILURE_TIMEOUT_SECONDS)
        finally:
            self.finished.set()


@pytest.fixture
def blocking_services(monkeypatch: pytest.MonkeyPatch) -> BlockingServices:
    """Point every router at services whose gallery/image reads block until released.

    Patching the attribute on the class itself covers all routers at once - they share the
    single `ApiDependencies` object rather than importing their own copy.
    """
    invoker = MagicMock()
    # A bare MagicMock attribute is truthy, which would put the auth dependencies into
    # multiuser mode and answer every request with 401 before the route is ever reached.
    invoker.services.configuration.multiuser = False

    services = BlockingServices(invoker=invoker)

    def slow_list_item_names(**_: object) -> GalleryItemNamesResult:
        services.block()
        return GalleryItemNamesResult(items=[], starred_count=0, total_count=0)

    def slow_get_image_names(**_: object) -> ImageNamesResult:
        services.block()
        return ImageNamesResult(image_names=[], starred_count=0, total_count=0)

    def slow_list_items(**_: object) -> OffsetPaginatedResults[GalleryItem]:
        services.block()
        return OffsetPaginatedResults[GalleryItem](limit=10, offset=0, total=0, items=[])

    def slow_get_item_names(**_: object) -> GalleryItemNames:
        services.block()
        return GalleryItemNames(item_names=[], starred_count=0, total_count=0)

    def slow_queue_item_summaries(**_: object) -> list[SessionQueueItemSummary]:
        services.block()
        return []

    invoker.services.gallery.list_item_names.side_effect = slow_list_item_names
    invoker.services.gallery.get_item_names.side_effect = slow_get_item_names
    invoker.services.gallery.list_items.side_effect = slow_list_items
    invoker.services.images.get_image_names.side_effect = slow_get_image_names
    invoker.services.session_queue.get_queue_item_summaries_by_ids.side_effect = slow_queue_item_summaries

    monkeypatch.setattr(ApiDependencies, "invoker", invoker, raising=False)
    return services


@pytest.mark.anyio
@pytest.mark.parametrize(
    "slow_route,params,json_body",
    [
        ("/api/v1/gallery/items/names", {}, None),
        ("/api/v1/gallery/items/names", {"search_term": "anything"}, None),
        ("/api/v1/gallery/item_names", {}, None),
        ("/api/v1/gallery/items/", {}, None),
        ("/api/v1/images/names", {}, None),
        ("/api/v1/queue/default/item_summaries_by_ids", {}, {"item_ids": [1, 2, 3]}),
    ],
)
async def test_slow_gallery_read_leaves_the_event_loop_free(
    blocking_services: BlockingServices, slow_route: str, params: dict, json_body: dict | None
) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        if json_body is None:
            slow_request = asyncio.create_task(client.get(slow_route, params=params))
        else:
            slow_request = asyncio.create_task(client.post(slow_route, params=params, json=json_body))

        try:
            # Wait until the slow handler is actually executing before probing, so the probe
            # is provably concurrent with it. `to_thread` keeps this wait off the event loop:
            # if the handler regressed to running ON the loop, this await can only resume
            # after the handler's timeout backstop - which the finished-flag assertion below
            # then reports as the failure, rather than the test deadlocking.
            handler_started = await asyncio.to_thread(blocking_services.started.wait, FAILURE_TIMEOUT_SECONDS)
            assert handler_started, f"{slow_route} never reached its (patched) service call."

            response = await client.get(PROBE_ROUTE)

            assert response.status_code == 200
            assert not blocking_services.finished.is_set(), (
                f"{PROBE_ROUTE} could not be answered until {slow_route}'s handler had finished, and the "
                f"handler is only released by this test AFTER the probe - so the route is executing its "
                f"blocking database work on the event loop, and the server answers nothing else while it "
                f"runs. Declare the route `def` instead of `async def`."
            )
            assert not slow_request.done(), (
                "The slow request completed before the probe, so nothing was measured concurrently - "
                "the event loop was blocked for its full duration."
            )
        finally:
            # Whatever happened above, let the handler (and the pending request) finish so
            # neither a threadpool worker nor the task outlives the test.
            blocking_services.release.set()
            await slow_request
