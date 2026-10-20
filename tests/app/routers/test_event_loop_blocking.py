"""Guards that slow gallery queries do not stall the whole server.

The gallery list/name routes run synchronous SQLite work. If such a route is declared
`async def`, that work executes *on the event loop*, so for its whole duration the process
serves nothing else - no other HTTP request, no socket.io progress event. On a large
library a single search can take minutes, which users experience as the backend being
dead rather than as a slow search.

Declaring these routes `def` hands them to Starlette's threadpool instead, leaving the
loop free. These tests pin that property down: the slowness is simulated with a
synchronous sleep in the service layer, so they assert the *dispatch mechanism* rather
than any particular query's speed, and stay fast and deterministic in CI.
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api_app import app
from invokeai.app.services.gallery.gallery_common import GalleryItem, GalleryItemNames, GalleryItemNamesResult
from invokeai.app.services.image_records.image_records_common import ImageNamesResult
from invokeai.app.services.session_queue.session_queue_common import SessionQueueItemSummary
from invokeai.app.services.shared.pagination import OffsetPaginatedResults

# Long enough that a blocked event loop is unmistakable, short enough to keep the suite fast.
BLOCKING_SECONDS = 1.0

# A trivial route with no auth dependency and no database access. If the loop is free, this
# answers in single-digit milliseconds no matter what else the server is doing.
PROBE_ROUTE = "/api/v1/app/version"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def blocking_invoker(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Point every router at services whose gallery/image reads block for BLOCKING_SECONDS.

    Patching the attribute on the class itself covers all routers at once - they share the
    single `ApiDependencies` object rather than importing their own copy.
    """
    invoker = MagicMock()
    # A bare MagicMock attribute is truthy, which would put the auth dependencies into
    # multiuser mode and answer every request with 401 before the route is ever reached.
    invoker.services.configuration.multiuser = False

    def slow_list_item_names(**_: object) -> GalleryItemNamesResult:
        time.sleep(BLOCKING_SECONDS)
        return GalleryItemNamesResult(items=[], starred_count=0, total_count=0)

    def slow_get_image_names(**_: object) -> ImageNamesResult:
        time.sleep(BLOCKING_SECONDS)
        return ImageNamesResult(image_names=[], starred_count=0, total_count=0)

    def slow_list_items(**_: object) -> OffsetPaginatedResults[GalleryItem]:
        time.sleep(BLOCKING_SECONDS)
        return OffsetPaginatedResults[GalleryItem](limit=10, offset=0, total=0, items=[])

    def slow_get_item_names(**_: object) -> GalleryItemNames:
        time.sleep(BLOCKING_SECONDS)
        return GalleryItemNames(item_names=[], starred_count=0, total_count=0)

    def slow_queue_item_summaries(**_: object) -> list[SessionQueueItemSummary]:
        time.sleep(BLOCKING_SECONDS)
        return []

    invoker.services.gallery.list_item_names.side_effect = slow_list_item_names
    invoker.services.gallery.get_item_names.side_effect = slow_get_item_names
    invoker.services.gallery.list_items.side_effect = slow_list_items
    invoker.services.images.get_image_names.side_effect = slow_get_image_names
    invoker.services.session_queue.get_queue_item_summaries_by_ids.side_effect = slow_queue_item_summaries

    monkeypatch.setattr(ApiDependencies, "invoker", invoker, raising=False)
    return invoker


async def _probe_latency_while_busy(
    client: AsyncClient, slow_route: str, params: dict, json_body: dict | None = None
) -> tuple[float, asyncio.Task]:
    """Start `slow_route`, then time a probe request issued while it is still running.

    The clock starts before yielding to the slow request, so a blocked loop shows up as
    probe latency even though the probe itself never got a chance to be dispatched.
    """
    started = time.perf_counter()
    if json_body is None:
        slow_request = asyncio.create_task(client.get(slow_route, params=params))
    else:
        slow_request = asyncio.create_task(client.post(slow_route, params=params, json=json_body))
    # Hand control to the slow request so it reaches its route handler before we probe.
    for _ in range(10):
        await asyncio.sleep(0)

    response = await client.get(PROBE_ROUTE)
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    return elapsed, slow_request


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
    blocking_invoker: MagicMock, slow_route: str, params: dict, json_body: dict | None
) -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        elapsed, slow_request = await _probe_latency_while_busy(client, slow_route, params, json_body)

        assert elapsed < BLOCKING_SECONDS / 2, (
            f"{PROBE_ROUTE} took {elapsed:.2f}s while {slow_route} was running. The slow route is "
            f"executing its blocking database work on the event loop, so the server answers "
            f"nothing else until it finishes. Declare the route `def` instead of `async def`."
        )
        assert not slow_request.done(), (
            "The slow request finished before the probe was even dispatched, so nothing was "
            "measured concurrently - the event loop was blocked for its full duration."
        )

        await slow_request
