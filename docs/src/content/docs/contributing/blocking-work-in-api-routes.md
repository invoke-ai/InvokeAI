---
title: Blocking Work in API Routes
---

Almost every service in the backend is synchronous — the database layer, the model
manager, the file stores. The API layer in front of them is asynchronous. Getting the
boundary between the two wrong does not produce a slow endpoint; it produces a server
that stops answering entirely.

## The rule

**A route handler that only calls synchronous services must be declared `def`, not
`async def`.**

```python
# Correct — Starlette runs this in a worker thread.
@gallery_router.get("/items/names")
def get_gallery_item_names(current_user: CurrentUserOrDefault) -> GalleryItemNamesResult:
    return ApiDependencies.invoker.services.gallery.list_item_names(...)
```

```python
# Wrong — the database query runs on the event loop.
@gallery_router.get("/items/names")
async def get_gallery_item_names(current_user: CurrentUserOrDefault) -> GalleryItemNamesResult:
    return ApiDependencies.invoker.services.gallery.list_item_names(...)
```

The same rule applies to **dependencies**, not just handlers. A dependency declared
`async def` that performs a synchronous database lookup blocks the loop on every request
that uses it.

## Why it matters

The server runs as a single process with a single event loop. Anything executed directly
on that loop has the whole process to itself until it returns. Blocking work on the loop
therefore does not just delay its own response — for its entire duration the process
serves **no** other HTTP request and delivers **no** socket.io event. Users do not
experience this as one slow endpoint; they experience it as the application freezing,
typically mid-generation, because progress events stop arriving too.

The cost scales with the user's library, not with the developer's. A gallery query that
returns in milliseconds against a test database can take minutes against a multi-gigabyte
one — for example a metadata search, which has to read every row's metadata blob.

Declaring the handler `def` makes FastAPI dispatch it to a worker thread instead, leaving
the loop free to serve everything else.

## When `async def` is right

Use `async def` when the body actually awaits something — streaming a response, awaiting
another async API, or coordinating tasks. If such a handler *also* performs blocking work,
that work must be wrapped explicitly:

```python
from starlette.concurrency import run_in_threadpool

user = await run_in_threadpool(ApiDependencies.invoker.services.users.get, user_id)
```

`async def` with no `await` in the body is always a mistake: it gains nothing and costs
the loop.

## What this does not fix

Moving work to the threadpool does not make it faster, and it does not make it parallel.
The SQLite layer uses a single connection behind a process-wide lock, so database work
remains serialized regardless of which thread requests it. The benefit is confined to —
and this is the point — keeping everything *else* responsive while it runs.

## Testing it

Two tests cover this, and they do different jobs.

`tests/app/routers/test_no_blocking_async_routes.py` **enforces the rule**: it parses every
router module and fails if any route handler is `async def` without awaiting anything. This
is the one that catches a new route — a per-route test cannot, because the route does not
exist when the test is written.

`tests/app/routers/test_event_loop_blocking.py` **proves the effect** for a few
representative routes. It stubs a service method to block synchronously, issues a request
against the route under test, and asserts that an unrelated trivial route still answers
while that request is in flight.

Note what the second one measures: not the slow request's own duration, which the fix does
not change, but the latency of other requests during it. A benchmark of the slow endpoint
alone will show no improvement and is the wrong instrument here.

If you call a route handler directly from a test, call it like the plain function it now is
— no `await`, no `asyncio.run`.
