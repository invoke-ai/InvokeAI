from invokeai.app.services.events.events_common import InvocationProgressEvent
from invokeai.app.services.progress_previews.progress_previews_default import (
    PROGRESS_IMAGE_MIN_INTERVAL_S,
    MemoryProgressPreviews,
)

_COMMON_FIELDS = {
    "queue_id": "default",
    "batch_id": "batch-1",
    "invocation": {"type": "add", "id": "node-1", "a": 1, "b": 2},
    "invocation_source_id": "node-1",
    "message": "Denoising",
    "image": {"width": 64, "height": 64, "dataURL": "data:image/jpeg;base64,frame"},
}


def _event(item_id: int, user_id: str = "owner-1", session_id: str = "session-1", **overrides):
    return InvocationProgressEvent(
        **{**_COMMON_FIELDS, "item_id": item_id, "user_id": user_id, "session_id": session_id, **overrides}
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_revisions_are_monotonic_per_item_and_restart_with_a_new_session() -> None:
    clock = _Clock()
    store = MemoryProgressPreviews(clock=clock)

    first = store.reserve_revision(1, "session-1", 0.1)
    clock.now += 1
    second = store.reserve_revision(1, "session-1", 0.2)
    clock.now += 1
    other_item = store.reserve_revision(2, "session-2", 0.1)
    clock.now += 1
    new_session = store.reserve_revision(1, "session-1b", 0.1)

    assert (first, second) == (1, 2)
    assert other_item == 1
    # A new session on the same item is a new epoch: the client resets its gate on session change.
    assert new_session == 1


def test_frames_inside_the_interval_are_dropped_except_final_and_indeterminate_ones() -> None:
    clock = _Clock()
    store = MemoryProgressPreviews(clock=clock)

    assert store.reserve_revision(1, "session-1", 0.1) == 1
    clock.now += PROGRESS_IMAGE_MIN_INTERVAL_S / 2
    assert store.reserve_revision(1, "session-1", 0.2) is None
    # A final frame is never dropped: it is the one the preview settles on.
    assert store.reserve_revision(1, "session-1", 1.0) == 2
    # Nor an indeterminate one.
    assert store.reserve_revision(1, "session-1", None) == 3
    clock.now += PROGRESS_IMAGE_MIN_INTERVAL_S * 1.5
    assert store.reserve_revision(1, "session-1", 0.3) == 4


def test_throttle_is_per_item() -> None:
    clock = _Clock()
    store = MemoryProgressPreviews(clock=clock)

    assert store.reserve_revision(1, "session-1", 0.1) == 1
    # Another item (another GPU) is not held back by the first one's frame.
    assert store.reserve_revision(2, "session-2", 0.1) == 1


def test_records_the_latest_event_and_lists_only_the_owners_running_items() -> None:
    store = MemoryProgressPreviews()

    store.record(_event(1, revision=1))
    store.record(_event(1, revision=2))
    store.record(_event(2, user_id="owner-2", session_id="session-2", revision=1))
    store.record(_event(3, queue_id="other", session_id="session-3", revision=1))

    assert store.get(1) is not None and store.get(1).revision == 2
    assert [event.item_id for event in store.list_for_user("owner-1")] == [1, 3]
    assert [event.item_id for event in store.list_for_user("owner-1", "default")] == [1]
    assert [event.item_id for event in store.list_for_user("owner-2")] == [2]
    assert store.list_for_user("stranger") == []


def test_clear_forgets_the_frame_but_keeps_the_revision_counter() -> None:
    # A workflow-call parent suspends (session leg ends, frame cleared) and later resumes with
    # the same item and session. Its next frames must outrank the ones a client already holds.
    clock = _Clock()
    store = MemoryProgressPreviews(clock=clock)

    assert store.reserve_revision(1, "session-1", 0.1) == 1
    store.record(_event(1, revision=1))

    store.clear(1)

    assert store.get(1) is None
    assert store.list_for_user("owner-1") == []
    # No throttle carries over, but the counter does.
    assert store.reserve_revision(1, "session-1", 0.1) == 2

    store.record(_event(1, revision=2))
    store.clear_all()
    assert store.list_for_user("owner-1") == []
    assert store.reserve_revision(1, "session-1", 0.1) == 1


def test_clear_node_drops_the_frame_only_if_that_node_produced_it() -> None:
    # Once the denoise node completes, replaying its frame would show a finished denoise as
    # still running. A later node's frame must not be dropped by the earlier node's completion.
    store = MemoryProgressPreviews()
    store.record(_event(1, revision=1))

    store.clear_node(1, "other-node")
    assert store.get(1) is not None

    store.clear_node(1, "node-1")
    assert store.get(1) is None
    assert store.list_for_user("owner-1") == []


def test_frameless_counters_are_bounded() -> None:
    from invokeai.app.services.progress_previews import progress_previews_default

    store = MemoryProgressPreviews()
    limit = progress_previews_default.COUNTER_RETENTION_LIMIT
    for item_id in range(limit + 10):
        store.reserve_revision(item_id, f"session-{item_id}", 0.1)
        store.clear(item_id)

    assert len(store._items) == limit
    # The oldest counters went; the newest survive.
    assert 0 not in store._items
    assert limit + 9 in store._items


def test_reserved_revision_survives_a_record_with_the_same_session() -> None:
    clock = _Clock()
    store = MemoryProgressPreviews(clock=clock)

    revision = store.reserve_revision(1, "session-1", 0.1)
    store.record(_event(1, revision=revision))
    clock.now += 1

    assert store.reserve_revision(1, "session-1", 0.2) == 2
