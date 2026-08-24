from invokeai.app.services.events.events_common import EventBase, UserAccessChangedEvent


def test_get_events_excludes_server_internal_events() -> None:
    """Server-internal events must stay out of the generated client API schema.

    `EventBase.get_events()` feeds the OpenAPI generator, so any event it returns ends up
    in `openapi.json`/`schema.ts`. `UserAccessChangedEvent` is dispatched only between
    server components and is never delivered to clients.
    """
    assert UserAccessChangedEvent not in EventBase.get_events()


def test_get_events_includes_client_facing_events() -> None:
    event_names = {event.__event_name__ for event in EventBase.get_events()}

    assert "invocation_complete" in event_names
    assert "queue_item_status_changed" in event_names
