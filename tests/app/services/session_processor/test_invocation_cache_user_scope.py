"""The session runner scopes the shared invocation cache to each queue item's owner in multiuser mode."""

import pytest

from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
from tests.app.services.workflow_call_test_utils import _build_queue_item, _build_workflow_runner
from tests.test_invocation_cache_memory import CACHE_SCOPE_NODE_CLASSES, CacheScopeRecorderInvocation


# Node classes cover the default cache path and overrides that reach the cache through the dispatch
# services argument or through context._services, so run_node must scope both.
@pytest.mark.parametrize("node_class", CACHE_SCOPE_NODE_CLASSES)
@pytest.mark.parametrize("multiuser,expected_calls", [(True, ["alice", "bob"]), (False, ["alice"])])
def test_runner_scopes_cache_entries_to_the_queue_owner_only_in_multiuser(
    monkeypatch: pytest.MonkeyPatch,
    multiuser: bool,
    expected_calls: list[str],
    node_class: type[CacheScopeRecorderInvocation],
) -> None:
    runner, events, _ = _build_workflow_runner(monkeypatch)
    services = runner._services
    services.configuration.node_cache_size = 5
    services.configuration.multiuser = multiuser
    services.invocation_cache = MemoryInvocationCache(max_cache_size=5)
    CacheScopeRecorderInvocation.calls.clear()

    outputs = []
    for user_id in ("alice", "bob"):
        invocation = node_class(id=f"node-{user_id}", prompt="p")
        queue_item = _build_queue_item(invocation)
        queue_item.user_id = user_id
        runner.run_node(invocation=invocation, queue_item=queue_item)
        outputs.append(events.completed[-1][2].prompt)

    assert events.errors == []
    assert CacheScopeRecorderInvocation.calls == expected_calls
    # Single-user mode keeps sharing: bob's run is served alice's cached output.
    assert outputs == [f"p for {user}" for user in (["alice", "bob"] if multiuser else ["alice", "alice"])]
