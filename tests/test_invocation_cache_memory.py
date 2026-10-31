# pyright: reportPrivateUsage=false
from contextlib import suppress
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.baseinvocation import BaseInvocation, invocation, services_for_queue_user
from invokeai.app.invocations.fields import ImageField, InputField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
from invokeai.app.services.shared.invocation_context import InvocationContext
from tests.test_nodes import PromptTestInvocation, PromptTestInvocationOutput


def test_invocation_cache_memory_max_cache_size():
    cache = MemoryInvocationCache()
    assert cache._max_cache_size == 0
    output_1 = ImageOutput(image=ImageField(image_name="foo"), width=512, height=512)
    cache.save(1, output_1)
    assert cache.get(1) is None
    assert cache._hits == 0
    assert cache._misses == 0  # TODO: when cache size is zero, should we consider it a miss?
    assert len(cache._cache) == 0


def test_invocation_cache_memory_creates_deterministic_keys():
    hash1 = MemoryInvocationCache.create_key(PromptTestInvocation(prompt="foo"))
    hash2 = MemoryInvocationCache.create_key(PromptTestInvocation(prompt="foo"))
    hash3 = MemoryInvocationCache.create_key(PromptTestInvocation(prompt="bar"))

    assert hash1 == hash2
    assert hash1 != hash3


def test_invocation_cache_memory_adds_invocation():
    output_1 = ImageOutput(image=ImageField(image_name="foo"), width=512, height=512)
    output_2 = ImageOutput(image=ImageField(image_name="bar"), width=512, height=512)
    cache = MemoryInvocationCache(max_cache_size=5)
    cache.save(1, output_1)
    cache.save(2, output_2)
    assert cache.get(1) == output_1
    assert cache.get(2) == output_2


def test_invocation_cache_memory_tracks_hits():
    output_1 = ImageOutput(image=ImageField(image_name="foo"), width=512, height=512)
    cache = MemoryInvocationCache(max_cache_size=5)
    cache.save(1, output_1)
    cache.get(1)  # hit
    cache.get(1)  # hit
    cache.get(1)  # hit
    cache.get(2)  # miss
    cache.get(2)  # miss
    assert cache._hits == 3
    assert cache._misses == 2


def test_invocation_cache_memory_is_lru():
    output_1 = ImageOutput(image=ImageField(image_name="foo"), width=512, height=512)
    output_2 = ImageOutput(image=ImageField(image_name="bar"), width=512, height=512)
    output_3 = ImageOutput(image=ImageField(image_name="baz"), width=512, height=512)
    cache = MemoryInvocationCache(max_cache_size=2)
    cache.save(1, output_1)
    cache.save(2, output_2)
    cache.save(3, output_3)
    assert cache.get(1) is None
    assert cache.get(2) == output_2
    assert cache.get(3) == output_3
    assert len(cache._cache) == 2
    assert list(cache._cache.keys()) == [2, 3]
    cache.get(2)
    assert list(cache._cache.keys()) == [3, 2]


def test_invocation_cache_memory_disables_and_enables():
    output_1 = ImageOutput(image=ImageField(image_name="foo"), width=512, height=512)
    output_2 = ImageOutput(image=ImageField(image_name="bar"), width=512, height=512)
    cache = MemoryInvocationCache(max_cache_size=2)
    cache.save(1, output_1)
    cache.disable()
    assert cache.get(1) is None
    cache.save(2, output_2)
    assert cache.get(2) is None
    assert len(cache._cache) == 1
    assert cache._hits == 0
    assert cache._misses == 0
    cache.enable()
    cache.save(2, output_2)
    assert cache.get(2) is output_2
    assert len(cache._cache) == 2
    assert cache._hits == 1
    assert cache._misses == 0


def test_invocation_cache_memory_deletes_by_match():
    # The _delete_by_match method attempts to log but the logger is not set up in the test environment
    with suppress(AttributeError):
        cache = MemoryInvocationCache(max_cache_size=5)
        output_1 = ImageOutput(image=ImageField(image_name="foo"), width=512, height=512)
        output_2 = ImageOutput(image=ImageField(image_name="bar"), width=512, height=512)
        output_3 = ImageOutput(image=ImageField(image_name="baz"), width=512, height=512)
        cache.save(1, output_1)
        cache.save(2, output_2)
        cache.save(3, output_3)
        cache._delete_by_match("bar")
        assert cache.get(1) == output_1
        assert cache.get(2) is None
        assert cache.get(3) == output_3
        assert len(cache._cache) == 2
        assert list(cache._cache.keys()) == [1, 3]
        cache._delete_by_match("foo")
        assert cache.get(1) is None
        assert cache.get(2) is None
        assert cache.get(3) == output_3
        assert len(cache._cache) == 1
        assert list(cache._cache.keys()) == [3]
        cache._delete_by_match("baz")
        assert cache.get(1) is None
        assert cache.get(2) is None
        assert cache.get(3) is None
        assert len(cache._cache) == 0
        assert list(cache._cache.keys()) == []
        # shouldn't raise on empty cache
        cache._delete_by_match("foo")


def test_invocation_cache_memory_clears():
    cache = MemoryInvocationCache(max_cache_size=5)
    output_1 = ImageOutput(image=ImageField(image_name="foo"), width=512, height=512)
    output_2 = ImageOutput(image=ImageField(image_name="bar"), width=512, height=512)
    output_3 = ImageOutput(image=ImageField(image_name="baz"), width=512, height=512)
    cache.save(1, output_1)
    cache.save(2, output_2)
    cache.save(3, output_3)
    cache.get(1)
    cache.get(2)
    cache.get(3)
    cache.get("foo")  # miss
    cache.get("bar")  # miss
    cache.clear()
    assert len(cache._cache) == 0
    assert cache._hits == 0
    assert cache._misses == 0
    assert cache._misses == 0
    assert cache.get(1) is None
    assert cache.get(2) is None
    assert cache.get(3) is None


def test_invocation_cache_memory_status():
    cache = MemoryInvocationCache(max_cache_size=5)
    output_1 = ImageOutput(image=ImageField(image_name="foo"), width=512, height=512)
    output_2 = ImageOutput(image=ImageField(image_name="bar"), width=512, height=512)
    output_3 = ImageOutput(image=ImageField(image_name="baz"), width=512, height=512)
    cache.save(1, output_1)
    cache.save(2, output_2)
    cache.save(3, output_3)
    cache.get(1)
    cache.get(2)
    cache.get(3)
    cache.get("foo")  # miss
    cache.get("bar")  # miss
    status = cache.get_status()
    assert status.hits == 3
    assert status.misses == 2
    assert status.enabled
    assert status.size == 3
    assert status.max_size == 5
    cache.disable()
    status = cache.get_status()
    assert not status.enabled
    cache.enable()
    status = cache.get_status()
    assert status.enabled
    cache.clear()
    status = cache.get_status()
    assert status.size == 0
    assert status.hits == 0
    assert status.misses == 0
    assert status.enabled
    assert status.max_size == 5
    cache._max_cache_size = 0  # cache should be disabled when max_cache_size is zero
    status = cache.get_status()
    assert not status.enabled
    assert status.size == 0
    assert status.hits == 0
    assert status.misses == 0
    assert status.max_size == 0


def test_invocation_cache_memory_purges_video_outputs_on_video_deletion():
    """Regression: without the videos.on_deleted hook, a cached VideoOutput outlives the
    video's deletion — re-running an identical graph then "succeeds" with a 404 video."""
    from unittest.mock import MagicMock

    from invokeai.app.invocations.fields import VideoField
    from invokeai.app.invocations.primitives import VideoOutput

    cache = MemoryInvocationCache(max_cache_size=5)
    invoker = MagicMock()
    cache.start(invoker)
    # The video service's on_deleted must be wired to the same purge callback as images.
    invoker.services.videos.on_deleted.assert_called_once_with(cache._delete_by_match)
    invoker.services.images.on_deleted.assert_called_once_with(cache._delete_by_match)

    output = VideoOutput(
        video=VideoField(video_name="clip.mp4"), width=832, height=480, num_frames=81, fps=16.0, duration=5.0625
    )
    cache.save(1, output)
    cache._delete_by_match("clip.mp4")
    assert cache.get(1) is None
    assert len(cache._cache) == 0


@invocation("test_cache_scope_recorder", version="1.0.0", use_cache=True)
class CacheScopeRecorderInvocation(BaseInvocation):
    """Records which queueing user each real invoke() ran for."""

    prompt: str = InputField(default="")
    calls: ClassVar[list] = []

    def invoke(self, context: InvocationContext) -> PromptTestInvocationOutput:
        user_id = context._data.queue_item.user_id
        type(self).calls.append(user_id)
        return PromptTestInvocationOutput(prompt=f"{self.prompt} for {user_id}")


@invocation("test_cache_scope_custom_override", version="1.0.0", use_cache=True)
class CacheScopeCustomOverrideInvocation(CacheScopeRecorderInvocation):
    """A node pack that owns its cache logic, keying it through services.invocation_cache.create_key."""

    def invoke_internal(self, context: InvocationContext, services) -> PromptTestInvocationOutput:
        key = services.invocation_cache.create_key(self)
        cached = services.invocation_cache.get(key)
        if cached is None:
            cached = self.invoke(context)
            services.invocation_cache.save(key, cached)
        return cached


@invocation("test_cache_scope_context_services_override", version="1.0.0", use_cache=True)
class CacheScopeContextServicesOverrideInvocation(CacheScopeRecorderInvocation):
    """A node pack that owns its cache logic through context._services, ignoring the services argument."""

    def invoke_internal(self, context: InvocationContext, services) -> PromptTestInvocationOutput:
        cache = context._services.invocation_cache
        key = cache.create_key(self)
        cached = cache.get(key)
        if cached is None:
            cached = self.invoke(context)
            cache.save(key, cached)
        return cached


CACHE_SCOPE_NODE_CLASSES = [
    CacheScopeRecorderInvocation,
    CacheScopeCustomOverrideInvocation,
    CacheScopeContextServicesOverrideInvocation,
]


def _cache_services(*, multiuser: bool) -> SimpleNamespace:
    return SimpleNamespace(
        configuration=SimpleNamespace(node_cache_size=5, multiuser=multiuser),
        invocation_cache=MemoryInvocationCache(max_cache_size=5),
        intermediates=MagicMock(),
        logger=MagicMock(),
    )


def _run(
    services: SimpleNamespace,
    user_id: str | None,
    session_id: str = "session",
    node_class: type[CacheScopeRecorderInvocation] = CacheScopeRecorderInvocation,
) -> str:
    """Dispatch one node the way the session processor does, for a queue item owned by `user_id`."""
    node = node_class(id=f"node-{len(node_class.calls)}", prompt="p")
    node_services = services_for_queue_user(services, user_id)  # type: ignore[arg-type]
    # Like run_node, the context carries the same per-user services as the dispatch.
    context = SimpleNamespace(
        _data=SimpleNamespace(queue_item=SimpleNamespace(user_id=user_id, session_id=session_id)),
        _services=node_services,
        _skip_invocation_cache=False,
    )
    return node.invoke_internal_with_effects(context, node_services).output.prompt  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _reset_recorder_calls():
    CacheScopeRecorderInvocation.calls.clear()


@pytest.mark.parametrize("node_class", CACHE_SCOPE_NODE_CLASSES)
def test_multiuser_cache_hits_never_cross_users(node_class: type[CacheScopeRecorderInvocation]):
    services = _cache_services(multiuser=True)

    assert _run(services, "alice", node_class=node_class) == "p for alice"
    # Bob's identical node must run invoke() rather than receive alice's output.
    assert _run(services, "bob", node_class=node_class) == "p for bob"
    assert CacheScopeRecorderInvocation.calls == ["alice", "bob"]


@pytest.mark.parametrize("node_class", CACHE_SCOPE_NODE_CLASSES)
def test_multiuser_cache_still_hits_for_the_same_user(node_class: type[CacheScopeRecorderInvocation]):
    services = _cache_services(multiuser=True)

    assert _run(services, "alice", "s1", node_class) == "p for alice"
    assert _run(services, "alice", "s2", node_class) == "p for alice"
    assert CacheScopeRecorderInvocation.calls == ["alice"]


@pytest.mark.parametrize("node_class", CACHE_SCOPE_NODE_CLASSES)
@pytest.mark.parametrize("user_id", [None, ""])
def test_multiuser_cache_is_bypassed_without_a_queueing_user(
    user_id: str | None, node_class: type[CacheScopeRecorderInvocation]
):
    services = _cache_services(multiuser=True)

    _run(services, user_id, node_class=node_class)
    _run(services, user_id, node_class=node_class)
    assert CacheScopeRecorderInvocation.calls == [user_id, user_id]
    assert len(services.invocation_cache._cache) == 0


def test_single_user_cache_is_shared_across_sessions():
    services = _cache_services(multiuser=False)

    assert services_for_queue_user(services, "system") is services  # type: ignore[arg-type]
    assert _run(services, "system", "s1") == "p for system"
    assert _run(services, "other", "s2") == "p for system"
    assert CacheScopeRecorderInvocation.calls == ["system"]
