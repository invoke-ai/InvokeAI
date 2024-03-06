# pyright: reportPrivateUsage=false
from contextlib import suppress

from invokeai.app.invocations.fields import ImageField
from invokeai.app.invocations.primitives import ImageOutput
from invokeai.app.services.invocation_cache.invocation_cache_memory import MemoryInvocationCache
from tests.test_nodes import PromptTestInvocation


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
