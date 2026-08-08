import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from threading import Event, get_ident

import pytest
import torch

from invokeai.app.services.object_serializer.object_serializer_common import ObjectNotFoundError
from invokeai.app.services.object_serializer.object_serializer_disk import ObjectSerializerDisk
from invokeai.app.services.object_serializer.object_serializer_forward_cache import ObjectSerializerForwardCache


@dataclass
class MockDataclass:
    foo: str


def count_files(path: Path):
    return len(list(path.iterdir()))


@pytest.fixture
def obj_serializer(tmp_path: Path):
    return ObjectSerializerDisk[MockDataclass](tmp_path, safe_globals=[MockDataclass])


@pytest.fixture
def fwd_cache(tmp_path: Path):
    return ObjectSerializerForwardCache(
        ObjectSerializerDisk[MockDataclass](tmp_path, safe_globals=[MockDataclass]), max_cache_size=2
    )


def test_obj_serializer_disk_initializes(tmp_path: Path):
    obj_serializer = ObjectSerializerDisk[MockDataclass](tmp_path, safe_globals=[MockDataclass])
    assert obj_serializer._output_dir == tmp_path


def test_obj_serializer_disk_saves(obj_serializer: ObjectSerializerDisk[MockDataclass]):
    obj_1 = MockDataclass(foo="bar")
    obj_1_name = obj_serializer.save(obj_1)
    assert Path(obj_serializer._output_dir, obj_1_name).exists()

    obj_2 = MockDataclass(foo="baz")
    obj_2_name = obj_serializer.save(obj_2)
    assert Path(obj_serializer._output_dir, obj_2_name).exists()


def test_obj_serializer_disk_loads(obj_serializer: ObjectSerializerDisk[MockDataclass]):
    obj_1 = MockDataclass(foo="bar")
    obj_1_name = obj_serializer.save(obj_1)
    assert obj_serializer.load(obj_1_name).foo == "bar"

    obj_2 = MockDataclass(foo="baz")
    obj_2_name = obj_serializer.save(obj_2)
    assert obj_serializer.load(obj_2_name).foo == "baz"

    with pytest.raises(ObjectNotFoundError):
        obj_serializer.load("nonexistent_object_name")


def test_obj_serializer_disk_deletes(obj_serializer: ObjectSerializerDisk[MockDataclass]):
    obj_1 = MockDataclass(foo="bar")
    obj_1_name = obj_serializer.save(obj_1)

    obj_2 = MockDataclass(foo="bar")
    obj_2_name = obj_serializer.save(obj_2)

    obj_serializer.delete(obj_1_name)
    assert not Path(obj_serializer._output_dir, obj_1_name).exists()
    assert Path(obj_serializer._output_dir, obj_2_name).exists()


def test_obj_serializer_disk_delete_is_noop_when_object_is_missing(
    obj_serializer: ObjectSerializerDisk[MockDataclass],
):
    obj_serializer.delete("missing_object_name")


def test_obj_serializer_ephemeral_creates_tempdir(tmp_path: Path):
    obj_serializer = ObjectSerializerDisk[MockDataclass](tmp_path, safe_globals=[MockDataclass], ephemeral=True)
    assert isinstance(obj_serializer._tempdir, tempfile.TemporaryDirectory)
    assert obj_serializer._base_output_dir == tmp_path
    assert obj_serializer._output_dir != tmp_path
    assert obj_serializer._output_dir == Path(obj_serializer._tempdir.name)


def test_obj_serializer_ephemeral_deletes_tempdir(tmp_path: Path):
    obj_serializer = ObjectSerializerDisk[MockDataclass](tmp_path, safe_globals=[MockDataclass], ephemeral=True)
    tempdir_path = obj_serializer._output_dir
    del obj_serializer
    assert not tempdir_path.exists()


def test_obj_serializer_ephemeral_deletes_tempdir_on_stop(tmp_path: Path):
    obj_serializer = ObjectSerializerDisk[MockDataclass](tmp_path, safe_globals=[MockDataclass], ephemeral=True)
    tempdir_path = obj_serializer._output_dir
    obj_serializer.stop(None)  # pyright: ignore [reportArgumentType]
    assert not tempdir_path.exists()


def test_obj_serializer_ephemeral_writes_to_tempdir(tmp_path: Path):
    obj_serializer = ObjectSerializerDisk[MockDataclass](tmp_path, safe_globals=[MockDataclass], ephemeral=True)
    obj_1 = MockDataclass(foo="bar")
    obj_1_name = obj_serializer.save(obj_1)
    assert Path(obj_serializer._output_dir, obj_1_name).exists()
    assert not Path(tmp_path, obj_1_name).exists()


def test_obj_serializer_ephemeral_deletes_dangling_tempdirs_on_init(tmp_path: Path):
    tempdir = tmp_path / "tmpdir"
    tempdir.mkdir()
    ObjectSerializerDisk[MockDataclass](tmp_path, safe_globals=[MockDataclass], ephemeral=True)
    assert not tempdir.exists()


def test_obj_serializer_does_not_delete_tempdirs_on_init(tmp_path: Path):
    tempdir = tmp_path / "tmpdir"
    tempdir.mkdir()
    ObjectSerializerDisk[MockDataclass](tmp_path, safe_globals=[MockDataclass], ephemeral=False)
    assert tempdir.exists()


def test_obj_serializer_disk_different_types(tmp_path: Path):
    obj_serializer_1 = ObjectSerializerDisk[MockDataclass](tmp_path, safe_globals=[MockDataclass])
    obj_1 = MockDataclass(foo="bar")
    obj_1_name = obj_serializer_1.save(obj_1)
    obj_1_loaded = obj_serializer_1.load(obj_1_name)
    assert obj_serializer_1._obj_class_name == "MockDataclass"
    assert isinstance(obj_1_loaded, MockDataclass)
    assert obj_1_loaded.foo == "bar"
    assert obj_1_name.startswith("MockDataclass_")

    obj_serializer_2 = ObjectSerializerDisk[int](tmp_path, safe_globals=[int])
    obj_2_name = obj_serializer_2.save(9001)
    assert obj_serializer_2._obj_class_name == "int"
    assert obj_serializer_2.load(obj_2_name) == 9001
    assert obj_2_name.startswith("int_")

    obj_serializer_3 = ObjectSerializerDisk[str](tmp_path, safe_globals=[str])
    obj_3_name = obj_serializer_3.save("foo")
    assert obj_serializer_3._obj_class_name == "str"
    assert obj_serializer_3.load(obj_3_name) == "foo"
    assert obj_3_name.startswith("str_")

    obj_serializer_4 = ObjectSerializerDisk[torch.Tensor](tmp_path, safe_globals=[torch.Tensor])
    obj_4_name = obj_serializer_4.save(torch.tensor([1, 2, 3]))
    obj_4_loaded = obj_serializer_4.load(obj_4_name)
    assert obj_serializer_4._obj_class_name == "Tensor"
    assert isinstance(obj_4_loaded, torch.Tensor)
    assert torch.equal(obj_4_loaded, torch.tensor([1, 2, 3]))
    assert obj_4_name.startswith("Tensor_")


def test_obj_serializer_fwd_cache_initializes(obj_serializer: ObjectSerializerDisk[MockDataclass]):
    fwd_cache = ObjectSerializerForwardCache(obj_serializer)
    assert fwd_cache._underlying_storage == obj_serializer


def test_obj_serializer_fwd_cache_saves_and_loads(fwd_cache: ObjectSerializerForwardCache[MockDataclass]):
    obj = MockDataclass(foo="bar")
    obj_name = fwd_cache.save(obj)
    obj_loaded = fwd_cache.load(obj_name)
    obj_underlying = fwd_cache._underlying_storage.load(obj_name)
    assert obj_loaded == obj_underlying
    assert obj_loaded.foo == "bar"


def test_obj_serializer_fwd_cache_respects_cache_size(fwd_cache: ObjectSerializerForwardCache[MockDataclass]):
    obj_1 = MockDataclass(foo="bar")
    obj_1_name = fwd_cache.save(obj_1)
    obj_2 = MockDataclass(foo="baz")
    obj_2_name = fwd_cache.save(obj_2)
    obj_3 = MockDataclass(foo="qux")
    obj_3_name = fwd_cache.save(obj_3)
    assert obj_1_name not in fwd_cache._cache
    assert obj_2_name in fwd_cache._cache
    assert obj_3_name in fwd_cache._cache
    # apparently qsize is "not reliable"?
    assert fwd_cache._cache_ids.qsize() == 2


def test_obj_serializer_fwd_cache_calls_delete_callback(fwd_cache: ObjectSerializerForwardCache[MockDataclass]):
    called_name = None
    obj_1 = MockDataclass(foo="bar")

    def on_deleted(name: str):
        nonlocal called_name
        called_name = name

    fwd_cache.on_deleted(on_deleted)
    obj_1_name = fwd_cache.save(obj_1)
    fwd_cache.delete(obj_1_name)
    assert called_name == obj_1_name


def test_obj_serializer_fwd_cache_removes_deleted_ids_from_eviction_queue(
    fwd_cache: ObjectSerializerForwardCache[MockDataclass],
):
    obj_1_name = fwd_cache.save(MockDataclass(foo="bar"))
    obj_2_name = fwd_cache.save(MockDataclass(foo="baz"))
    fwd_cache.delete(obj_1_name)

    obj_3_name = fwd_cache.save(MockDataclass(foo="qux"))

    assert obj_1_name not in fwd_cache._cache
    assert obj_2_name in fwd_cache._cache
    assert obj_3_name in fwd_cache._cache
    assert fwd_cache._cache_ids.qsize() == 2


def test_obj_serializer_fwd_cache_cleans_up_when_storage_object_is_missing(
    fwd_cache: ObjectSerializerForwardCache[MockDataclass],
):
    called_names: list[str] = []
    fwd_cache.on_deleted(called_names.append)
    obj_name = fwd_cache.save(MockDataclass(foo="bar"))
    underlying_storage = fwd_cache._underlying_storage
    assert isinstance(underlying_storage, ObjectSerializerDisk)
    underlying_storage._get_path(obj_name).unlink()

    fwd_cache.delete(obj_name)

    assert obj_name not in fwd_cache._cache
    assert fwd_cache._cache_ids.qsize() == 0
    assert called_names == [obj_name]


def test_obj_serializer_fwd_cache_preserves_fifo_order_after_deletion(tmp_path: Path):
    fwd_cache = ObjectSerializerForwardCache(
        ObjectSerializerDisk[MockDataclass](tmp_path, safe_globals=[MockDataclass]), max_cache_size=3
    )
    obj_1_name = fwd_cache.save(MockDataclass(foo="one"))
    obj_2_name = fwd_cache.save(MockDataclass(foo="two"))
    obj_3_name = fwd_cache.save(MockDataclass(foo="three"))
    fwd_cache.delete(obj_2_name)
    obj_4_name = fwd_cache.save(MockDataclass(foo="four"))
    obj_5_name = fwd_cache.save(MockDataclass(foo="five"))

    assert obj_1_name not in fwd_cache._cache
    assert obj_2_name not in fwd_cache._cache
    assert obj_3_name in fwd_cache._cache
    assert obj_4_name in fwd_cache._cache
    assert obj_5_name in fwd_cache._cache
    assert fwd_cache._cache_ids.qsize() == 3


def test_obj_serializer_fwd_cache_delete_of_evicted_object_is_noop(
    fwd_cache: ObjectSerializerForwardCache[MockDataclass],
):
    obj_1_name = fwd_cache.save(MockDataclass(foo="one"))
    obj_2_name = fwd_cache.save(MockDataclass(foo="two"))
    obj_3_name = fwd_cache.save(MockDataclass(foo="three"))
    cache_before_delete = fwd_cache._cache.copy()
    queue_size_before_delete = fwd_cache._cache_ids.qsize()

    fwd_cache.delete(obj_1_name)

    assert fwd_cache._cache == cache_before_delete
    assert fwd_cache._cache_ids.qsize() == queue_size_before_delete
    assert obj_2_name in fwd_cache._cache
    assert obj_3_name in fwd_cache._cache


def test_obj_serializer_fwd_cache_concurrent_deletes_do_not_leave_stale_eviction_ids(
    fwd_cache: ObjectSerializerForwardCache[MockDataclass], monkeypatch: pytest.MonkeyPatch
):
    obj_1_name = fwd_cache.save(MockDataclass(foo="one"))
    obj_2_name = fwd_cache.save(MockDataclass(foo="two"))
    first_delete_thread_id: int | None = None
    first_drained_queue = Event()
    release_first_delete = Event()
    second_delete_started = Event()
    second_delete_finished = Event()
    original_get_nowait = fwd_cache._cache_ids.get_nowait

    def controlled_get_nowait():
        try:
            return original_get_nowait()
        except Empty:
            if get_ident() == first_delete_thread_id:
                first_drained_queue.set()
                assert release_first_delete.wait(timeout=5)
            raise

    monkeypatch.setattr(fwd_cache._cache_ids, "get_nowait", controlled_get_nowait)

    def delete_first():
        nonlocal first_delete_thread_id
        first_delete_thread_id = get_ident()
        fwd_cache.delete(obj_1_name)

    def delete_second():
        second_delete_started.set()
        fwd_cache.delete(obj_2_name)
        second_delete_finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_delete = executor.submit(delete_first)
        assert first_drained_queue.wait(timeout=5)
        second_delete = executor.submit(delete_second)
        assert second_delete_started.wait(timeout=5)
        try:
            assert not second_delete_finished.wait(timeout=0.1)
        finally:
            release_first_delete.set()
        first_delete.result(timeout=5)
        second_delete.result(timeout=5)

    assert fwd_cache._cache == {}
    assert fwd_cache._cache_ids.qsize() == 0

    obj_3_name = fwd_cache.save(MockDataclass(foo="three"))
    obj_4_name = fwd_cache.save(MockDataclass(foo="four"))

    assert set(fwd_cache._cache) == {obj_3_name, obj_4_name}
    assert fwd_cache._cache_ids.qsize() == 2
