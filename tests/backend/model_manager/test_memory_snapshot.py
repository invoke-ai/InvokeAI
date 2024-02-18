import pytest

from invokeai.backend.model_manager.load.memory_snapshot import MemorySnapshot, get_pretty_snapshot_diff
from invokeai.backend.model_manager.util.libc_util import Struct_mallinfo2


def test_memory_snapshot_capture():
    """Smoke test of MemorySnapshot.capture()."""
    snapshot = MemorySnapshot.capture()

    # We just check process_ram, because it is the only field that should be supported on all platforms.
    assert snapshot.process_ram > 0


snapshots = [
    MemorySnapshot(process_ram=1, vram=2, malloc_info=Struct_mallinfo2()),
    MemorySnapshot(process_ram=1, vram=2, malloc_info=None),
    MemorySnapshot(process_ram=1, vram=None, malloc_info=Struct_mallinfo2()),
    MemorySnapshot(process_ram=1, vram=None, malloc_info=None),
    None,
]


@pytest.mark.parametrize("snapshot_1", snapshots)
@pytest.mark.parametrize("snapshot_2", snapshots)
def test_get_pretty_snapshot_diff(snapshot_1, snapshot_2):
    """Test that get_pretty_snapshot_diff() works with various combinations of missing MemorySnapshot fields."""
    msg = get_pretty_snapshot_diff(snapshot_1, snapshot_2)
    print(msg)

    expected_lines = 0
    if snapshot_1 is not None and snapshot_2 is not None:
        expected_lines += 1
        if snapshot_1.vram is not None and snapshot_2.vram is not None:
            expected_lines += 1
        if snapshot_1.malloc_info is not None and snapshot_2.malloc_info is not None:
            expected_lines += 5

    assert len(msg.splitlines()) == expected_lines
