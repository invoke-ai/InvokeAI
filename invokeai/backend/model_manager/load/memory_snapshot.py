import gc
from typing import Optional

import psutil
import torch
from typing_extensions import Self

from invokeai.backend.model_management.libc_util import LibcUtil, Struct_mallinfo2

GB = 2**30  # 1 GB


class MemorySnapshot:
    """A snapshot of RAM and VRAM usage. All values are in bytes."""

    def __init__(self, process_ram: int, vram: Optional[int], malloc_info: Optional[Struct_mallinfo2]):
        """Initialize a MemorySnapshot.

        Most of the time, `MemorySnapshot` will be constructed with `MemorySnapshot.capture()`.

        Args:
            process_ram (int): CPU RAM used by the current process.
            vram (Optional[int]): VRAM used by torch.
            malloc_info (Optional[Struct_mallinfo2]): Malloc info obtained from LibcUtil.
        """
        self.process_ram = process_ram
        self.vram = vram
        self.malloc_info = malloc_info

    @classmethod
    def capture(cls, run_garbage_collector: bool = True) -> Self:
        """Capture and return a MemorySnapshot.

        Note: This function has significant overhead, particularly if `run_garbage_collector == True`.

        Args:
            run_garbage_collector (bool, optional): If true, gc.collect() will be run before checking the process RAM
                usage. Defaults to True.

        Returns:
            MemorySnapshot
        """
        if run_garbage_collector:
            gc.collect()

        # According to the psutil docs (https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info), rss is
        # supported on all platforms.
        process_ram = psutil.Process().memory_info().rss

        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated()
        else:
            # TODO: We could add support for mps.current_allocated_memory() as well. Leaving out for now until we have
            # time to test it properly.
            vram = None

        try:
            malloc_info = LibcUtil().mallinfo2()  # type: ignore
        except (OSError, AttributeError):
            # OSError: This is expected in environments that do not have the 'libc.so.6' shared library.
            # AttributeError: This is expected in environments that have `libc.so.6` but do not have the `mallinfo2` (e.g. glibc < 2.33)
            # TODO: Does `mallinfo` work?
            malloc_info = None

        return cls(process_ram, vram, malloc_info)


def get_pretty_snapshot_diff(snapshot_1: Optional[MemorySnapshot], snapshot_2: Optional[MemorySnapshot]) -> str:
    """Get a pretty string describing the difference between two `MemorySnapshot`s."""

    def get_msg_line(prefix: str, val1: int, val2: int) -> str:
        diff = val2 - val1
        return f"{prefix: <30} ({(diff/GB):+5.3f}): {(val1/GB):5.3f}GB -> {(val2/GB):5.3f}GB\n"

    msg = ""

    if snapshot_1 is None or snapshot_2 is None:
        return msg

    msg += get_msg_line("Process RAM", snapshot_1.process_ram, snapshot_2.process_ram)

    if snapshot_1.malloc_info is not None and snapshot_2.malloc_info is not None:
        msg += get_msg_line("libc mmap allocated", snapshot_1.malloc_info.hblkhd, snapshot_2.malloc_info.hblkhd)

        msg += get_msg_line("libc arena used", snapshot_1.malloc_info.uordblks, snapshot_2.malloc_info.uordblks)

        msg += get_msg_line("libc arena free", snapshot_1.malloc_info.fordblks, snapshot_2.malloc_info.fordblks)

        libc_total_allocated_1 = snapshot_1.malloc_info.arena + snapshot_1.malloc_info.hblkhd
        libc_total_allocated_2 = snapshot_2.malloc_info.arena + snapshot_2.malloc_info.hblkhd
        msg += get_msg_line("libc total allocated", libc_total_allocated_1, libc_total_allocated_2)

        libc_total_used_1 = snapshot_1.malloc_info.uordblks + snapshot_1.malloc_info.hblkhd
        libc_total_used_2 = snapshot_2.malloc_info.uordblks + snapshot_2.malloc_info.hblkhd
        msg += get_msg_line("libc total used", libc_total_used_1, libc_total_used_2)

    if snapshot_1.vram is not None and snapshot_2.vram is not None:
        msg += get_msg_line("VRAM", snapshot_1.vram, snapshot_2.vram)

    return "\n"+msg if len(msg)>0 else msg
