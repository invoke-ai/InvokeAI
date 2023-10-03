import gc
from typing import Optional

import psutil
import torch

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
    def capture(cls, run_garbage_collector: bool = True):
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

        malloc_info = None
        try:
            malloc_info = LibcUtil().mallinfo2()
        except Exception:
            # TODO(ryand): Catch a more specific exception.
            # This is expected in environments that do not have the 'libc.so.6' shared library.
            pass

        return cls(process_ram, vram, malloc_info)


def get_pretty_snapshot_diff(snapshot_1: MemorySnapshot, snapshot_2: MemorySnapshot) -> str:
    """Get a pretty string describing the difference between two `MemorySnapshot`s."""
    ram_diff = snapshot_2.process_ram - snapshot_1.process_ram
    msg = f"RAM ({(ram_diff/GB):+.2f}): {(snapshot_1.process_ram/GB):.2f}GB -> {(snapshot_2.process_ram/GB):.2f}GB"

    vram_diff = None
    if snapshot_1.vram is not None and snapshot_2.vram is not None:
        vram_diff = snapshot_2.vram - snapshot_1.vram

    msg += f", VRAM ({(vram_diff/GB):+.2f}): {(snapshot_1.vram/GB):.2f}GB -> {(snapshot_2.vram/GB):.2f}GB"

    return msg
