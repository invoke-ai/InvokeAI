"""Probes on a safetensors file a loader has already read."""

import os
from pathlib import Path


def overwrite_payloads(path: Path, byte: bytes = b"\x3f") -> None:
    """Rewrite every tensor byte of ``path`` in place, leaving the header as it was.

    A loaded tensor that still reads the file -- a view of a memory mapping -- shows the new bytes; one that was
    copied does not. Mappings are coherent with writes through a file handle on Windows, Linux and macOS, so this
    tells the two apart on every platform, where comparing storages cannot: every tensor ``safetensors`` maps from
    one file reports a storage of its own, the size of the tensor.
    """
    with open(path, "r+b") as f:
        start = 8 + int.from_bytes(f.read(8), "little")
        size = f.seek(0, os.SEEK_END) - start
        f.seek(start)
        f.write(byte * size)
