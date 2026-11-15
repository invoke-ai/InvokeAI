"""Probes on a safetensors file a loader has already read."""

import functools
import itertools
import os
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def overwrite_payloads(path: Path, byte: bytes = b"\x3f") -> None:
    """Rewrite every tensor byte of ``path`` in place, leaving the header as it was.

    Where safetensors' mapping shows writes made through a file handle (`mapping_shows_file_writes`), a loaded
    tensor that still reads the file -- a view of the mapping -- shows the new bytes, and one that was copied does
    not. Comparing storages cannot tell the two apart: every tensor ``safetensors`` maps from one file reports a
    storage of its own, the size of the tensor.
    """
    with open(path, "r+b") as f:
        start = 8 + int.from_bytes(f.read(8), "little")
        size = f.seek(0, os.SEEK_END) - start
        f.seek(start)
        f.write(byte * size)


@functools.cache
def mapping_shows_file_writes() -> bool:
    """Whether a tensor safetensors maps from a file sees `overwrite_payloads`. It does on Windows and Linux; on the
    macOS CI runners it keeps the bytes it was mapped with, so there the probe cannot tell a view from a copy."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        path = Path(directory) / "probe.safetensors"
        save_file({"x": torch.zeros(1024)}, path)
        with safe_open(path, framework="pt") as f:
            tensor = f.get_tensor("x")
        overwrite_payloads(path)
        changed = bool(tensor.any())
        del tensor
    return changed


needs_mapping_that_shows_file_writes = pytest.mark.skipif(
    not mapping_shows_file_writes(),
    reason="safetensors' mapping does not show in-place writes here, so overwrite_payloads can't tell a view from a copy",
)


def tensors_reading_the_file(module: torch.nn.Module, path: Path) -> list[str]:
    """The parameters and buffers of ``module`` that are still views of ``path``'s mapping, by `overwrite_payloads`.

    Each one pins the whole file for as long as the model is cached, outside the cache's size calculation. Bytes are
    compared, so a float8 tensor, which `torch.equal` cannot compare on CPU, is checked like any other. Not
    `state_dict()`: it leaves out non-persistent buffers, and the scales of kept fp8 layers are registered as those.
    """

    def as_bytes() -> dict[str, torch.Tensor]:
        tensors = itertools.chain(
            module.named_parameters(remove_duplicate=False), module.named_buffers(remove_duplicate=False)
        )
        return {name: tensor.detach().contiguous().reshape(-1).view(torch.uint8).clone() for name, tensor in tensors}

    before = as_bytes()
    overwrite_payloads(path)
    after = as_bytes()
    return [name for name in before if not torch.equal(before[name], after[name])]
