"""Probes on a safetensors file a loader has already read."""

import functools
import itertools
import os
import sys
import tempfile
from collections.abc import Callable, Mapping
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


#: Only the macOS runners keep the bytes a tensor was mapped with. Anywhere else a probe that cannot tell a view from
#: a copy is a broken test environment, which `tensors_reading_the_file` reports instead of passing silently.
needs_mapping_that_shows_file_writes = pytest.mark.skipif(
    sys.platform == "darwin",
    reason="safetensors' mapping does not show in-place writes on macOS, so overwrite_payloads can't tell a view from a copy",
)


def tensors_reading_the_file(source: torch.nn.Module | Mapping[str, torch.Tensor], path: Path) -> list[str]:
    """The tensors of ``source`` that are still views of ``path``'s mapping, by `overwrite_payloads`.

    ``source`` is a loaded state dict, or a module, whose parameters and buffers are checked: each one that is a view
    pins the whole file for as long as the model is cached, outside the cache's size calculation. Bytes are
    compared, so a float8 tensor, which `torch.equal` cannot compare on CPU, is checked like any other. Not
    `state_dict()`: it leaves out non-persistent buffers, and the scales of kept fp8 layers are registered as those.
    """
    assert mapping_shows_file_writes(), "a mapped tensor does not show a write to its file here, so this finds nothing"

    def as_bytes() -> dict[str, torch.Tensor]:
        if isinstance(source, torch.nn.Module):
            tensors = itertools.chain(
                source.named_parameters(remove_duplicate=False), source.named_buffers(remove_duplicate=False)
            )
        else:
            tensors = iter(source.items())
        return {name: tensor.detach().contiguous().reshape(-1).view(torch.uint8).clone() for name, tensor in tensors}

    before = as_bytes()
    overwrite_payloads(path)
    after = as_bytes()
    return [name for name in before if not torch.equal(before[name], after[name])]


def served_checkpoint(state_dict: Mapping[str, torch.Tensor]) -> Callable[..., dict[str, torch.Tensor]]:
    """A stand-in for `load_safetensors_checkpoint` that serves ``state_dict`` without a file.

    It applies ``select`` the way the real reader does, to the keys alone, so a loader's drops are exercised and
    the tensors it never selects are never handed over.
    """

    def load(_path, *, select=None, owned: bool = False) -> dict[str, torch.Tensor]:
        if select is None:
            return state_dict  # type: ignore[return-value]  # the very dict, as a loader that clears it is tested on
        return {key: state_dict[key] for key in select({key: key for key in state_dict}).values()}

    return load


class ReadLog:
    """What `record_reads` saw: the tensors each ``get_tensor`` returned, and the ``backend`` each open asked for."""

    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        self.backends: list[str] = []
        #: ``("open", backend)`` and ``("read", key)`` in the order they happened, for ordering assertions.
        self.events: list[tuple[str, str]] = []


def record_reads(monkeypatch: pytest.MonkeyPatch) -> ReadLog:
    """Record every tensor read through ``safetensors.safe_open``, which is how the loaders read a checkpoint.

    The handle is the real one, so the loader gets what it would have got; only its ``get_tensor`` is watched.
    It patches the ``safetensors`` package attribute, so it sees only code that imports ``safe_open`` when it runs,
    as `load_safetensors_checkpoint` does. A module that imports it at module level (`moss_image.py`,
    `qwen3_5_encoder.py`) keeps the original and is not recorded.
    """
    log = ReadLog()

    class _Handle:
        def __init__(self, handle) -> None:
            self._handle = handle

        def get_tensor(self, key: str) -> torch.Tensor:
            tensor = self._handle.get_tensor(key)
            log.tensors[key] = tensor
            log.events.append(("read", key))
            return tensor

        def __getattr__(self, name: str):
            return getattr(self._handle, name)

    class _Open:
        def __init__(self, *args, **kwargs) -> None:
            self._cm = safe_open(*args, **kwargs)
            log.backends.append(kwargs.get("backend", "mmap"))
            log.events.append(("open", log.backends[-1]))

        def __enter__(self) -> _Handle:
            return _Handle(self._cm.__enter__())

        def __exit__(self, *exc) -> object:
            return self._cm.__exit__(*exc)

    monkeypatch.setattr("safetensors.safe_open", _Open)
    return log
