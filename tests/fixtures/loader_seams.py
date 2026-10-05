"""Drive a real loader over a toy checkpoint, and record what it did on the way.

Sixteen places in this tree take a checkpoint apart before a model exists to put it into: strip
prefixes, read markers, read the header, pick a quantization branch, convert key names, reserve
cache room, swap modules in. Each has grown its own test driver -- twenty files build a loader with
``object.__new__`` and most define a helper to do it -- and each driver stubs a slightly different
set of the loader's surroundings. The cost is not the duplication. It is that a driver which stubs
one thing too many turns its tests green without the loader having done the work, and nothing
points that out.

So the *surroundings* are described here, once, as data: a :class:`Seam` says which of them a given
loader reaches for, and :func:`prepare` installs those and no others.

The loader's own attributes -- cache handle, logger, device -- are a separate matter and are always
supplied: ``object.__new__`` leaves an instance with none of them, so putting them back is
reconstituting the object rather than standing in for something outside it. Not every loader reads
all three (the Mistral encoder builds its own logger), but a switch per attribute would cost more
than the attribute.

What a run records is as much the point as what it returns. ``reserved``, ``dtypes_at_make_room``
and ``order`` make the ordering constraint visible -- room has to be reserved *before* the weights
widen, or the reservation is for a smaller model than the one about to exist -- and that constraint
is held at every seam today by the order the calls happen to be written in. An end-state assertion
cannot see it. These can.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.util.devices import TorchDevice

#: Distinguishes "do not touch `read_safetensors_metadata`" from "make it return nothing". Both are
#: real cases: a loader can be tested against a file whose header it really reads, and against that
#: read yielding nothing, and those are different paths through it.
UNSET = object()


@dataclass(frozen=True)
class Seam:
    """Which of a loader's surroundings a test has to stand in for."""

    loader: type
    #: Where the loader's ``from x import y`` names live, and so where a patch has to land.
    module: ModuleType
    entry: str = "_load_from_singlefile"
    #: Only for a loader that calls ``safetensors.torch.load_file`` through the package rather than
    #: importing the name. Patching the wrong host patches a name nobody reads, and the loader then
    #: quietly reads whatever file the config points at.
    load_file_host: ModuleType | None = None
    compute_dtype: torch.dtype = torch.float32
    #: ``TorchDevice`` is one class shared by every loader module, so this is a switch rather than a
    #: host: patching it is process-wide for the test's duration, not scoped to ``module``.
    patches_device: bool = False
    #: Loaders that never read ``self._torch_dtype``. Setting it anyway would hide one that started.
    sets_torch_dtype: bool = True
    #: False for loaders with no FP8 Storage pass, and for those whose tests want the real one:
    #: stubbing ``_apply_fp8_layerwise_casting`` by default would take real code out of their path.
    casts_fp8_storage: bool = True


@dataclass
class SeamRun:
    """A loader standing ready over a checkpoint, plus the record of what it does when run.

    Kept separate from the call so that a refusal can be asserted with the record still in hand:
    "it raised" and "it raised *before* reserving cache room for a load that cannot finish" are
    different claims, and only the second one is worth making.
    """

    seam: Seam
    loader: Any
    #: Every ``make_room`` argument, in call order. A loader that reserves twice is reserving
    #: absolutely rather than additively, so the last call has to cover the peak on its own.
    reserved: list[int] = field(default_factory=list)
    #: The dtypes of the dict *the loader is working on* when room was first reserved. A weight
    #: already widened here was widened before it was paid for. Empty when the loader reads a real
    #: file, because then there is no served dict to watch.
    dtypes_at_make_room: dict[str, torch.dtype] = field(default_factory=dict)
    #: ``(name, rooms_reserved)`` for each observed step, in call order -- how many times
    #: ``make_room`` had been called by the time that step ran.
    order: list[tuple[str, int]] = field(default_factory=list)
    casting_calls: int = 0

    def load(self, *args: Any) -> Any:
        """Call the seam's entry point. Most take the config alone; ``_load_model`` also takes the
        submodel, which is the argument deciding which part of a checkpoint is being asked for."""
        return getattr(self.loader, self.seam.entry)(*args)


def prepare(
    seam: Seam,
    monkeypatch: pytest.MonkeyPatch,
    *,
    state_dict: Mapping[str, torch.Tensor] | None = None,
    metadata: Mapping[str, str] | None | object = UNSET,
    geometry: Callable[[pytest.MonkeyPatch], None] = lambda _monkeypatch: None,
    observe: tuple[str, ...] = (),
) -> SeamRun:
    """Stand ``seam``'s loader up over ``state_dict``, or over whatever file the config points at.

    Each read gets its own copy of ``state_dict``, because loaders rebind and clear the dict they
    are handed and a test that loads twice must not be reading the first load's leavings. The copy
    the loader is working on is the one ``dtypes_at_make_room`` watches: watching the caller's dict
    instead would make that record a restatement of the test's own input, which is a probe that
    cannot fail.

    Leaving ``state_dict`` out lets the loader read an actual checkpoint, which is the only way to
    exercise ``read_safetensors_metadata`` for real. Seven loaders call it and almost every test
    stubs it out; of the seams driven here today only the Mistral encoder writes a real file.

    ``observe`` names module-level steps of the loader to wrap, so that :attr:`SeamRun.order` says
    how much room had been reserved by the time each one ran.
    """
    run = SeamRun(seam=seam, loader=None)
    served: list[dict[str, torch.Tensor]] = []

    def load_file(_path):
        served.append(dict(state_dict or {}))
        return served[-1]

    def make_room(reserve: int) -> None:
        if not run.reserved and served:
            run.dtypes_at_make_room = {key: value.dtype for key, value in served[-1].items()}
        run.reserved.append(reserve)

    def apply_casting(model: torch.nn.Module, _config: Any, _submodel: Any) -> torch.nn.Module:
        run.casting_calls += 1
        return model

    loader = object.__new__(seam.loader)
    loader._ram_cache = MagicMock(make_room=make_room)
    loader._logger = MagicMock()
    loader._torch_device = torch.device("cpu")
    if seam.sets_torch_dtype:
        loader._torch_dtype = seam.compute_dtype
    if seam.casts_fp8_storage:
        loader._apply_fp8_layerwise_casting = apply_casting
    run.loader = loader

    if state_dict is not None:
        monkeypatch.setattr(seam.load_file_host or seam.module, "load_file", load_file)
    if metadata is not UNSET:
        header = dict(metadata) if metadata is not None else None
        monkeypatch.setattr(seam.module, "read_safetensors_metadata", lambda _path, _logger: header)
    if seam.patches_device:
        monkeypatch.setattr(TorchDevice, "choose_torch_device", staticmethod(lambda: torch.device("cpu")))
        monkeypatch.setattr(TorchDevice, "choose_bfloat16_safe_dtype", staticmethod(lambda _d: seam.compute_dtype))

    # Before the wrappers, so a geometry that replaces an observed name cannot silently discard the
    # observation: this way each wrapper closes over whatever geometry left in place.
    geometry(monkeypatch)

    for name in observe:
        original = getattr(seam.module, name)

        def step(*args, _name=name, _original=original, **kwargs):
            run.order.append((_name, len(run.reserved)))
            return _original(*args, **kwargs)

        monkeypatch.setattr(seam.module, name, step)

    return run
