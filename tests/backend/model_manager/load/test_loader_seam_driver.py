"""The seam driver's own records, against a loader small enough to be wrong on purpose.

`tests/fixtures/loader_seams.py` is fixture code, and fixture code that lies is worse than no
fixture: every seam driven through it inherits the lie, and the tests keep passing. That happened
once already -- `dtypes_at_make_room` watched the caller's dict while the loader worked on a copy,
so it reported the test's own input whatever the loader did, and the one ordering test relying on it
became unfalsifiable. Three reviewers found it; nothing in the suite did.

So the records are pinned here against toy loaders that deliberately get the order wrong. These are
not tests of a loader. They are the reason a loader's test can be believed.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.fixtures.loader_seams import Seam, prepare


class _WidensBeforeReserving:
    """The regression every seam's ordering assertion exists to catch: the checkpoint is widened to
    the compute dtype, and only then is the cache asked to make room for it."""

    def _load_from_singlefile(self, _config):
        _stubbed.widen(_stubbed.load_file("ignored"))
        self._ram_cache.make_room(1)
        return torch.nn.Module()


class _ReservesFirst:
    def _load_from_singlefile(self, _config):
        state_dict = _stubbed.load_file("ignored")
        self._ram_cache.make_room(1)
        _stubbed.widen(state_dict)
        return torch.nn.Module()


#: What each `widen` call was handed, so a test can say what the loader actually received rather
#: than inferring it from a dict the loader never touched.
SEEN: list[dict[str, torch.dtype]] = []


def _widen(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The step `observe` watches: a stand-in for the fold or the cast that widens a checkpoint."""
    SEEN.append({key: value.dtype for key, value in state_dict.items()})
    for key in list(state_dict):
        state_dict[key] = state_dict[key].float()
    return state_dict


@pytest.fixture(autouse=True)
def _forget_what_was_seen():
    SEEN.clear()
    yield
    SEEN.clear()


#: Stands in for a loader module: `prepare` patches these names here, and the toy loaders read them
#: back the way a real loader reads its own module-level imports.
_stubbed = SimpleNamespace(load_file=lambda _path: {}, widen=_widen)

PACKED = {"lin.weight": torch.zeros(4, 4, dtype=torch.float8_e4m3fn)}


@pytest.mark.parametrize(
    ("loader", "expected"),
    [(_WidensBeforeReserving, torch.float32), (_ReservesFirst, torch.float8_e4m3fn)],
    ids=["widened_before_reserving", "reserved_first"],
)
def test_the_dtype_record_sees_what_the_loader_had_when_it_asked_for_room(
    monkeypatch: pytest.MonkeyPatch, loader: type, expected: torch.dtype
) -> None:
    """The two cases have to come out differently, or the record is not an observation.

    A driver that hands the loader a copy and then reads its own dict reports `float8_e4m3fn` for
    both -- which is what shipped in the first draft of this fixture, and what made the ideogram4
    ordering test pass no matter where `make_room` was called.
    """
    run = prepare(Seam(loader=loader, module=_stubbed), monkeypatch, state_dict=PACKED)

    run.load(None)

    assert run.dtypes_at_make_room == {"lin.weight": expected}


def test_each_load_gets_its_own_copy_of_the_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Loaders rebind and clear the dict they are handed, so a second load must not be reading the
    first one's leavings.

    Asserted on what the loader was actually handed, not on the caller's dict: the caller's dict
    survives either way, so a driver that made one copy up front and served it to both loads would
    satisfy that half while the second load started from the first load's float32 weights.
    """
    run = prepare(Seam(loader=_ReservesFirst, module=_stubbed), monkeypatch, state_dict=PACKED)

    run.load(None)
    run.load(None)

    assert SEEN == [{"lin.weight": torch.float8_e4m3fn}] * 2, SEEN
    assert PACKED["lin.weight"].dtype is torch.float8_e4m3fn, "the loader mutated the test's own dict"


@pytest.mark.parametrize(
    ("loader", "expected"),
    [(_WidensBeforeReserving, [("widen", 0)]), (_ReservesFirst, [("widen", 1)])],
    ids=["widened_before_reserving", "reserved_first"],
)
def test_the_step_record_says_how_much_room_had_been_reserved(
    monkeypatch: pytest.MonkeyPatch, loader: type, expected: list[tuple[str, int]]
) -> None:
    """`observe` is the other half of the ordering claim, and the half that works on a seam reading
    a real file, where there is no served dict to watch. It has to separate the same two loaders."""
    run = prepare(Seam(loader=loader, module=_stubbed), monkeypatch, state_dict=PACKED, observe=("widen",))

    run.load(None)

    assert run.order == expected


def test_a_seam_installs_no_stub_it_did_not_ask_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """The failure this fixture exists to stop: a driver that stubs one thing too many lets a test
    pass without the loader having done the work. Only the plumbing every `object.__new__` loader
    needs -- cache, logger, device -- is unconditional."""
    declined = Seam(loader=_ReservesFirst, module=_stubbed, sets_torch_dtype=False, casts_fp8_storage=False)
    asked = Seam(loader=_ReservesFirst, module=_stubbed)

    without = prepare(declined, monkeypatch, state_dict=PACKED)
    with_stubs = prepare(asked, monkeypatch, state_dict=PACKED)

    assert not hasattr(without.loader, "_torch_dtype")
    assert not hasattr(without.loader, "_apply_fp8_layerwise_casting")
    assert with_stubs.loader._torch_dtype is torch.float32
    assert callable(with_stubs.loader._apply_fp8_layerwise_casting)
    # `read_safetensors_metadata` is left alone unless a test names it, so that a seam can be driven
    # against a real header.
    assert not hasattr(_stubbed, "read_safetensors_metadata")
