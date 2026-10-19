"""The process-wide owner of torch's SDPA backend selection.

torch's own `sdpa_kernel` saves and restores process-global state non-atomically, so two threads
whose windows overlap leave it permanently changed. These pin that `sdpa_policy` does not, because
the leak escapes the architecture that caused it: every other model in the process then picks
kernels by the leaked order, and `backend.util.attention` sizes working memory from that order.
"""

import logging
import threading

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import invokeai.backend.util.sdpa_scope as sdpa_scope
from invokeai.backend.util.sdpa_scope import foreign_window_entries, sdpa_policy

KREA2 = (
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
)
ANIMA = (
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
)
EXCLUSIVE = (SDPBackend.MATH,)


def _global_state() -> tuple[list[int], set[SDPBackend]]:
    """Everything `sdpa_kernel` mutates: the priority order and the per-backend enable flags."""
    from torch.nn.attention import _cur_sdpa_kernel_backends

    return list(torch._C._get_sdp_priority_order()), set(_cur_sdpa_kernel_backends())


@pytest.fixture(autouse=True)
def _restore_global_sdpa_state():
    """These mutate process-global torch state, and pytest-xdist gives a worker arbitrary files.

    Without this, a leak here would be indistinguishable from the bug under test in whatever file
    the worker runs next.
    """
    priority, backends = _global_state()
    yield
    torch._C._set_sdp_priority_order(priority)
    sdpa_kernel(list(backends)).__enter__()


def _run_interleaved(scope, outer: tuple[SDPBackend, ...], inner: tuple[SDPBackend, ...], set_priority: bool = True):
    """A enters, B enters, A exits, B exits -- the ordering torch's save/restore gets wrong.

    Returns what each thread saw while it was inside its own window.
    """
    a_inside, b_inside, a_left = threading.Event(), threading.Event(), threading.Event()
    seen: dict[str, tuple[list[int], set[SDPBackend]]] = {}

    def a():
        with scope(outer, set_priority=set_priority):
            a_inside.set()
            b_inside.wait(timeout=5)

    def b():
        a_inside.wait(timeout=5)
        with scope(inner, set_priority=set_priority):
            b_inside.set()
            a_left.wait(timeout=5)
            seen["b"] = _global_state()

    ta, tb = threading.Thread(target=a), threading.Thread(target=b)
    ta.start()
    tb.start()
    ta.join(timeout=10)
    a_left.set()
    tb.join(timeout=10)
    assert not ta.is_alive() and not tb.is_alive()
    return seen


class TestTheLeakIsClosed:
    def test_torchs_own_scope_leaks_when_two_windows_overlap(self):
        """The defect this module exists for, pinned against torch so the fix is not mistaken for one.

        If a future torch makes `sdpa_kernel` thread-safe this fails, and `sdpa_policy` can go.
        """
        before = _global_state()
        _run_interleaved(lambda b, set_priority: sdpa_kernel(list(b), set_priority=set_priority), KREA2, ANIMA)
        assert _global_state() != before

    def test_interleaved_windows_restore_the_global_state(self):
        before = _global_state()
        _run_interleaved(sdpa_policy, KREA2, ANIMA)
        assert _global_state() == before

    def test_an_exclusive_policy_does_not_leave_backends_disabled(self):
        """The expensive half: an exclusive override leaks three *disabled* backends, not an order.

        Every later attention in the process then falls to math, which is the O(seq^2) path the
        Krea-2 processor exists to avoid.
        """
        before = _global_state()
        _run_interleaved(sdpa_policy, KREA2, EXCLUSIVE, set_priority=False)
        assert _global_state() == before


class TestWhileAWindowIsOpen:
    def test_the_open_policy_wins_over_a_later_one(self):
        """Two concurrent sessions cannot hold different values of a process-global setting.

        That is torch's constraint; what this module chooses is *which* one applies and that it is
        restored exactly once. A later window must not overwrite the open one -- overwriting is how
        the leak happens.
        """
        seen = _run_interleaved(sdpa_policy, KREA2, ANIMA)
        krea2_order = [int(b) for b in KREA2]
        assert seen["b"][0][: len(krea2_order)] == krea2_order

    def test_a_single_window_applies_its_policy_and_restores_it(self):
        before = _global_state()
        with sdpa_policy(EXCLUSIVE, set_priority=False):
            _, enabled = _global_state()
            assert enabled == set(EXCLUSIVE)
        assert _global_state() == before

    def test_nesting_in_one_thread_restores_once(self):
        before = _global_state()
        with sdpa_policy(KREA2) as outer_owns:
            inside_outer = _global_state()
            with sdpa_policy(ANIMA) as inner_owns:
                # The inner window must neither apply its own policy nor restore anything: the outer
                # one is still open and owns the globals.
                assert _global_state() == inside_outer
            # And exiting it must not have restored either -- that is how the leak used to happen.
            assert _global_state() == inside_outer
        assert (outer_owns, inner_owns) == (True, False)
        assert _global_state() == before

    def test_an_exception_inside_the_window_still_restores(self):
        before = _global_state()
        with pytest.raises(RuntimeError, match="boom"):
            with sdpa_policy(EXCLUSIVE, set_priority=False):
                raise RuntimeError("boom")
        assert _global_state() == before


class TestAWindowThatDidNotApply:
    """An exclusive override exists so that a completed run proves which kernel served it.

    A window that did not own the globals voids that, and the benchmark reads this counter to say
    so rather than print a label for a kernel that may not have run.
    """

    def test_a_non_owning_window_is_counted(self):
        before = foreign_window_entries()
        with sdpa_policy(KREA2):
            with sdpa_policy(EXCLUSIVE, set_priority=False) as owned:
                assert owned is False
        assert foreign_window_entries() == before + 1

    def test_a_window_that_asked_for_what_is_already_set_is_not_counted(self):
        before = foreign_window_entries()
        with sdpa_policy(KREA2):
            with sdpa_policy(KREA2):
                pass
        assert foreign_window_entries() == before

    def test_an_override_that_did_not_apply_is_reported_loudly(self, caplog, monkeypatch):
        """At WARNING, and once per distinct pair: reporting once per process would let a benign
        default-vs-default collision swallow the line that invalidates a measurement."""
        monkeypatch.setattr(sdpa_scope, "_reported", set())
        with caplog.at_level(logging.WARNING, logger=sdpa_scope.logger.name):
            with sdpa_policy(KREA2):
                with sdpa_policy(EXCLUSIVE, set_priority=False):
                    pass
        assert len(caplog.records) == 1
        assert "did not get the kernel it asked for" in caplog.records[0].message
        assert "MATH" in caplog.records[0].message

    def test_a_default_ranking_collision_is_not_a_warning(self, caplog, monkeypatch):
        monkeypatch.setattr(sdpa_scope, "_reported", set())
        with caplog.at_level(logging.INFO, logger=sdpa_scope.logger.name):
            with sdpa_policy(KREA2):
                with sdpa_policy(ANIMA):
                    pass
        assert [r.levelno for r in caplog.records] == [logging.INFO]
