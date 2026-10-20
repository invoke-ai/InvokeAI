"""The process-wide owner of torch's SDPA backend selection.

It sets the priority order every model runs under, once at startup, and demotes cuDNN in it when a
cuDNN call fails. Windows only set the enable flags, and torch's own `sdpa_kernel` saves and restores
those non-atomically, so two threads whose windows overlap leave them permanently changed. These pin
that `sdpa_policy` does not, because the leak escapes the code that caused it: every other model in
the process then picks kernels by the leaked flags, and `backend.util.attention` sizes working memory
from them.
"""

import logging
import threading

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, _cur_sdpa_kernel_backends, _sdpa_kernel, sdpa_kernel

import invokeai.backend.util.sdpa_scope as sdpa_scope
from invokeai.backend.util.sdpa_scope import demote_cudnn_attention, install_default_sdpa_priority, sdpa_policy

MATH_ONLY = (SDPBackend.MATH,)
EFFICIENT_ONLY = (SDPBackend.EFFICIENT_ATTENTION,)

# Written out rather than read from the module under test: torch's own order, and the one InvokeAI installs.
TORCH_ORDER = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.OVERRIDEABLE,
]
PROCESS_ORDER = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
    SDPBackend.OVERRIDEABLE,
]


def _global_state() -> tuple[list[int], set[SDPBackend]]:
    """Everything `sdpa_kernel` can mutate: the priority order and the per-backend enable flags."""
    return list(torch._C._get_sdp_priority_order()), set(_cur_sdpa_kernel_backends())


def _priority() -> list[SDPBackend]:
    return [SDPBackend(backend) for backend in torch._C._get_sdp_priority_order()]


def _set_priority(order: list[SDPBackend]) -> None:
    torch._C._set_sdp_priority_order([int(backend) for backend in order])


@pytest.fixture(autouse=True)
def _restore_global_sdpa_state():
    """These mutate process-global torch state, and pytest-xdist gives a worker arbitrary files.

    Without this, a leak here would be indistinguishable from the bug under test in whatever file
    the worker runs next.
    """
    priority, backends = _global_state()
    yield
    torch._C._set_sdp_priority_order(priority)
    # The setter `sdpa_kernel` uses, not `sdpa_kernel` itself: an entered window that is then dropped is
    # finalized at once, and its exit puts back the very flags this is meant to undo.
    _sdpa_kernel(list(backends))


def _run_interleaved(scope, outer: tuple[SDPBackend, ...], inner: tuple[SDPBackend, ...]):
    """A enters, B enters, A exits, B exits -- the ordering torch's save/restore gets wrong.

    Returns what B saw while it was inside its own window.
    """
    a_inside, b_inside, a_left = threading.Event(), threading.Event(), threading.Event()
    seen: dict[str, tuple[list[int], set[SDPBackend]]] = {}

    def a():
        with scope(outer):
            a_inside.set()
            b_inside.wait(timeout=5)

    def b():
        a_inside.wait(timeout=5)
        with scope(inner):
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
        _run_interleaved(lambda backends: sdpa_kernel(list(backends)), MATH_ONLY, EFFICIENT_ONLY)
        assert _global_state() != before

    def test_interleaved_windows_do_not_leave_backends_disabled(self):
        """The expensive leak: every later attention in the process falls to whatever the window left
        enabled -- math, the O(seq^2) path, for an exclusive MATH override."""
        before = _global_state()
        _run_interleaved(sdpa_policy, MATH_ONLY, EFFICIENT_ONLY)
        assert _global_state() == before


class TestWhileAWindowIsOpen:
    def test_the_open_window_wins_over_a_later_one(self):
        """Two concurrent sessions cannot hold different values of a process-global setting.

        That is torch's constraint; what this module chooses is *which* one applies and that it is
        restored exactly once. A later window must not overwrite the open one -- overwriting is how
        the leak happens.
        """
        seen = _run_interleaved(sdpa_policy, MATH_ONLY, EFFICIENT_ONLY)
        assert seen["b"][1] == set(MATH_ONLY)

    def test_a_single_window_applies_its_backends_and_restores_them(self):
        before = _global_state()
        with sdpa_policy(MATH_ONLY):
            assert set(_cur_sdpa_kernel_backends()) == set(MATH_ONLY)
        assert _global_state() == before

    def test_nesting_in_one_thread_restores_once(self):
        before = _global_state()
        with sdpa_policy(MATH_ONLY):
            inside_outer = _global_state()
            with sdpa_policy(EFFICIENT_ONLY):
                # The inner window must neither apply its own backends nor restore anything: the outer
                # one is still open and owns the flags.
                assert _global_state() == inside_outer
            # And exiting it must not have restored either -- that is how the leak used to happen.
            assert _global_state() == inside_outer
        assert _global_state() == before

    def test_an_exception_inside_the_window_still_restores(self):
        before = _global_state()
        with pytest.raises(RuntimeError, match="boom"):
            with sdpa_policy(MATH_ONLY):
                raise RuntimeError("boom")
        assert _global_state() == before

    def test_a_window_leaves_the_order_alone(self):
        """The order is the process's: a window that restored it would undo a cuDNN demotion that
        happened while it was open."""
        install_default_sdpa_priority()
        with sdpa_policy(MATH_ONLY):
            assert demote_cudnn_attention() is True
        assert _priority() == TORCH_ORDER


class TestTheProcessDefault:
    """The order every model in the process runs under, installed once at startup.

    torch's own order ranks cuDNN behind the always-eligible math kernel, so it practically never runs;
    this one puts it second, behind flash, which leads wherever a build has it.
    """

    def test_it_ranks_flash_cudnn_efficient_math_ahead_of_the_rest(self):
        _set_priority(TORCH_ORDER)
        install_default_sdpa_priority()
        assert _priority() == PROCESS_ORDER

    def test_a_backend_outside_the_ranking_is_kept_behind_it(self):
        """OVERRIDEABLE is the one backend torch lists beyond the four ranked ones. Starting it at the
        front shows it moved behind the ranking rather than happening to be last already."""
        _set_priority(
            [
                SDPBackend.OVERRIDEABLE,
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
            ]
        )
        install_default_sdpa_priority()
        assert _priority() == PROCESS_ORDER

    def test_installing_it_twice_is_the_same_as_once(self):
        _set_priority(TORCH_ORDER)
        install_default_sdpa_priority()
        install_default_sdpa_priority()
        assert _priority() == PROCESS_ORDER

    def test_the_enable_flags_are_left_alone(self):
        torch.backends.cuda.enable_cudnn_sdp(False)
        before = set(_cur_sdpa_kernel_backends())
        install_default_sdpa_priority()
        assert set(_cur_sdpa_kernel_backends()) == before
        assert SDPBackend.CUDNN_ATTENTION not in before

    @pytest.mark.parametrize("change", ["removed", "signature changed"])
    def test_a_torch_whose_private_priority_api_changed_keeps_its_order_and_says_so(self, change, monkeypatch, caplog):
        def changed_signature(order, extra):
            raise AssertionError("never reached: the call itself fails on the signature")

        _set_priority(TORCH_ORDER)
        # Scoped, not test-wide: this file's teardown restores the order through the very call replaced here.
        with monkeypatch.context() as patch, caplog.at_level(logging.WARNING, logger=sdpa_scope.logger.name):
            if change == "removed":
                patch.delattr(torch._C, "_set_sdp_priority_order")
            else:
                patch.setattr(torch._C, "_set_sdp_priority_order", changed_signature)
            install_default_sdpa_priority()

        assert _priority() == TORCH_ORDER
        assert len(caplog.records) == 1
        assert "_set_sdp_priority_order" in caplog.records[0].message

    def test_startup_installs_it_and_the_cudnn_fallback(self, monkeypatch):
        import invokeai.backend.util.attention as attention
        from invokeai.app.util.startup_utils import apply_monkeypatches

        # Pin the entry points apply_monkeypatches rebinds so monkeypatch restores them afterwards.
        monkeypatch.setattr(F, "scaled_dot_product_attention", F.scaled_dot_product_attention)
        monkeypatch.setattr(torch.cuda, "empty_cache", torch.cuda.empty_cache)
        # The fallback installs on CUDA builds only; look like one on a CPU-only runner as well.
        monkeypatch.setattr(torch.version, "cuda", "13.0")
        monkeypatch.setattr(torch.version, "hip", None)
        _set_priority(TORCH_ORDER)

        apply_monkeypatches()

        assert _priority() == PROCESS_ORDER
        assert getattr(F.scaled_dot_product_attention, attention._CUDNN_FALLBACK_SENTINEL, False)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to exercise the SDPA dispatcher")
    def test_a_call_cudnn_cannot_serve_falls_through_to_the_next_kernel(self):
        """The property the whole order rests on: a kernel that cannot serve a call is skipped, not
        raised on. If a future torch made a ranked order raise instead, ranking cuDNN second would turn
        every call it cannot serve into a failed generation."""
        # fp32 is refused by the fused kernels, which is a portable way to make cuDNN unusable on a card
        # where it otherwise works.
        t = torch.randn(1, 24, 512, 128, device="cuda", dtype=torch.float32)
        params = torch.backends.cuda.SDPAParams(t, t, t, None, 0.0, False, False)
        if torch.backends.cuda.can_use_cudnn_attention(params):
            pytest.skip("This build serves fp32 with cuDNN, so it cannot stand in for an unusable backend")

        install_default_sdpa_priority()
        out = F.scaled_dot_product_attention(t, t, t)

        assert torch.isfinite(out).all()


class TestDemotingCudnn:
    def test_cudnn_moves_directly_behind_math_and_nothing_else_moves(self):
        install_default_sdpa_priority()
        assert demote_cudnn_attention() is True
        assert _priority() == TORCH_ORDER

    def test_only_the_first_of_two_demotions_moves_it(self):
        """Two GPU workers whose cuDNN calls fail at once: exactly one of them demotes, and warns."""
        install_default_sdpa_priority()
        assert [demote_cudnn_attention(), demote_cudnn_attention()] == [True, False]
        assert _priority() == TORCH_ORDER

    def test_the_enable_flags_are_left_alone(self):
        install_default_sdpa_priority()
        torch.backends.cuda.enable_flash_sdp(False)
        before = set(_cur_sdpa_kernel_backends())
        demote_cudnn_attention()
        assert set(_cur_sdpa_kernel_backends()) == before
