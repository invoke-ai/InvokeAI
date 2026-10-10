"""Hiding integrated GPUs from HIP on Windows ROCm before torch initializes it."""

import logging
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from invokeai.app.util import rocm_integrated_gpu
from invokeai.app.util.rocm_integrated_gpu import (
    VISIBILITY_ENV_VARS,
    ProbeFailed,
    discrete_device_visibility,
    hide_integrated_gpus_on_rocm_windows,
    probe_integrated_flags,
)

MODULE = "invokeai.app.util.rocm_integrated_gpu"


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([False, True], "0"),
        ([True, False], "1"),
        ([False, True, False], "0,2"),
        ([False, False], None),
        # An APU on its own is what the machine generates on.
        ([True], None),
        ([], None),
    ],
)
def test_discrete_device_visibility(flags, expected):
    assert discrete_device_visibility(flags) == expected


def _child_runs(monkeypatch: pytest.MonkeyPatch, code: str, timeout: float = 60) -> None:
    """Make the probe start a real child running `code` instead of asking torch."""
    monkeypatch.setattr(rocm_integrated_gpu, "_PROBE", code)
    monkeypatch.setattr(rocm_integrated_gpu, "_PROBE_TIMEOUT_SECONDS", timeout)


def test_probe_skips_what_the_child_prints_before_its_answer(monkeypatch: pytest.MonkeyPatch):
    _child_runs(monkeypatch, "print('W1002 flop_counter.py:29] triton not found'); print('1'); print('[false, true]')")
    assert probe_integrated_flags() == [False, True]


def test_an_answer_followed_by_a_hang_on_exit_returns_at_once(monkeypatch: pytest.MonkeyPatch):
    """HIP's teardown can hang after the answer (seen on Windows); startup must not wait out the timeout for it."""
    _child_runs(monkeypatch, "import sys, time; print('[false, true]'); sys.stdout.flush(); time.sleep(600)")
    started = time.monotonic()
    assert probe_integrated_flags() == [False, True]
    assert time.monotonic() - started < 30


@pytest.mark.parametrize(
    ("code", "timeout", "reason"),
    [
        ("import sys; sys.exit(3)", 60, "exited with code 3"),
        ("print('not json')", 60, "gave no answer"),
        ("print('{\"devices\": 2}')", 60, "gave no answer"),
        ("import time; time.sleep(600)", 1, "no answer within 1 s"),
    ],
    ids=["child-failed", "garbage", "not-a-list", "hangs-before-answering"],
)
def test_probe_says_why_the_child_could_not_answer(monkeypatch: pytest.MonkeyPatch, code, timeout, reason):
    _child_runs(monkeypatch, code, timeout)
    with pytest.raises(ProbeFailed, match=reason):
        probe_integrated_flags()


def test_probe_says_when_the_child_cannot_start(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rocm_integrated_gpu.sys, "executable", "Z:/no/such/python.exe")
    with pytest.raises(ProbeFailed, match="could not start it"):
        probe_integrated_flags()


def test_the_real_probe_answers_and_exits(monkeypatch: pytest.MonkeyPatch):
    """The probe code itself, in a real child: it prints a JSON list and exits (without a ROCm GPU, maybe [])."""
    # A cold torch import on a busy CI runner may take longer than startup should wait; this tests the code, not speed.
    monkeypatch.setattr(rocm_integrated_gpu, "_PROBE_TIMEOUT_SECONDS", 300)
    flags = probe_integrated_flags()
    assert isinstance(flags, list)
    assert all(isinstance(flag, bool) for flag in flags)


@pytest.fixture
def windows_rocm(monkeypatch: pytest.MonkeyPatch):
    """Windows, a ROCm torch not yet imported, no visibility variable set."""
    for var in VISIBILITY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(rocm_integrated_gpu.sys, "platform", "win32")
    monkeypatch.setattr(rocm_integrated_gpu, "_installed_torch_is_rocm", lambda: True)
    monkeypatch.setattr(rocm_integrated_gpu, "_torch_is_imported", lambda: False)
    yield monkeypatch
    # hide_integrated_gpus_on_rocm_windows writes os.environ directly.
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)


def _hide(flags):
    """Run the startup step with the probe answering `flags`, or failing when `flags` is an exception."""
    logger = MagicMock(spec=logging.Logger)
    kwargs = {"side_effect": flags} if isinstance(flags, BaseException) else {"return_value": flags}
    with patch(f"{MODULE}.probe_integrated_flags", **kwargs) as probe:
        hide_integrated_gpus_on_rocm_windows(logger)
    return logger, probe


def test_the_ryzen_igpu_next_to_a_radeon_is_hidden(windows_rocm):
    """The measured machine: RX 9060 XT at 0, Ryzen iGPU (gfx1036) at 1."""
    logger, _ = _hide([False, True])
    assert os.environ["HIP_VISIBLE_DEVICES"] == "0"
    assert "HIP_VISIBLE_DEVICES=0" in logger.info.call_args.args[0]


@pytest.mark.parametrize("flags", [[False], [False, False], [True]], ids=["one", "two-discrete", "apu"])
def test_nothing_is_hidden_without_an_igpu_next_to_a_discrete_gpu(windows_rocm, flags):
    _hide(flags)
    assert "HIP_VISIBLE_DEVICES" not in os.environ


def test_a_failed_probe_leaves_every_gpu_visible_and_says_so(windows_rocm):
    logger, _ = _hide(ProbeFailed("no answer within 60 s"))
    assert "HIP_VISIBLE_DEVICES" not in os.environ
    logger.warning.assert_called_once()
    assert "no answer within 60 s" in logger.warning.call_args.args[0]


@pytest.mark.parametrize("var", VISIBILITY_ENV_VARS)
def test_a_visibility_variable_already_set_wins(windows_rocm, var):
    windows_rocm.setenv(var, "1")
    _, probe = _hide([False, True])
    probe.assert_not_called()
    assert os.environ[var] == "1"


def test_other_platforms_and_builds_are_left_alone(windows_rocm):
    windows_rocm.setattr(rocm_integrated_gpu.sys, "platform", "linux")
    _, probe = _hide([False, True])
    probe.assert_not_called()

    windows_rocm.setattr(rocm_integrated_gpu.sys, "platform", "win32")
    windows_rocm.setattr(rocm_integrated_gpu, "_installed_torch_is_rocm", lambda: False)
    _, probe = _hide([False, True])
    probe.assert_not_called()


def test_too_late_once_torch_is_imported(windows_rocm):
    windows_rocm.setattr(rocm_integrated_gpu, "_torch_is_imported", lambda: True)
    logger, probe = _hide([False, True])
    probe.assert_not_called()
    logger.warning.assert_called_once()
