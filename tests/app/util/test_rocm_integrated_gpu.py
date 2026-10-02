"""Hiding integrated GPUs from HIP on Windows ROCm before torch initializes it."""

import logging
import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from invokeai.app.util import rocm_integrated_gpu
from invokeai.app.util.rocm_integrated_gpu import (
    VISIBILITY_ENV_VARS,
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


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


def test_probe_reads_the_last_line_past_torch_warnings():
    stdout = "W1002 flop_counter.py:29] triton not found\n[false, true]\n"
    with patch(f"{MODULE}.subprocess.run", return_value=_completed(stdout)):
        assert probe_integrated_flags() == [False, True]


@pytest.mark.parametrize(
    "outcome",
    [
        _completed("", returncode=1),
        _completed("not json\n"),
        _completed('{"devices": 2}\n'),
        _completed(""),
        subprocess.TimeoutExpired(cmd="python", timeout=120),
        OSError("no interpreter"),
    ],
    ids=["child-failed", "garbage", "not-a-list", "empty", "timeout", "oserror"],
)
def test_probe_answers_none_when_the_child_cannot(outcome):
    kwargs = {"side_effect": outcome} if isinstance(outcome, BaseException) else {"return_value": outcome}
    with patch(f"{MODULE}.subprocess.run", **kwargs):
        assert probe_integrated_flags() is None


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


def _hide(flags, device="auto", generation_devices="auto"):
    logger = MagicMock(spec=logging.Logger)
    with patch(f"{MODULE}.probe_integrated_flags", return_value=flags) as probe:
        hide_integrated_gpus_on_rocm_windows(device, generation_devices, logger)
    return logger, probe


def test_the_ryzen_igpu_next_to_a_radeon_is_hidden(windows_rocm):
    """The measured machine: RX 9060 XT at 0, Ryzen iGPU (gfx1036) at 1."""
    logger, _ = _hide([False, True])
    assert os.environ["HIP_VISIBLE_DEVICES"] == "0"
    assert "HIP_VISIBLE_DEVICES=0" in logger.info.call_args.args[0]


@pytest.mark.parametrize(
    "flags", [[False], [False, False], [True], None], ids=["one", "two-discrete", "apu", "unknown"]
)
def test_nothing_is_hidden_without_an_igpu_next_to_a_discrete_gpu(windows_rocm, flags):
    _hide(flags)
    assert "HIP_VISIBLE_DEVICES" not in os.environ


@pytest.mark.parametrize("var", VISIBILITY_ENV_VARS)
def test_a_visibility_variable_already_set_wins(windows_rocm, var):
    windows_rocm.setenv(var, "1")
    _, probe = _hide([False, True])
    probe.assert_not_called()
    assert os.environ[var] == "1"


@pytest.mark.parametrize(
    ("device", "generation_devices"),
    [("cuda:1", "auto"), ("auto", ["cuda:0", "cuda:1"])],
    ids=["legacy-device", "explicit-list"],
)
def test_explicit_device_indices_are_not_renumbered(windows_rocm, device, generation_devices):
    """`cuda:N` counts in HIP's full enumeration; hiding a device would point it at another GPU."""
    _, probe = _hide([False, True], device=device, generation_devices=generation_devices)
    probe.assert_not_called()


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
