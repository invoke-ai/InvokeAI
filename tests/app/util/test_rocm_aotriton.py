"""The `rocm_aotriton_experimental` setting: which builds and GPUs get AOTriton's experimental attention kernels."""

import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from invokeai.app.util import rocm_aotriton
from invokeai.app.util.rocm_aotriton import (
    AOTRITON_EXPERIMENTAL_ENV,
    apply_rocm_aotriton_setting,
    resolve_aotriton_experimental,
    rocm_major_version,
)


@pytest.mark.parametrize(
    ("rocm", "version", "expected"),
    [
        # AMD's wheels: both agree.
        ("10.0.0", "2.13.0+rocm10.0.0", 10),
        # A build that leaves torch.version.rocm unset: the label tells.
        (None, "2.13.0+rocm7.2", 7),
        (None, "2.10.0+rocm7.1", 7),
        # A locally built ROCm torch has no label; torch.version.rocm still names the ROCm.
        ("10.0.0", "2.8.0a0+gitfc14c65", 10),
        (None, "2.8.0a0+gitfc14c65", None),
        (None, "2.13.0+cu130", None),
        (None, "2.13.0", None),
    ],
)
def test_rocm_major_version_prefers_torch_version_rocm(rocm: str | None, version: str, expected: int | None):
    assert rocm_major_version(rocm, version) == expected


def test_auto_turns_the_kernels_on_for_rocm_10_on_gfx1200():
    """The measured case: RX 9060 XT, torch 2.13.0+rocm10.0.0."""
    value, reason = resolve_aotriton_experimental("auto", None, 10, {"gfx1200"})
    assert value == "1"
    assert "gfx1200" in reason


def test_auto_leaves_rocm_7_alone_where_the_switch_broke_every_fused_call():
    """torch 2.12+rocm7.14.1 on the same gfx1200: every flash/mem-efficient call failed with the switch on."""
    assert resolve_aotriton_experimental("auto", None, 7, {"gfx1200"})[0] is None


def test_auto_leaves_an_unlabelled_build_alone():
    assert resolve_aotriton_experimental("auto", None, None, {"gfx1200"})[0] is None


def test_auto_needs_every_generation_gpu_to_be_measured():
    """The switch is process-wide, so one unmeasured GPU keeps it off for all."""
    value, reason = resolve_aotriton_experimental("auto", None, 10, {"gfx1200", "gfx1101"})
    assert value is None
    assert "gfx1101" in reason


def test_auto_without_a_rocm_gpu_does_nothing():
    assert resolve_aotriton_experimental("auto", None, 10, set())[0] is None


@pytest.mark.parametrize(("setting", "expected"), [("on", "1"), ("off", "0")])
@pytest.mark.parametrize("rocm_major", [None, 7, 10])
def test_on_and_off_decide_for_any_build_and_gpu(setting, expected, rocm_major):
    assert resolve_aotriton_experimental(setting, None, rocm_major, {"gfx1101"})[0] == expected


@pytest.mark.parametrize("setting", ["auto", "on", "off"])
def test_a_value_in_the_environment_always_wins(setting):
    assert resolve_aotriton_experimental(setting, "0", 10, {"gfx1200"})[0] is None
    assert resolve_aotriton_experimental(setting, "1", 7, {"gfx1101"})[0] is None


# ===== applying it at startup ===============================================


@pytest.fixture
def no_exported_switch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(AOTRITON_EXPERIMENTAL_ENV, raising=False)
    yield
    # apply_rocm_aotriton_setting writes os.environ directly; monkeypatch restores it on teardown.
    monkeypatch.delenv(AOTRITON_EXPERIMENTAL_ENV, raising=False)


def _rocm_build(version: str = "2.13.0+rocm10.0.0", arch: str = "gfx1200", rocm: str | None = "10.0.0"):
    """Patches standing in for a ROCm torch with one GPU of `arch` to generate on.

    Both version sources are patched, so the host's own torch (CUDA, CPU or another ROCm) cannot decide the outcome.
    """
    return (
        patch("torch.version.hip", "7.15.26333"),
        patch.object(torch, "__version__", version),
        patch("torch.version.rocm", rocm, create=True),
        patch(
            "invokeai.backend.util.devices.TorchDevice.get_generation_devices",
            return_value=[torch.device("cuda", 0)],
        ),
        patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(gcnArchName=f"{arch}:xnack-")),
    )


def _apply(setting, patches, logger=None):
    logger = logger or MagicMock(spec=logging.Logger)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        apply_rocm_aotriton_setting(setting, "auto", logger)
    return logger


def test_apply_exports_the_switch_for_the_measured_case(no_exported_switch):
    logger = _apply("auto", _rocm_build())
    assert os.environ[AOTRITON_EXPERIMENTAL_ENV] == "1"
    assert "are on" in logger.info.call_args.args[0]


def test_apply_strips_the_feature_suffix_from_the_arch_name(no_exported_switch):
    """`gcnArchName` carries target features (`gfx1200:xnack-`); the measured list holds bare names."""
    _apply("auto", _rocm_build(arch="gfx1200"))
    assert os.environ.get(AOTRITON_EXPERIMENTAL_ENV) == "1"


@pytest.mark.parametrize(
    ("version", "rocm", "exported"),
    [
        # A local ROCm 10 build: no label, but torch.version.rocm names it.
        ("2.8.0a0+gitfc14c65", "10.0.0", "1"),
        # torch.version.rocm wins over the label.
        ("2.13.0+rocm10.0.0", "7.2.0", None),
        # Without torch.version.rocm the label decides.
        ("2.13.0+rocm10.0.0", None, "1"),
    ],
    ids=["local-build", "attribute-wins", "label-fallback"],
)
def test_apply_reads_the_rocm_version_from_torch(no_exported_switch, version, rocm, exported):
    _apply("auto", _rocm_build(version=version, rocm=rocm))
    assert os.environ.get(AOTRITON_EXPERIMENTAL_ENV) == exported


def test_apply_leaves_an_unmeasured_gpu_unset_and_says_how_to_opt_in(no_exported_switch):
    logger = _apply("auto", _rocm_build(arch="gfx1151"))
    assert AOTRITON_EXPERIMENTAL_ENV not in os.environ
    message = logger.info.call_args.args[0]
    assert "are off" in message and "rocm_aotriton_experimental: on" in message


def test_apply_keeps_an_exported_value_and_warns_when_the_setting_disagrees(monkeypatch):
    monkeypatch.setenv(AOTRITON_EXPERIMENTAL_ENV, "0")
    logger = _apply("on", _rocm_build())
    assert os.environ[AOTRITON_EXPERIMENTAL_ENV] == "0"
    assert AOTRITON_EXPERIMENTAL_ENV in logger.warning.call_args.args[0]


def test_apply_does_nothing_on_a_non_rocm_build(no_exported_switch):
    logger = MagicMock(spec=logging.Logger)
    with patch("torch.version.hip", None):
        apply_rocm_aotriton_setting("auto", "auto", logger)
        apply_rocm_aotriton_setting("on", "auto", logger)
    assert AOTRITON_EXPERIMENTAL_ENV not in os.environ
    logger.info.assert_not_called()
    # Only the explicit 'on' is worth a word.
    assert logger.warning.call_count == 1


def test_the_measured_list_only_holds_measured_architectures():
    """Widening `auto` needs a measurement on that GPU; this pins the list so a change is a deliberate edit."""
    assert rocm_aotriton.AUTO_MEASURED_ARCHS == frozenset({"gfx1200"})
    assert rocm_aotriton.MIN_AUTO_ROCM_MAJOR == 10
