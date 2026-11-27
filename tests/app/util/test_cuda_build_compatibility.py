"""The startup check names the two ways a CUDA build fails on an NVIDIA machine, and stays quiet otherwise.

Hardware is faked: CI has no GPU, and the cases that matter (an R570 driver, a Pascal card) are not on the dev machine
either. The fakes stand in for torch's device queries and the driver probe only; the decision is the code under test.
"""

import logging

import pytest
import torch

from invokeai.app.util import startup_utils
from invokeai.app.util.startup_utils import check_cuda_build_compatibility

LOGGER = "test_cuda_build_compatibility"


@pytest.fixture
def cuda_build(monkeypatch: pytest.MonkeyPatch):
    def configure(*, cuda: str | None = "13.0", hip: str | None = None, available: bool, driver: int | None = None):
        monkeypatch.setattr(torch.version, "cuda", cuda)
        monkeypatch.setattr(torch.version, "hip", hip)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
        monkeypatch.setattr(startup_utils, "nvidia_driver_cuda_version", lambda: driver)

    return configure


@pytest.fixture
def gpus(monkeypatch: pytest.MonkeyPatch):
    def configure(*devices: tuple[str, tuple[int, int]], arch_list: list[str]):
        monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: arch_list)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: len(devices))
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: devices[index][0])
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: devices[index][1])

    return configure


def errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelno == logging.ERROR]


def run(caplog: pytest.LogCaptureFixture) -> list[str]:
    with caplog.at_level(logging.INFO, logger=LOGGER):
        check_cuda_build_compatibility(logging.getLogger(LOGGER))
    return errors(caplog)


def test_a_driver_too_old_for_the_build_is_named_with_both_versions(cuda_build, caplog) -> None:
    cuda_build(available=False, driver=12080)

    [message] = run(caplog)

    assert "CUDA 13.0" in message and "CUDA 12.8" in message and "R580" in message


@pytest.mark.parametrize("driver", [None, 13000, 13030])
def test_no_error_without_a_driver_or_with_a_new_enough_one(cuda_build, caplog, driver: int | None) -> None:
    # No driver: a CPU-only machine that got the CUDA build (PyPI's Linux torch is one). New enough: no NVIDIA GPU.
    cuda_build(available=False, driver=driver)

    assert run(caplog) == []


def test_a_gpu_below_the_lowest_built_architecture_is_named(cuda_build, gpus, caplog) -> None:
    cuda_build(available=True)
    gpus(("NVIDIA GeForce GTX 1080", (6, 1)), ("NVIDIA GeForce RTX 4090", (8, 9)), arch_list=["sm_75", "sm_120"])

    [message] = run(caplog)

    assert "GTX 1080" in message and "6.1" in message and "7.5" in message


def test_supported_gpus_pass_and_ptx_entries_do_not_lower_the_floor(cuda_build, gpus, caplog) -> None:
    cuda_build(available=True)
    gpus(("NVIDIA GeForce GTX 1660", (7, 5)), arch_list=["sm_75", "sm_120", "compute_50"])

    assert run(caplog) == []


def test_suffixed_architectures_from_custom_builds_are_understood(cuda_build, gpus, caplog) -> None:
    cuda_build(available=True)
    gpus(("NVIDIA TITAN V", (7, 0)), arch_list=["sm_75", "sm_90a", "sm_100f"])

    [message] = run(caplog)

    assert "TITAN V" in message and "7.5" in message


@pytest.mark.parametrize(("cuda", "hip"), [(None, None), (None, "7.15.26333")])
def test_cpu_and_rocm_builds_are_not_checked(cuda_build, caplog, cuda, hip) -> None:
    cuda_build(cuda=cuda, hip=hip, available=False, driver=12080)

    assert run(caplog) == []
