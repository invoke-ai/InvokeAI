"""Device classes and environments: what a measurement is keyed by, from scripted device properties."""

import platform
import sys
from types import SimpleNamespace

import pytest
import torch

from invokeai.backend.util import nvml
from invokeai.backend.working_memory.fingerprint import DeviceClass, device_class_for, environment_for, os_family

GIB = 2**30
CUDA0 = torch.device("cuda", 0)


def _cuda_cards(monkeypatch: pytest.MonkeyPatch, cards: list[tuple[str, SimpleNamespace]], hip: str | None) -> None:
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: cards[torch.device(device).index][1])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: cards[torch.device(device).index][0])


def _props(total_bytes: int, bus: int, minor: int = 9, gfx: str = "") -> SimpleNamespace:
    return SimpleNamespace(total_memory=total_bytes, major=8, minor=minor, pci_bus_id=bus, gcnArchName=gfx)


def test_identical_cards_share_a_device_class_and_other_cards_do_not(monkeypatch: pytest.MonkeyPatch):
    rtx_4090 = "NVIDIA GeForce RTX 4090"
    _cuda_cards(
        monkeypatch,
        [
            (rtx_4090, _props(24 * GIB, bus=1)),
            (rtx_4090, _props(24 * GIB, bus=2)),  # the same card in another slot
            ("NVIDIA GeForce RTX 4080", _props(16 * GIB, bus=3)),
        ],
        hip=None,
    )

    first, second, other = (device_class_for(torch.device("cuda", index)) for index in range(3))

    assert first == second and first.id == second.id
    assert other.id != first.id
    assert (first.vendor, first.name, first.total_mib, first.backend, first.arch) == (
        "nvidia",
        rtx_4090,
        24 * 1024,
        "cuda",
        "sm_89",
    )
    assert DeviceClass.model_validate_json(first.model_dump_json()).id == first.id


def test_a_hip_device_class_names_its_gfx_architecture(monkeypatch: pytest.MonkeyPatch):
    _cuda_cards(monkeypatch, [("AMD Radeon RX 9060 XT", _props(16 * GIB, bus=3, gfx="gfx1200"))], hip="7.14.0")

    device_class = device_class_for(CUDA0)

    assert (device_class.vendor, device_class.backend, device_class.arch) == ("amd", "hip", "gfx1200")


def test_a_cpu_has_no_device_class():
    with pytest.raises(ValueError, match="not a GPU"):
        device_class_for(torch.device("cpu"))


_SETTINGS = {
    "pytorch_cuda_alloc_conf": None,
    "attention_type": "sdpa",
    "attention_slice_size": "auto",
    "sequential_guidance": False,
    "device_working_mem_gb": 3.0,
}


@pytest.fixture
def cuda_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(nvml, "driver_version", lambda: "581.57")
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)


def test_settings_are_hard_facts_and_full_versions_soft_ones(cuda_build):
    environment = environment_for(CUDA0, _SETTINGS)

    assert environment == environment_for(CUDA0, dict(_SETTINGS)), "deterministic"
    assert {key: environment.hard[f"setting.{key}"] for key in _SETTINGS} == {
        "pytorch_cuda_alloc_conf": "None",
        "attention_type": "sdpa",
        "attention_slice_size": "auto",
        "sequential_guidance": "False",
        "device_working_mem_gb": "3.0",
    }
    assert {
        "torch",
        "cuda",
        "cudnn",
        "allocator_backend",
        "rocm_sdpa_guard",
        "env.PYTORCH_CUDA_ALLOC_CONF",
        "env.TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL",
        "env.INVOKEAI_ROCM_CONV3D",
    } <= set(environment.hard)
    assert environment.hard["torch"].count(".") == 1, "major.minor only"
    assert set(environment.soft) == {"torch_build", "driver", "os_build", "invokeai"}
    assert environment.soft["os_build"] == platform.platform()
    assert environment.soft["driver"] == "581.57"


def test_only_hard_facts_change_the_environment_id(cuda_build, monkeypatch: pytest.MonkeyPatch):
    reference = environment_for(CUDA0, _SETTINGS).id

    monkeypatch.setattr(platform, "platform", lambda: "Windows-11-10.0.26100-SP0")
    monkeypatch.setattr(nvml, "driver_version", lambda: "590.01")
    assert environment_for(CUDA0, _SETTINGS).id == reference

    assert environment_for(CUDA0, {**_SETTINGS, "attention_type": "sliced"}).id != reference

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    changed = environment_for(CUDA0, _SETTINGS)
    assert changed.hard["env.PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    assert changed.id != reference


@pytest.mark.parametrize(
    ("platform_name", "family"),
    [("win32", "windows"), ("linux", "linux"), ("linux2", "linux"), ("darwin", "other"), ("freebsd14", "other")],
)
def test_the_os_family_is_named_from_sys_platform(platform_name: str, family: str):
    assert os_family(platform_name) == family


def test_the_operating_system_is_a_hard_fact(cuda_build, monkeypatch: pytest.MonkeyPatch):
    ids = {}
    for platform_name, family in (("win32", "windows"), ("linux", "linux"), ("darwin", "other")):
        monkeypatch.setattr(sys, "platform", platform_name)
        environment = environment_for(CUDA0, _SETTINGS)
        assert environment.hard["os"] == family
        ids[family] = environment.id

    assert len(set(ids.values())) == 3, "the same card and software on another OS is another environment"
