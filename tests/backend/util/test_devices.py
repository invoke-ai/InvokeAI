"""
Test abstract device class.
"""

import ctypes
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from invokeai.app.services.config import get_config
from invokeai.backend.util.devices import TorchDevice, choose_precision, choose_torch_device, torch_dtype

devices = ["cpu", "cuda:0", "cuda:1", "cuda:2", "mps"]
device_types_cpu = [("cpu", torch.float32), ("cuda:0", torch.float32), ("mps", torch.float32)]
device_types_cuda = [("cpu", torch.float32), ("cuda:0", torch.float16), ("mps", torch.float32)]
device_types_mps = [("cpu", torch.float32), ("cuda:0", torch.float32), ("mps", torch.float16)]


@pytest.mark.parametrize("device_name", devices)
def test_device_choice(device_name):
    config = get_config()
    config.device = device_name
    torch_device = TorchDevice.choose_torch_device()
    assert torch_device == torch.device(device_name)


# ===== per-thread session device (multi-GPU worker pinning) ================


def test_session_device_overrides_config():
    """A per-thread session device takes precedence over the global config.device."""
    config = get_config()
    config.device = "cpu"
    try:
        TorchDevice.set_session_device("cuda:1")
        assert TorchDevice.choose_torch_device() == torch.device("cuda:1")
    finally:
        TorchDevice.clear_session_device()
    # Once cleared, we fall back to the global config.
    assert TorchDevice.choose_torch_device() == torch.device("cpu")


def test_session_device_is_thread_local():
    """Each thread sees only its own pinned device; the main thread is unaffected."""
    config = get_config()
    config.device = "cpu"
    results: dict[str, torch.device] = {}
    barrier = threading.Barrier(2)

    def worker(name: str, device: str):
        TorchDevice.set_session_device(device)
        # Wait so both threads have set their device before either reads it, proving isolation.
        barrier.wait()
        results[name] = TorchDevice.choose_torch_device()
        TorchDevice.clear_session_device()

    t0 = threading.Thread(target=worker, args=("a", "cuda:0"))
    t1 = threading.Thread(target=worker, args=("b", "cuda:1"))
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    assert results["a"] == torch.device("cuda:0")
    assert results["b"] == torch.device("cuda:1")
    # The main thread never set a session device, so it still uses the global config.
    assert TorchDevice.get_session_device() is None
    assert TorchDevice.choose_torch_device() == torch.device("cpu")


# ===== generation_devices resolution (config -> concrete device list) =======


def test_get_generation_devices_auto_expands_to_all_cuda():
    """`auto` enumerates every visible CUDA device (when the legacy `device` is not pinned)."""
    config = get_config()
    config.device = "auto"
    with (
        patch("invokeai.backend.util.devices.torch.cuda.is_available", return_value=True),
        patch("invokeai.backend.util.devices.torch.cuda.device_count", return_value=3),
    ):
        assert TorchDevice.get_generation_devices("auto") == [
            torch.device("cuda:0"),
            torch.device("cuda:1"),
            torch.device("cuda:2"),
        ]


def test_get_generation_devices_auto_respects_pinned_legacy_device():
    """An upgraded install that pinned the legacy `device` setting (e.g. to keep generation off a
    display GPU) must not silently start workers on every GPU: `auto` resolves to the pinned device
    only. An explicit generation_devices list still overrides the pin."""
    config = get_config()
    config.device = "cuda:1"
    try:
        with (
            patch("invokeai.backend.util.devices.torch.cuda.is_available", return_value=True),
            patch("invokeai.backend.util.devices.torch.cuda.device_count", return_value=3),
        ):
            assert TorchDevice.get_generation_devices("auto") == [torch.device("cuda:1")]
            # An explicit list wins over the legacy pin.
            assert TorchDevice.get_generation_devices(["cuda:0", "cuda:2"]) == [
                torch.device("cuda:0"),
                torch.device("cuda:2"),
            ]
    finally:
        config.device = "auto"


def test_get_generation_devices_auto_without_cuda():
    """`auto` falls back to the single best device when no accelerator is available."""
    config = get_config()
    config.device = "auto"
    with (
        patch("invokeai.backend.util.devices.torch.cuda.is_available", return_value=False),
        patch("invokeai.backend.util.devices._xpu_is_available", return_value=False),
        patch("invokeai.backend.util.devices.torch.backends.mps.is_available", return_value=False),
    ):
        assert TorchDevice.get_generation_devices("auto") == [torch.device("cpu")]


def test_get_generation_devices_explicit_list_is_deduplicated():
    """An explicit list is normalized and deduplicated, preserving order."""
    # Mock CUDA as present so the device-existence validation passes on CPU-only runners.
    with (
        patch("invokeai.backend.util.devices.torch.cuda.is_available", return_value=True),
        patch("invokeai.backend.util.devices.torch.cuda.device_count", return_value=2),
    ):
        assert TorchDevice.get_generation_devices(["cuda:0", "cuda:0", "cuda:1"]) == [
            torch.device("cuda:0"),
            torch.device("cuda:1"),
        ]


@pytest.mark.parametrize("value", [None, []])
def test_get_generation_devices_empty(value):
    """`None` or an empty list resolves to an empty list (caller handles the single-device fallback)."""
    assert TorchDevice.get_generation_devices(value) == []


def test_generation_devices_summary_suffix_tied_to_full_device_set():
    """A device's `#N` suffix stays tied to its cuda index even when other devices are disabled.

    Regression test: with 4 identically-named GPUs, disabling cuda:1 must leave cuda:2/cuda:3 as
    #3/#4 (not renumber them to #2/#3), so the backend log matches the frontend's labeling.
    """
    with (
        patch("invokeai.backend.util.devices.torch.cuda.is_available", return_value=True),
        patch("invokeai.backend.util.devices.torch.cuda.device_count", return_value=4),
        patch("invokeai.backend.util.devices.torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 4090"),
    ):
        summary = TorchDevice.get_generation_devices_summary(["cuda:0", "cuda:2", "cuda:3"])
        assert summary == (
            "[NVIDIA GeForce RTX 4090 #1 (cuda:0), "
            "NVIDIA GeForce RTX 4090 #3 (cuda:2), "
            "NVIDIA GeForce RTX 4090 #4 (cuda:3)]"
        )


def test_generation_devices_summary_single_device_has_no_suffix():
    """A single resolved device is summarized by bare name, with no bracket or `#N` suffix."""
    with (
        patch("invokeai.backend.util.devices.torch.cuda.is_available", return_value=True),
        patch("invokeai.backend.util.devices.torch.cuda.device_count", return_value=4),
        patch("invokeai.backend.util.devices.torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 4090"),
    ):
        assert TorchDevice.get_generation_devices_summary(["cuda:2"]) == "NVIDIA GeForce RTX 4090"


@pytest.mark.parametrize("device_dtype_pair", device_types_cpu)
def test_device_dtype_cpu(device_dtype_pair):
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        device_name, dtype = device_dtype_pair
        config = get_config()
        config.device = device_name
        torch_dtype = TorchDevice.choose_torch_dtype()
        assert torch_dtype == dtype


@pytest.mark.parametrize("device_dtype_pair", device_types_cuda)
def test_device_dtype_cuda(device_dtype_pair):
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_name", return_value="RTX4070"),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        device_name, dtype = device_dtype_pair
        config = get_config()
        config.device = device_name
        torch_dtype = TorchDevice.choose_torch_dtype()
        assert torch_dtype == dtype


@pytest.mark.parametrize("device_dtype_pair", device_types_mps)
def test_device_dtype_mps(device_dtype_pair):
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        device_name, dtype = device_dtype_pair
        config = get_config()
        config.device = device_name
        torch_dtype = TorchDevice.choose_torch_dtype()
        assert torch_dtype == dtype


@pytest.mark.parametrize("device_dtype_pair", device_types_cuda)
def test_device_dtype_override(device_dtype_pair):
    with (
        patch("torch.cuda.get_device_name", return_value="RTX4070"),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        device_name, dtype = device_dtype_pair
        config = get_config()
        config.device = device_name
        config.precision = "float32"
        torch_dtype = TorchDevice.choose_torch_dtype()
        assert torch_dtype == torch.float32


def test_normalize():
    assert (
        TorchDevice.normalize("cuda") == torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cuda")
    )
    assert (
        TorchDevice.normalize("cuda:0") == torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cuda")
    )
    assert (
        TorchDevice.normalize("cuda:1") == torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cuda")
    )
    assert TorchDevice.normalize("mps") == torch.device("mps")
    assert TorchDevice.normalize("cpu") == torch.device("cpu")


@pytest.mark.parametrize("device_name", devices)
def test_legacy_device_choice(device_name):
    config = get_config()
    config.device = device_name
    with pytest.deprecated_call():
        torch_device = choose_torch_device()
    assert torch_device == torch.device(device_name)


@pytest.mark.parametrize("device_dtype_pair", device_types_cpu)
def test_legacy_device_dtype_cpu(device_dtype_pair):
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
        patch("torch.cuda.get_device_name", return_value="RTX9090"),
    ):
        device_name, dtype = device_dtype_pair
        config = get_config()
        config.device = device_name
        with pytest.deprecated_call():
            torch_device = choose_torch_device()
            returned_dtype = torch_dtype(torch_device)
        assert returned_dtype == dtype


def test_legacy_precision_name():
    config = get_config()
    config.precision = "auto"
    with (
        pytest.deprecated_call(),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.backends.mps.is_available", return_value=True),
        patch("torch.cuda.get_device_name", return_value="RTX9090"),
    ):
        assert "float16" == choose_precision(torch.device("cuda"))
        assert "float16" == choose_precision(torch.device("mps"))
        assert "float32" == choose_precision(torch.device("cpu"))


# ===== choose_anima_inference_dtype (config.precision honoring) ============


def test_choose_anima_inference_dtype_float16():
    """precision='float16' returns torch.float16 without touching hardware."""
    config = get_config()
    config.precision = "float16"
    result = TorchDevice.choose_anima_inference_dtype(torch.device("cpu"))
    assert result is torch.float16


def test_choose_anima_inference_dtype_bfloat16():
    """precision='bfloat16' returns torch.bfloat16 without touching hardware."""
    config = get_config()
    config.precision = "bfloat16"
    result = TorchDevice.choose_anima_inference_dtype(torch.device("cpu"))
    assert result is torch.bfloat16


def test_choose_anima_inference_dtype_float32():
    """precision='float32' returns torch.float32 without touching hardware."""
    config = get_config()
    config.precision = "float32"
    result = TorchDevice.choose_anima_inference_dtype(torch.device("cpu"))
    assert result is torch.float32


def test_choose_anima_inference_dtype_auto_delegates_to_safe_dtype():
    """precision='auto' delegates to choose_bfloat16_safe_dtype (current behavior)."""
    config = get_config()
    config.precision = "auto"
    device = torch.device("cpu")
    sentinel = torch.bfloat16
    with patch.object(TorchDevice, "choose_bfloat16_safe_dtype", return_value=sentinel) as mock_safe:
        result = TorchDevice.choose_anima_inference_dtype(device)
    assert result is sentinel
    mock_safe.assert_called_once_with(device)


@patch("torch.cuda.device_count", return_value=2)
@patch("torch.cuda.is_available", return_value=True)
def test_get_generation_devices_rejects_out_of_range_cuda(mock_avail, mock_count):
    # cuda:2 does not exist on a 2-GPU machine — fail fast instead of deferring to first allocation.
    with pytest.raises(ValueError, match="only 2 CUDA"):
        TorchDevice.get_generation_devices(["cuda:2"])


@patch("torch.cuda.device_count", return_value=2)
@patch("torch.cuda.is_available", return_value=True)
def test_get_generation_devices_accepts_in_range_cuda(mock_avail, mock_count):
    assert [str(d) for d in TorchDevice.get_generation_devices(["cuda:1"])] == ["cuda:1"]


@patch("torch.cuda.is_available", return_value=False)
def test_get_generation_devices_rejects_cuda_when_unavailable(mock_avail):
    with pytest.raises(ValueError, match="no CUDA"):
        TorchDevice.get_generation_devices(["cuda:0"])


@patch("torch.backends.mps.is_available", return_value=False)
def test_get_generation_devices_rejects_mps_when_unavailable(mock_mps):
    # e.g. Linux, or an unsupported macOS build — fail fast instead of starting a worker and cache
    # pinned to a device that only errors at the first tensor operation.
    with pytest.raises(ValueError, match="MPS is not available"):
        TorchDevice.get_generation_devices(["mps"])


@patch("torch.backends.mps.is_available", return_value=True)
def test_get_generation_devices_accepts_mps_when_available(mock_mps):
    assert [str(d) for d in TorchDevice.get_generation_devices(["mps"])] == ["mps"]


# ===== XPU (Intel GPU) ======================================================

device_types_xpu = [
    ("cpu", torch.float32),
    ("cuda:0", torch.float32),
    ("mps", torch.float32),
    ("xpu", torch.float16),
    ("xpu:0", torch.float16),
]


@pytest.mark.parametrize("device_name", ["xpu:0", "xpu:1"])
def test_device_choice_xpu(device_name):
    """An explicit xpu:N device in the config is honored verbatim."""
    config = get_config()
    config.device = device_name
    assert TorchDevice.choose_torch_device() == torch.device(device_name)


def test_auto_device_prefers_xpu_over_cpu():
    """With no CUDA/MPS and an XPU present, `auto` selects (and normalizes) xpu."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
        patch("invokeai.backend.util.devices._xpu_is_available", return_value=True),
        patch("torch.xpu.current_device", return_value=0, create=True),
    ):
        config = get_config()
        config.device = "auto"
        assert TorchDevice.choose_torch_device() == torch.device("xpu", 0)


def test_auto_device_prefers_cuda_over_xpu():
    """CUDA outranks XPU in auto selection."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.current_device", return_value=0),
        patch("invokeai.backend.util.devices._xpu_is_available", return_value=True),
    ):
        config = get_config()
        config.device = "auto"
        assert TorchDevice.choose_torch_device().type == "cuda"


@pytest.mark.parametrize("device_dtype_pair", device_types_xpu)
def test_device_dtype_xpu(device_dtype_pair):
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
        patch("invokeai.backend.util.devices._xpu_is_available", return_value=True),
        patch("torch.xpu.current_device", return_value=0, create=True),
    ):
        device_name, dtype = device_dtype_pair
        config = get_config()
        config.device = device_name
        config.precision = "auto"
        assert TorchDevice.choose_torch_dtype() == dtype


def test_normalize_xpu():
    with (
        patch("invokeai.backend.util.devices._xpu_is_available", return_value=True),
        patch("torch.xpu.current_device", return_value=0, create=True),
    ):
        assert TorchDevice.normalize("xpu") == torch.device("xpu", 0)
        assert TorchDevice.normalize("xpu:1") == torch.device("xpu", 1)
    with patch("invokeai.backend.util.devices._xpu_is_available", return_value=False):
        assert TorchDevice.normalize("xpu") == torch.device("xpu")


# ===== TorchDevice.xpu_mem_get_info fallback ================================


def test_xpu_mem_get_info_native():
    """When torch.xpu.mem_get_info works, its result is passed through."""
    with patch.object(torch.xpu, "mem_get_info", return_value=(5, 10), create=True):
        assert TorchDevice.xpu_mem_get_info(torch.device("xpu")) == (5, 10)


def _install_fake_sysman(free: int, size: int, indices=(0,)):
    """Point the Sysman cache at a stub that reports fixed figures for the given device indices."""
    import invokeai.backend.util.level_zero as level_zero

    def fake_get_state(_handle, state_ref):
        state = state_ref._obj
        state.free = free
        state.size = size
        return 0

    lib = SimpleNamespace(zesMemoryGetState=fake_get_state)
    level_zero._sysman_attempted = True
    level_zero._sysman = (lib, {i: [ctypes.c_void_p(1)] for i in indices})
    return level_zero


def test_xpu_memory_info_resolves_an_index_less_device():
    """An index-less device must resolve to the current device rather than returning None.

    Returning None would silently skip the driver-global query and drop to the blind estimate --
    an accuracy regression with no visible symptom.
    """
    from invokeai.backend.util.level_zero import reset_cache, xpu_memory_info

    gib = 1 << 30
    level_zero = _install_fake_sysman(free=5 * gib, size=8 * gib, indices=(0,))
    try:
        with patch.object(torch.xpu, "current_device", return_value=0, create=True):
            assert xpu_memory_info(torch.device("xpu")) == (5 * gib, 8 * gib)
        assert xpu_memory_info(torch.device("xpu", 0)) == (5 * gib, 8 * gib)
        # A device Sysman has no module for yields None rather than another device's numbers.
        assert xpu_memory_info(torch.device("xpu", 3)) is None
        assert xpu_memory_info(torch.device("cpu")) is None
    finally:
        reset_cache()
        assert level_zero._sysman is None


def test_get_device_name_degrades_instead_of_raising():
    """Naming is for labels and logs only, so a backend that cannot answer must not take the
    caller down. torch.xpu.get_device_name goes through _lazy_init, which raises AssertionError
    (not RuntimeError) on a build without XPU."""
    with patch.object(
        torch.xpu, "get_device_name", side_effect=AssertionError("Torch not compiled with XPU enabled"), create=True
    ):
        assert TorchDevice.get_device_name(torch.device("xpu", 1)) == "xpu:1"
    with patch.object(torch.cuda, "get_device_name", side_effect=RuntimeError("no driver")):
        assert TorchDevice.get_device_name(torch.device("cuda", 0)) == "cuda:0"


def test_get_device_name_returns_backend_name_when_available():
    with patch.object(torch.xpu, "get_device_name", return_value="Intel(R) Arc(TM) Pro B70 Graphics", create=True):
        assert TorchDevice.get_device_name(torch.device("xpu", 0)) == "Intel(R) Arc(TM) Pro B70 Graphics"
    assert TorchDevice.get_device_name(torch.device("cpu")) == "CPU"


def test_xpu_mem_get_info_falls_back_to_sysman_before_estimating():
    """Level Zero Sysman is driver-global like mem_get_info, so it must be preferred over the
    process-local estimate, which is blind to VRAM held by other processes."""
    gib = 1 << 30
    with (
        patch.object(torch.xpu, "mem_get_info", side_effect=RuntimeError("aspect missing"), create=True),
        patch("invokeai.backend.util.devices.xpu_memory_info", return_value=(15 * gib, 32 * gib)) as mock_sysman,
        patch.object(
            torch.xpu, "get_device_properties", return_value=SimpleNamespace(total_memory=32 * gib), create=True
        ),
        patch.object(torch.xpu, "memory_reserved", return_value=0, create=True),
    ):
        # The blind estimate would say 32 GiB free here; Sysman's 15 GiB must win.
        assert TorchDevice.xpu_mem_get_info(torch.device("xpu")) == (15 * gib, 32 * gib)
    mock_sysman.assert_called_once()


def test_xpu_mem_get_info_unknown_total_raises():
    """An unreadable total_memory propagates rather than being reported as (0, 0).

    Returning (0, 0) would make ModelCache._get_vram_available collapse to a constant
    -working_mem budget for the life of the process (the vram_allocated term cancels
    against _get_vram_in_use), silently forcing per-layer autocast forever.
    """
    with (
        patch.object(torch.xpu, "mem_get_info", side_effect=RuntimeError(), create=True),
        patch.object(torch.xpu, "get_device_properties", side_effect=RuntimeError("unreadable"), create=True),
        patch.object(torch.xpu, "memory_reserved", side_effect=RuntimeError(), create=True),
    ):
        with pytest.raises(RuntimeError, match="unreadable"):
            TorchDevice.xpu_mem_get_info(torch.device("xpu"))


@pytest.mark.parametrize(
    "native_error",
    [
        RuntimeError("aspect missing"),
        # torch.xpu._lazy_init raises AssertionError, not RuntimeError, on a build without XPU.
        AssertionError("Torch not compiled with XPU enabled"),
    ],
    ids=["missing-sycl-aspect", "torch-without-xpu"],
)
def test_xpu_mem_get_info_estimates_when_native_and_sysman_both_fail(native_error):
    """Last resort: total_memory minus this process's reserved bytes.

    The native failure type moves between torch releases, so both spellings must reach the
    estimate rather than propagating.
    """
    gib = 1 << 30
    with (
        patch.object(torch.xpu, "mem_get_info", side_effect=native_error, create=True),
        patch("invokeai.backend.util.devices.xpu_memory_info", return_value=None),
        patch.object(
            torch.xpu, "get_device_properties", return_value=SimpleNamespace(total_memory=32 * gib), create=True
        ),
        patch.object(torch.xpu, "memory_reserved", return_value=2 * gib, create=True),
    ):
        assert TorchDevice.xpu_mem_get_info(torch.device("xpu")) == (30 * gib, 32 * gib)


def test_get_generation_devices_auto_expands_to_all_xpu():
    """With no CUDA, `auto` enumerates every visible XPU device."""
    config = get_config()
    config.device = "auto"
    with (
        patch("invokeai.backend.util.devices.torch.cuda.is_available", return_value=False),
        patch("invokeai.backend.util.devices._xpu_is_available", return_value=True),
        patch("torch.xpu.device_count", return_value=2, create=True),
    ):
        assert TorchDevice.get_generation_devices("auto") == [torch.device("xpu:0"), torch.device("xpu:1")]


def _auto_xpu_devices(integrated_map: dict[int, bool | None]):
    """Resolve `auto` with N XPU devices whose integrated-ness is given per index."""
    config = get_config()
    config.device = "auto"

    def fake_is_integrated(device: torch.device):
        return integrated_map.get(device.index)

    with (
        patch("invokeai.backend.util.devices.torch.cuda.is_available", return_value=False),
        patch("invokeai.backend.util.devices._xpu_is_available", return_value=True),
        patch("torch.xpu.device_count", return_value=len(integrated_map), create=True),
        patch("invokeai.backend.util.devices.xpu_device_is_integrated", side_effect=fake_is_integrated),
    ):
        return TorchDevice.get_generation_devices("auto")


def test_auto_excludes_integrated_gpu_when_a_discrete_one_exists():
    """The mainstream Arc config is iGPU + discrete card; `auto` must not dispatch to the iGPU."""
    assert _auto_xpu_devices({0: True, 1: False}) == [torch.device("xpu:1")]


def test_auto_keeps_integrated_gpu_when_it_is_the_only_device():
    """Dropping the only GPU would leave nothing to generate on."""
    assert _auto_xpu_devices({0: True}) == [torch.device("xpu:0")]


def test_auto_keeps_devices_of_unknown_type():
    """A None answer from the Level Zero probe must not narrow the device list on a guess."""
    assert _auto_xpu_devices({0: None, 1: None}) == [torch.device("xpu:0"), torch.device("xpu:1")]


def test_explicit_generation_devices_can_still_select_an_integrated_gpu():
    """The exclusion applies to `auto` only; naming a device opts into it."""
    config = get_config()
    config.device = "auto"
    with (
        patch("invokeai.backend.util.devices.torch.cuda.is_available", return_value=False),
        patch("invokeai.backend.util.devices._xpu_is_available", return_value=True),
        patch("torch.xpu.device_count", return_value=2, create=True),
        patch("invokeai.backend.util.devices.xpu_device_is_integrated", return_value=True),
    ):
        assert TorchDevice.get_generation_devices(["xpu:0"]) == [torch.device("xpu:0")]


def test_get_generation_devices_rejects_out_of_range_xpu():
    with (
        patch("invokeai.backend.util.devices._xpu_is_available", return_value=True),
        patch("torch.xpu.device_count", return_value=2, create=True),
    ):
        with pytest.raises(ValueError, match="only 2 XPU"):
            TorchDevice.get_generation_devices(["xpu:2"])


def test_get_generation_devices_rejects_xpu_when_unavailable():
    with patch("invokeai.backend.util.devices._xpu_is_available", return_value=False):
        with pytest.raises(ValueError, match="no XPU"):
            TorchDevice.get_generation_devices(["xpu:0"])


def test_generation_devices_summary_xpu_pair():
    """Two identically-named XPUs get stable #N suffixes, same as CUDA."""
    with (
        patch("invokeai.backend.util.devices.torch.cuda.is_available", return_value=False),
        patch("invokeai.backend.util.devices._xpu_is_available", return_value=True),
        patch("torch.xpu.device_count", return_value=2, create=True),
        patch("torch.xpu.get_device_name", return_value="Intel Arc Pro B70", create=True),
    ):
        summary = TorchDevice.get_generation_devices_summary(["xpu:0", "xpu:1"])
        assert summary == "[Intel Arc Pro B70 #1 (xpu:0), Intel Arc Pro B70 #2 (xpu:1)]"


def test_session_device_index_on_xpu():
    """Worker labels resolve the device index for XPU sessions too."""
    try:
        TorchDevice.set_session_device("xpu:1")
        assert TorchDevice.get_session_device_index() == 1
        assert TorchDevice.get_session_device_label() == " (#1)"
    finally:
        TorchDevice.clear_session_device()
