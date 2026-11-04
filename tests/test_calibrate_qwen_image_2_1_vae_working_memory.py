"""scripts/calibrate_qwen_image_2_1_vae_working_memory.py: the points it plans, and that the builder reads what it writes.

The measurement needs a CUDA or ROCm device and runs by hand (see the script's docstring)."""

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from invokeai.backend.qwen_image_2_1.vae import SPEC
from invokeai.backend.working_memory.coefficients import find_entry, shipped_data
from invokeai.backend.working_memory.counters import CounterReading
from invokeai.backend.working_memory.measurement import fresh_window
from scripts.working_memory import fresh_points
from scripts.working_memory.build_shipped import merge_points, read_point_files

pytestmark = pytest.mark.platform_independent


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "calibrate_qwen_image_2_1_vae_working_memory.py"
    spec = spec_from_file_location("calibrate_qwen_image_2_1_vae_working_memory", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()


def test_the_plan_runs_each_operation_and_dtype_untiled_and_at_each_tile_of_each_frame() -> None:
    points = script.plan("vae", ["decode"], ["bf16", "fp32"], [(512, 512), (896, 1152)], [2048], [512, 1024])
    assert [(p.dtype, p.height, p.width, p.tile) for p in points] == [
        ("bf16", 512, 512, None),
        ("bf16", 896, 1152, None),
        ("bf16", 2048, 2048, 512),
        ("bf16", 2048, 2048, 1024),
        ("fp32", 512, 512, None),
        ("fp32", 896, 1152, None),
        ("fp32", 2048, 2048, 512),
        ("fp32", 2048, 2048, 1024),
    ]


def test_a_size_is_a_side_or_a_shape() -> None:
    assert script.shapes("512,896x1152") == [(512, 512), (896, 1152)]
    with pytest.raises(ValueError):
        script.shapes("1x2x3")


def _reading(*, reserved: int, peak_reserved: int, driver: int, retries: int = 0) -> CounterReading:
    return CounterReading(
        reserved=reserved,
        # Live tensors stay under the allocator's blocks, so reading allocated growth instead of reserved fails.
        allocated=reserved // 2,
        peak_reserved=peak_reserved,
        peak_allocated=peak_reserved // 2,
        num_ooms=0,
        num_alloc_retries=retries,
        driver=driver,
        peak_driver=None,
        paged=None,
        device_used=None,
        method="torch-only",
    )


FEATURES = {
    "operation": "decode",
    "height": 2048,
    "width": 2048,
    "tile": 512,
    "dtype": "bf16",
    "conv_backend": "miopen",
}
WEIGHTS = 2**30


def _line(platform: dict[str, str], device: str, measured: int, *, retries: int = 0, spill: bool = False) -> dict:
    """A line as the script writes it, for the 2048px frame at a 512px bf16 tile. The driver reads 300 MiB above the
    reserved growth, as on HIP."""
    point = script.Point("vae", "decode", "bf16", 2048, 2048, 512)
    common = {
        "estimator": SPEC.id,
        "version": SPEC.version,
        "platform": platform,
        "features": FEATURES,
        "device": device,
    }
    facts = {
        "device": device,
        "torch": "2.13.0+rocm10.0.0",
        "hip": "7.15.26333",
        "os": platform["os"],
        "allocator_backend": "native",
        "alloc_conf": {},
        "miopen_find_mode": "FAST",
        "aotriton_experimental": None,
        "rocm_sdpa_guard": True,
    }
    first = _reading(reserved=WEIGHTS, peak_reserved=WEIGHTS, driver=WEIGHTS)
    top = WEIGHTS + measured
    final = _reading(reserved=WEIGHTS, peak_reserved=top, driver=top + 300 * 2**20, retries=retries)
    window = fresh_window(first, final, oom=False, spill=spill)
    return script.point_line(point, common, window, facts, "2026-11-03")


def _record(platform: dict[str, str], device: str, measured: int, tmp_path: Path) -> Path:
    points_file = tmp_path / "points.jsonl"
    record = fresh_points.file_record(_line(platform, device, measured))
    points_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return points_file


LINUX = {"conv_backend": "miopen", "allocator": "native", "os": "linux"}


def test_a_valid_window_is_its_reserved_growth_as_the_shipped_points_read() -> None:
    line = _line(LINUX, "AMD Radeon Pro W7900 Dual Slot (gfx1100)", 3 * 2**30)
    assert (line["status"], line["measured_bytes"], line["method"]) == ("point", 3 * 2**30, "reserved_peak")


@pytest.mark.parametrize(
    ("kwargs", "reason"), [({"retries": 1}, "alloc_retry"), ({"spill": True}, "spill")], ids=["retry", "spill"]
)
def test_a_voided_window_is_no_point_and_stays_out_of_the_file(kwargs: dict, reason: str) -> None:
    line = _line(LINUX, "AMD Radeon Pro W7900 Dual Slot (gfx1100)", 3 * 2**30, **kwargs)
    assert (line["status"], line["reasons"]) == ("invalid", [reason])
    assert "measured_bytes" not in line and fresh_points.file_record(line) is None


def test_a_w7900_record_naming_its_os_remeasures_the_shipped_point(tmp_path: Path) -> None:
    """The script records the OS `platform_for` gives; the shipped W7900 entry and points name none. Remeasured at the
    shipped bytes on the same card, the record merges into its point rather than joining the data beside it."""
    data = shipped_data()
    entries = [entry for file in data.values() for entry in file.coefficients]
    platform = {"conv_backend": "miopen", "allocator": "native", "os": "linux"}
    points_file = _record(platform, "AMD Radeon Pro W7900 Dual Slot (gfx1100)", 3592421376, tmp_path)

    read = read_point_files([points_file], {SPEC.id: SPEC}, entries).points

    assert len(read) == 1 and read[0].holds_upper_bound and read[0].in_fit
    assert read[0].platform == {"conv_backend": "miopen", "allocator": "native"}
    assert find_entry(entries, SPEC.id, SPEC.version, read[0].platform, "test").platform == read[0].platform
    merged = merge_points(data, read)["qwen_image_2_1_vae.json"]
    assert merged.points == data["qwen_image_2_1_vae.json"].points


def test_a_record_on_a_windows_entry_keeps_its_os(tmp_path: Path) -> None:
    entries = [entry for file in shipped_data().values() for entry in file.coefficients]
    platform = {"conv_backend": "miopen", "allocator": "expandable", "os": "windows"}
    points_file = _record(platform, "RX 9060 XT (gfx1200)", 3 * 2**30, tmp_path)

    read = read_point_files([points_file], {SPEC.id: SPEC}, entries).points

    assert len(read) == 1 and read[0].measured_bytes == 3 * 2**30 and read[0].platform == platform
