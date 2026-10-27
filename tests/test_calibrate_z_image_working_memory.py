"""scripts/calibrate_z_image_working_memory.py: the forwards it plans, the runtime it measures them in, and the lines it
makes of their windows.

The measurement needs a CUDA or ROCm device and runs by hand (see the script's docstring). These cover what decides the
lines it writes: which forwards a run measures, that a measuring process applies the server's runtime with the
allocator it was asked for, and how a cold and a warm forward become a `MeasuredPoint` in the point file, or an invalid
line that stays out of it. What one window measures is `measurement`'s, tested with the probe's windows.
"""

import dataclasses
import json
import re
import shutil
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from invokeai.backend.working_memory import coefficients, counters
from invokeai.backend.working_memory.coefficients import MeasuredPoint
from invokeai.backend.working_memory.counters import CounterReading
from invokeai.backend.working_memory.measurement import FreshWindow, fresh_window, measure
from invokeai.backend.working_memory.platform import Platform
from invokeai.backend.z_image.denoise_working_memory import estimate_z_image_denoise_working_memory
from scripts.working_memory import build_shipped

pytestmark = pytest.mark.platform_independent

SCRIPT = Path(__file__).parents[1] / "scripts" / "calibrate_z_image_working_memory.py"
MIB = 2**20
GIB = 2**30


def _load_script():
    spec = spec_from_file_location("calibrate_z_image_working_memory", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load_script()
Point = script.Point

GRID = {
    "text_tokens": 512,
    "short_text_tokens": 77,
    "blocks": 2,
    "extra_blocks": 0,
    "control_blocks": 15,
}


def _reading(**values) -> CounterReading:
    """A reading as a measuring process takes it: 1000 bytes reserved, 800 allocated, 1200 at the driver, nothing
    paged, no failed or retried allocation."""
    fields = {
        "reserved": 1000,
        "allocated": 800,
        "peak_reserved": 1000,
        "peak_allocated": 800,
        "num_ooms": 0,
        "num_alloc_retries": 0,
        "driver": 1200,
        "peak_driver": None,
        "paged": 0,
        "device_used": None,
        "method": "windows-d3dkmt",
    }
    return CounterReading(**(fields | values))


def _window(oom: bool = False, spill: bool = False, **final) -> FreshWindow:
    return fresh_window(_reading(), _reading(**final), oom=oom, spill=spill)


def _common() -> dict:
    """The fields of a point the measurement does not decide, as the measuring process takes them."""
    estimate = estimate_z_image_denoise_working_memory(
        torch.nn.Module(),
        image_seq_len=script.image_tokens(1024),
        text_seq_len=512,
        lora_count=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return {
        "estimator": estimate.estimator,
        "version": estimate.version,
        "platform": dataclasses.asdict(Platform("cudnn", "native", "windows")),
        "features": dict(estimate.features),
        "device": "NVIDIA GeForce RTX 4090",
    }


def _measured_point(line: dict) -> MeasuredPoint:
    return MeasuredPoint.model_validate(
        {key: value for key, value in line.items() if key in MeasuredPoint.model_fields}
    )


def test_the_fit_grid_runs_every_size_in_every_dtype_and_the_short_caption_at_its_sizes() -> None:
    points = script.plan(
        "fit", sizes=[512, 1024, 1440], dtypes=["bfloat16", "float32"], short_text_sizes=[1024, 1440, 2048], **GRID
    )

    expected = [
        (dtype, px, text)
        for dtype in ("bfloat16", "float32")
        for px, text in ((512, 512), (1024, 512), (1440, 512), (1024, 77), (1440, 77))
    ]
    # 2048px runs no caption at all: it is not among the sizes.
    assert [(point.dtype, point.px, point.text_tokens) for point in points] == expected
    assert all(
        point.fit and point.blocks == 2 and not point.force_math and not point.control_blocks for point in points
    )


def test_extra_blocks_add_one_forward_outside_the_fit_at_1024px_in_the_first_dtype() -> None:
    points = script.plan(
        "fit", sizes=[512], dtypes=["float16", "bfloat16"], short_text_sizes=[], **(GRID | {"extra_blocks": 4})
    )

    assert [point for point in points if not point.fit] == [Point("blocks", 1024, 512, "float16", 4)]
    assert points[-1].kind == "blocks"


def test_each_check_pairs_its_forward_with_a_default_kernel_forward_in_every_dtype() -> None:
    dtypes = ["bfloat16", "float32"]
    checks = GRID | {"extra_blocks": 4, "control_blocks": 6}
    math = script.plan("math", sizes=[512], dtypes=dtypes, short_text_sizes=[], **checks)
    control = script.plan("control", sizes=[512], dtypes=dtypes, short_text_sizes=[], **checks)

    assert math == [
        Point("reference", 1024, 512, "bfloat16", 2),
        Point("math", 1024, 512, "bfloat16", 2, force_math=True),
        Point("reference", 1024, 512, "float32", 2),
        Point("math", 1024, 512, "float32", 2, force_math=True),
    ]
    assert control == [
        Point("reference", 1024, 512, "bfloat16", 2),
        Point("control", 1024, 512, "bfloat16", 2, control_blocks=6),
        Point("reference", 1024, 512, "float32", 2),
        Point("control", 1024, 512, "float32", 2, control_blocks=6),
    ]
    assert not any(point.fit for point in math + control)


@pytest.mark.parametrize(
    "alloc_conf, configured",
    [("backend:cudaMallocAsync", "backend:cudaMallocAsync"), ("", None)],
    ids=["--alloc-conf", "platform-default"],
)
def test_a_measuring_process_applies_the_server_runtime_with_the_allocator_it_measures(
    monkeypatch: pytest.MonkeyPatch, alloc_conf: str, configured
) -> None:
    from invokeai.app import run_app
    from invokeai.app.services.config.config_default import get_config

    applied = []
    monkeypatch.setattr(run_app, "apply_server_runtime", lambda config, logger: applied.append(config))

    script._apply_server_runtime(alloc_conf)

    (config,) = applied
    assert config.pytorch_cuda_alloc_conf == configured
    # Every other setting as a server without a config file reads it.
    assert config.model_dump(exclude={"pytorch_cuda_alloc_conf"}) == get_config().model_dump(
        exclude={"pytorch_cuda_alloc_conf"}
    )


WEIGHTS = 1463092352
"""The bf16 weights of the measured transformer, 2 main blocks: allocated and reserved when a window opens."""
CONTEXT = 256 * MIB
"""What the driver counts beyond them at that point: the context and the libraries' workspaces."""

RECORDED = {
    "RTX 4090, native allocator, 512px": (
        (367001600, 350224384, 195313664, 8388608),
        (350224384, 350224384, 186466304, 0),
        367001600,
    ),
    # HIP's retained pool serves the warm forward: its driver figure does not grow, its reserved bytes decide.
    "RX 9060 XT, expandable segments, 512px": (
        (500629504, 316669952, 266485760, 149520384),
        (0, 232783872, 186466304, 0),
        500629504,
    ),
    # Here the warm forward demands more than the cold one.
    "RTX 4090, cudaMallocAsync, 1024px": (
        (754974720, 738197504, 568122368, 8388608),
        (838860800, 838860800, 559275008, 0),
        838860800,
    ),
}
"""Per run, bf16 with a 512-token caption: the growth of each counter of its cold and its warm forward (driver,
reserved, allocated and shared-memory bytes) and the point's `measured_bytes`, as the point files of this script's runs
on these GPUs recorded them."""


class _RecordedCounters:
    """`counters.DeviceCounters` replaying a forward's readings: its baseline, then its close."""

    def __init__(self, first: CounterReading, final: CounterReading):
        self._readings = [first, final]

    def read(self) -> CounterReading:
        return self._readings.pop(0) if len(self._readings) > 1 else self._readings[0]

    def synchronize(self) -> None:
        pass

    def reset_peaks(self) -> None:
        pass

    def close(self) -> None:
        pass


def _recorded_forward(monkeypatch: pytest.MonkeyPatch, growth: tuple[int, int, int, int]) -> FreshWindow:
    """A forward measured as the script measures it, on readings that grow by `growth` over the baseline."""
    driver, reserved, allocated, paged = growth
    first = _reading(
        reserved=WEIGHTS,
        allocated=WEIGHTS,
        peak_reserved=WEIGHTS,
        peak_allocated=WEIGHTS,
        driver=WEIGHTS + CONTEXT,
        paged=0,
    )
    final = _reading(
        reserved=WEIGHTS + reserved,
        allocated=WEIGHTS,
        peak_reserved=WEIGHTS + reserved,
        peak_allocated=WEIGHTS + allocated,
        driver=WEIGHTS + CONTEXT + driver,
        paged=paged,
    )
    monkeypatch.setattr(counters, "DeviceCounters", lambda device: _RecordedCounters(first, final))
    return measure(torch.device("cpu"), lambda: None)


@pytest.mark.parametrize("run", RECORDED)
def test_recorded_counters_give_the_demand_and_the_method_of_the_shipped_points(
    monkeypatch: pytest.MonkeyPatch, run: str
) -> None:
    cold_growth, warm_growth, measured_bytes = RECORDED[run]
    cold, warm = _recorded_forward(monkeypatch, cold_growth), _recorded_forward(monkeypatch, warm_growth)

    line = script.point_line(
        Point("fit", 512, 512, "bfloat16", 2), common=_common(), provenance="m", windows=[cold, warm], raw={}
    )

    # The paged growth stays below the spill tolerance, 143 MiB on the RX 9060 XT included.
    assert cold.reasons == warm.reasons == ()
    assert cold.counter_method == warm.counter_method == "windows-d3dkmt"
    point = _measured_point(line)
    assert (point.measured_bytes, point.method) == (measured_bytes, "demand_peak")


def test_a_point_is_the_larger_demand_of_its_cold_and_warm_forward_and_the_file_gets_it_as_a_measured_point() -> None:
    cold = _window(peak_reserved=5000, driver=6200)
    warm = _window(peak_reserved=5400, driver=5300)
    common = _common()

    line = script.point_line(
        Point("fit", 1024, 512, "float32", 2), common=common, provenance="measured", windows=[cold, warm], raw={"x": 1}
    )
    record = script.file_record(line)

    assert line["status"] == "point" and record is not None and "status" not in record
    point = _measured_point(record)
    assert (point.measured_bytes, point.method, point.in_fit) == (5000, "demand_peak", True)
    assert point.estimator == "denoise.z_image" and point.features == common["features"]
    assert point.platform == {"conv_backend": "cudnn", "allocator": "native", "os": "windows"}
    assert record["fit"] is True
    assert record["raw"] == {"x": 1, "cold": dataclasses.asdict(cold), "warm": dataclasses.asdict(warm)}
    assert record["point"] == {
        "kind": "fit",
        "px": 1024,
        "text_tokens": 512,
        "dtype": "float32",
        "blocks": 2,
        "force_math": False,
        "control_blocks": 0,
    }


def test_a_check_point_stays_out_of_the_fit() -> None:
    windows = [_window(peak_reserved=5000), _window(peak_reserved=5000)]

    line = script.point_line(
        Point("math", 1024, 512, "float32", 2, force_math=True),
        common=_common(),
        provenance="measured",
        windows=windows,
        raw={},
    )

    assert line["fit"] is False
    assert _measured_point(line).in_fit is False


def test_a_fit_grid_forward_its_line_does_not_admit_ships_out_of_the_fit_and_an_admitted_one_stays_bounded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """On a build whose SDPA kernel materializes the score matrix, a fit-grid forward carries a score term its line
    does not price: it measures that term, as a check does. The data builder takes it out of the fit, so the estimate,
    its line plus the term, only has to cover the measured peak. The same peak of a forward its line admits stays in
    the fit, and the upper bound still refuses an estimate over 1.3x of it."""
    # A card of its own: no shipped point of the same measurement keeps its place.
    common = _common() | {"device": "Test GPU"}
    with_scores = common | {"features": common["features"] | {"score_matrix_bytes": GIB}}
    windows = [_window(peak_reserved=GIB), _window(peak_reserved=GIB)]
    data_dir = tmp_path / "data"
    shutil.copytree(build_shipped.DATA_DIR, data_dir)

    def build(name: str, common: dict, *flags: str) -> tuple[dict, int]:
        line = script.point_line(
            Point("fit", 1024, 512, "float32", 2), common=common, provenance="m", windows=windows, raw={}
        )
        points = tmp_path / f"{name}.jsonl"
        points.write_text(json.dumps(script.file_record(line)) + "\n", encoding="utf-8")
        return line, build_shipped.main(["--points", str(points), "--data-dir", str(data_dir), *flags])

    admitted, refused = build("admitted", common)
    assert admitted["in_fit"] is True and refused == 1
    point_id = "denoise.z_image-cudnn,native-denoise-fp32-4096+512tok-TestGPU-demand_peak"
    assert re.search(
        rf"{re.escape(point_id)}: reserves [0-9.]+x the measured peak, over 1\.3x", capsys.readouterr().err
    )

    off_line, written = build("with_scores", with_scores, "--write")
    assert written == 0, capsys.readouterr().err
    assert off_line["fit"] is True and off_line["in_fit"] is False
    points = coefficients.load_data(data_dir)["denoise.json"].points
    assert [point.in_fit for point in points if point.device == "Test GPU"] == [False]


def test_without_the_drivers_figure_a_point_is_a_reserved_peak_of_torchs_counters_alone() -> None:
    with_driver = _window(peak_reserved=3000, driver=9000)
    without = _window(peak_reserved=3500, peak_allocated=3600, driver=None, method="torch-only")

    line = script.point_line(
        Point("fit", 1024, 512, "float32", 2), common=_common(), provenance="m", windows=[with_driver, without], raw={}
    )

    # The driver's 7800 bytes of the cold forward are not torch's: the bytes are what the method says they are.
    assert (line["measured_bytes"], line["method"]) == (2800, "reserved_peak")


def test_an_invalid_forward_is_no_measured_point_and_stays_out_of_the_file() -> None:
    clean = _window(peak_reserved=3000)
    retried = _window(num_alloc_retries=1, peak_reserved=2500)
    out_of_memory = _window(oom=True, peak_reserved=9000)
    point = Point("fit", 2048, 512, "float32", 2)

    retried_line = script.point_line(point, common=_common(), provenance="m", windows=[clean, retried], raw={})
    oom_line = script.point_line(point, common=_common(), provenance="m", windows=[out_of_memory], raw={})

    assert (retried_line["status"], retried_line["reasons"]) == ("invalid", ["alloc_retry"])
    assert (oom_line["status"], oom_line["reasons"], set(oom_line["raw"])) == ("invalid", ["oom"], {"cold"})
    for line in (retried_line, oom_line):
        assert not {"measured_bytes", "method", "provenance"} & set(line)
        with pytest.raises(ValidationError):
            _measured_point(line)
        # The data builder refuses a point file record that is no measured point.
        assert script.file_record(line) is None
    assert script.file_record(script.failed_line(point, "exit code 1")) is None


def test_the_runtime_facts_name_the_estimator_file_and_feed_the_provenance_and_the_header(monkeypatch) -> None:
    # The questions only a GPU answers; the rest is this process's own runtime.
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")

    facts = script._runtime_facts(torch.device("cuda"), "windows")

    module = (Path(__file__).parents[1] / "invokeai" / "backend" / "z_image" / "denoise_working_memory.py").resolve()
    assert Path(facts["estimator_module"]) == module
    provenance = script.provenance(Point("fit", 1024, 512, "bfloat16", 2), facts, "2026-10-18")
    assert provenance.startswith("scripts/calibrate_z_image_working_memory.py (2026-10-18): NVIDIA GeForce RTX 4090, ")
    header = script._runtime_header(facts, {"conv_backend": "cudnn", "allocator": "native", "os": "windows"})
    assert header.endswith(f"estimator from {module}")


def test_the_script_leaves_torch_unimported_so_a_measuring_process_can_configure_the_allocator_first() -> None:
    # A measuring process runs the script; `apply_rocm_windows_allocator_default` skips with a warning, and
    # `configure_torch_cuda_allocator` raises, once torch is imported.
    program = (
        "import sys\n"
        "from importlib.util import module_from_spec, spec_from_file_location\n"
        f"spec = spec_from_file_location('calibrate_z_image_working_memory', {str(SCRIPT)!r})\n"
        "spec.loader.exec_module(module_from_spec(spec))\n"
        "print(sorted(name for name in sys.modules if name == 'torch' or name.startswith('torch.')))\n"
    )

    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=300)

    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip() == "[]"
