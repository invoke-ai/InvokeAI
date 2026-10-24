"""scripts/working_memory/fresh_points.py: the plumbing the fresh-process calibration scripts share.

The measurement needs a CUDA or ROCm device and runs by hand. These cover what decides the lines the scripts write: that
a measuring process applies the server's runtime with the allocator it was asked for and reports on the child protocol,
how a cold and a warm forward become a `MeasuredPoint` in the point file or an invalid line that stays out of it, and that
both scripts' records are what `scripts/working_memory/build_shipped.py --points` reads. Which of a point's windows decides
its bytes is `measurement`'s, tested there; each script's grid, model and inputs are its own test file's.
"""

import dataclasses
import json
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from invokeai.backend.working_memory import counters
from invokeai.backend.working_memory.coefficients import MeasuredPoint, shipped_data
from invokeai.backend.working_memory.counters import CounterReading
from invokeai.backend.working_memory.measurement import FreshWindow, fresh_window, measure
from invokeai.backend.working_memory.platform import Platform
from invokeai.backend.working_memory.spec import import_estimator_modules
from invokeai.backend.z_image.denoise_working_memory import estimate_z_image_denoise_working_memory
from scripts.working_memory import fresh_points
from scripts.working_memory.build_shipped import read_point_files

pytestmark = pytest.mark.platform_independent

SCRIPTS = Path(__file__).parents[1] / "scripts"
MIB = 2**20


def _load(name: str):
    spec = spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


z_image = _load("calibrate_z_image_working_memory")
flux2 = _load("calibrate_flux2_denoise_working_memory")
ZPoint = z_image.Point


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


def _common(features: dict, estimator: str = "denoise.z_image") -> dict:
    """The fields of a point the measurement does not decide, as a measuring process takes them."""
    return {
        "estimator": estimator,
        "version": 1,
        "platform": dataclasses.asdict(Platform("cudnn", "native", "windows")),
        "features": features,
        "device": "NVIDIA GeForce RTX 4090",
    }


def _z_image_features() -> dict:
    estimate = estimate_z_image_denoise_working_memory(
        torch.nn.Module(),
        image_seq_len=z_image.image_tokens(1024),
        text_seq_len=512,
        lora_count=0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return dict(estimate.features)


def _measured_point(line: dict) -> MeasuredPoint:
    return MeasuredPoint.model_validate(
        {key: value for key, value in line.items() if key in MeasuredPoint.model_fields}
    )


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

    fresh_points.apply_server_runtime(alloc_conf, "test")

    (config,) = applied
    assert config.pytorch_cuda_alloc_conf == configured
    # Every other setting as a server without a config file reads it.
    assert config.model_dump(exclude={"pytorch_cuda_alloc_conf"}) == get_config().model_dump(
        exclude={"pytorch_cuda_alloc_conf"}
    )


class _Exit(Exception):
    def __init__(self, code: int):
        self.code = code


@pytest.mark.parametrize("fails", [False, True], ids=["measured", "raised"])
def test_a_measuring_process_reports_its_line_on_the_result_line_and_leaves_with_its_status(
    monkeypatch: pytest.MonkeyPatch, capsys, fails: bool
) -> None:
    """It leaves through `os._exit`, past Python's shutdown, whose HIP teardown can hang on Windows."""
    applied: list[str] = []
    monkeypatch.setattr(fresh_points, "apply_server_runtime", lambda alloc_conf, logger: applied.append(alloc_conf))
    monkeypatch.setattr(fresh_points.os, "_exit", lambda code: (_ for _ in ()).throw(_Exit(code)))
    point = ZPoint("fit", 1024, 512, "bfloat16", 2)

    def measure_point(measured, force):
        if fails:
            raise RuntimeError("no device")
        return {"status": "point", "point": dataclasses.asdict(measured), "force": force}

    request = json.dumps({"point": dataclasses.asdict(point), "alloc_conf": "backend:cudaMallocAsync", "force": True})
    with pytest.raises(_Exit) as left:
        fresh_points.measure_in_this_process(request, point_type=ZPoint, measure=measure_point, logger_name="test")

    out = capsys.readouterr()
    assert applied == ["backend:cudaMallocAsync"]
    if fails:
        assert left.value.code == 1 and fresh_points.RESULT_PREFIX not in out.out and "no device" in out.err
    else:
        assert left.value.code == 0
        assert out.out == fresh_points.RESULT_PREFIX + json.dumps(measure_point(point, True)) + "\n"


def test_the_driving_process_reads_the_childs_result_line_or_records_why_there_is_none(tmp_path: Path) -> None:
    """The child's other output is not the result; a child that leaves without one is a failed line, with the last
    line of its stderr."""
    child = tmp_path / "child.py"
    child.write_text(
        "import json, sys\n"
        "request = json.loads(sys.argv[2])\n"
        "if request['force']:\n"
        "    sys.exit('the device is gone')\n"
        "print('a log line')\n"
        f"print({fresh_points.RESULT_PREFIX!r} + json.dumps({{'status': 'point', 'point': request['point']}}))\n",
        encoding="utf-8",
    )
    point = ZPoint("fit", 1024, 512, "bfloat16", 2)

    measured = fresh_points.run_point(child, point, "", force=False)
    failed = fresh_points.run_point(child, point, "", force=True)

    assert measured == {"status": "point", "point": dataclasses.asdict(point)}
    assert failed == fresh_points.failed_line(point, "exit code 1: the device is gone")
    assert fresh_points.file_record(failed) is None


WEIGHTS = 1463092352
"""The bf16 weights of the measured Z-Image transformer, 2 main blocks: allocated and reserved when a window opens."""
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
"""Per Z-Image run, bf16 with a 512-token caption: the growth of each counter of its cold and its warm forward (driver,
reserved, allocated and shared-memory bytes) and the point's `measured_bytes`, as the point files of the runs on these
GPUs recorded them."""


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
    """A forward measured as the scripts measure it, on readings that grow by `growth` over the baseline."""
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

    line = fresh_points.point_line(
        ZPoint("fit", 512, 512, "bfloat16", 2),
        common=_common(_z_image_features()),
        provenance="m",
        windows=[cold, warm],
        raw={},
    )

    # The paged growth stays below the spill tolerance, 143 MiB on the RX 9060 XT included.
    assert cold.reasons == warm.reasons == ()
    assert cold.counter_method == warm.counter_method == "windows-d3dkmt"
    point = _measured_point(line)
    assert (point.measured_bytes, point.method) == (measured_bytes, "demand_peak")


def test_a_valid_point_reaches_the_file_as_a_measured_point_with_the_scripts_own_keys() -> None:
    """Which window decides the bytes is `measurement.measured_bytes_and_method`'s, tested with it."""
    cold = _window(peak_reserved=5000, driver=6200)
    warm = _window(peak_reserved=5400, driver=5300)
    common = _common(_z_image_features())

    line = fresh_points.point_line(
        ZPoint("fit", 1024, 512, "float32", 2), common=common, provenance="measured", windows=[cold, warm], raw={"x": 1}
    )
    record = fresh_points.file_record(line)

    assert line["status"] == "point" and record is not None and "status" not in record
    point = _measured_point(record)
    assert (point.measured_bytes, point.method, point.in_fit) == (5000, "demand_peak", True)
    assert point.estimator == "denoise.z_image" and point.features == common["features"]
    assert point.platform == {"conv_backend": "cudnn", "allocator": "native", "os": "windows"}
    assert record["fit"] is True
    assert record["raw"] == {"x": 1, "cold": dataclasses.asdict(cold), "warm": dataclasses.asdict(warm)}
    assert record["point"] == dataclasses.asdict(ZPoint("fit", 1024, 512, "float32", 2))


def test_a_check_point_stays_out_of_the_fit() -> None:
    windows = [_window(peak_reserved=5000), _window(peak_reserved=5000)]

    line = fresh_points.point_line(
        ZPoint("math", 1024, 512, "float32", 2, force_math=True),
        common=_common(_z_image_features()),
        provenance="measured",
        windows=windows,
        raw={},
    )

    assert line["fit"] is False
    assert _measured_point(line).in_fit is False


def test_an_invalid_forward_is_no_measured_point_and_stays_out_of_the_file() -> None:
    clean = _window(peak_reserved=3000)
    retried = _window(num_alloc_retries=1, peak_reserved=2500)
    out_of_memory = _window(oom=True, peak_reserved=9000)
    point = ZPoint("fit", 2048, 512, "float32", 2)
    common = _common(_z_image_features())

    retried_line = fresh_points.point_line(point, common=common, provenance="m", windows=[clean, retried], raw={})
    oom_line = fresh_points.point_line(point, common=common, provenance="m", windows=[out_of_memory], raw={})

    assert (retried_line["status"], retried_line["reasons"]) == ("invalid", ["alloc_retry"])
    assert (oom_line["status"], oom_line["reasons"], set(oom_line["raw"])) == ("invalid", ["oom"], {"cold"})
    for line in (retried_line, oom_line):
        assert not {"measured_bytes", "method", "provenance"} & set(line)
        with pytest.raises(ValidationError):
            _measured_point(line)
        # The data builder refuses a point file record that is no measured point.
        assert fresh_points.file_record(line) is None
    assert fresh_points.file_record(fresh_points.failed_line(point, "exit code 1")) is None


@pytest.mark.parametrize("script", [z_image, flux2], ids=["z_image", "flux2"])
def test_the_runtime_facts_name_the_estimator_file_and_feed_each_scripts_provenance_and_the_header(
    monkeypatch, script
) -> None:
    # The questions only a GPU answers; the rest is this process's own runtime.
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: "NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    family = "z_image" if script is z_image else "flux2"
    module = (Path(__file__).parents[1] / "invokeai" / "backend" / family / "denoise_working_memory.py").resolve()

    facts = fresh_points.runtime_facts(torch.device("cuda"), "windows", str(module))

    assert Path(facts["estimator_module"]) == module
    point = ZPoint("fit", 1024, 512, "bfloat16", 2) if script is z_image else flux2.Point("fit", 4096, 1024, 512)
    assert script.provenance(point, facts, "2026-10-18").startswith(
        f"scripts/{Path(script.__file__).name} (2026-10-18): NVIDIA GeForce RTX 4090, torch {torch.__version__}, "
        "Windows, no allocator configuration (torch reports the native allocator)"
    )
    header = fresh_points.runtime_header(facts, {"conv_backend": "cudnn", "allocator": "native", "os": "windows"})
    assert header.endswith(f"estimator from {module}")


def test_both_scripts_records_are_points_the_builder_reads(tmp_path: Path) -> None:
    """A record as each script's measuring process writes it, its features the script's own estimate's: the builder
    takes it as a fresh-process point on the platform the data keys it by."""
    z_point = ZPoint("fit", 1024, 512, "bfloat16", 2)
    f_point = flux2.Point("fit", 4096, 1024, 512, references=1, reference_px=2024)
    features = [_z_image_features(), dict(flux2._estimate(f_point, torch.device("cpu")).features)]
    window = _window(peak_reserved=5000, driver=6200)
    records = [
        fresh_points.file_record(
            fresh_points.point_line(
                point, common=_common(feats, estimator), provenance="test", windows=[window, window], raw={}
            )
        )
        for point, feats, estimator in (
            (z_point, features[0], "denoise.z_image"),
            (f_point, features[1], "denoise.flux2"),
        )
    ]
    path = tmp_path / "points.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    entries = [entry for data in shipped_data().values() for entry in data.coefficients]
    built = read_point_files([path], import_estimator_modules(), entries)

    assert [(point.estimator, point.features, point.in_fit) for point in built.points] == [
        ("denoise.z_image", features[0], True),
        ("denoise.flux2", features[1], True),
    ]
    assert all(point.platform == {"conv_backend": "cudnn", "allocator": "native"} for point in built.points)
    assert built.skipped == {}


def test_the_scripts_leave_torch_unimported_so_a_measuring_process_can_configure_the_allocator_first() -> None:
    # A measuring process runs a script; `apply_rocm_windows_allocator_default` skips with a warning, and
    # `configure_torch_cuda_allocator` raises, once torch is imported.
    program = "import sys\nfrom importlib.util import module_from_spec, spec_from_file_location\n"
    for name in ("calibrate_z_image_working_memory", "calibrate_flux2_denoise_working_memory"):
        program += (
            f"spec = spec_from_file_location({name!r}, {str(SCRIPTS / f'{name}.py')!r})\n"
            "spec.loader.exec_module(module_from_spec(spec))\n"
        )
    program += "print(sorted(name for name in sys.modules if name == 'torch' or name.startswith('torch.')))\n"

    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=300)

    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip() == "[]"
