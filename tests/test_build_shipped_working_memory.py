"""scripts/working_memory/build_shipped.py: calibration store exports into shipped working-memory points and fits."""

import platform as stdlib_platform
import shutil
from collections.abc import Callable, Iterator
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import torch

from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    CalibrationDeviceClass,
    CalibrationEnvironment,
    CalibrationPoint,
    CalibrationRun,
    CalibrationStoreFile,
)
from invokeai.backend.util import nvml
from invokeai.backend.working_memory import coefficients
from invokeai.backend.working_memory import platform as working_memory_platform
from invokeai.backend.working_memory.coefficients import MeasuredPoint, load_data
from invokeai.backend.working_memory.fingerprint import DeviceClass, Environment, environment_for
from invokeai.backend.working_memory.platform import platform_for
from invokeai.backend.working_memory.probe import CounterDelta
from invokeai.backend.working_memory.spec import import_estimator_modules
from scripts.working_memory.build_shipped import (
    DATA_DIR,
    BuildError,
    Spread,
    build_points,
    export_paths,
    main,
    merge_points,
    point_platform,
)

CUDA0 = torch.device("cuda", 0)
HOST_OS = working_memory_platform._host_os
"""Bound at import, before the suite's fixture pins it."""
SYS_PLATFORMS = {"windows": "win32", "linux": "linux"}
RTX_4090 = DeviceClass(vendor="nvidia", name="NVIDIA GeForce RTX 4090", total_mib=24564, backend="cuda", arch="sm_89")
RX_9060_XT = DeviceClass(vendor="amd", name="AMD Radeon RX 9060 XT", total_mib=16304, backend="hip", arch="gfx1200")
SETTINGS = {"pytorch_cuda_alloc_conf": None, "attention_type": "sdpa"}
GIB = 2**30


def _decode(px: int, **changes: object) -> dict[str, object]:
    return {"operation": "decode", "height": px, "width": px, "tile": None, "batch_size": 1, "score_matrix_bytes": 0,
            "dtype": "bf16"} | changes  # fmt: skip


def _delta(value: int) -> CounterDelta:
    return CounterDelta(baseline=GIB, peak=GIB + value, offset=0, delta=value)


def _window(
    item: int,
    features: dict,
    demand: int,
    *,
    valid=True,
    fit_eligible=True,
    reasons=(),
    version=2,
    estimator="vae.flux2",
    allocated: int | None = None,
):
    delta = _delta(demand)
    return {
        "run_id": "calibration-run",
        "queue_item_id": item,
        "node_id": "l2i",
        "node_type": "flux2_vae_decode",
        "model_key": "flux2-vae",
        "device": "cuda:0",
        "estimator": estimator,
        "version": version,
        "features": features,
        "terms": (),
        "reservation_bytes": demand,
        "source": "shipped:conv_backend=cudnn,allocator=native",
        "reservation_applied_bytes": demand + GIB,
        "driver": delta,
        "reserved": delta,
        "allocated": None if allocated is None else _delta(allocated),
        "demand_bytes": demand,
        "method": "windows-d3dkmt",
        "verified": True,
        "resident_fraction": 1.0 if fit_eligible else 0.5,
        "co_locked_models": [],
        "segments": 1,
        "duration_s": 0.5,
        "valid": valid,
        "fit_eligible": fit_eligible,
        "reasons": list(reasons),
    }


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch):
    """Scripts this process as a GPU machine: `as_machine(hip, os)` makes `environment_for` and `platform_for` read
    the build and OS given, with the native allocator or the allocator configuration given."""

    def as_machine(hip: str | None, os_name: str, alloc_conf: str | None = None) -> None:
        if alloc_conf is None:
            monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        else:
            monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", alloc_conf)
        monkeypatch.setattr(torch.version, "hip", hip)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
        monkeypatch.setattr(nvml, "driver_version", lambda: None)
        # Both `environment_for` and `platform_for` read the OS from `sys.platform`; the suite's pin is lifted.
        monkeypatch.setattr(working_memory_platform, "_host_os", HOST_OS)
        monkeypatch.setattr(working_memory_platform.sys, "platform", SYS_PLATFORMS[os_name])
        monkeypatch.setattr(stdlib_platform, "platform", lambda: "Test-1.0")

    return as_machine


def _export(path: Path, device_class: DeviceClass, windows: list[dict], *, os_fact: bool = True) -> Path:
    """A store as the scripted machine records it; without `os_fact`, as a server that records no OS did."""
    recorded = environment_for(CUDA0, SETTINGS)
    hard = recorded.hard if os_fact else {key: value for key, value in recorded.hard.items() if key != "os"}
    env = CalibrationEnvironment.of(Environment(hard=hard, soft=recorded.soft))
    store = CalibrationStoreFile(
        schema_version=1,
        device_class=CalibrationDeviceClass.of(device_class),
        runs=[
            CalibrationRun(
                run_id="calibration-run",
                model_key="flux2-vae",
                model_hash=None,
                user_id="system",
                created_at="2026-10-17T00:00:00Z",
                env_id=env.id,
            )
        ],
        points=[CalibrationPoint(**window, env_id=env.id) for window in windows],
        envs=[env],
    )
    path.write_text(store.model_dump_json(indent=2), encoding="utf-8")
    return path


@pytest.fixture
def exports(tmp_path: Path, machine) -> tuple[Path, Path]:
    """A Linux RTX 4090 with two repetitions of a 1024px decode, an invalid and a partially resident window, and a
    Windows RX 9060 XT on MIOpen's native allocator, saved as a family E2E run saves them, next to a status file."""
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "20261017-120000-box-1-status.json").write_text('{"available": true}', encoding="utf-8")
    machine(None, "linux")
    nvidia = _export(
        calibration / f"20261017-120000-box-1-{RTX_4090.id}.json",
        RTX_4090,
        [
            _window(1, _decode(1024), 4_600_000_000),
            _window(2, _decode(512), 1_150_000_000),
            _window(3, _decode(1024), 4_700_000_000),
            _window(4, _decode(1024), 9 * GIB, valid=False, fit_eligible=False, reasons=["oom"]),
            _window(5, _decode(1024), 9 * GIB, fit_eligible=False, reasons=["partial_residency"]),
        ],
    )
    machine("7.14.0", "windows")
    amd = _export(
        calibration / f"20261017-120000-box-2-{RX_9060_XT.id}.json",
        RX_9060_XT,
        [_window(1, _decode(1024), 10_900_000_000)],
    )
    return nvidia, amd


@pytest.fixture
def entries():
    return [entry for data in coefficients.shipped_data().values() for entry in data.coefficients]


@pytest.fixture
def specs():
    return import_estimator_modules()


def test_valid_fit_eligible_windows_become_points_at_their_largest_repetition(exports, entries, specs) -> None:
    build = build_points(exports, specs, entries)

    assert build.skipped == {"invalid": 1, "not fit-eligible": 1}
    got = sorted((p.device, p.platform, p.features["height"], p.measured_bytes, p.method) for p in build.points)
    assert got == [
        ("AMD Radeon RX 9060 XT", {"conv_backend": "miopen", "allocator": "native", "os": "windows"}, 1024,
         10_900_000_000, "probe_demand"),
        ("NVIDIA GeForce RTX 4090", {"conv_backend": "cudnn", "allocator": "native"}, 512, 1_150_000_000,
         "probe_demand"),
        ("NVIDIA GeForce RTX 4090", {"conv_backend": "cudnn", "allocator": "native"}, 1024, 4_700_000_000,
         "probe_demand"),
    ]  # fmt: skip
    largest = next(p for p in build.points if p.measured_bytes == 4_700_000_000)
    assert "largest of 2 valid fit-eligible windows" in largest.provenance
    assert "item 3" in largest.provenance and "NVIDIA GeForce RTX 4090" in largest.provenance
    assert exports[0].name in largest.provenance


def test_the_same_store_given_twice_counts_once(exports, entries, specs) -> None:
    once = build_points(exports, specs, entries)
    twice = build_points([*exports, exports[0]], specs, entries)

    assert twice.points == once.points
    assert twice.spreads == once.spreads
    assert twice.skipped["duplicate"] == 5


@pytest.mark.parametrize(
    "hip, os_name", [(None, "linux"), (None, "windows"), ("7.14.0", "windows"), ("7.14.0", "linux")]
)
def test_a_points_platform_is_the_one_platform_for_gives_on_the_measuring_machine(
    hip, os_name, machine, tmp_path: Path, entries, specs
) -> None:
    machine(hip, os_name)
    device_class = RX_9060_XT if hip else RTX_4090
    export = _export(tmp_path / "export.json", device_class, [_window(1, _decode(1024), 4 * GIB)])
    measuring_machine = point_platform(platform_for(CUDA0), entries)
    # The build runs on another OS than the export was recorded on.
    machine(hip, "linux" if os_name == "windows" else "windows")

    [point] = build_points([export], specs, entries).points

    assert point.platform == measuring_machine
    # Only MIOpen's native allocator has an entry of its own per OS; elsewhere a point names no OS.
    assert ("os" in point.platform) == (hip is not None and os_name == "windows")


def test_a_window_whose_environment_lacks_a_platform_fact_is_left_out(machine, tmp_path: Path, entries, specs) -> None:
    machine(None, "linux")
    complete = _export(tmp_path / "complete.json", RTX_4090, [_window(1, _decode(1024), 4 * GIB)])
    without_os = _export(tmp_path / "old.json", RTX_4090, [_window(2, _decode(512), GIB)], os_fact=False)

    build = build_points([complete, without_os], specs, entries)

    assert [p.features["height"] for p in build.points] == [1024]
    assert build.skipped == {"no platform facts": 1}


@pytest.mark.parametrize(
    "window, message",
    [
        pytest.param(_window(1, _decode(1024), 4 * GIB, version=1), "version 1, the registered estimator is version 2",
                     id="stale-version"),
        pytest.param(_window(1, {"operation": "decode", "height": 1024}, 4 * GIB), "cannot evaluate its features",
                     id="features-the-terms-cannot-read"),
    ],
)  # fmt: skip
def test_an_export_the_estimator_cannot_price_is_refused(
    window, message, machine, tmp_path: Path, entries, specs
) -> None:
    machine(None, "linux")
    export = _export(tmp_path / "export.json", RTX_4090, [window])

    with pytest.raises(BuildError, match=message):
        build_points([export], specs, entries)


def test_a_directory_stands_for_its_exports_in_lifetime_order(tmp_path: Path) -> None:
    for name in ("run-10-0123456789abcdef.json", "run-2-0123456789abcdef.json", "run-2-status.json", "notes.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")

    assert [p.name for p in export_paths([tmp_path])] == ["run-2-0123456789abcdef.json", "run-10-0123456789abcdef.json"]


def _probe_point(measured_bytes: int) -> MeasuredPoint:
    return MeasuredPoint(
        estimator="vae.flux2",
        version=2,
        platform=CUDNN_NATIVE,
        features=_decode(1024),
        measured_bytes=measured_bytes,
        method="probe_demand",
        device="NVIDIA GeForce RTX 4090",
        provenance="-",
    )


@pytest.mark.parametrize(
    "shipped, new, kept", [(5 * GIB, 4 * GIB, 5 * GIB), (4 * GIB, 5 * GIB, 5 * GIB)], ids=["lower", "higher"]
)
def test_a_remeasured_point_keeps_the_larger_demand(shipped: int, new: int, kept: int) -> None:
    [data] = coefficients.shipped_data().values()
    with_point = {"vae.json": data.model_copy(update={"points": [*data.points, _probe_point(shipped)]})}

    merged = merge_points(with_point, [_probe_point(new)])["vae.json"]

    same = [p for p in merged.points if p.method == "probe_demand"]
    assert [p.measured_bytes for p in same] == [kept]
    assert len(merged.points) == len(with_point["vae.json"].points)


@pytest.fixture
def shipped_checks(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[Path], None]]:
    """`check(data_dir)` runs the data test's checks on `data_dir` instead of the shipped files."""

    def check(data_dir: Path) -> None:
        data = load_data(data_dir)
        monkeypatch.setattr(coefficients, "shipped_data", lambda: data)
        coefficients.shipped_coefficients.cache_clear()
        path = Path(__file__).parent / "backend" / "working_memory" / "test_shipped_coefficients.py"
        spec = spec_from_file_location("shipped_checks_on_built_data", path)
        assert spec is not None and spec.loader is not None
        checks = module_from_spec(spec)
        spec.loader.exec_module(checks)
        built = [point for file in data.values() for point in file.points]
        assert sorted(map(repr, checks.ALL_POINTS)) == sorted(map(repr, built)), "the checks read the built data"
        checks.test_every_data_file_parses_and_declares_coefficients()
        checks.test_every_coefficient_entry_resolves_for_its_spec_on_a_measured_platform()
        checks.test_every_fitted_value_is_decided_by_an_in_fit_point()
        checks.test_every_fitted_value_is_the_shipped_fit_of_its_points()
        checks.test_a_stand_in_is_a_whole_line_no_in_fit_point_measures()
        checks.test_the_out_of_fit_points_are_the_listed_ones()
        checks.test_every_listed_upper_bound_exception_is_a_shipped_point_that_needs_it()
        checks.test_a_point_bound_is_never_looser_than_the_default()
        for estimator in checks.SPECS.values():
            checks.test_every_estimator_has_coefficients_on_every_measured_platform(estimator)
            checks.test_the_os_only_matters_where_an_entry_names_it(estimator)
            checks.test_an_unmeasured_platform_is_priced_as_the_measured_one_at_least_and_says_so(estimator)
        for point in checks.ALL_POINTS:
            checks.test_every_measured_point_is_bounded_by_its_estimate(point)

    try:
        yield check
    finally:
        # Resolved from the built data; the next caller must resolve from the shipped files again.
        coefficients.shipped_coefficients.cache_clear()


MIB = 2**20
CUDNN_NATIVE = {"conv_backend": "cudnn", "allocator": "native"}
MIOPEN_EXPANDABLE_WINDOWS = {"conv_backend": "miopen", "allocator": "expandable", "os": "windows"}


def _carried(data_dir: Path, estimator: str, platform: dict[str, str]) -> dict[str, float]:
    [entry] = [
        e for e in load_data(data_dir)["vae.json"].coefficients if e.estimator == estimator and e.platform == platform
    ]
    assert entry.values is not None
    return dict(entry.values)


def _data_copy(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    data_dir = tmp_path / "data"
    shutil.copytree(DATA_DIR, data_dir)
    return data_dir, {path.name: path.read_bytes() for path in data_dir.iterdir()}


def _files(data_dir: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in data_dir.iterdir()}


def test_written_data_holds_the_points_and_passes_the_shipped_data_checks(
    exports, tmp_path: Path, shipped_checks, capsys
) -> None:
    data_dir, before = _data_copy(tmp_path)

    assert main([*map(str, exports), "--data-dir", str(data_dir)]) == 0
    assert _files(data_dir) == before, "nothing written without --write"
    preview = capsys.readouterr().out
    # The 4.7 GB decode lies above the cuDNN line all FLUX.2 points share with vae.sd3: it is refitted.
    shipped_k = _carried(DATA_DIR, "vae.sd3", CUDNN_NATIVE)["k_decode_bf16"]
    assert f"vae.sd3 {CUDNN_NATIVE} k_decode_bf16 {shipped_k:.0f} ->" in preview

    # A run's calibration/ directory stands for its exports.
    assert main([str(exports[0].parent), "--data-dir", str(data_dir), "--write"]) == 0
    assert capsys.readouterr().out.splitlines()[1:] == preview.splitlines()[1:] + [f"wrote {data_dir / 'vae.json'}"]
    built = load_data(data_dir)["vae.json"]
    probe_points = [p for p in built.points if p.method == "probe_demand"]
    assert sorted(p.measured_bytes for p in probe_points) == [1_150_000_000, 4_700_000_000, 10_900_000_000]
    assert _carried(data_dir, "vae.sd3", CUDNN_NATIVE)["k_decode_bf16"] > shipped_k
    shipped_checks(data_dir)

    # Built again from the same exports, the data does not change.
    written = _files(data_dir)
    assert main([*map(str, exports), "--data-dir", str(data_dir), "--write"]) == 0
    assert "no changes" in capsys.readouterr().out
    assert _files(data_dir) == written


def _sd_decode(px: int, dtype: str) -> dict[str, object]:
    """An SD1/SDXL decode as the RX 9060 XT recorded it: the ROCm SDPA guard chunks the mid-block scores."""
    return {"operation": "decode", "height": px, "width": px, "tile": None, "dtype": dtype, "upcast_weight_bytes": 0,
            "score_matrix_bytes": 285212672 if px == 512 else GIB, "vae_class": "AutoencoderKL"}  # fmt: skip


def _amd_export(tmp_path: Path, machine, windows: list[dict]) -> Path:
    machine("7.14.0", "windows", "expandable_segments:True")
    return _export(tmp_path / f"20261017-160953-box-1-{RX_9060_XT.id}.json", RX_9060_XT, windows)


def test_new_readings_below_the_shipped_ones_add_points_without_moving_their_line(
    tmp_path: Path, machine, capsys
) -> None:
    """The RX 9060 XT's fp16 SDXL probe decodes, all below the shipped expandable-segments fp16 points (460 MiB under
    940 MiB at 512px): lower bounds only, they neither choose the line nor are capped by it."""
    windows = [
        _window(item, _sd_decode(px, "fp16"), mib * MIB, estimator="vae.sd")
        for item, (px, mib) in enumerate([(512, 460), (768, 1120), (1024, 1960), (1280, 2880)], start=1)
    ]
    export = _amd_export(tmp_path, machine, windows)
    data_dir, _ = _data_copy(tmp_path)
    shipped = _carried(DATA_DIR, "vae.sd3", MIOPEN_EXPANDABLE_WINDOWS)

    assert main([str(export), "--data-dir", str(data_dir), "--write"]) == 0

    captured = capsys.readouterr()
    assert "line " not in captured.out
    assert "not written" not in captured.err
    assert _carried(data_dir, "vae.sd3", MIOPEN_EXPANDABLE_WINDOWS) == shipped
    written = [p for p in load_data(data_dir)["vae.json"].points if p.method == "probe_demand"]
    assert sorted(p.measured_bytes // MIB for p in written) == [460, 1120, 1960, 2880]


def _fp32_1280(first_mib: int, largest_mib: int) -> list[dict]:
    """Two fp32 SDXL decodes at 1280px in one long-running server, allocated flat at the first one's bytes."""
    return [
        _window(1, _sd_decode(1280, "fp32"), first_mib * MIB, estimator="vae.sd", allocated=first_mib * MIB),
        _window(2, _sd_decode(1280, "fp32"), largest_mib * MIB, estimator="vae.sd", allocated=first_mib * MIB),
    ]


def test_a_raise_that_pushes_fresh_process_points_over_their_bound_is_refused_even_with_consent(
    tmp_path: Path, machine, capsys
) -> None:
    """The RX 9060 XT's fp32 SDXL decode at 1280px: allocated stays at 8020 MiB while reserved grows to 13000 MiB. The
    line that covers it over-reserves the fresh-process points of the same line beyond 1.3x."""
    export = _amd_export(tmp_path, machine, _fp32_1280(8020, 13000))
    data_dir, before = _data_copy(tmp_path)

    for consent in ([], ["--accept-fragmentation"]):
        assert main([str(export), "--data-dir", str(data_dir), "--write", *consent]) == 1
        captured = capsys.readouterr()
        assert "[2 windows, MiB 8020, 13000; max/first 1.62]" in captured.out
        assert "reserved MiB 8020, 13000 while allocated MiB 8020, 8020" in captured.out
        assert "decode_fp32: k 5988 -> " in captured.out and "1 flagged as fragmentation" in captured.out
        assert "the bounds of these points reject the result" in captured.err
        assert "vae.sd-miopen,expandable,windows-decode-fp32-1280x1280-RX9060XT-demand_peak: reserves" in captured.err
        assert _files(data_dir) == before


def test_a_raise_on_a_fragmentation_flagged_point_needs_consent(
    tmp_path: Path, machine, shipped_checks, capsys
) -> None:
    """A smaller growth (8020 to 9300 MiB reserved at flat allocated bytes) that raises the line within the bounds."""
    export = _amd_export(tmp_path, machine, _fp32_1280(8020, 9300))
    data_dir, before = _data_copy(tmp_path)
    shipped_k = _carried(DATA_DIR, "vae.sd3", MIOPEN_EXPANDABLE_WINDOWS)["k_decode_fp32"]

    assert main([str(export), "--data-dir", str(data_dir), "--write"]) == 1
    captured = capsys.readouterr()
    assert "warning: fragmentation?" in captured.out and "1 flagged as fragmentation" in captured.out
    assert "--accept-fragmentation" in captured.err
    assert _files(data_dir) == before

    assert main([str(export), "--data-dir", str(data_dir), "--write", "--accept-fragmentation"]) == 0
    assert _carried(data_dir, "vae.sd3", MIOPEN_EXPANDABLE_WINDOWS)["k_decode_fp32"] > shipped_k
    shipped_checks(data_dir)


@pytest.mark.parametrize(
    "reserved, allocated, fragmented",
    [
        pytest.param((8020, 13000), (8020, 8020), True, id="reserved-grows-allocated-flat"),
        # The RTX 4090's first fp16 512px decode: allocated grows with reserved, a first-run effect.
        pytest.param((576, 736, 736), (450, 609, 609), False, id="both-grow"),
        pytest.param((1120, 1020, 1020), (864, 864, 864), False, id="the-first-is-the-largest"),
        pytest.param((2887, 3160, 3160), (2887, 2887, 2887), False, id="growth-below-ten-percent"),
    ],
)
def test_a_spread_is_flagged_as_fragmentation_only_when_reserved_grows_around_flat_allocations(
    reserved, allocated, fragmented
) -> None:
    spread = Spread(demand=reserved, reserved=reserved, allocated=allocated)
    assert spread.fragmented is fragmented
