"""scripts/working_memory/build_shipped.py: calibration store exports and point files into shipped working-memory
points and fits."""

import itertools
import json
import platform as stdlib_platform
import re
import shutil
from collections.abc import Callable, Iterator
from fractions import Fraction
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
from invokeai.backend.util.vae_working_memory import VAE_FLUX2_SPEC, VAE_SD_SPEC
from invokeai.backend.working_memory import coefficients, fragmentation
from invokeai.backend.working_memory import platform as working_memory_platform
from invokeai.backend.working_memory.bounds import bound_violation, evaluate_point, point_id
from invokeai.backend.working_memory.coefficients import CoefficientEntry, MeasuredPoint, WorkingMemoryData, load_data
from invokeai.backend.working_memory.fingerprint import DeviceClass, Environment, environment_for
from invokeai.backend.working_memory.fit import (
    base_values,
    carrier_of,
    convolution_line,
    fit_points,
    shipped_two_phase,
)
from invokeai.backend.working_memory.fragmentation import Measurement
from invokeai.backend.working_memory.platform import platform_for
from invokeai.backend.working_memory.probe import CounterDelta
from invokeai.backend.working_memory.spec import EstimatorSpec, import_estimator_modules
from invokeai.backend.z_image.denoise_working_memory import Z_IMAGE_DENOISE_SPEC
from scripts.working_memory.build_shipped import (
    DATA_DIR,
    RAISE_NOTE,
    BuildError,
    bound_violations,
    build_points,
    export_paths,
    legacy_fit,
    main,
    merge_points,
    point_identity,
    point_platform,
    read_point_files,
    refit,
    write_data,
)

CUDA0 = torch.device("cuda", 0)
HOST_OS = working_memory_platform._host_os
"""Bound at import, before the suite's fixture pins it."""
SYS_PLATFORMS = {"windows": "win32", "linux": "linux"}
RTX_4090 = DeviceClass(vendor="nvidia", name="NVIDIA GeForce RTX 4090", total_mib=24564, backend="cuda", arch="sm_89")
RX_9060_XT = DeviceClass(vendor="amd", name="AMD Radeon RX 9060 XT", total_mib=16304, backend="hip", arch="gfx1200")
SETTINGS = {"pytorch_cuda_alloc_conf": None, "attention_type": "sdpa"}
GIB = 2**30
VAE_VERSION = VAE_FLUX2_SPEC.version
"""The version the VAE windows here are recorded at: the FLUX.2 and SD1/SDXL estimators', which move together."""
assert VAE_SD_SPEC.version == VAE_VERSION


def _decode(px: int, **changes: object) -> dict[str, object]:
    return {"operation": "decode", "height": px, "width": px, "tile": None, "batch_size": 1, "score_matrix_bytes": 0,
            "dtype": "bf16"} | changes  # fmt: skip


def _sd_fp16_decode(px: int) -> dict[str, object]:
    """An SD1/SDXL fp16 decode as the RTX 4090 records it: cuDNN's attention materializes no score matrix."""
    return {"operation": "decode", "height": px, "width": px, "tile": None, "dtype": "fp16", "upcast_weight_bytes": 0,
            "score_matrix_bytes": 0, "vae_class": "AutoencoderKL"}  # fmt: skip


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
    version=VAE_VERSION,
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
    """A Linux RTX 4090 with two repetitions of a 1024px SD fp16 decode, an invalid and a partially resident window,
    and a Windows RX 9060 XT on MIOpen's native allocator, saved as a family E2E run saves them, next to a status file.
    The RTX 4090's decodes are SD's: the FLUX.2 decode lines on its platform have no headroom for a raise
    (`bounds.UPPER_BOUND_EXCEPTIONS`)."""
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    (calibration / "20261017-120000-box-1-status.json").write_text('{"available": true}', encoding="utf-8")
    machine(None, "linux")
    nvidia = _export(
        calibration / f"20261017-120000-box-1-{RTX_4090.id}.json",
        RTX_4090,
        [
            _window(1, _sd_fp16_decode(1024), 4_600_000_000, estimator="vae.sd"),
            _window(2, _sd_fp16_decode(512), 1_150_000_000, estimator="vae.sd"),
            _window(3, _sd_fp16_decode(1024), 4_700_000_000, estimator="vae.sd"),
            _window(
                4, _sd_fp16_decode(1024), 9 * GIB, valid=False, fit_eligible=False, reasons=["oom"], estimator="vae.sd"
            ),
            _window(
                5, _sd_fp16_decode(1024), 9 * GIB, fit_eligible=False, reasons=["partial_residency"], estimator="vae.sd"
            ),
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
    amd, nvidia = f"AMD Radeon RX 9060 XT ({RX_9060_XT.id})", f"NVIDIA GeForce RTX 4090 ({RTX_4090.id})"
    assert got == [
        (amd, {"conv_backend": "miopen", "allocator": "native", "os": "windows"}, 1024, 10_900_000_000,
         "probe_demand"),
        (nvidia, {"conv_backend": "cudnn", "allocator": "native"}, 512, 1_150_000_000, "probe_demand"),
        (nvidia, {"conv_backend": "cudnn", "allocator": "native"}, 1024, 4_700_000_000, "probe_demand"),
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


@pytest.mark.parametrize("hip, os_name", [(None, "linux"), ("7.14.0", "windows")])
def test_a_point_of_an_estimator_keyed_by_no_platform_names_no_platform(
    hip, os_name, machine, tmp_path: Path, entries, specs
) -> None:
    # MOSS-Image's denoise entry reads alike on every platform; ROCm on Windows names even its OS for a VAE point.
    machine(hip, os_name)
    features = {"height": 1024, "width": 1024, "text_tokens": 64, "score_matrix_bytes": 0, "dequant_bytes": 0}
    window = _window(1, features, 4 * GIB, estimator="moss_image.denoise", version=1)
    export = _export(tmp_path / "export.json", RX_9060_XT if hip else RTX_4090, [window])

    [point] = build_points([export], specs, entries).points

    assert point.platform == {}


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
        pytest.param(_window(1, _decode(1024), 4 * GIB, version=VAE_VERSION - 1),
                     f"version {VAE_VERSION - 1}, the registered estimator is version {VAE_VERSION}", id="stale-version"),
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
        version=VAE_VERSION,
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
    data = coefficients.shipped_data()["vae.json"]
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
        checks.test_only_the_listed_entries_ship_hand_set_values()
        checks.test_a_stand_in_is_a_whole_line_its_fresh_process_points_cannot_fit()
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
    # The 4.7 GB decode lies above the cuDNN line all SD fp16 points share with vae.sd3: it is raised.
    assert f"line vae.json: vae.sd3 {CUDNN_NATIVE} decode_fp16: " in preview
    assert "; raised, 2 new points on it" in preview

    # A run's calibration/ directory stands for its exports.
    assert main([str(exports[0].parent), "--data-dir", str(data_dir), "--write"]) == 0
    assert capsys.readouterr().out.splitlines()[1:] == preview.splitlines()[1:] + [f"wrote {data_dir / 'vae.json'}"]
    built = load_data(data_dir)["vae.json"]
    probe_points = [p for p in built.points if p.method == "probe_demand"]
    assert sorted(p.measured_bytes for p in probe_points) == [1_150_000_000, 4_700_000_000, 10_900_000_000]
    shipped, raised = (_carried(d, "vae.sd3", CUDNN_NATIVE) for d in (DATA_DIR, data_dir))
    at_1024 = [line["k_decode_fp16"] * 1024 * 1024 + line["b_decode_fp16"] for line in (shipped, raised)]
    assert at_1024[0] < 1.05 * 4_700_000_000 <= at_1024[1]
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


MIOPEN_NATIVE = {"conv_backend": "miopen", "allocator": "native"}


@pytest.mark.parametrize(
    "per_pixel, shipped_k", [(2000, 7200), (8000, 8400)], ids=["below-the-stand-in", "above-the-stand-in"]
)
def test_probe_points_on_a_stand_in_line_raise_it_but_never_lower_it(
    per_pixel: int, shipped_k: int, tmp_path: Path, machine, shipped_checks
) -> None:
    """fp16 SDXL probe decodes on Linux with MIOpen's native allocator, a line no fresh-process point measures: its
    stand-in of 7200 bytes per pixel stays the floor until one does. Warm windows below it would fit 2100."""
    machine("7.14.0", "linux")
    windows = [
        _window(item, _sd_decode(px, "fp16"), per_pixel * px * px, estimator="vae.sd")
        for item, px in enumerate([512, 1024], start=1)
    ]
    export = _export(tmp_path / f"20261017-160953-box-1-{RX_9060_XT.id}.json", RX_9060_XT, windows)
    data_dir, _ = _data_copy(tmp_path)
    shipped = _carried(DATA_DIR, "vae.sd3", MIOPEN_NATIVE)
    assert (shipped["k_decode_fp16"], shipped["b_decode_fp16"]) == (7200, 0)

    assert main([str(export), "--data-dir", str(data_dir), "--write"]) == 0

    carried = _carried(data_dir, "vae.sd3", MIOPEN_NATIVE)
    assert (carried["k_decode_fp16"], carried["b_decode_fp16"]) == (shipped_k, 0)
    assert carried == shipped | {"k_decode_fp16": shipped_k}
    [entry] = [
        e
        for e in load_data(data_dir)["vae.json"].coefficients
        if e.estimator == "vae.sd3" and e.platform == MIOPEN_NATIVE
    ]
    assert {"k_decode_fp16", "b_decode_fp16"} <= set(entry.stand_ins)
    assert (RAISE_NOTE in entry.provenance) == (shipped_k > 7200)
    shipped_checks(data_dir)


def _miopen_native_point(
    estimator: str, features: dict[str, object], measured_bytes: int, method: str
) -> MeasuredPoint:
    return MeasuredPoint(
        estimator=estimator,
        version=import_estimator_modules()[estimator].version,
        platform=MIOPEN_NATIVE,
        features=features,
        measured_bytes=measured_bytes,
        method=method,
        device="AMD Radeon RX 9060 XT",
        provenance="-",
    )


def _refit_shipped(
    points: list[MeasuredPoint], specs, carrier: str, platform: dict[str, str] = MIOPEN_NATIVE
) -> CoefficientEntry:
    """The shipped data with `points` merged and refitted, checked against every point's bounds; returns `carrier`'s
    entry for `platform`, MIOpen's native allocator off Windows unless given."""
    after = refit(merge_points(coefficients.shipped_data(), points), specs)
    assert bound_violations(after, specs) == []
    [entry] = [
        entry
        for file in after.values()
        for entry in file.coefficients
        if entry.estimator == carrier and entry.platform == platform
    ]
    return entry


def test_a_stand_in_stays_until_its_fresh_process_points_span_the_sizes_of_a_fit(specs) -> None:
    """One fresh-process fp16 decode at 1024px and a warm probe at 512px: together they span 4x, but the fresh size
    alone pins nothing at 512px, where a cold run's base can lie far above the warm window. The fit through them
    (7010 bytes per pixel) would undercut the 7200 stand-in."""
    fresh = _miopen_native_point("vae.sd", _sd_decode(1024, "fp16"), 7_000_000_000, "demand_peak")
    probe = _miopen_native_point("vae.sd", _sd_decode(512, "fp16"), 1_000_000_000, "probe_demand")

    entry = _refit_shipped([fresh, probe], specs, "vae.sd3")

    assert entry.values is not None
    assert (entry.values["k_decode_fp16"], entry.values["b_decode_fp16"]) == (7200, 0)
    assert {"k_decode_fp16", "b_decode_fp16"} <= set(entry.stand_ins)


def _cogview4_probes(per_pixel: int) -> list[MeasuredPoint]:
    """Probe decodes of CogView4's VAE at 512px and 1024px, a 1 GiB base and `per_pixel` bytes per pixel."""
    return [
        _miopen_native_point("vae.cogview4", _sd_decode(px, "bf16"), GIB + per_pixel * px * px, "probe_demand")
        for px in (512, 1024)
    ]


def test_probes_a_stand_in_covers_by_the_margin_leave_it_as_it_is(specs) -> None:
    """CogView4's bf16 decode stand-in off Windows, 13720 bytes per pixel and a 14 MiB base, reserves 3444 MiB at
    512px and 13734 MiB at 1024px. Probes of a 1 GiB base and 7000 bytes per pixel read 2774 and 8024 MiB: covered by
    far more than 5%. Their own fit (7350 bytes per pixel, a 1076 MiB base) must not lift the base by 1062 MiB."""
    entry = _refit_shipped(_cogview4_probes(7000), specs, "vae.cogview4")

    assert entry.values is not None
    assert (entry.values["k_decode_bf16"], entry.values["b_decode_bf16"]) == (13720, 14 * MIB)
    assert RAISE_NOTE not in entry.provenance


def test_a_stand_in_short_of_a_probe_is_raised_by_the_least_its_bounds_allow(specs) -> None:
    """Probes of a 1 GiB base and 10000 bytes per pixel read 3524 MiB at 512px, above the stand-in's 3444 MiB, and
    11024 MiB at 1024px, below its 13734. Covering 512px by 5% takes 3700.2 MiB there: 256.2 MiB more base, or 1025
    more bytes per pixel. Averaged over the measured 512px to 1024px, the slope costs 2.5 times as much (the middle,
    655360 px, over 512px's 262144), so the base takes the raise, to 270.2 MiB. Of the lines as they ship, whole bytes
    per pixel and whole MiB of base, 13720 with 271 MiB averages 8846 MiB over that range; 13721 with 270 MiB still
    covers 512px and averages 8845.625: a byte per pixel more costs 0.625 MiB at the middle, a MiB of base less saves
    a whole one."""
    entry = _refit_shipped(_cogview4_probes(10_000), specs, "vae.cogview4")

    assert entry.values is not None
    k, b = entry.values["k_decode_bf16"], entry.values["b_decode_bf16"]
    assert (k, b) == (13721, 270 * MIB)
    assert k * 512 * 512 + b >= 1.05 * (GIB + 10_000 * 512 * 512) > k * 512 * 512 + b - MIB
    assert {"k_decode_bf16", "b_decode_bf16"} <= set(entry.stand_ins)
    assert RAISE_NOTE in entry.provenance


def test_probe_tiled_decodes_raise_a_tiled_stand_in_and_its_assembly_line_above_the_tile(specs) -> None:
    """The SD estimator's fp16 tiled decode off Windows is a stand-in, tiled line and assembly line alike, over the
    borrowed untiled stand-in of 7200 bytes per pixel. No untiled SD point measures the 512px tile there, so probe
    decodes at that tile are read above what the untiled line prices it at. Four of them, 1024 to 8192px, the largest
    needing the assembly phase, raise the tiled slope and the assembly base, fitted together; the halves they lie
    under stay. Read on their raw bytes, the raise would price the tile twice."""
    tile_price = 7200 * 512 * 512
    above_the_tile = {1024: 1600 * MIB, 2048: 1700 * MIB, 4096: 2200 * MIB, 8192: 9000 * MIB}
    probes = [
        _miopen_native_point("vae.sd", _sd_decode(px, "fp16") | {"tile": 512}, tile_price + mib, "probe_demand")
        for px, mib in above_the_tile.items()
    ]
    names = [f"{part}_{line}_fp16" for line in ("tiled_decode", "assembled_decode") for part in ("k", "b")]
    shipped = _carried(DATA_DIR, "vae.sd", MIOPEN_NATIVE)

    entry = _refit_shipped(probes, specs, "vae.sd")

    readings = [
        (px * px, tile_price + mib, Fraction(tile_price), Fraction(tile_price)) for px, mib in above_the_tile.items()
    ]
    tiled, assembled = shipped_two_phase(readings)
    assert entry.values is not None
    raised = [entry.values[name] for name in names]
    assert raised == [max(shipped[name], fit) for name, fit in zip(names, (*tiled, *assembled), strict=True)]
    assert raised[0] > shipped[names[0]] and raised[3] > shipped[names[3]], "the tiled slope and the assembly base"
    assert raised[1] == shipped[names[1]] and raised[2] == shipped[names[2]]
    raw_tiled, raw_assembled = shipped_two_phase([reading[:2] for reading in readings])
    assert (*raw_tiled, *raw_assembled) != (*tiled, *assembled)
    assert set(names) <= set(entry.stand_ins)
    assert RAISE_NOTE in entry.provenance


def test_probes_on_the_untiled_line_raise_it_and_may_shift_tile_cost_off_a_tiled_line_without_breaking_a_bound(
    specs,
) -> None:
    """Warm RTX 4090 SD fp16 decodes at 512 and 1024 px under the native allocator raise the untiled line they
    measure, and never lower it. The tiled SD decodes sit on that line at their tile. Those with an untiled reference at
    their tile are covered above it, so the raise adds to their estimates byte for byte; those without one are covered
    above the raised line, and the fit of the lines above takes less for them: a tiled or assembly value falls. Every
    shipped point stays covered and within its bound. SD fp16, as FLUX's decode lines on this platform have no headroom
    for a raise (`bounds.UPPER_BOUND_EXCEPTIONS`); a 1024 px probe of 4.8 GB, as 4.9 GB breaks a tiled bound (below)."""
    native = {"conv_backend": "cudnn", "allocator": "native"}
    probes = [
        MeasuredPoint(
            estimator="vae.sd",
            version=VAE_VERSION,
            platform=native,
            features=_sd_fp16_decode(px),
            measured_bytes=demand,
            method="probe_demand",
            device="NVIDIA GeForce RTX 4090",
            provenance="-",
        )
        for px, demand in ((512, 1_300_000_000), (1024, 4_800_000_000))
    ]
    before = coefficients.shipped_data()
    entries_before = [entry for file in before.values() for entry in file.coefficients]

    after = refit(merge_points(before, probes), specs)

    entries_after = [entry for file in after.values() for entry in file.coefficients]
    untiled_before = carrier_of(entries_before, probes[0], "decode_fp16")
    untiled_after = carrier_of(entries_after, probes[0], "decode_fp16")
    assert untiled_before.values is not None and untiled_after.values is not None
    halves = ("k_decode_fp16", "b_decode_fp16")
    assert all(untiled_after.values[name] >= untiled_before.values[name] for name in halves)
    assert any(untiled_after.values[name] > untiled_before.values[name] for name in halves), "the probes raise it"

    for point in (point for file in after.values() for point in file.points):
        spec = specs[point.estimator]
        estimate = evaluate_point(point, spec, after.values())
        # Covered: an in-fit point by its fitted lines alone; an out-of-fit check, by the floor too (`bounds.py`).
        assert not point.in_fit or estimate.reservation_bytes >= point.measured_bytes, point_id(point, spec)
        assert bound_violation(point, spec, estimate) is None, point_id(point, spec)

    def sd_decode(entries: list[CoefficientEntry]) -> dict[str, float]:
        [entry] = [entry for entry in entries if entry.estimator == "vae.sd" and entry.platform == native]
        assert entry.values is not None
        return {
            name: value
            for name, value in entry.values.items()
            if name.endswith(("tiled_decode_fp16", "assembled_decode_fp16"))
        }

    shifted = sd_decode(entries_after)
    assert any(shifted[name] < value for name, value in sd_decode(entries_before).items()), "tile cost moved off"


def test_an_untiled_raise_that_breaks_a_tiled_bound_is_refused_though_a_steeper_raise_would_keep_it(
    tmp_path: Path, machine, capsys
) -> None:
    """A known limitation, not the behaviour wanted: the warm SD fp16 decodes of the test above with a 1024 px probe
    of 4.9 GB. `fit.raised_pair` raises the untiled line by the least its own points' bounds allow, the base, without
    the tiled bounds (`bounds.line_bounds` gives none), which lifts the decodes at a 512 px tile at 1536 and 3072 px to
    1.30x and 1.31x of their measured peaks, over their bound. The builder refuses it and writes nothing, though a
    steeper raise would keep every tiled bound: the earlier rule, the larger per coefficient of the fit on all points
    and on the fresh-process ones, accepted these probes. A joint fit of the untiled and tiled lines would fix it."""
    machine(None, "linux")
    windows = [
        _window(1, _sd_fp16_decode(512), 1_300_000_000, estimator="vae.sd"),
        _window(2, _sd_fp16_decode(1024), 4_900_000_000, estimator="vae.sd"),
    ]
    export = _export(tmp_path / f"20261017-120000-box-1-{RTX_4090.id}.json", RTX_4090, windows)
    data_dir, before = _data_copy(tmp_path)

    assert main([str(export), "--data-dir", str(data_dir), "--write"]) == 1

    captured = capsys.readouterr()
    assert "the bounds of these points reject the result" in captured.err
    for side in (1536, 3072):
        violation = f"vae.sd-cudnn,native-decode-fp16-{side}x{side}-tile512-RTX4090-demand_peak: reserves"
        assert violation in captured.err
    assert _files(data_dir) == before


def test_a_probe_no_hull_edge_covers_within_the_bounds_raises_the_line_between_them(specs) -> None:
    """A CogView4 probe decode of 4000 MiB at 640px on the RTX 4090, above its cuDNN bf16 line (8696 bytes per pixel, a
    63 MiB base: 3460 MiB there). Every line through it and a fresh-process point or the origin reserves more than a
    fresh-process point priced on the line allows, the 512px decodes among them; a line between them covers it by 5%
    and keeps every bound. No tiled decode sits on this line, whose bounds the raise would not see
    (`bounds.line_bounds`)."""
    probe = MeasuredPoint(
        estimator="vae.cogview4",
        version=specs["vae.cogview4"].version,
        platform=CUDNN_NATIVE,
        features=_sd_decode(640, "bf16"),
        measured_bytes=4000 * MIB,
        method="probe_demand",
        device="NVIDIA GeForce RTX 4090",
        provenance="-",
    )

    entry = _refit_shipped([probe], specs, "vae.cogview4", CUDNN_NATIVE)

    assert entry.values is not None
    k, b = entry.values["k_decode_bf16"], entry.values["b_decode_bf16"]
    shipped = _carried(DATA_DIR, "vae.cogview4", CUDNN_NATIVE)
    assert k >= shipped["k_decode_bf16"] and b >= shipped["b_decode_bf16"]
    assert (k, b) != (shipped["k_decode_bf16"], shipped["b_decode_bf16"]), "the probe raises the line"
    assert k * 640 * 640 + b >= 1.05 * 4000 * MIB


def _fp32_1280(first_mib: int, largest_mib: int) -> list[dict]:
    """Two fp32 SDXL decodes at 1280px in one long-running server, allocated flat at the first one's bytes."""
    return [
        _window(1, _sd_decode(1280, "fp32"), first_mib * MIB, estimator="vae.sd", allocated=first_mib * MIB),
        _window(2, _sd_decode(1280, "fp32"), largest_mib * MIB, estimator="vae.sd", allocated=first_mib * MIB),
    ]


def test_a_probe_no_line_can_cover_within_the_fresh_process_bounds_is_refused_even_with_consent(
    tmp_path: Path, machine, capsys
) -> None:
    """The RX 9060 XT's fp32 SDXL decode at 1280px: allocated stays at 8020 MiB while reserved grows to 13000 MiB. No
    line at or above the fit covers it by 5% within 1.3x of the fresh-process decodes at 1280px: the data conflicts
    with the bounds. The refusal shows the point with its windows and the fragmentation flag that explains it, and
    names the probe and a bound it conflicts with, by id, measured bytes and the most it allows."""
    export = _amd_export(tmp_path, machine, _fp32_1280(8020, 13000))
    data_dir, before = _data_copy(tmp_path)
    shipped_k = _carried(DATA_DIR, "vae.sd3", MIOPEN_EXPANDABLE_WINDOWS)["k_decode_fp32"]

    for consent in ([], ["--accept-fragmentation"]):
        assert main([str(export), "--data-dir", str(data_dir), "--write", *consent]) == 1
        captured = capsys.readouterr()
        assert "[2 windows, MiB 8020, 13000; max/first 1.62]" in captured.out
        assert "reserved MiB 8020, 13000 while allocated MiB 8020, 8020" in captured.out
        assert "error: not written: the points conflict with the bounds: the points cannot fit a line" in captured.err
        assert f"decode_fp32: no line at or above k={shipped_k:.0f}" in captured.err
        assert f"covers 13000 MiB at size {1280 * 1280} by 5% within the upper bound of vae." in captured.err
        assert re.search(r"-1280x1280-scores-RX9060XT-demand_peak \(measured \d+ MiB, at most \d+ MiB\)", captured.err)
        assert _files(data_dir) == before


def test_a_stand_in_no_line_can_raise_within_its_bounds_is_refused(tmp_path: Path, machine, capsys) -> None:
    """CogView4's bf16 decode stand-in on Linux MIOpen (13720 bytes per pixel, a 14 MiB base) with one fresh-process
    decode of 3000 MiB at 512px, which holds it to 3900 MiB there. A 30000 MiB probe at 1024px needs 31500 MiB: the
    least slope that reaches it with the stand-in's base puts 512px at 7886 MiB. Nothing is written."""
    data_dir, _ = _data_copy(tmp_path)
    data = load_data(data_dir)["vae.json"]
    fresh = _miopen_native_point("vae.cogview4", _sd_decode(512, "bf16"), 3000 * MIB, "demand_peak")
    write_data(data_dir / "vae.json", data.model_copy(update={"points": [*data.points, fresh]}))
    before = _files(data_dir)
    machine("7.14.0", "linux")
    window = _window(
        1,
        _sd_decode(1024, "bf16"),
        30000 * MIB,
        estimator="vae.cogview4",
        version=import_estimator_modules()["vae.cogview4"].version,
    )
    export = _export(tmp_path / f"20261017-160953-box-1-{RX_9060_XT.id}.json", RX_9060_XT, [window])

    assert main([str(export), "--data-dir", str(data_dir), "--write"]) == 1

    captured = capsys.readouterr()
    assert f"+ {30000 * MIB}" in captured.out, "the probe is shown"
    stand_in = "the points cannot raise a stand-in: vae.cogview4 on {'conv_backend': 'miopen', 'allocator': 'native'}"
    assert stand_in in captured.err
    assert "decode_bf16: no line at or above k=13720, b=14 MiB covers 30000 MiB at size 1048576 by 5%" in captured.err
    assert "within the upper bound of vae.cogview4-miopen,native-decode-bf16-512x512-" in captured.err
    assert "(measured 3000 MiB, at most 3900 MiB)" in captured.err
    assert _files(data_dir) == before


def test_two_owners_attempts_with_one_id_are_measured_apart(tmp_path: Path, machine, entries, specs) -> None:
    """The two fp32 1280px decodes of `_fp32_1280`, each another owner's attempt "1": reserved grew from one to the
    other around flat allocations, which within one attempt is fragmentation. Across two it is not."""
    windows = [
        {**window, "run_id": run_id} for window, run_id in zip(_fp32_1280(8020, 9300), ("run-a", "run-b"), strict=True)
    ]
    export = _amd_export(tmp_path, machine, windows)
    store = json.loads(export.read_text(encoding="utf-8"))
    [run] = store["runs"]
    store["runs"] = [
        {**run, "run_id": run_id, "user_id": owner, "attempt_id": "1", "attempt_steps": 1}
        for run_id, owner in (("run-a", "admin-a"), ("run-b", "admin-b"))
    ]
    export.write_text(json.dumps(store), encoding="utf-8")

    build = build_points([export], specs, entries)

    assert [spread.fragmented for spread in build.spreads.values()] == [False]


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


RTX_4060_TI_16GB = DeviceClass(
    vendor="nvidia", name="NVIDIA GeForce RTX 4060 Ti", total_mib=16380, backend="cuda", arch="sm_89"
)
RTX_4060_TI_8GB = DeviceClass(
    vendor="nvidia", name="NVIDIA GeForce RTX 4060 Ti", total_mib=8188, backend="cuda", arch="sm_89"
)


def test_cards_of_one_name_are_separate_points_and_spreads(
    tmp_path: Path, machine, entries, specs, shipped_checks, capsys
) -> None:
    """The 16 GB and the 8 GB RTX 4060 Ti report one name. An SD fp16 decode at 1024px on each allocates 4.2 GB and
    reserves 4.2 GB on the one card, 4.7 GB on the other: two cards' readings, not a server's cache growing around the
    same tensors. The 4.7 GB raises the cuDNN line, and nothing flags it as fragmentation."""
    machine(None, "linux")
    sixteen = _export(
        tmp_path / f"20261017-120000-box-1-{RTX_4060_TI_16GB.id}.json",
        RTX_4060_TI_16GB,
        [_window(1, _sd_fp16_decode(1024), 4_200_000_000, estimator="vae.sd", allocated=4_200_000_000)],
    )
    eight = _export(
        tmp_path / f"20261018-120000-box-1-{RTX_4060_TI_8GB.id}.json",
        RTX_4060_TI_8GB,
        [_window(1, _sd_fp16_decode(1024), 4_700_000_000, estimator="vae.sd", allocated=4_200_000_000)],
    )
    data_dir, _ = _data_copy(tmp_path)

    assert main([str(sixteen), str(eight), "--data-dir", str(data_dir), "--write"]) == 0

    captured = capsys.readouterr()
    assert "fragmentation" not in captured.out + captured.err
    assert "raised, 2 new points on it" in captured.out
    expected = [
        (f"NVIDIA GeForce RTX 4060 Ti ({RTX_4060_TI_16GB.id})", 4_200_000_000),
        (f"NVIDIA GeForce RTX 4060 Ti ({RTX_4060_TI_8GB.id})", 4_700_000_000),
    ]
    written = load_data(data_dir)["vae.json"].points
    assert sorted((p.device, p.measured_bytes) for p in written if p.method == "probe_demand") == sorted(expected)
    shipped_checks(data_dir)
    build = build_points([sixteen, eight], specs, entries)
    assert sorted((p.device, p.measured_bytes) for p in build.points) == sorted(expected)
    assert sorted(spread.demand for spread in build.spreads.values()) == [(4_200_000_000,), (4_700_000_000,)]


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
def test_a_measurement_is_flagged_as_fragmentation_only_when_reserved_grows_around_flat_allocations(
    reserved, allocated, fragmented
) -> None:
    measurement = Measurement(line="decode_fp32", size=1280**2, demand=reserved, reserved=reserved, allocated=allocated)
    assert fragmentation.fragmented([measurement]) == [fragmented]


def test_a_size_whose_one_window_stands_out_from_the_smaller_sizes_is_flagged_without_any_growth(
    tmp_path: Path, machine, entries, specs, capsys
) -> None:
    """The RX 9060 XT's fp32 SDXL decodes of one attempt (real reserved and allocated bytes): 576px and 832px reserve
    1.11 and 1.10 times what they allocate, the one fully resident 1216px window 1.47 times, 1.33 times their median.
    It has no repetition to grow over, so only the ratio rule can flag it -- as the user fit does."""
    windows = [
        _window(item, _sd_decode(px, "fp32"), reserved, estimator="vae.sd", allocated=allocated)
        for item, (px, reserved, allocated) in enumerate(
            [
                (576, 1887436800, 1702840320),
                (576, 1887436800, 1702840320),
                (832, 3900702720, 3552839680),
                (832, 3900702720, 3552839680),
                (1216, 11135877120, 7589201920),
            ],
            start=1,
        )
    ]
    export = _amd_export(tmp_path, machine, windows)

    build = build_points([export], specs, entries)

    flagged = {point.features["width"] for point in build.points if build.spreads[point_identity(point)].fragmented}
    assert flagged == {1216}
    main([str(export), "--data-dir", str(DATA_DIR)])
    assert "warning: fragmentation? vae.sd" in capsys.readouterr().out


DENOISE_SLOPE, DENOISE_BASE = 225280, 100 * MIB
"""The line the Z-Image denoise points below lie on in bf16 and fp16, 220 KiB per attended token plus 100 MiB; fp32
measures twice that. The shipped fit adds 5%: 236544 B per token and 105 MiB, and 473088 B and 210 MiB."""
FITTED = {"bf16": (236544, 105 * MIB), "fp16": (236544, 105 * MIB), "fp32": (473088, 210 * MIB)}


def _forward_record(image_tokens: int, dtype: str, *, check: bool = False, **changes: object) -> dict:
    """A point file record of a plain Z-Image forward as `scripts/calibrate_z_image_working_memory.py` writes one: the
    platform as `platform_for` gives it, and the script's own keys beside the point's. A check's point is out of the
    fit."""
    tokens = image_tokens + 512
    features = {"image_tokens": image_tokens, "text_tokens": 512, "tokens": tokens, "dtype": dtype,
                "control_blocks": 0, "has_attention_mask": False, "regional_bias_bytes": 0, "score_matrix_bytes": 0,
                "lora_count": 0, "dequant_transient_bytes": 0, "fp8_cast_transient_bytes": 0}  # fmt: skip
    measured = (DENOISE_SLOPE * tokens + DENOISE_BASE) * (2 if dtype == "fp32" else 1)
    return {
        "estimator": Z_IMAGE_DENOISE_SPEC.id,
        "version": Z_IMAGE_DENOISE_SPEC.version,
        "platform": CUDNN_NATIVE | {"os": "linux"},
        "features": features,
        "measured_bytes": measured,
        "method": "demand_peak",
        "device": "NVIDIA GeForce RTX 4090",
        "in_fit": not check,
        "provenance": "scripts/calibrate_z_image_working_memory.py, test",
        "fit": not check,
        "point": {"kind": "check" if check else "fit", "image_tokens": image_tokens, "dtype": dtype},
        "raw": {"cold": {"reserved": measured}, "warm": {"reserved": measured - MIB}},
    } | changes


def _point_file(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return path


def _denoise_entries(data_dir: Path) -> dict[str, coefficients.CoefficientEntry]:
    """The Z-Image denoise entries by platform; the file declares the FLUX.2 denoise's too."""
    return {
        ",".join(entry.platform.values()): entry
        for entry in load_data(data_dir)["denoise.json"].coefficients
        if entry.estimator == Z_IMAGE_DENOISE_SPEC.id
    }


LEGACY_DENOISE = {f"{part}_denoise_{dtype}": value for dtype in FITTED for part, value in (("k", 262144), ("b", GIB))}
"""The hand-set constants of a `legacy` Z-Image denoise entry."""


def _legacy_data_copy(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    """The shipped data as it was before the RTX 4090's native-allocator Z-Image denoise points shipped: that entry
    hand-set and `legacy` again, without its points."""
    data_dir, _ = _data_copy(tmp_path)
    data = load_data(data_dir)["denoise.json"]

    def reverted(entry: coefficients.CoefficientEntry | MeasuredPoint) -> bool:
        return entry.estimator == Z_IMAGE_DENOISE_SPEC.id and entry.platform == CUDNN_NATIVE

    entries = [
        entry.model_copy(update={"values": LEGACY_DENOISE, "legacy": True}) if reverted(entry) else entry
        for entry in data.coefficients
    ]
    points = [point for point in data.points if not reverted(point)]
    write_data(data_dir / "denoise.json", data.model_copy(update={"coefficients": entries, "points": points}))
    return data_dir, _files(data_dir)


def _test_points(data_dir: Path) -> list[MeasuredPoint]:
    """The Z-Image denoise points the test's records or export wrote: their device is not the shipped Z-Image points'
    `RTX 4090`. A record names the card as the calibration script does, an export's point by name and device class
    id."""
    devices = {RTX_4090.name, f"{RTX_4090.name} ({RTX_4090.id})"}
    return [
        point
        for point in load_data(data_dir)["denoise.json"].points
        if point.estimator == Z_IMAGE_DENOISE_SPEC.id and point.device in devices
    ]


def test_fresh_process_points_replace_the_legacy_entry_they_fit_and_leave_the_others_as_they_were(
    tmp_path: Path, shipped_checks, capsys
) -> None:
    """RTX 4090 points under the native allocator, all three dtypes at three sizes: that platform's legacy entry
    becomes their fit and passes the data checks. Every other entry stays as shipped, the one still legacy included,
    which no point measures. The script's own keys in a record are ignored."""
    forwards = [_forward_record(image, dtype) for dtype in FITTED for image in (1024, 4096, 16384)]
    points = _point_file(tmp_path / "z_image_points.jsonl", forwards)
    data_dir, _ = _legacy_data_copy(tmp_path)

    assert main(["--points", str(points), "--data-dir", str(data_dir), "--write"]) == 0

    out = capsys.readouterr().out
    assert "9 fresh-process points from 1 point files; records left out: none" in out
    assert f"denoise.json: denoise.z_image {CUDNN_NATIVE} is fitted now, no longer legacy" in out
    assert (
        "{'conv_backend': 'miopen', 'allocator': 'native'} stays legacy: no in-fit point measures this platform" in out
    )
    entries, shipped = _denoise_entries(data_dir), _denoise_entries(DATA_DIR)
    native, shipped_native = entries.pop("cudnn,native"), shipped.pop("cudnn,native")
    assert not native.legacy and native.values == {
        f"{part}_denoise_{dtype}": value
        for dtype, line in FITTED.items()
        for part, value in zip("kb", line, strict=True)
    }
    # What it floors the platforms it prices unmeasured at stays as declared.
    assert native.unmeasured_floor == shipped_native.unmeasured_floor != {}
    assert entries == shipped
    # The points name the platform as the data keys it: no OS where no entry names one.
    written = _test_points(data_dir)
    assert len(written) == 9 and {tuple(point.platform) for point in written} == {("conv_backend", "allocator")}
    shipped_checks(data_dir)


def test_a_check_ships_out_of_the_fit_beside_the_fit_point_whose_features_it_shares(tmp_path: Path) -> None:
    """The script's checks test an assumption or an analytic term, not the line: a forward with four blocks instead
    of two, whose features are a fit point's, and one forced onto the math kernel, priced with the score matrix that
    builds. Both ship out of the fit, bounding their estimates from below only. The four-block forward measured more
    than the two-block one, yet sits beside it rather than in its place, and neither moves the fit."""
    forwards = [_forward_record(image, dtype) for dtype in FITTED for image in (1024, 4096, 16384)]
    two_blocks = forwards[1]
    four_blocks = _forward_record(4096, "bf16", check=True, measured_bytes=two_blocks["measured_bytes"] + 20 * MIB)
    forced_math = _forward_record(4096, "bf16", check=True, measured_bytes=9 * GIB)
    forced_math["features"] |= {"score_matrix_bytes": 30 * 4608 * 4608 * 17}
    points = _point_file(tmp_path / "with_checks.jsonl", [*forwards, four_blocks, forced_math])
    data_dir, _ = _legacy_data_copy(tmp_path)

    assert main(["--points", str(points), "--data-dir", str(data_dir), "--write"]) == 0

    written = _test_points(data_dir)
    shared = [p for p in written if p.features == two_blocks["features"]]
    assert sorted((p.in_fit, p.measured_bytes) for p in shared) == [
        (False, four_blocks["measured_bytes"]),
        (True, two_blocks["measured_bytes"]),
    ]
    assert [p.in_fit for p in written].count(False) == 2
    native = _denoise_entries(data_dir)["cudnn,native"]
    assert native.values == {
        f"{part}_denoise_{dtype}": value
        for dtype, line in FITTED.items()
        for part, value in zip("kb", line, strict=True)
    }


def test_a_legacy_entry_keeps_its_values_unless_the_points_fit_every_line_and_its_points_bound_them(
    tmp_path: Path, capsys
) -> None:
    """bf16 points alone fit no fp16 or fp32 line, so the entry stays legacy. Its hand-set values then reserve over 3x
    what the points measured, which the bound rule refuses to write."""
    points = _point_file(tmp_path / "bf16.jsonl", [_forward_record(image, "bf16") for image in (1024, 4096, 16384)])
    data_dir, before = _legacy_data_copy(tmp_path)

    assert main(["--points", str(points), "--data-dir", str(data_dir), "--write"]) == 1

    captured = capsys.readouterr()
    assert (
        f"note: denoise.z_image {CUDNN_NATIVE} stays legacy: denoise_fp16: no in-fit point measures it; "
        "denoise_fp32: no in-fit point measures it"
    ) in captured.out
    assert "the bounds of these points reject the result" in captured.err
    assert (
        "denoise.z_image-cudnn,native-denoise-bf16-1024+512tok-NVIDIAGeForceRTX4090-demand_peak: reserves 3.27x"
    ) in captured.err
    assert _files(data_dir) == before


def test_probe_windows_alone_never_replace_a_legacy_entrys_hand_set_values(tmp_path: Path, machine, capsys) -> None:
    """RTX 4090 windows of every dtype at three sizes would fit each line the legacy entry carries. But a window is a
    warm reading in a running server, which can lie below the cold first run, so it bounds a line from below only: the
    entry keeps its hand-set values, the windows ship as lower bounds under them, and the note says why."""
    windows = []
    for item, (dtype, image) in enumerate(itertools.product(FITTED, (1024, 4096, 16384)), start=1):
        features = _forward_record(image, dtype)["features"]
        demand = DENOISE_SLOPE * features["tokens"] + DENOISE_BASE
        windows.append(
            _window(item, features, demand, estimator=Z_IMAGE_DENOISE_SPEC.id, version=Z_IMAGE_DENOISE_SPEC.version)
        )
    machine(None, "linux")
    export = _export(tmp_path / f"20261018-120000-box-1-{RTX_4090.id}.json", RTX_4090, windows)
    data_dir, _ = _legacy_data_copy(tmp_path)

    assert main([str(export), "--data-dir", str(data_dir), "--write"]) == 0

    reasons = "; ".join(
        f"denoise_{dtype}: only probe_demand points measure it, which bound it from below only" for dtype in FITTED
    )
    assert f"note: denoise.z_image {CUDNN_NATIVE} stays legacy: {reasons}" in capsys.readouterr().out
    native = _denoise_entries(data_dir)["cudnn,native"]
    assert native.legacy and native.values == LEGACY_DENOISE
    assert [p.method for p in _test_points(data_dir)] == ["probe_demand"] * len(windows)


def test_a_legacy_entry_whose_fit_conflicts_with_the_bounds_stays_legacy(tmp_path: Path, machine, capsys) -> None:
    """Fresh RTX 4090 forwards of every dtype at three sizes would replace the legacy entry with their fit, as above.
    A warm window 2 GiB above the bf16 line at 16384 image tokens (5778 MiB) needs 6067 MiB there, but the fresh
    forward of that size holds the line to 1.3x its 3730 MiB, 4849 MiB: no bf16 line keeps both. The entry keeps its
    hand-set values, the note names the conflict, and those values then over-reserve the fresh forwards, which the
    bound rule refuses to write."""
    records = [_forward_record(image, dtype) for dtype in FITTED for image in (1024, 4096, 16384)]
    points = _point_file(tmp_path / "z_image_points.jsonl", records)
    features = _forward_record(16384, "bf16")["features"]
    demand = DENOISE_SLOPE * features["tokens"] + DENOISE_BASE + 2 * GIB
    window = _window(1, features, demand, estimator=Z_IMAGE_DENOISE_SPEC.id, version=Z_IMAGE_DENOISE_SPEC.version)
    machine(None, "linux")
    export = _export(tmp_path / f"20261018-120000-box-1-{RTX_4090.id}.json", RTX_4090, [window])
    data_dir, before = _legacy_data_copy(tmp_path)

    assert main([str(export), "--points", str(points), "--data-dir", str(data_dir), "--write"]) == 1

    captured = capsys.readouterr()
    assert (
        f"note: denoise.z_image {CUDNN_NATIVE} stays legacy: denoise_bf16: no line at or above k=236544, b=105 MiB "
        "covers 5778 MiB at size 16896 by 5% within the upper bound of "
        "denoise.z_image-cudnn,native-denoise-bf16-16384+512tok-NVIDIAGeForceRTX4090-demand_peak "
        "(measured 3730 MiB, at most 4849 MiB)"
    ) in captured.out
    assert "is fitted now, no longer legacy" not in captured.out
    assert "the bounds of these points reject the result" in captured.err
    assert "conflict with the bounds" not in captured.err
    assert _files(data_dir) == before


def test_a_measurement_a_point_file_gives_twice_keeps_its_larger_reading(tmp_path: Path, entries, specs) -> None:
    """A rerun that reads less, recorded after the first run, does not take its place."""
    larger = _forward_record(4096, "bf16")
    smaller = larger | {"measured_bytes": larger["measured_bytes"] - 64 * MIB}

    build = read_point_files([_point_file(tmp_path / "twice.jsonl", [larger, smaller])], specs, entries)

    assert [point.measured_bytes for point in build.points] == [larger["measured_bytes"]]
    assert build.skipped == {"remeasured, the larger kept": 1}


@pytest.mark.parametrize(
    "changes, message",
    [
        pytest.param({"method": "probe_demand"}, "bounds its estimate from below only", id="probe-window"),
        pytest.param(
            {"platform": {"conv_backend": "cudnn", "allocator": "expandable", "os": "linux"}},
            "which `platform.MEASURED` does not list",
            id="unmeasured-platform",
        ),
    ],
)
def test_a_record_that_is_no_fresh_process_point_of_a_measured_platform_is_refused(
    changes: dict, message: str, tmp_path: Path, capsys
) -> None:
    points = _point_file(tmp_path / "refused.jsonl", [_forward_record(4096, "bf16", **changes)])

    assert main(["--points", str(points), "--data-dir", str(DATA_DIR)]) == 2
    assert message in capsys.readouterr().err


def test_a_remeasurement_below_the_shipped_point_keeps_it_and_says_so(tmp_path: Path, capsys) -> None:
    """The shipped point is the larger reading, so it stays; the build names the lower one rather than drop it unseen."""
    shipped = load_data(DATA_DIR)["denoise.json"].points[0]
    rerun = shipped.model_dump(mode="json") | {"measured_bytes": shipped.measured_bytes - 64 * MIB}
    points = _point_file(tmp_path / "rerun.jsonl", [rerun])
    data_dir, before = _data_copy(tmp_path)

    assert main(["--points", str(points), "--data-dir", str(data_dir), "--write"]) == 0

    out = capsys.readouterr().out
    assert f"kept shipped {shipped.measured_bytes} over remeasured {shipped.measured_bytes - 64 * MIB}" in out
    assert _files(data_dir) == before


def _toy_terms(features, values):
    return ()


def test_a_legacy_entrys_tiled_line_is_fitted_on_the_fit_of_the_line_beneath(tmp_path: Path) -> None:
    """`legacy_fit` rates a line that sits on another with that line's fit, as `fitted_values` does, rather than
    leaving it out of the fit that would replace the hand-set values."""
    spec = EstimatorSpec(
        id="toy", version=1, coefficients=frozenset(), platform_keys=("conv_backend",), terms=_toy_terms,
        line=convolution_line,
    )  # fmt: skip
    platform = {"conv_backend": "cudnn"}

    def point(px: int, mib: int, tile: int | None = None) -> MeasuredPoint:
        features = {"operation": "encode", "dtype": "bf16", "height": px, "width": px, "tile": tile}
        return MeasuredPoint(estimator="toy", version=1, platform=platform, features=features,
                             measured_bytes=mib * 2**20, method="demand_peak", device="-", provenance="-")  # fmt: skip

    lines = {f"{part}_{line}_bf16": 0.0 for line in ("encode", "tiled_encode") for part in ("k", "b")}
    entry = CoefficientEntry(estimator="toy", version=1, platform=platform, values=lines, legacy=True, provenance="-")
    data = WorkingMemoryData(
        schema_version=1,
        coefficients=[entry],
        points=[point(512, 1100), point(1024, 4400), point(2048, 1250, 512), point(4096, 1500, 512)],
    )
    groups = fit_points([data], {"toy": spec})

    values, problems = legacy_fit(entry, groups, base_values([entry], groups, {}), {})

    assert not problems and set(values) == set(lines)
