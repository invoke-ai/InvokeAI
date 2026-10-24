"""The shipped working-memory data, checked against the estimators that read it.

Every coefficient entry resolves for the spec it names on a measured platform. Every value an entry carries is either a
declared stand-in or decided by at least one in-fit point and the shipped fit (`fit.py`) of the points it is resolved
for. Every point bounds its estimate as `working_memory/bounds.py` says: from below always, from above where a fitted
term decides an in-fit point that holds an upper bound (a fresh-process peak, not a `probe_demand` window). The points no
line can hold within 1.3x are listed in `bounds.UPPER_BOUND_EXCEPTIONS`; the out-of-fit ones in `OUT_OF_FIT`.
"""

import collections
import itertools
import platform as stdlib_platform
import re
from dataclasses import replace
from pathlib import Path
from typing import get_args, get_type_hints

import pytest
import torch
from pydantic import ValidationError

import invokeai
from invokeai.app.util.torch_cuda_allocator import ROCM_WINDOWS_ALLOC_CONF
from invokeai.backend.util import nvml
from invokeai.backend.working_memory import platform as platform_module
from invokeai.backend.working_memory.bounds import (
    MAX_OVER_RESERVATION,
    UPPER_BOUND_EXCEPTIONS,
    bound_violation,
    evaluate_point,
    point_id,
)
from invokeai.backend.working_memory.coefficients import (
    CoefficientEntry,
    MeasuredPoint,
    ResolvedCoefficients,
    ValuesFrom,
    WorkingMemoryData,
    load_data,
    resolve_coefficients,
    shipped_coefficients,
    shipped_data,
)
from invokeai.backend.working_memory.counters import backend_for
from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate
from invokeai.backend.working_memory.fingerprint import environment_for
from invokeai.backend.working_memory.fit import capping_spans, carrier_of, entry_key, fit_points, fitted_values
from invokeai.backend.working_memory.platform import (
    MEASURED,
    Platform,
    is_measured,
    platform_for,
    platform_for_environment,
    priced_as,
)
from invokeai.backend.working_memory.spec import ESTIMATOR_MODULES, EstimatorSpec, import_estimator_modules

OUT_OF_FIT = {((("allocator", "native"), ("conv_backend", "cudnn")), "RTX 4090", "reserved_peak"): 6}
"""The out-of-fit points by platform, device and method: the six CUDA forced-math-kernel decodes and encodes, which
check the score-matrix term. A point that leaves the fit needs an edit here."""

# Imported deliberately rather than relying on whichever modules ran first: a spec missing from the registry is a spec
# whose data nobody checks.
SPECS = import_estimator_modules()

EVERY_PLATFORM = [
    Platform(**dict(zip(get_type_hints(Platform), values, strict=True)))
    for values in itertools.product(*(get_args(hint) for hint in get_type_hints(Platform).values()))
]
MEASURED_PLATFORMS = [platform for platform in EVERY_PLATFORM if is_measured(platform)]
UNMEASURED_PLATFORMS = [platform for platform in EVERY_PLATFORM if not is_measured(platform)]


def _keys(platform: Platform, spec: EstimatorSpec) -> dict[str, str]:
    return {key: getattr(platform, key) for key in spec.platform_keys}


def _evaluate(point: MeasuredPoint) -> WorkingMemoryEstimate:
    spec = SPECS[point.estimator]
    assert point.version == spec.version
    return evaluate_point(point, spec, shipped_data().values())


ALL_POINTS = [point for data in shipped_data().values() for point in data.points]
POINTS = [pytest.param(point, id=point_id(point)) for point in ALL_POINTS]
ENTRIES = [entry for data in shipped_data().values() for entry in data.coefficients]


def test_every_data_file_parses_and_declares_coefficients() -> None:
    for name, data in shipped_data().items():
        assert data.coefficients, f"{name} declares no coefficients"


def test_every_coefficient_entry_resolves_for_its_spec_on_a_measured_platform() -> None:
    """Resolving checks what an entry provides against what its spec reads, the borrowed values included. An entry for
    an unmeasured platform would never be read: the resolver prices that platform as `priced_as` says. Only the OS may
    refine the spec's keys."""
    seen: set[tuple[str, int, tuple[tuple[str, str], ...]]] = set()
    for entry in ENTRIES:
        where = f"{entry.estimator} on {entry.platform}"
        spec = SPECS.get(entry.estimator)
        assert spec is not None, f"{where}: no estimator registers this id"
        assert entry.version == spec.version, f"{where}: version {entry.version}, the estimator is {spec.version}"
        assert set(entry.platform) - {"os"} == set(spec.platform_keys), f"{where}: the spec keys {spec.platform_keys}"
        platforms = [
            p for p in MEASURED_PLATFORMS if all(getattr(p, key) == value for key, value in entry.platform.items())
        ]
        assert platforms, f"{where}: not a measured platform"
        assert entry_key(entry) not in seen, f"{where}: declared twice"
        seen.add(entry_key(entry))
        for platform in platforms:
            shipped_coefficients(spec, platform)


@pytest.mark.parametrize("spec", sorted(SPECS.values(), key=lambda spec: spec.id), ids=lambda spec: spec.id)
def test_every_estimator_has_coefficients_on_every_measured_platform(spec: EstimatorSpec) -> None:
    for platform in MEASURED_PLATFORMS:
        assert set(shipped_coefficients(spec, platform).values) == spec.coefficients


@pytest.mark.parametrize("spec", sorted(SPECS.values(), key=lambda spec: spec.id), ids=lambda spec: spec.id)
def test_the_os_only_matters_where_an_entry_names_it(spec: EstimatorSpec) -> None:
    """cuDNN and MIOpen with expandable segments are priced the same on every OS."""
    os_specific = {
        (entry.platform["conv_backend"], entry.platform["allocator"]) for entry in ENTRIES if "os" in entry.platform
    }
    assert os_specific == {("miopen", "native"), ("miopen", "expandable")}
    for platform in MEASURED_PLATFORMS:
        if (platform.conv_backend, platform.allocator) in os_specific:
            continue
        values = [shipped_coefficients(spec, replace(platform, os=os)).values for os in ("windows", "linux", "other")]
        assert all(dict(v) == dict(values[0]) for v in values), platform


@pytest.mark.parametrize("spec", sorted(SPECS.values(), key=lambda spec: spec.id), ids=lambda spec: spec.id)
def test_an_unmeasured_platform_is_priced_as_the_measured_one_at_least_and_says_so(spec: EstimatorSpec) -> None:
    for platform in UNMEASURED_PLATFORMS:
        measured = priced_as(platform)
        assert is_measured(measured) and measured.os == platform.os
        resolved = shipped_coefficients(spec, platform)
        borrowed = shipped_coefficients(spec, measured)
        assert all(resolved.values[name] >= value for name, value in borrowed.values.items())
        where = ",".join(f"{key}={value}" for key, value in _keys(measured, spec).items())
        assert resolved.source.startswith(f"shipped:conv_backend={platform.conv_backend},")
        assert f"(priced as {where}" in resolved.source


def test_every_fitted_value_is_decided_by_an_in_fit_point() -> None:
    """A value no point decides is unguarded: any number would pass. An estimator without points of its own takes a
    measured one's values with `values_from`, and is then held to that one's points. Only declared stand-ins are
    exempt, and they are listed in the entry."""
    decided: set[tuple[tuple[str, int, tuple[tuple[str, str], ...]], str]] = set()
    for point in ALL_POINTS:
        if point.in_fit:
            owner = entry_key(carrier_of(ENTRIES, point))
            decided |= {(owner, t.coefficient) for t in _evaluate(point).terms if t.included and t.coefficient}

    undecided = [
        f"{entry.estimator} {entry.platform} {name}"
        for entry in ENTRIES
        for name in entry.values or {}
        if (entry_key(entry), name) not in decided and name not in entry.stand_ins
    ]
    assert not undecided, f"no measured point decides these shipped values: {undecided}"


def test_every_fitted_value_is_the_shipped_fit_of_its_points() -> None:
    """The values are reproducible from the points rather than hand-tuned. A new point that moves a fit fails here,
    naming the values to ship."""
    fitted = fitted_values(shipped_data().values())
    stale = {}
    for entry in ENTRIES:
        if entry.values is None:
            continue
        expected = fitted.get(entry_key(entry), {})
        differs = {
            name: expected.get(name)
            for name, value in entry.values.items()
            if name not in entry.stand_ins and expected.get(name) != value
        }
        if differs:
            stale[f"{entry.estimator} {entry.platform}"] = differs
    assert not stale, f"ship these fitted values: {stale}"


def test_a_stand_in_is_a_whole_line_its_fresh_process_points_cannot_fit() -> None:
    """A stand-in is for what a platform has not measured. Once in-fit fresh-process points measure its line at sizes
    a fit needs (`capping_spans`), the line is fitted instead, and this names it. `probe_demand` points bound a
    stand-in from below only: warm windows never fit a line below what a cold first run needs."""
    groups = fit_points(shipped_data().values())
    for entry in ENTRIES:
        lines = {name[2:] for name in entry.stand_ins}
        assert {f"{part}_{line}" for line in lines for part in ("k", "b")} == set(entry.stand_ins), entry.platform
        measured = sorted(line for line in lines if capping_spans(groups.get((entry_key(entry), line), [])))
        assert not measured, f"{entry.estimator} on {entry.platform}: fit {measured}, they are measured now"


def test_the_out_of_fit_points_are_the_listed_ones() -> None:
    """Leaving the fit removes a point's upper bound, so it is never a quiet flag flip."""
    out_of_fit = collections.Counter(
        (tuple(sorted(point.platform.items())), point.device, point.method) for point in ALL_POINTS if not point.in_fit
    )
    assert out_of_fit == OUT_OF_FIT


def test_every_listed_upper_bound_exception_is_a_shipped_point_that_needs_it() -> None:
    """A stale exception would let a new point at that id over-reserve unseen."""
    ids = {point_id(point): point for point in ALL_POINTS}
    for listed, ratio in UPPER_BOUND_EXCEPTIONS.items():
        point = ids.get(listed)
        assert point is not None, f"{listed}: no such point"
        assert point.in_fit
        reserved = _evaluate(point).reservation_bytes / point.measured_bytes
        assert MAX_OVER_RESERVATION < reserved <= ratio, f"{listed}: {reserved:.3f}x"


@pytest.mark.parametrize("point", POINTS)
def test_every_measured_point_is_bounded_by_its_estimate(point: MeasuredPoint) -> None:
    assert bound_violation(point, _evaluate(point)) is None


def test_a_point_bound_is_never_looser_than_the_default() -> None:
    """`max_ratio` tightens the bound for a point the estimate has to stay close to; it never excuses one."""
    for point in ALL_POINTS:
        if point.max_ratio is not None:
            assert 1.0 <= point.max_ratio < MAX_OVER_RESERVATION, point_id(point)


@pytest.mark.parametrize(
    "device, hip, conv_backend",
    [
        (torch.device("cuda", 0), "7.2.0", "miopen"),
        (torch.device("cuda", 0), None, "cudnn"),
        # A HIP build reports its GPU as "cuda"; anything else on that build is not on MIOpen.
        (torch.device("cpu"), "7.2.0", "cpu"),
        (torch.device("mps"), "7.2.0", "mps"),
        (torch.device("xpu"), None, "xpu"),
        (torch.device("meta"), None, "cpu"),
    ],
)
def test_the_platform_is_the_convolution_backend_the_device_really_runs(device, hip, conv_backend, monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", hip)
    assert platform_for(device).conv_backend == conv_backend


HOST_OS = platform_module._host_os
"""Bound at import, before the suite's fixture replaces it."""

ALLOC_VARS = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF")


@pytest.mark.parametrize("var", ALLOC_VARS)
@pytest.mark.parametrize(
    "backend, alloc_conf, allocator",
    [
        ("native", None, "native"),
        ("native", "expandable_segments: True", "expandable"),
        # What Invoke sets for a ROCm build on Windows.
        ("native", ROCM_WINDOWS_ALLOC_CONF, "expandable"),
        ("native", "expandable_segments:False", "native"),
        # The driver's pool ignores expandable segments.
        ("cudaMallocAsync", "backend:cudaMallocAsync,expandable_segments:True", "cudaMallocAsync"),
    ],
)
def test_the_platform_is_the_allocator_a_gpu_allocates_through(
    var, backend, alloc_conf, allocator, monkeypatch
) -> None:
    for name in ALLOC_VARS:
        monkeypatch.delenv(name, raising=False)
    if alloc_conf is not None:
        monkeypatch.setenv(var, alloc_conf)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: backend)

    assert platform_for(torch.device("cuda", 0)).allocator == allocator
    # Nothing but a CUDA/HIP GPU allocates through torch's caching allocator.
    assert platform_for(torch.device("cpu")).allocator == "native"
    assert platform_for(torch.device("mps")).allocator == "native"


@pytest.mark.parametrize(
    "sys_platform, os", [("win32", "windows"), ("linux", "linux"), ("darwin", "other"), ("freebsd14", "other")]
)
def test_the_platform_is_the_os_the_process_runs_on(sys_platform, os, monkeypatch) -> None:
    # The suite pins `_host_os` to Linux (tests/conftest.py); this is the function it replaces.
    monkeypatch.setattr(platform_module.sys, "platform", sys_platform)
    assert HOST_OS() == os
    monkeypatch.setattr(platform_module, "_host_os", HOST_OS)
    assert platform_for(torch.device("cpu")).os == os


def _recorded_hard_facts(device: torch.device) -> dict[str, str]:
    """The hard facts a calibration store records for `device` in this process."""
    return dict(environment_for(device, {"attention_type": "sdpa"}).hard)


@pytest.fixture
def host_os_unpinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """The platform reads the OS from `sys.platform` again, as `environment_for` does, instead of the suite's pin."""
    monkeypatch.setattr(platform_module, "_host_os", HOST_OS)


@pytest.mark.parametrize("hip", [None, "7.14.0"], ids=["cuda", "hip"])
@pytest.mark.parametrize(
    "allocator_backend, alloc_var, alloc_conf",
    [
        ("native", None, None),
        ("native", "PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True"),
        ("native", "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False"),
        ("cudaMallocAsync", "PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync,expandable_segments:True"),
    ],
)
@pytest.mark.parametrize("sys_platform", ["win32", "linux", "darwin"])
def test_a_recorded_gpu_environment_names_the_platform_its_operations_are_priced_on(
    hip, allocator_backend, alloc_var, alloc_conf, sys_platform, host_os_unpinned, monkeypatch
) -> None:
    """Shipped points are keyed by the platform their calibration store's environment names; it must be the one
    `platform_for` prices the same machine on."""
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: allocator_backend)
    monkeypatch.setattr(nvml, "driver_version", lambda: None)
    monkeypatch.setattr(platform_module.sys, "platform", sys_platform)
    monkeypatch.setattr(stdlib_platform, "platform", lambda: "Test-1.0")
    if alloc_var is not None:
        monkeypatch.setenv(alloc_var, alloc_conf)
    cuda = torch.device("cuda", 0)

    assert platform_for_environment(backend_for(cuda), _recorded_hard_facts(cuda)) == platform_for(cuda)


@pytest.mark.parametrize("device", [torch.device("mps"), torch.device("xpu")], ids=str)
def test_a_recorded_environment_of_a_device_without_a_caching_allocator_names_its_platform(
    device, host_os_unpinned, monkeypatch
) -> None:
    # An allocator variable configures CUDA/HIP only.
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    assert platform_for_environment(backend_for(device), _recorded_hard_facts(device)) == platform_for(device)


def test_an_environment_without_the_platforms_facts_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", None)
    # A CPU wheel (macOS, Windows) has no allocator binding to ask.
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(nvml, "driver_version", lambda: None)
    hard = _recorded_hard_facts(torch.device("cuda", 0))

    with pytest.raises(ValueError, match="no operating system"):
        platform_for_environment("cuda", {key: value for key, value in hard.items() if key != "os"})
    with pytest.raises(ValueError, match="no allocator_backend"):
        platform_for_environment("cuda", {key: value for key, value in hard.items() if key != "allocator_backend"})


@pytest.mark.parametrize(
    "platform, measured",
    [
        (("mps", "native", "linux"), ("cudnn", "native", "linux")),
        (("xpu", "native", "windows"), ("cudnn", "native", "windows")),
        (("cpu", "native", "other"), ("cudnn", "native", "other")),
        (("cudnn", "expandable", "windows"), ("cudnn", "native", "windows")),
        (("miopen", "cudaMallocAsync", "linux"), ("miopen", "native", "linux")),
        # Expandable segments on MIOpen are measured on Windows only, where they are Invoke's default.
        (("miopen", "expandable", "windows"), ("miopen", "expandable", "windows")),
        (("miopen", "expandable", "linux"), ("miopen", "native", "linux")),
        (("miopen", "expandable", "other"), ("miopen", "native", "other")),
    ],
)
def test_an_unmeasured_platform_takes_cudnn_or_its_backends_native_allocator(platform, measured) -> None:
    assert priced_as(Platform(*platform)) == Platform(*measured)
    assert is_measured(Platform(*measured))
    assert {m for m in MEASURED if m[2] is not None} == {("miopen", "expandable", "windows")}


def _no_terms(features, values):
    return ()


def _spec(spec_id: str, version: int = 1, coefficients: frozenset[str] = frozenset({"k"})) -> EstimatorSpec:
    return EstimatorSpec(
        id=spec_id, version=version, coefficients=coefficients, platform_keys=("conv_backend",), terms=_no_terms
    )


def _entry(
    estimator: str, *, version: int = 1, values=None, values_from: str | None = None, floor=None, os=None
) -> CoefficientEntry:
    return CoefficientEntry(
        estimator=estimator,
        version=version,
        platform={"conv_backend": "cudnn"} | ({"os": os} if os else {}),
        values=values,
        values_from=ValuesFrom(estimator=values_from, version=1) if values_from else None,
        unmeasured_floor=floor or {},
        provenance="-",
    )


def _data(*entries: CoefficientEntry) -> list[WorkingMemoryData]:
    return [WorkingMemoryData(schema_version=1, coefficients=list(entries), points=[])]


CUDNN = Platform("cudnn", "native", "linux")


def test_a_missing_or_stale_entry_names_the_estimator_and_platform() -> None:
    """No fallback to another platform or version: coefficients fitted for something else are the shortfall the data
    exists to prevent."""
    spec = _spec("test.estimator", version=2)

    assert resolve_coefficients(
        _data(_entry(spec.id, version=2, values={"k": 7.0})), spec, CUDNN, origin="test"
    ) == ResolvedCoefficients(values={"k": 7.0}, source="test:conv_backend=cudnn")
    with pytest.raises(LookupError, match=r"'test\.estimator' version 2 on conv_backend=miopen\."):
        resolve_coefficients(
            _data(_entry(spec.id, version=2, values={"k": 7.0})), spec, replace(CUDNN, conv_backend="miopen"), "test"
        )
    with pytest.raises(LookupError, match=r"'test\.estimator' version 2 on conv_backend=cudnn \(found only version"):
        resolve_coefficients(_data(_entry(spec.id, version=1, values={"k": 7.0})), spec, CUDNN, "test")
    with pytest.raises(LookupError, match=r"declared 2 times"):
        resolve_coefficients(
            _data(_entry(spec.id, version=2, values={"k": 7.0}), _entry(spec.id, version=2, values={"k": 8.0})),
            spec,
            CUDNN,
            "test",
        )


def test_an_entry_that_names_the_os_takes_precedence_there_and_only_there() -> None:
    data = _data(
        _entry("test.lender", values={"k": 7.0}),
        _entry("test.lender", values={"k": 9.0}, os="windows"),
        _entry("test.borrower", values_from="test.lender"),
    )
    on_windows = resolve_coefficients(data, _spec("test.borrower"), replace(CUDNN, os="windows"), origin="test")
    assert on_windows == ResolvedCoefficients(
        values={"k": 9.0}, source="test:conv_backend=cudnn,os=windows (values of test.lender)"
    )
    on_linux = resolve_coefficients(data, _spec("test.borrower"), CUDNN, origin="test")
    assert dict(on_linux.values) == {"k": 7.0}


def test_borrowed_values_resolve_one_hop_and_the_source_names_both_borrowings() -> None:
    data = _data(_entry("test.lender", values={"k": 7.0}), _entry("test.borrower", values_from="test.lender"))

    on_cudnn = resolve_coefficients(data, _spec("test.borrower"), CUDNN, origin="test")
    assert on_cudnn == ResolvedCoefficients(values={"k": 7.0}, source="test:conv_backend=cudnn (values of test.lender)")
    on_xpu = resolve_coefficients(data, _spec("test.borrower"), replace(CUDNN, conv_backend="xpu"), origin="test")
    assert on_xpu == ResolvedCoefficients(
        values={"k": 7.0}, source="test:conv_backend=xpu (priced as conv_backend=cudnn, values of test.lender)"
    )


def test_a_borrower_reads_the_lines_it_runs_of_a_lender_that_carries_more() -> None:
    """A family member that runs fewer dtypes borrows from the one that runs them all; it sees only its own."""
    data = _data(
        _entry("test.lender", values={"k": 7.0, "k_other": 9.0}), _entry("test.borrower", values_from="test.lender")
    )
    resolved = resolve_coefficients(data, _spec("test.borrower"), CUDNN, origin="test")
    assert dict(resolved.values) == {"k": 7.0}
    # An entry's own values are exactly what its spec reads: an extra one would be a value nothing checks.
    with pytest.raises(LookupError, match=r"provide \['k', 'k_other'\]"):
        resolve_coefficients(data, _spec("test.lender"), CUDNN, origin="test")


def test_an_unmeasured_platform_raises_a_value_to_its_floor_and_a_measured_one_does_not() -> None:
    data = _data(_entry("test.lender", values={"k": 7.0, "j": 5.0}, floor={"k": 8.0, "j": 4.0}))
    spec = _spec("test.lender", coefficients=frozenset({"k", "j"}))

    assert dict(resolve_coefficients(data, spec, CUDNN, origin="test").values) == {"k": 7.0, "j": 5.0}
    on_mps = resolve_coefficients(data, spec, replace(CUDNN, conv_backend="mps"), origin="test")
    assert dict(on_mps.values) == {"k": 8.0, "j": 5.0}
    assert on_mps.source == "test:conv_backend=mps (priced as conv_backend=cudnn, floored)"


@pytest.mark.parametrize(
    "entries, message",
    [
        pytest.param(
            (_entry("a", values_from="b"), _entry("b", values_from="c"), _entry("c", values={"k": 1.0})),
            "borrows in turn",
            id="chain",
        ),
        pytest.param((_entry("a", values_from="b"), _entry("b", values_from="a")), "borrows in turn", id="cycle"),
        pytest.param(
            (_entry("a", values_from="b"),), r"No working-memory coefficients for estimator 'b'", id="no-lender"
        ),
        pytest.param(
            (_entry("a", values_from="b"), _entry("b", values={"other": 1.0})), r"provide \['other'\]", id="mismatch"
        ),
    ],
)
def test_a_borrowing_that_is_not_one_hop_to_matching_values_is_refused(entries, message) -> None:
    with pytest.raises(LookupError, match=message):
        resolve_coefficients(_data(*entries), _spec("a"), CUDNN, origin="test")


@pytest.mark.parametrize("values, values_from", [({"k": 1.0}, "b"), (None, None)], ids=["both", "neither"])
def test_an_entry_carries_exactly_one_of_values_and_values_from(values, values_from) -> None:
    with pytest.raises(ValidationError, match="exactly one of"):
        _entry("a", values=values, values_from=values_from)


def test_a_floor_or_stand_in_names_a_value_the_entry_carries() -> None:
    with pytest.raises(ValidationError, match=r"unmeasured floors \['j'\]"):
        _entry("a", values={"k": 1.0}, floor={"j": 2.0})
    with pytest.raises(ValidationError, match=r"stand-ins \['j'\]"):
        CoefficientEntry(estimator="a", version=1, platform={}, values={"k": 1.0}, stand_ins=["j"], provenance="-")


def test_a_data_directory_without_data_files_is_named(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=re.escape(str(tmp_path))):
        load_data(tmp_path)
    with pytest.raises(FileNotFoundError, match=re.escape(str(tmp_path / "missing"))):
        load_data(tmp_path / "missing")
    (tmp_path / "broken.json").write_text('{"schema_version": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="broken.json"):
        load_data(tmp_path)


@pytest.mark.platform_independent
def test_every_module_that_declares_an_estimator_is_listed() -> None:
    """The registry fills as modules import, so a spec declared outside `ESTIMATOR_MODULES` is missing from every
    check above until something else happens to import it -- and a ROCm user is the first to find its gap."""
    package = Path(invokeai.__file__).parent
    declaring = sorted(
        ".".join(path.relative_to(package.parent).with_suffix("").parts)
        for owner in ("app", "backend")
        for path in (package / owner).rglob("*.py")
        if re.search(r"\bEstimatorSpec\(", path.read_text(encoding="utf-8"))
    )
    assert declaring, "found no estimator modules -- the scan is broken"
    unlisted = sorted(set(declaring) - set(ESTIMATOR_MODULES))
    stale = sorted(set(ESTIMATOR_MODULES) - set(declaring))
    assert not unlisted and not stale, f"add to ESTIMATOR_MODULES: {unlisted}; declares no spec: {stale}"


def _measured(method: str, max_ratio: float | None = None) -> MeasuredPoint:
    return MeasuredPoint(
        estimator="vae.sd3",
        version=2,
        platform={"conv_backend": "cudnn", "allocator": "native"},
        features={},
        measured_bytes=1,
        method=method,
        device="-",
        provenance="-",
        max_ratio=max_ratio,
    )


@pytest.mark.parametrize("method", get_args(get_type_hints(MeasuredPoint)["method"]))
def test_only_fresh_process_points_bound_an_estimate_from_above(method: str) -> None:
    """A probe window is a warm operation in a running server; the cold first run after a server start can exceed it,
    so it may raise a line but never caps one."""
    assert _measured(method).holds_upper_bound == (method != "probe_demand")
    if method == "probe_demand":
        with pytest.raises(ValidationError, match="from below only"):
            _measured(method, max_ratio=1.2)
    else:
        assert _measured(method, max_ratio=1.2).max_ratio == 1.2
