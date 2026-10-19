"""The shipped working-memory data, checked against the estimators that read it.

Every coefficient entry resolves for the spec it names on a measured platform value, and every value an entry carries
is decided by at least one measured point. A point is peak *reserved* bytes, one fresh process per point (see its
`provenance`), and bounds its estimate from both sides. The model cache consumes an estimate as `free >= estimate`, so
the estimate must be at least the point. Where a fitted term decides it, it must also be at most 1.3 times the point,
or the point's tighter `max_ratio`: a reservation the operation never uses is model weights evicted for nothing. Where
an analytic term (a materialized score matrix) decides it, that term bounds rather than describes, so it gets no upper
bound, and the cache's `device_working_mem_gb` floor counts toward the lower one.
"""

import itertools
import re
from dataclasses import asdict
from pathlib import Path
from typing import get_args, get_type_hints

import pytest
import torch
from pydantic import ValidationError

import invokeai
from invokeai.app.services.config.config_default import InvokeAIAppConfig
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
from invokeai.backend.working_memory.estimate import WorkingMemoryEstimate
from invokeai.backend.working_memory.platform import UNMEASURED_PRICED_AS, Platform, platform_for
from invokeai.backend.working_memory.spec import ESTIMATOR_MODULES, EstimatorSpec, import_estimator_modules

MAX_OVER_RESERVATION = 1.3
"""How far above a measured point a reservation may sit where a fitted coefficient decides it, unless the point
carries a tighter `max_ratio`."""

WORKING_MEMORY_FLOOR = int(InvokeAIAppConfig.model_fields["device_working_mem_gb"].default * 2**30)
"""What `ModelCache._working_mem_reserve` floors every reservation at. The field default, not a config instance an
environment variable could move."""

# Imported deliberately rather than relying on whichever modules ran first: a spec missing from the registry is a spec
# whose data nobody checks.
SPECS = import_estimator_modules()


def _platforms(*, measured: bool) -> list[Platform]:
    """Every platform built from measured values only, or every platform with at least one unmeasured value."""
    every = {key: get_args(hint) for key, hint in get_type_hints(Platform).items()}
    measured_values = {
        key: [v for v in values if v not in UNMEASURED_PRICED_AS.get(key, {})] for key, values in every.items()
    }
    chosen = measured_values if measured else every
    platforms = [Platform(**dict(zip(chosen, values, strict=True))) for values in itertools.product(*chosen.values())]
    if measured:
        return platforms
    return [p for p in platforms if any(value in UNMEASURED_PRICED_AS.get(key, {}) for key, value in asdict(p).items())]


def _platform_key(platform: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(platform.items()))


def _point_id(point: MeasuredPoint) -> str:
    features = point.features
    shape = f"{features.get('height')}x{features.get('width')}"
    tile = f"-tile{features['tile']}" if features.get("tile") is not None else ""
    batch = f"-b{features['batch_size']}" if features.get("batch_size", 1) != 1 else ""
    scores = "-scores" if features.get("score_matrix_bytes") else ""
    platform = ",".join(point.platform.values())
    card = point.device.split(" (")[0].replace(" ", "")
    return f"{point.estimator}-{platform}-{features.get('operation')}-{shape}{tile}{batch}{scores}-{card}"


def _evaluate(point: MeasuredPoint) -> WorkingMemoryEstimate:
    spec = SPECS[point.estimator]
    assert point.version == spec.version
    return spec.evaluate(point.features, shipped_coefficients(spec, Platform(**point.platform)))


def _fitted_decides(estimate: WorkingMemoryEstimate) -> bool:
    return any(term.included and term.coefficient is not None for term in estimate.terms)


POINTS = [pytest.param(point, id=_point_id(point)) for data in shipped_data().values() for point in data.points]


def test_every_data_file_parses_and_declares_coefficients() -> None:
    for name, data in shipped_data().items():
        assert data.coefficients, f"{name} declares no coefficients"


def test_every_coefficient_entry_resolves_for_its_spec_on_a_measured_platform_value() -> None:
    """Resolving checks what an entry provides against what its spec reads, the borrowed values included. An entry for
    an unmeasured value would never be read: the resolver prices that value as `UNMEASURED_PRICED_AS` says."""
    seen: set[tuple[str, int, tuple[tuple[str, str], ...]]] = set()
    for name, data in shipped_data().items():
        for entry in data.coefficients:
            where = f"{name}: {entry.estimator} on {entry.platform}"
            spec = SPECS.get(entry.estimator)
            assert spec is not None, f"{where}: no estimator registers this id"
            assert entry.version == spec.version, f"{where}: version {entry.version}, the estimator is {spec.version}"
            assert set(entry.platform) == set(spec.platform_keys), f"{where}: the spec keys {spec.platform_keys}"
            for key, value in entry.platform.items():
                assert value not in UNMEASURED_PRICED_AS.get(key, {}), f"{where}: {key}={value} is not measured"
            key = (entry.estimator, entry.version, _platform_key(entry.platform))
            assert key not in seen, f"{where}: declared twice"
            seen.add(key)
            shipped_coefficients(spec, Platform(**entry.platform))


@pytest.mark.parametrize("spec", sorted(SPECS.values(), key=lambda spec: spec.id), ids=lambda spec: spec.id)
def test_every_estimator_has_coefficients_on_every_measured_platform_value(spec: EstimatorSpec) -> None:
    for platform in _platforms(measured=True):
        assert set(shipped_coefficients(spec, platform).values) == spec.coefficients


@pytest.mark.parametrize("spec", sorted(SPECS.values(), key=lambda spec: spec.id), ids=lambda spec: spec.id)
def test_an_unmeasured_platform_value_is_priced_as_the_measured_one_and_says_so(spec: EstimatorSpec) -> None:
    for platform in _platforms(measured=False):
        priced_as = Platform(
            **{key: UNMEASURED_PRICED_AS.get(key, {}).get(value, value) for key, value in asdict(platform).items()}
        )
        resolved = shipped_coefficients(spec, platform)
        assert resolved.values == shipped_coefficients(spec, priced_as).values
        assert resolved.source.startswith(
            f"shipped:conv_backend={platform.conv_backend} ({priced_as.conv_backend} values"
        )


def test_every_carried_value_is_decided_by_a_measured_point() -> None:
    """A value no point decides is unguarded: any number would pass. An estimator without points of its own takes a
    measured one's values with `values_from`, and is then held to that one's points."""
    entries = {
        (entry.estimator, entry.version, _platform_key(entry.platform)): entry
        for data in shipped_data().values()
        for entry in data.coefficients
    }
    decided: set[tuple[tuple[str, int, tuple[tuple[str, str], ...]], str]] = set()
    for data in shipped_data().values():
        for point in data.points:
            key = (point.estimator, point.version, _platform_key(point.platform))
            lender = entries[key].values_from
            owner = (lender.estimator, lender.version, key[2]) if lender is not None else key
            estimate = _evaluate(point)
            decided |= {(owner, term.coefficient) for term in estimate.terms if term.included and term.coefficient}

    undecided = [
        f"{entry.estimator} {entry.platform} {name}"
        for key, entry in entries.items()
        for name in entry.values or {}
        if (key, name) not in decided
    ]
    assert not undecided, f"no measured point decides these shipped values: {undecided}"


@pytest.mark.parametrize("point", POINTS)
def test_every_measured_point_is_bounded_by_its_estimate(point: MeasuredPoint) -> None:
    estimate = _evaluate(point)
    fitted = _fitted_decides(estimate)

    reserved = estimate.reservation_bytes if fitted else max(estimate.reservation_bytes, WORKING_MEMORY_FLOOR)
    assert reserved >= point.measured_bytes, (
        f"reserves {reserved / 2**30:.3f} GiB for a measured {point.measured_bytes / 2**30:.3f} GiB"
    )
    if point.max_ratio is not None or fitted:
        max_ratio = point.max_ratio if point.max_ratio is not None else MAX_OVER_RESERVATION
        assert estimate.reservation_bytes <= max_ratio * point.measured_bytes, (
            f"reserves {estimate.reservation_bytes / point.measured_bytes:.2f}x the measured peak, over {max_ratio}x"
        )


def test_a_point_bound_is_never_looser_than_the_default() -> None:
    """`max_ratio` tightens the bound for a point the estimate has to stay close to; it never excuses one."""
    for name, data in shipped_data().items():
        for point in data.points:
            if point.max_ratio is not None:
                assert 1.0 <= point.max_ratio < MAX_OVER_RESERVATION, f"{name}: {_point_id(point)}"


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
    assert platform_for(device) == Platform(conv_backend=conv_backend)


def _no_terms(features, values):
    return ()


def _spec(spec_id: str, version: int = 1) -> EstimatorSpec:
    return EstimatorSpec(
        id=spec_id, version=version, coefficients=frozenset({"k"}), platform_keys=("conv_backend",), terms=_no_terms
    )


def _entry(estimator: str, *, version: int = 1, values=None, values_from: str | None = None) -> CoefficientEntry:
    return CoefficientEntry(
        estimator=estimator,
        version=version,
        platform={"conv_backend": "cudnn"},
        values=values,
        values_from=ValuesFrom(estimator=values_from, version=1) if values_from else None,
        provenance="-",
    )


def _data(*entries: CoefficientEntry) -> list[WorkingMemoryData]:
    return [WorkingMemoryData(schema_version=1, coefficients=list(entries), points=[])]


def test_a_missing_or_stale_entry_names_the_estimator_and_platform() -> None:
    """No fallback to another platform or version: coefficients fitted for something else are the shortfall the data
    exists to prevent."""
    spec = _spec("test.estimator", version=2)

    assert resolve_coefficients(
        _data(_entry(spec.id, version=2, values={"k": 7.0})), spec, Platform("cudnn"), origin="test"
    ) == ResolvedCoefficients(values={"k": 7.0}, source="test:conv_backend=cudnn")
    with pytest.raises(LookupError, match=r"'test\.estimator' version 2 on conv_backend=miopen\."):
        resolve_coefficients(_data(_entry(spec.id, version=2, values={"k": 7.0})), spec, Platform("miopen"), "test")
    with pytest.raises(LookupError, match=r"'test\.estimator' version 2 on conv_backend=cudnn \(found only version"):
        resolve_coefficients(_data(_entry(spec.id, version=1, values={"k": 7.0})), spec, Platform("cudnn"), "test")
    with pytest.raises(LookupError, match=r"declared 2 times"):
        resolve_coefficients(
            _data(_entry(spec.id, version=2, values={"k": 7.0}), _entry(spec.id, version=2, values={"k": 8.0})),
            spec,
            Platform("cudnn"),
            "test",
        )


def test_borrowed_values_resolve_one_hop_and_the_source_names_both_borrowings() -> None:
    data = _data(_entry("test.lender", values={"k": 7.0}), _entry("test.borrower", values_from="test.lender"))

    on_cudnn = resolve_coefficients(data, _spec("test.borrower"), Platform("cudnn"), origin="test")
    assert on_cudnn == ResolvedCoefficients(values={"k": 7.0}, source="test:conv_backend=cudnn (values of test.lender)")
    on_xpu = resolve_coefficients(data, _spec("test.borrower"), Platform("xpu"), origin="test")
    assert on_xpu == ResolvedCoefficients(
        values={"k": 7.0}, source="test:conv_backend=xpu (cudnn values of test.lender)"
    )


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
        resolve_coefficients(_data(*entries), _spec("a"), Platform("cudnn"), origin="test")


@pytest.mark.parametrize("values, values_from", [({"k": 1.0}, "b"), (None, None)], ids=["both", "neither"])
def test_an_entry_carries_exactly_one_of_values_and_values_from(values, values_from) -> None:
    with pytest.raises(ValidationError, match="exactly one of"):
        _entry("a", values=values, values_from=values_from)


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
