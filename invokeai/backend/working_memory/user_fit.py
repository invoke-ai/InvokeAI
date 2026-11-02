"""A user's own calibration, fitted per line and read by estimates in place of the shipped values where it measured.

The app keeps a device class's calibration windows (`app/services/working_memory_calibration/`); this module owns the
rule that turns them into lines and the rule by which an estimate uses them. Lines are those of `fit.py`: a window
joins the line its estimator resolves to on the platform it ran on -- the lender's, where the estimator borrows its
values -- so a calibration of one model's decode moves every estimator that prices with the same line. Only the
estimators that opt in (`EstimatorSpec.user_calibrated`) take part: the others' windows fit no user line, and no user
line moves their estimates. Only the lines whose demand is their own take part (`fit.Line.own_demand`): a line that
sits on another, a tiled VAE operation's, is fitted on what its points measured above their untiled demand at the
tile, a tiled decode's together with its assembly line: its windows fit no user line. Its estimate prices its tile on
the larger of the user and the shipped untiled line (`resolve`), and what tiling adds stays shipped.

An estimate reads a user line through the provider seam: the app registers one (`set_user_fit_provider`) that answers
from memory, and `EstimatorSpec.estimate` asks it per estimate. Without a provider, or when it has no fit for the
device, the estimate is the shipped one, bit for bit. With one, `resolve` applies the user line where it was measured:

- at or below the largest size the line measured (`UserLine.x_max`), the user line replaces the shipped one;
- above it, the larger of the user and the shipped estimate, since nothing measured says how the line continues.
"""

import functools
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Optional

import torch

from invokeai.backend.util.logging import InvokeAILogger
from invokeai.backend.working_memory.coefficients import ResolvedCoefficients, find_entry, shipped_data
from invokeai.backend.working_memory.estimate import FeatureValue, WorkingMemoryEstimate
from invokeai.backend.working_memory.fit import EntryKey, LineKey, carried_lines, entry_key, user_pair
from invokeai.backend.working_memory.platform import Platform, priced_as

if TYPE_CHECKING:
    from invokeai.backend.working_memory.spec import EstimatorSpec


@dataclass(frozen=True)
class UserPoint:
    """One calibration window as a fit reads it."""

    estimator: str
    version: int
    features: Mapping[str, FeatureValue]
    demand_bytes: int


@dataclass(frozen=True)
class UserLine:
    """A line fitted on a user's windows (`fit.user_pair`), and the sizes it was fitted on."""

    k: int
    """Bytes per unit of the line's size (`fit.Line.size`)."""
    b: int
    """Bytes whatever the size."""
    x_min: int
    """The smallest size measured."""
    x_max: int
    """The largest size measured."""
    points: int


@dataclass(frozen=True)
class UserFit:
    """A device class's user lines in one environment, on the platform they were measured on."""

    label: str
    """Names the fit in an estimate's `source`, e.g. ``user:<device class id>:<environment id>``."""
    platform: Platform
    lines: Mapping[LineKey, UserLine]
    _by_spec: dict[tuple[str, int], Mapping[str, UserLine]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def lines_for(self, spec: "EstimatorSpec") -> Mapping[str, UserLine]:
        """The user lines `spec` prices with on this fit's platform, by suffix. Resolved once per spec."""
        key = (spec.id, spec.version)
        found = self._by_spec.get(key)
        if found is None:
            carrier = carrier_key(spec.id, spec.version, self.platform)
            found = MappingProxyType({suffix: line for (entry, suffix), line in self.lines.items() if entry == carrier})
            self._by_spec[key] = found
        return found


@functools.cache
def carrier_key(estimator: str, version: int, platform: Platform) -> Optional[EntryKey]:
    """The shipped entry that carries the lines a user calibration of `estimator` `version` fits on `platform`: its own,
    or the one it borrows from. An entry's own values beside borrowed ones are lines that sit on another and their
    assembly lines, which no user calibration fits. None where the shipped data has no entry for it."""
    full = {key: str(value) for key, value in asdict(priced_as(platform)).items()}
    entries = [entry for file in shipped_data().values() for entry in file.coefficients]
    where = f"{estimator} on {full}"
    try:
        entry = find_entry(entries, estimator, version, full, where)
        if entry.values_from is not None:
            entry = find_entry(entries, entry.values_from.estimator, entry.values_from.version, full, where)
    except LookupError:
        return None
    return entry_key(entry)


def _carried_lines(key: EntryKey) -> frozenset[str]:
    entries = [entry for file in shipped_data().values() for entry in file.coefficients]
    for entry in entries:
        if entry_key(entry) == key and entry.values is not None:
            return frozenset(carried_lines(entry))
    return frozenset()


def fit_user_lines(
    points: Iterable[UserPoint], platform: Platform, specs: Mapping[str, "EstimatorSpec"]
) -> tuple[Mapping[LineKey, UserLine], Mapping[LineKey, str]]:
    """The user lines `points`, measured on `platform`, fit; and, per line they measure but cannot fit, why.

    A point joins a line where its estimator, as `specs` registers it, is `user_calibrated` and its `line` admits the
    point's features as its own demand (`fit.Line.own_demand`), its estimator version has shipped values on the
    platform, the entry carrying them has the line, and it demanded more than nothing. Each line is `fit.user_pair` of
    its points: two sizes at least `fit.MIN_AREA_RATIO` apart, or no line.
    """
    groups: dict[LineKey, list[tuple[int, int]]] = {}
    for point in points:
        spec = specs.get(point.estimator)
        if spec is None or not spec.user_calibrated or point.demand_bytes <= 0:
            continue
        found = spec.line(point.features)
        if found is None or not found.own_demand:
            continue
        carrier = carrier_key(point.estimator, point.version, platform)
        if carrier is None or found.suffix not in _carried_lines(carrier):
            continue
        groups.setdefault((carrier, found.suffix), []).append((found.size, point.demand_bytes))

    lines: dict[LineKey, UserLine] = {}
    problems: dict[LineKey, str] = {}
    for key, measured in groups.items():
        try:
            k, b = user_pair(measured)
        except ValueError as e:
            problems[key] = str(e)
            continue
        sizes = [size for size, _ in measured]
        lines[key] = UserLine(k=k, b=b, x_min=min(sizes), x_max=max(sizes), points=len(measured))
    return MappingProxyType(lines), MappingProxyType(problems)


def resolve(
    spec: "EstimatorSpec",
    features: Mapping[str, FeatureValue],
    shipped: ResolvedCoefficients,
    fit: UserFit,
) -> WorkingMemoryEstimate:
    """The estimate for `features` with `fit`'s line in place of the shipped one, where `fit` has the line and `spec`
    is `user_calibrated`.

    The line and its size are read as `fit.py` reads a point's (`EstimatorSpec.line`): for the convolution lines the
    output's pixels. At or below `UserLine.x_max` the user line decides; above, the larger estimate wins, the shipped
    one on a tie.

    A tiled operation prices its tile on the untiled line beneath its tiled line (`fit.Line.sits_on`), and takes the
    larger estimate at every size, the rule above `x_max`: its tiled and assembly lines, which no user line measures
    (`fit_user_lines`), are fitted over the untiled demand measured where the shipped line was, which a lower user line
    beneath them could undercut, while a card whose untiled decode needs more than the shipped line needs it for each
    tile too. `source` names the user fit and the line wherever the user line decides.
    """
    found = spec.line(features) if spec.user_calibrated else None
    if found is None:
        return spec.evaluate(features, shipped)
    tiled = found.sits_on is not None
    suffix, size = found.sits_on if found.sits_on is not None else (found.suffix, found.size)
    line = fit.lines_for(spec).get(suffix)
    if line is None:
        return spec.evaluate(features, shipped)
    values = MappingProxyType({**shipped.values, f"k_{suffix}": line.k, f"b_{suffix}": line.b})
    within = not tiled and size <= line.x_max
    if tiled:
        where = "at the tile, over shipped"
    else:
        where = "within its measured sizes" if within else "above its measured sizes, over shipped"
    user = spec.evaluate(
        features, ResolvedCoefficients(values=values, source=f"{fit.label} ({suffix} {where}) + {shipped.source}")
    )
    if within:
        return user
    shipped_estimate = spec.evaluate(features, shipped)
    if user.reservation_bytes > shipped_estimate.reservation_bytes:
        return user
    over = " at the tile" if tiled else ", above its measured sizes"
    return replace(shipped_estimate, source=f"{shipped.source} (over {fit.label} {suffix}{over})")


UserFitProvider = Callable[["EstimatorSpec", Platform, torch.device], Optional[UserFit]]
"""Answers, from memory, the user fit an estimate of `spec` on `device` reads, or None. Called per estimate, so it
reads no file and touches no device."""

_provider: Optional[UserFitProvider] = None
_logged_failure: bool = False

_LOGGER: Final = InvokeAILogger.get_logger(__name__)


def set_user_fit_provider(provider: UserFitProvider) -> None:
    """Make `provider` the one estimates ask. The app's calibration service registers it when it starts."""
    global _provider
    _provider = provider


def clear_user_fit_provider(provider: UserFitProvider) -> None:
    """Stop asking `provider`, if it is the registered one: a service that stops must not unregister its successor."""
    global _provider
    if _provider is provider:
        _provider = None


def user_fit_for(spec: "EstimatorSpec", platform: Platform, device: torch.device) -> Optional[UserFit]:
    """The registered provider's answer, None without one. A provider that raises leaves the estimate shipped: an
    estimate must not fail over a calibration."""
    global _logged_failure
    provider = _provider
    if provider is None:
        return None
    try:
        return provider(spec, platform, device)
    except Exception:
        if not _logged_failure:
            _logged_failure = True
            _LOGGER.warning(
                "The working-memory user fit could not be read; estimates use the shipped values.", exc_info=True
            )
        return None
