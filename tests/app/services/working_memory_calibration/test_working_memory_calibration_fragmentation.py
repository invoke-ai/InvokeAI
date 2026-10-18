"""Which calibration windows count toward a user fit, on the windows a real SDXL calibration measured.

The numbers are an RX 9060 XT export's (16 GB, ROCm on Windows, `vae.sd` decode, one UI attempt of 18 windows), cut
to the fields the rules read. Its fp32 1216px windows ran on a fragmented allocator: their allocated bytes match to the
byte, while reserved stands 1.30 to 1.47 times allocated against 1.10 at 576px and 832px. Fitted on, the fp32 line
priced a 1024px decode at 8.3 GiB, 2.4 GiB over the shipped estimate.
"""

import datetime
import math
from fractions import Fraction
from typing import Optional

import pytest

from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    CalibrationDeviceClass,
    CalibrationEnvironment,
    CalibrationPoint,
    CalibrationRun,
    CalibrationStoreFile,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_fits import (
    exclusions,
    model_lines,
    user_fit_of,
)
from invokeai.backend.util.vae_working_memory import VAE_SD_SPEC
from invokeai.backend.working_memory import user_fit
from invokeai.backend.working_memory.coefficients import shipped_coefficients
from invokeai.backend.working_memory.fingerprint import DeviceClass, Environment
from invokeai.backend.working_memory.platform import Platform
from invokeai.backend.working_memory.probe import CounterDelta
from invokeai.backend.working_memory.spec import import_estimator_modules
from tests.fixtures.working_memory_calibration import measured_window

MIB = 2**20
SPECS = import_estimator_modules()
PLATFORM = Platform("miopen", "expandable", "windows")
CARD = DeviceClass(vendor="amd", name="AMD Radeon RX 9060 XT", total_mib=16304, backend="hip", arch="gfx1200")
ENV = Environment(hard={"os": "windows", "torch": "2.13", "hip": "7.15"}, soft={"driver": "test"})

# dtype, side, upcast_weight_bytes, score_matrix_bytes, fit_eligible, demand, reserved delta, allocated delta; in the
# order they ran. Not fit-eligible means partial_residency.
WINDOWS = [
    ("fp16", 1024, 0, 1073741824, True, 1992294400, 1992294400, 1610874880),
    ("fp16", 1024, 0, 1073741824, True, 1887436800, 1887436800, 1610874880),
    ("fp16", 1024, 0, 1073741824, True, 1887436800, 1887436800, 1610874880),
    ("fp16", 1472, 0, 1073741824, True, 4047503360, 4047503360, 3328721920),
    ("fp16", 1472, 0, 1073741824, True, 3902799872, 3902799872, 3328721920),
    ("fp16", 1472, 0, 1073741824, True, 3902799872, 3902799872, 3328721920),
    ("fp16", 2112, 0, 1073741824, False, 8011120640, 8011120640, 6852510720),
    ("fp16", 2112, 0, 1073741824, True, 8875147264, 8875147264, 6852510720),
    ("fp16", 2112, 0, 1073741824, False, 8870952960, 8870952960, 6852510720),
    ("fp32", 576, 167307726, 456855552, True, 1954545664, 1900019712, 1870138368),
    ("fp32", 576, 0, 456855552, True, 1887436800, 1887436800, 1702840320),
    ("fp32", 576, 0, 456855552, True, 1887436800, 1887436800, 1702840320),
    ("fp32", 832, 0, 1073741824, True, 3900702720, 3900702720, 3552839680),
    ("fp32", 832, 0, 1073741824, True, 3900702720, 3900702720, 3552839680),
    ("fp32", 832, 0, 1073741824, True, 3900702720, 3900702720, 3552839680),
    ("fp32", 1216, 0, 1073741824, False, 9835642880, 9835642880, 7589201920),
    ("fp32", 1216, 0, 1073741824, True, 11135877120, 11135877120, 7589201920),
    ("fp32", 1216, 0, 1073741824, False, 10842275840, 10842275840, 7589201920),
]


def _delta(value: int) -> CounterDelta:
    return CounterDelta(baseline=0, peak=value, offset=0, delta=value)


def _file(attempts: Optional[list[tuple[str, list[tuple]]]] = None) -> CalibrationStoreFile:
    """The windows as one run's, or per attempt, each a run of its own carrying that `attempt_id`, in that order."""
    env = CalibrationEnvironment.of(ENV)
    points = []
    runs = []
    for hour, (attempt, windows) in enumerate(attempts or [("run", WINDOWS)]):
        for item, (dtype, side, upcast, scores, eligible, demand, reserved, allocated) in enumerate(windows):
            window = measured_window(
                attempt, 141 + item, model_key="sdxl:vae", estimator="vae.sd", version=VAE_SD_SPEC.version
            )
            features = {
                "operation": "decode",
                "height": side,
                "width": side,
                "tile": None,
                "dtype": dtype,
                "vae_class": "AutoencoderKL",
                "upcast_weight_bytes": upcast,
                "score_matrix_bytes": scores,
            }
            update = {
                "features": features,
                "demand_bytes": demand,
                "reserved": _delta(reserved),
                "allocated": _delta(allocated),
                "fit_eligible": eligible,
                "reasons": [] if eligible else ["partial_residency"],
            }
            points.append(CalibrationPoint(**window.model_copy(update=update).model_dump(), env_id=env.id))
        runs.append(
            CalibrationRun(
                run_id=attempt,
                model_key="sdxl",
                model_hash=None,
                user_id="system",
                created_at=datetime.datetime(2026, 10, 18, hour, tzinfo=datetime.timezone.utc),
                env_id=env.id,
                attempt_id=attempt if attempts else None,
            )
        )
    return CalibrationStoreFile(
        schema_version=1, device_class=CalibrationDeviceClass.of(CARD), runs=runs, points=points, envs=[env]
    )


def _features(dtype: str, side: int) -> dict:
    return {
        "operation": "decode",
        "height": side,
        "width": side,
        "tile": None,
        "dtype": dtype,
        "vae_class": "AutoencoderKL",
        "upcast_weight_bytes": 0,
        "score_matrix_bytes": 1073741824,
    }


def test_the_fragmented_fp32_1216px_windows_drop_out_and_the_others_count() -> None:
    reasons = exclusions(_file(), SPECS)

    # Rows 15-17: the one fit-eligible window is a ratio outlier (1.47 against 1.10 at the smaller sizes); its two
    # partly resident neighbours never counted. Rows 6 and 8 are partly resident; the fp16 2112px window that ran
    # fully resident (1.30 against 1.24 and 1.22) counts. Row 9 was upcast and measures no line at all.
    assert reasons == {6: "partial_residency", 8: "partial_residency", 15: "partial_residency", 16: "fragmented",
                       17: "partial_residency"}  # fmt: skip


def test_without_them_the_fp32_line_rests_on_576px_and_832px_and_1024px_prices_as_shipped() -> None:
    fit, problems = user_fit_of(_file(), CalibrationEnvironment.of(ENV).id, PLATFORM, SPECS)
    assert fit is not None, problems
    [fp32] = [line for (_, suffix), line in fit.lines.items() if suffix == "decode_fp32"]

    # The two counted sizes, by hand: the chord through their readings beats the edge from the origin, plus 5%.
    (x0, y0), (x1, y1) = (576**2, 1887436800), (832**2, 3900702720)
    k = Fraction(y1 - y0, x1 - x0)
    b = y0 - k * x0
    assert (fp32.k, fp32.b) == (math.ceil(k * Fraction(21, 20)), math.ceil(b * Fraction(21, 20) / MIB) * MIB)
    assert (fp32.x_min, fp32.x_max) == (576**2, 832**2)

    features = _features("fp32", 1024)
    shipped = shipped_coefficients(VAE_SD_SPEC, PLATFORM)
    shipped_bytes = VAE_SD_SPEC.evaluate(features, shipped).reservation_bytes
    # Above the measured sizes the larger estimate wins, and here that is the shipped one (5.9 GiB, not 8.3 GiB).
    assert fp32.k * 1024**2 + fp32.b < shipped_bytes
    assert user_fit.resolve(VAE_SD_SPEC, features, shipped, fit).reservation_bytes == shipped_bytes


def test_the_summary_shows_the_sizes_that_do_not_count_and_why() -> None:
    file = _file()

    lines, excluded = model_lines(
        file,
        "sdxl",
        ENV,
        lambda env: PLATFORM,
        import_estimator_modules(),
        priced_on=[],
        name_of=lambda key: "SDXL",
    )

    by_line = {line.line: line for line in lines}
    assert [
        (size.width, size.windows, size.counted, size.excluded_reason) for size in by_line["decode_fp32"].sizes
    ] == [
        (576, 2, True, None),
        (832, 3, True, None),
        (1216, 3, False, "fragmented"),
    ]
    assert [(size.width, size.windows, size.counted) for size in by_line["decode_fp16"].sizes] == [
        (1024, 3, True),
        (1472, 3, True),
        (2112, 1, True),
    ]
    fragmented = by_line["decode_fp32"].sizes[-1]
    assert fragmented.measured_bytes == 11135877120 and fragmented.user_bytes is not None
    assert by_line["decode_fp32"].own_windows == 5
    # Not counted: the upcast window, the four partly resident ones and the fragmented one.
    assert excluded == 6


def test_a_measurement_whose_reserve_grew_around_a_flat_allocation_drops_out_without_a_smaller_size() -> None:
    """The fp32 1216px windows as if all three had run fully resident, with no smaller size to compare: reserved
    grew 1.13 times from the first to the largest while allocated did not move a byte."""
    file = _file()
    file.points = [point.model_copy(update={"fit_eligible": True, "reasons": []}) for point in file.points[15:18]]

    assert exclusions(file, SPECS) == {0: "fragmented", 1: "fragmented", 2: "fragmented"}


# The RX 9060 XT's fp32 windows of two later attempts (store rows 28-35 and 46-53): the first of them fragmented at
# 1152px (the one fully resident window reserves 1.43 times what it allocates, beside two partly resident ones); the
# next, run with every other model moved out first, clean (1.00 to 1.10).
FRAGMENTED_ATTEMPT = [
    ("fp32", 576, 0, 456855552, True, 1887436800, 1887436800, 1702840320),
    ("fp32", 576, 0, 456855552, True, 1887436800, 1887436800, 1702840320),
    ("fp32", 768, 0, 1073741824, True, 3334471680, 3334471680, 3027271680),
    ("fp32", 768, 0, 1073741824, True, 3334471680, 3334471680, 3027271680),
    ("fp32", 768, 0, 1073741824, True, 3334471680, 3334471680, 3027271680),
    ("fp32", 1152, 0, 1073741824, False, 8157921280, 8157921280, 6811361280),
    ("fp32", 1152, 0, 1073741824, True, 9751756800, 9751756800, 6811361280),
    ("fp32", 1152, 0, 1073741824, False, 8724152320, 8724152320, 6811361280),
]
CLEAN_ATTEMPT = [
    ("fp32", 576, 0, 456855552, True, 1866465280, 1866465280, 1702840320),
    ("fp32", 576, 0, 456855552, True, 1866465280, 1866465280, 1702840320),
    ("fp32", 768, 0, 1073741824, True, 3313500160, 3313500160, 3027271680),
    ("fp32", 768, 0, 1073741824, True, 3313500160, 3313500160, 3027271680),
    ("fp32", 768, 0, 1073741824, True, 3313500160, 3313500160, 3027271680),
    ("fp32", 1152, 0, 1073741824, True, 6815744000, 6815744000, 6811361280),
    ("fp32", 1152, 0, 1073741824, True, 7465861120, 7465861120, 6811361280),
    ("fp32", 1152, 0, 1073741824, True, 7465861120, 7465861120, 6811361280),
]


def _reasons_by_attempt(file: CalibrationStoreFile, reasons: dict) -> dict[str, list]:
    found: dict[str, list] = {}
    for index, point in enumerate(file.points):
        if point.features["width"] == 1152:
            found.setdefault(point.run_id, []).append(reasons.get(index))
    return found


@pytest.mark.parametrize("order", ["fragmented-then-clean", "clean-then-fragmented"])
def test_each_attempt_at_a_size_is_judged_on_its_own(order: str) -> None:
    """Pooled, the clean 1152px windows would share the fragmented one's fate: as its ratio outlier, or, after them,
    as a reserve that grew 1.43 times over the first of theirs at the same allocation. Per attempt, the clean one
    counts."""
    attempts = [("fragmented", FRAGMENTED_ATTEMPT), ("clean", CLEAN_ATTEMPT)]
    file = _file(attempts if order == "fragmented-then-clean" else attempts[::-1])

    assert _reasons_by_attempt(file, exclusions(file, SPECS)) == {
        "fragmented": ["partial_residency", "fragmented", "partial_residency"],
        "clean": [None, None, None],
    }
    lines, _ = model_lines(
        file, "sdxl", ENV, lambda env: PLATFORM, import_estimator_modules(), priced_on=[], name_of=lambda key: None
    )
    [size] = [size for size in lines[0].sizes if size.width == 1152]
    assert (size.counted, size.windows, size.measured_bytes) == (True, 3, 7465861120)


def test_a_running_attempts_windows_set_no_standard_for_finished_sizes() -> None:
    """The fragmented attempt's 1152px window, with no smaller size of its own finished: while the clean attempt still
    runs, its 576px and 768px windows are no reference, so nothing finished flags it yet."""
    fragmented_1152 = [window for window in FRAGMENTED_ATTEMPT if window[1] == 1152]
    file = _file([("finished", fragmented_1152), ("running", CLEAN_ATTEMPT)])

    assert "fragmented" in exclusions(file, SPECS).values()
    assert "fragmented" not in exclusions(file, SPECS, live_runs={"running"}).values()
