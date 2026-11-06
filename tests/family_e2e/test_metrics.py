import math
from pathlib import Path

import numpy as np
import pytest

from scripts.family_e2e.metrics import (
    ImageSizeMismatchError,
    compare_peak,
    compare_time,
    load_rgb,
    mad,
    measured_values,
    median,
    psnr,
    write_diff_image,
)
from tests.family_e2e.test_records import MB, make_record, write_png


def _image(value: int, width: int = 8, height: int = 8) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_identical_images_have_infinite_psnr() -> None:
    assert psnr(_image(100), _image(100)) == math.inf


def test_off_by_one_in_every_channel_gives_20_log10_255() -> None:
    assert psnr(_image(100), _image(101)) == pytest.approx(48.1308, abs=1e-4)


def test_single_saturated_channel_gives_psnr_from_its_share_of_the_error() -> None:
    a = _image(0, width=4, height=4)
    b = a.copy()
    b[2, 1, 0] = 255
    # MSE = 255² / 48 values, so PSNR = 10·log10(48).
    assert psnr(a, b) == pytest.approx(16.8124, abs=1e-4)


def test_size_mismatch_is_an_explicit_error() -> None:
    with pytest.raises(ImageSizeMismatchError, match="size mismatch: 8x8 vs 8x4"):
        psnr(_image(0), _image(0, width=8, height=4))


def test_alpha_is_ignored_when_loading(tmp_path: Path) -> None:
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., :3] = 77
    rgba[..., 3] = 10
    write_png(tmp_path / "rgba.png", rgba)
    write_png(tmp_path / "rgb.png", _image(77, 4, 4))

    assert psnr(load_rgb(tmp_path / "rgba.png"), load_rgb(tmp_path / "rgb.png")) == math.inf


def test_diff_image_amplifies_and_saturates(tmp_path: Path) -> None:
    a = _image(100, 2, 1)
    b = a.copy()
    b[0, 0] = (101, 100, 70)

    write_diff_image(a, b, tmp_path / "diff" / "case.png")

    assert load_rgb(tmp_path / "diff" / "case.png").tolist() == [[[10, 0, 255], [0, 0, 0]]]


def test_median_and_mad_resist_a_single_extreme_value() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 100.0]
    assert median(values) == 3.0
    assert mad(values) == 1.0
    assert median([]) is None and mad([]) is None


@pytest.mark.parametrize(
    ("values", "tolerance", "outlier"),
    [
        ([11.0, 11.0, 11.0], 0.05, True),  # MAD 0: the 10 % deviation exceeds 5 %
        ([11.0, 11.0, 11.0], 0.15, False),  # within tolerance
        ([9.0, 12.0, 15.0], 0.05, False),  # +20 %, but within 3×MAD = 9 s of noise
        ([20.0, 21.0, 22.0], 0.05, True),  # +110 %, beyond 3×MAD = 3 s
        ([5.0, 5.0, 5.0], 0.10, True),  # a sudden speed-up is flagged too
    ],
)
def test_time_outlier_needs_tolerance_and_noise_exceeded(values: list[float], tolerance: float, outlier: bool) -> None:
    delta = compare_time(values, 10.0, tolerance)

    assert delta is not None
    assert delta.outlier is outlier
    assert delta.delta == median(values) - 10.0


@pytest.mark.parametrize(("baseline_mb", "outlier"), [(900, False), (600, True), (1300, True)])
def test_peak_outlier_uses_an_absolute_tolerance(baseline_mb: int, outlier: bool) -> None:
    delta = compare_peak([1000.0 * MB] * 3, baseline_mb * MB, tolerance_mb=256)

    assert delta is not None and delta.outlier is outlier


def test_comparison_without_current_values_is_none() -> None:
    assert compare_time([], 10.0, 0.1) is None
    assert compare_peak([], 10.0, 256) is None


def test_measured_values_use_each_metric_only_from_valid_completed_measurements() -> None:
    records = [
        make_record(rep=0, total_s=10.0, peak_bytes=100),
        make_record(rep=1, total_s=11.0, peak_bytes=200, valid_memory=False),
        make_record(rep=2, total_s=50.0, peak_bytes=300, valid_timing=False),
        make_record("zimage-t2i@8x8", "warmup", total_s=90.0, peak_bytes=900),
        make_record(rep=3, attempt=1, status="failed", total_s=70.0, peak_bytes=700),
    ]

    values = measured_values(records)

    assert values.total_s == [10.0, 11.0]
    assert values.process_peak_bytes == [100, 300]
    assert dict(values.node_peak_bytes) == {"denoise": [100, 300]}
