"""Image quality and statistics helpers for the family E2E report."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.family_e2e.records import Record

MB = 2**20
DIFF_GAIN = 10
"""Amplification of the absolute difference image; a 1-level difference shows as 10 levels."""
MAD_FACTOR = 3.0


class ImageSizeMismatchError(ValueError):
    def __init__(self, a: np.ndarray, b: np.ndarray) -> None:
        super().__init__(f"size mismatch: {a.shape[1]}x{a.shape[0]} vs {b.shape[1]}x{b.shape[0]}")


def load_rgb(path: Path) -> np.ndarray:
    """Returns the image as an HxWx3 uint8 array; alpha is dropped."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape != b.shape:
        raise ImageSizeMismatchError(a, b)
    return np.subtract(a, b, dtype=np.int16)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """PSNR in dB over all channels of two 8-bit RGB images; identical images give `math.inf`."""
    mse = float(np.mean(np.square(_difference(a, b), dtype=np.int32)))
    if mse == 0:
        return math.inf
    return 10 * math.log10(255**2 / mse)


def write_diff_image(a: np.ndarray, b: np.ndarray, path: Path) -> None:
    """Writes |a - b| amplified by DIFF_GAIN as a PNG."""
    diff = np.abs(_difference(a, b)).astype(np.int32) * DIFF_GAIN
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8)).save(path)


def median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def mad(values: Sequence[float]) -> float | None:
    """Median absolute deviation from the median."""
    center = median(values)
    if center is None:
        return None
    return statistics.median([abs(v - center) for v in values])


@dataclass(frozen=True)
class MeasuredValues:
    """Per-run values of completed measured records; each metric only from records valid for it."""

    total_s: list[float] = field(default_factory=list)
    per_step_s: list[float] = field(default_factory=list)
    process_peak_bytes: list[int] = field(default_factory=list)
    node_peak_bytes: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    node_baseline_bytes: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    node_duration_s: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))


def measured_values(records: Iterable[Record]) -> MeasuredValues:
    values = MeasuredValues()
    for record in records:
        if record.kind != "measure" or record.status != "completed":
            continue
        if record.valid_timing and record.timing is not None:
            values.total_s.append(record.timing.total_s)
            if record.timing.per_step_s is not None:
                values.per_step_s.append(record.timing.per_step_s)
            for node in record.timing.nodes:
                if node.t_end is not None:
                    values.node_duration_s[node.node_id].append(node.t_end - node.t_start)
        if record.valid_memory and record.memory is not None:
            if record.memory.process_peak_bytes is not None:
                values.process_peak_bytes.append(record.memory.process_peak_bytes)
            for node in record.memory.nodes:
                if node.peak_bytes is not None:
                    values.node_peak_bytes[node.node_id].append(node.peak_bytes)
                if node.baseline_bytes is not None:
                    values.node_baseline_bytes[node.node_id].append(node.baseline_bytes)
    return values


@dataclass(frozen=True)
class Delta:
    """The current median against a comparison value (reference or previous run)."""

    current: float
    baseline: float
    outlier: bool

    @property
    def delta(self) -> float:
        return self.current - self.baseline

    @property
    def relative(self) -> float | None:
        return self.delta / self.baseline if self.baseline else None


def _compare(values: Sequence[float], baseline: float, tolerance: float) -> Delta | None:
    current = median(values)
    if current is None:
        return None
    deviation = abs(current - baseline)
    # A deviation within the run-to-run noise of this run's own measurements is not an outlier.
    noise = MAD_FACTOR * (mad(values) or 0.0)
    return Delta(current=current, baseline=baseline, outlier=deviation > tolerance and deviation > noise)


def compare_time(values: Sequence[float], baseline: float, tolerance: float) -> Delta | None:
    """`tolerance` is relative to the baseline (0.10 = 10 %). Returns None without current values."""
    return _compare(values, baseline, tolerance * abs(baseline))


def compare_peak(values: Sequence[float], baseline: float, tolerance_mb: float) -> Delta | None:
    """Values and baseline in bytes, tolerance absolute in MB (2**20 bytes). Returns None without current values."""
    return _compare(values, baseline, tolerance_mb * MB)
