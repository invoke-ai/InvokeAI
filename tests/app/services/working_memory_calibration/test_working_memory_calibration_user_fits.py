"""User fits from the calibration store: what estimates read after a session start, and the per-model summary."""

import asyncio
import builtins
import datetime
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.math import AddInvocation
from invokeai.app.invocations.model import LoRACollectionLoader, LoRAField, ModelIdentifierField
from invokeai.app.invocations.sd.sdxl import SDXLModelLoaderInvocation
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_records import UnknownModelException
from invokeai.app.services.session_queue.session_queue_common import Batch, BatchDatum
from invokeai.app.services.session_queue.session_queue_default import SessionQueue
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.errors import DatabaseUnavailableError
from invokeai.app.services.shared.database.queries.session_queue import SessionQueueQueries
from invokeai.app.services.shared.graph import Graph
from invokeai.app.services.working_memory_calibration import (
    working_memory_calibration_default as calibration_default,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    CalibrationAttemptBusyError,
    CalibrationAttemptConflictError,
    CalibrationAttemptInterleavedError,
    CalibrationModelChangedError,
    CalibrationStoreError,
    CalibrationStoreLockTimeoutError,
    WorkingMemoryCalibrationRequest,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_default import (
    _ATTEMPTS_LOCK,
    ATTEMPT_IDLE_GRACE,
    DeviceMemory,
    SettingValue,
    WorkingMemoryCalibrationService,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_store import CalibrationStore
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.util.vae_working_memory import VAE_SD_SPEC
from invokeai.backend.working_memory.coefficients import shipped_coefficients
from invokeai.backend.working_memory.fingerprint import DeviceClass, Environment
from invokeai.backend.working_memory.platform import platform_for
from invokeai.backend.working_memory.probe import CounterDelta, MeasuredWindow
from tests.fixtures.working_memory_calibration import (
    GPU,
    HeldEnqueue,
    calibration_service,
    fake_environment,
    measured_window,
)

MIB = 2**20
PLATFORM = platform_for(GPU)


def _features(side: int, **extra: object) -> dict:
    return {
        "operation": "decode",
        "height": side,
        "width": side,
        "tile": None,
        "dtype": "fp16",
        "vae_class": "AutoencoderKL",
        "upcast_weight_bytes": 0,
        "score_matrix_bytes": 0,
        **extra,
    }


def _window(run_id: str, side: int, demand: int, **update: object) -> MeasuredWindow:
    window = measured_window(run_id, model_key="model-a:vae", estimator="vae.sd", version=VAE_SD_SPEC.version)
    return window.model_copy(update={"features": _features(side), "demand_bytes": demand, **update})


def _on_the_line(run_id: str, sides: tuple[int, ...] = (512, 768, 1024)) -> list[MeasuredWindow]:
    """Windows on 900 B/px + 400 MiB, which the user line prices at 945 B/px + 420 MiB."""
    return [_window(run_id, side, 900 * side**2 + 400 * MIB) for side in sides]


def _user_bytes(side: int) -> int:
    return 945 * side**2 + 420 * MIB


def _estimate(side: int = 1024) -> int:
    return VAE_SD_SPEC.estimate(_features(side), GPU).reservation_bytes


def _shipped(side: int = 1024) -> int:
    return VAE_SD_SPEC.evaluate(_features(side), shipped_coefficients(VAE_SD_SPEC, PLATFORM)).reservation_bytes


MODELS = {
    key: SimpleNamespace(
        key=key, name=key.upper(), hash=None, base=BaseModelType.StableDiffusionXL, type=ModelType.Main
    )
    for key in ("model-a", "model-b", "model-c")
}
"""Installed SDXL main models: their decode prices on vae.sd's fp16 and fp32 lines."""


def _get_model(key: str) -> SimpleNamespace:
    if key not in MODELS:
        raise UnknownModelException(key)
    return MODELS[key]


@pytest.fixture
def queue(mock_invoker: Invoker, mock_sqlite_database: Database) -> SessionQueue:
    queue = SessionQueue(mock_sqlite_database)
    mock_invoker.services.session_queue = queue
    model_manager = MagicMock()
    model_manager.store.get_model.side_effect = _get_model
    model_manager.store.all_models.return_value = list(MODELS.values())
    mock_invoker.services.model_manager = model_manager
    queue.start(mock_invoker)
    return queue


@pytest.fixture
def started(mock_invoker: Invoker, queue: SessionQueue) -> Iterator:
    """Start services on a root; each is stopped after the test, which unregisters its user-fit provider."""
    services: list[WorkingMemoryCalibrationService] = []

    def start(root: Path, **kwargs) -> WorkingMemoryCalibrationService:
        service = calibration_service(root, **kwargs)
        service.start(mock_invoker)
        # The queue's retries reach the latest one, as they reach the server's.
        mock_invoker.services.working_memory_calibration = service
        services.append(service)
        return service

    yield start
    for service in services:
        service.stop(mock_invoker)


def _create(
    service: WorkingMemoryCalibrationService,
    model_key: str = "model-a",
    *,
    running: bool = False,
    attempt_id: str | None = None,
    attempt_steps: int | None = None,
) -> str:
    """A run of `model_key`; finished (its item completed) unless `running`, so the fits take its windows."""
    graph = Graph()
    graph.add_node(AddInvocation(id="add", a=1, b=2))
    request = WorkingMemoryCalibrationRequest(graph=graph, attempt_id=attempt_id, attempt_steps=attempt_steps)
    created = asyncio.run(service.create_run(model_key, request, "system"))
    if not running:
        _finish(service, created.item_ids)
    return created.run_id


def _finish(service: WorkingMemoryCalibrationService, item_ids: list[int]) -> None:
    for item_id in item_ids:
        service._services.session_queue.complete_queue_item(item_id)


def _environment(hard: Mapping[str, str] | None = None, soft: Mapping[str, str] | None = None):
    def environment_for(device, settings: Mapping[str, SettingValue]) -> Environment:
        env = fake_environment(device, settings)
        return Environment(hard={**env.hard, **(hard or {})}, soft={**env.soft, **(soft or {})})

    return environment_for


def test_estimates_read_a_calibration_from_the_next_session_start_on_and_drop_it_on_reset(
    tmp_path: Path, started
) -> None:
    service = started(tmp_path)
    run_id = _create(service)
    service.record_windows(run_id, _on_the_line(run_id))

    assert _estimate() == _shipped()
    service.refresh_user_fits()
    assert _estimate() == _user_bytes(1024) and _estimate(256) == _user_bytes(256)

    # A reset takes effect at once, not at the next session start.
    service.reset_model("model-a")
    assert _estimate() == _shipped()


def test_a_plan_is_priced_with_the_store_as_it_is_now(tmp_path: Path, started) -> None:
    def reservation(estimator: str, height: int, width: int, dtype: str, device) -> int:
        return VAE_SD_SPEC.estimate(_features(height, dtype=dtype), device).reservation_bytes

    def largest_fp16(service: WorkingMemoryCalibrationService) -> int:
        return max(case.width for case in service.get_plan("model-c").cases if case.line == "decode_fp16")

    service = started(tmp_path, untiled_decode_reservation=reservation)
    shipped = largest_fp16(service)
    run_id = _create(service)
    # Steeper than the shipped line, so the largest size that decodes untiled shrinks.
    service.record_windows(run_id, [_window(run_id, side, 8000 * side**2) for side in (512, 1024)])

    assert largest_fp16(service) < shipped


CARD_16GB = DeviceClass(vendor="amd", name="AMD Radeon RX 9060 XT", total_mib=16304, backend="hip", arch="gfx1200")
# Driver and torch-reserved baselines of real windows on that card (rows 0, 9, 15 and 16 of one 18-window UI attempt):
# what the process held outside torch grew from 722 MiB at the first decode to 2628 MiB at the last sizes.
RX9060XT_BASELINES = [
    (7777943552, 7021264896),
    (7608401920, 7063207936),
    (4718526464, 1962934272),
    (4718526464, 1969225728),
]


def _baseline(value: int) -> CounterDelta:
    return CounterDelta(baseline=value, peak=value, offset=0, delta=0)


def test_a_fresh_servers_plan_sizes_for_the_outside_use_its_stored_windows_recorded(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh server reads 300 MiB outside torch where the stored windows recorded 2755592192 bytes, so the cache can
    give 16304 MiB less those: 14340390912 bytes, below 0.9 of the card (15386384793). A calibration lock moves every
    other model out first, so only the VAE stays beside the decode: less 1 GiB for it, a reservation (1.5x the
    estimate) may take 13266649088. fp16 at 2100 B/px: 2048px needs 13212057600, 2112px 14050713600. fp32 at 6000
    B/px: 1152px needs 11943936000, 1216px 13307904000. The reading alone would have allowed 2112px and 1216px."""
    per_pixel = {"fp16": 2100, "fp32": 6000}
    idle = DeviceMemory(capacity_bytes=(16304 - 300) * MIB, outside_torch_bytes=300 * MIB)
    service = started(
        tmp_path,
        device_class=CARD_16GB,
        untiled_decode_reservation=lambda estimator, height, width, dtype, device: per_pixel[dtype] * height * width,
        device_memory=lambda device: idle,
    )
    run_id = _create(service)
    windows = [
        measured_window(run_id).model_copy(update={"driver": _baseline(driver), "reserved": _baseline(reserved)})
        for driver, reserved in RX9060XT_BASELINES
    ]
    # An invalid window says nothing about what a decode finds.
    invalid = {"driver": _baseline(16 * 2**30), "valid": False, "fit_eligible": False, "reasons": ["oom"]}
    windows.append(windows[2].model_copy(update=invalid))
    service.record_windows(run_id, windows)
    reads: list[str] = []
    real_read = CalibrationStore.read
    monkeypatch.setattr(
        CalibrationStore, "read", lambda store, class_id: reads.append(class_id) or real_read(store, class_id)
    )

    plan = service.get_plan("model-c")

    assert [(case.line, case.width) for case in plan.cases] == [
        ("decode_fp16", 1024),
        ("decode_fp16", 1408),
        ("decode_fp16", 2048),
        ("decode_fp32", 576),
        ("decode_fp32", 768),
        ("decode_fp32", 1152),
    ]
    # The plan sizes from the store its refresh read, and reads it no second time.
    assert reads == [CARD_16GB.id]


def test_a_first_calibration_sizes_for_the_reading_alone(tmp_path: Path, started) -> None:
    """No stored window yet: 300 MiB outside, so 16004 MiB, above 0.9 of the card; the pretile share decides, as in
    the sizes the RX 9060 XT ran partly resident."""
    per_pixel = {"fp16": 2100, "fp32": 6000}
    idle = DeviceMemory(capacity_bytes=(16304 - 300) * MIB, outside_torch_bytes=300 * MIB)
    service = started(
        tmp_path,
        device_class=CARD_16GB,
        untiled_decode_reservation=lambda estimator, height, width, dtype, device: per_pixel[dtype] * height * width,
        device_memory=lambda device: idle,
    )

    plan = service.get_plan("model-c")

    assert [case.width for case in plan.cases] == [1024, 1472, 2112, 576, 832, 1216]


def test_an_unchanged_store_is_not_read_again_and_estimates_touch_no_file(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = started(tmp_path)
    run_id = _create(service)
    service.record_windows(run_id, _on_the_line(run_id))
    reads: list[str] = []
    real_read = CalibrationStore.read
    monkeypatch.setattr(
        CalibrationStore, "read", lambda store, class_id: reads.append(class_id) or real_read(store, class_id)
    )

    service.refresh_user_fits()
    service.refresh_user_fits()
    assert len(reads) == 1

    def no_io(*args, **kwargs):
        raise AssertionError("file access on the estimate path")

    monkeypatch.setattr(os, "stat", no_io)
    monkeypatch.setattr(builtins, "open", no_io)
    monkeypatch.setattr(Path, "stat", no_io)
    monkeypatch.setattr(CalibrationStore, "read", no_io)
    assert [_estimate(side) for side in (512, 1024)] == [_user_bytes(512), _user_bytes(1024)]


def test_a_hard_environment_change_falls_back_to_shipped_and_a_soft_one_keeps_the_fit(
    tmp_path: Path, started, mock_invoker: Invoker
) -> None:
    service = started(tmp_path)
    run_id = _create(service)
    service.record_windows(run_id, _on_the_line(run_id))
    service.stop(mock_invoker)

    soft = started(tmp_path, environment_for=_environment(soft={"driver": "2.0"}))
    soft.refresh_user_fits()
    assert _estimate() == _user_bytes(1024)
    [line] = soft.get_model_summary("model-a").lines
    assert (line.state, line.changed_facts) == ("stale", ["driver"])
    soft.stop(mock_invoker)

    hard = started(tmp_path, environment_for=_environment(hard={"torch": "2.8"}))
    hard.refresh_user_fits()
    assert _estimate() == _shipped()
    [line] = hard.get_model_summary("model-a").lines
    assert (line.state, line.changed_facts) == ("unusable", ["torch"])


def test_the_summary_shows_per_line_what_was_measured_and_what_estimates_make_of_it(tmp_path: Path, started) -> None:
    service = started(tmp_path)
    run_id = _create(service)
    windows = _on_the_line(run_id) + [
        _window(run_id, 1024, 900 * 1024**2 + 300 * MIB),
        _window(run_id, 1024, 9 * 10**9, valid=False, fit_eligible=False, reasons=["oom"]),
        _window(run_id, 2048, 9 * 10**9, features=_features(2048, tile=512)),
    ]
    service.record_windows(run_id, windows)
    other = _create(service, "model-b")
    service.record_windows(other, [_window(other, 256, 900 * 256**2 + 400 * MIB, model_key="model-b:vae")])

    summary = service.get_model_summary("model-a")

    assert summary.excluded_windows == 2
    assert [(run.run_id, run.status, run.point_count) for run in summary.runs] == [(run_id, "completed", 6)]
    [line] = summary.lines
    assert (line.estimator, line.line, line.carrier, line.state) == ("vae.sd", "decode_fp16", "vae.sd3", "in_use")
    # The line is fitted on every model's windows of it, and shows them: model-b's 256px window widens the range.
    assert (line.measured_range.min_pixels, line.measured_range.max_pixels) == (256**2, 1024**2)  # type: ignore[union-attr]
    assert [(size.width, size.windows, size.measured_bytes) for size in line.sizes] == [
        (256, 1, 900 * 256**2 + 400 * MIB),
        (512, 1, 900 * 512**2 + 400 * MIB),
        (768, 1, 900 * 768**2 + 400 * MIB),
        (1024, 2, 900 * 1024**2 + 400 * MIB),
    ]
    assert [size.shipped_bytes for size in line.sizes] == [_shipped(side) for side in (256, 512, 768, 1024)]
    assert [size.user_bytes for size in line.sizes] == [_user_bytes(side) for side in (256, 512, 768, 1024)]
    assert line.own_windows == 4
    assert _measured_by(line) == [("model-a", "MODEL-A", 4), ("model-b", "MODEL-B", 1)]


def _measured_by(line) -> list[tuple[str, str | None, int]]:
    return [(entry.model_key, entry.name, entry.windows) for entry in line.measured_by]


def test_a_model_never_calibrated_sees_the_line_its_decode_shares_with_another_models(tmp_path: Path, started) -> None:
    service = started(tmp_path)
    run_id = _create(service, "model-a")
    service.record_windows(run_id, _on_the_line(run_id))

    summary = service.get_model_summary("model-b")

    assert summary.runs == [] and summary.excluded_windows == 0
    # Only the fp16 line has windows; model-b's fp32 decode line has none and is not shown.
    [line] = summary.lines
    assert (line.line, line.state, line.own_windows) == ("decode_fp16", "in_use", 0)
    assert _measured_by(line) == [("model-a", "MODEL-A", 3)]
    assert [size.user_bytes for size in line.sizes] == [_user_bytes(side) for side in (512, 768, 1024)]


def test_after_a_reset_a_shared_line_stays_carried_by_the_other_models_windows(tmp_path: Path, started) -> None:
    service = started(tmp_path)
    run_a = _create(service, "model-a")
    service.record_windows(run_a, _on_the_line(run_a))
    run_b = _create(service, "model-b")
    service.record_windows(
        run_b, [_window(run_b, side, 900 * side**2 + 400 * MIB, model_key="model-b:vae") for side in (512, 1024)]
    )

    service.reset_model("model-a")

    [line] = service.get_model_summary("model-a").lines
    assert (line.state, line.own_windows) == ("in_use", 0)
    assert _measured_by(line) == [("model-b", "MODEL-B", 2)]
    assert _estimate() == _user_bytes(1024)


def test_resetting_a_model_removes_every_window_the_summary_attributes_to_it(tmp_path: Path, started) -> None:
    """Model A's VAE measured inside model B's run counts for A, as a submodel key (`model-a:vae`)."""
    service = started(tmp_path)
    run_b = _create(service, "model-b")
    service.record_windows(run_b, _on_the_line(run_b))
    [line] = service.get_model_summary("model-a").lines
    assert line.own_windows == 3

    removed = service.reset_model("model-a")

    assert (removed.runs_removed, removed.points_removed) == (0, 3)
    assert service.get_model_summary("model-a").lines == []


def test_recalibrating_under_new_soft_facts_clears_stale(tmp_path: Path, started, mock_invoker: Invoker) -> None:
    service = started(tmp_path)
    run_id = _create(service)
    service.record_windows(run_id, _on_the_line(run_id))
    service.stop(mock_invoker)

    updated = started(tmp_path, environment_for=_environment(soft={"driver": "2.0"}))
    [line] = updated.get_model_summary("model-a").lines
    assert (line.state, line.changed_facts) == ("stale", ["driver"])

    rerun = _create(updated)
    updated.record_windows(rerun, _on_the_line(rerun, sides=(512, 1024)))

    [line] = updated.get_model_summary("model-a").lines
    assert (line.state, line.changed_facts) == ("in_use", [])
    # The older windows of the same hard environment still feed the line.
    assert line.own_windows == 5


def test_a_run_still_running_feeds_no_fit_until_the_queue_finishes_it(tmp_path: Path, started) -> None:
    """Whoever refreshes meanwhile (a plan, another item, a reset): a running attempt's windows move no estimate.
    The summary shows what the run measured so far, and the user estimates estimates really read."""
    service = started(tmp_path)
    graph = Graph()
    graph.add_node(AddInvocation(id="add", a=1, b=2))
    created = asyncio.run(service.create_run("model-a", WorkingMemoryCalibrationRequest(graph=graph), "system"))
    service.record_windows(created.run_id, _on_the_line(created.run_id))

    service.refresh_user_fits()
    assert _estimate() == _shipped()
    [line] = service.get_model_summary("model-a").lines
    assert (line.state, line.own_windows, [size.windows for size in line.sizes]) == ("insufficient", 3, [1, 1, 1])
    assert line.measured_range is None and {size.user_bytes for size in line.sizes} == {None}
    assert "still measuring" in (line.problem or "")

    # Finished in the queue only: the file is unchanged, and the next refresh still notices.
    _finish(service, created.item_ids)
    service.refresh_user_fits()
    assert _estimate() == _user_bytes(1024)
    [line] = service.get_model_summary("model-a").lines
    assert line.state == "in_use"


def test_a_failed_run_feeds_no_fit_while_its_retry_waits_in_the_queue(tmp_path: Path, started) -> None:
    """A retry is a new item of the run's origin, which the run's record does not name until the runner admits it:
    queued behind other work, it must hold the failed run's windows back from the moment it is queued, and release
    them once it is canceled."""
    service = started(tmp_path)
    created = asyncio.run(service.create_run("model-a", WorkingMemoryCalibrationRequest(graph=_add_graph()), "system"))
    service.record_windows(created.run_id, _on_the_line(created.run_id))
    queue = service._services.session_queue
    [item_id] = created.item_ids
    queue.fail_queue_item(item_id, "OutOfMemoryError", "out of memory", "")
    # Failed with valid windows: the partial run feeds the fit, as every finished run does.
    service.refresh_user_fits()
    assert _estimate() == _user_bytes(1024)

    [retried] = queue.retry_items_by_id("default", [item_id]).retried_items
    service.refresh_user_fits()

    assert _estimate() == _shipped()
    [line] = service.get_model_summary("model-a").lines
    assert line.state == "insufficient" and "still measuring" in (line.problem or "")

    queue.cancel_queue_item(retried.item_id)
    service.refresh_user_fits()
    assert _estimate() == _user_bytes(1024)


def test_an_attempt_feeds_no_fit_until_every_run_of_it_finished_and_another_finished_attempt_does(
    tmp_path: Path, started, mock_invoker: Invoker
) -> None:
    service = started(tmp_path)
    finished_attempt = _create(service, attempt_id="earlier", attempt_steps=1)
    service.record_windows(finished_attempt, _on_the_line(finished_attempt))
    # The running attempt measures a steeper line: were its finished first step used, estimates would move.
    first = _create(service, attempt_id="current", attempt_steps=2)
    service.record_windows(first, [_window(first, side, 3000 * side**2) for side in (512, 1024)])
    second = asyncio.run(
        service.create_run(
            "model-a",
            WorkingMemoryCalibrationRequest(graph=_add_graph(), attempt_id="current", attempt_steps=2),
            "system",
        )
    )

    service.refresh_user_fits()
    assert _estimate() == _user_bytes(1024)

    _finish(service, second.item_ids)
    service.refresh_user_fits()
    assert _estimate() > _user_bytes(1024)


def _add_graph() -> Graph:
    graph = Graph()
    graph.add_node(AddInvocation(id="add", a=1, b=2))
    return graph


def test_an_attempt_stopped_before_it_queued_every_run_never_feeds_a_fit_and_the_summary_says_so(
    tmp_path: Path, started
) -> None:
    """Two of three planned runs, finished, whose windows alone span enough sizes to fit a line: Stop ended the attempt
    before its last step was queued, so it feeds nothing, then or later."""
    service = started(tmp_path)
    queued = [_create(service, attempt_id="stopped", attempt_steps=3) for _ in range(2)]
    for run_id in queued:
        service.record_windows(run_id, _on_the_line(run_id))

    service.refresh_user_fits()

    assert _estimate() == _shipped()
    summary = service.get_model_summary("model-a")
    assert {(run.run_id, run.attempt_steps, run.held_back) for run in summary.runs} == {
        (run_id, 3, "aborted_attempt") for run_id in queued
    }
    [line] = summary.lines
    assert line.state == "insufficient"
    assert "A stopped or interrupted calibration measured this line" in (line.problem or "")
    assert {(size.counted, size.excluded_reason) for size in line.sizes} == {(False, "aborted_attempt")}
    assert summary.excluded_windows == 6


def test_an_attempt_feeds_once_it_queued_every_planned_run_and_all_finished(tmp_path: Path, started) -> None:
    service = started(tmp_path)
    first = _create(service, attempt_id="whole", attempt_steps=2)
    service.record_windows(first, _on_the_line(first))
    # The first step finished before the second was posted: a prefix, which feeds nothing yet.
    service.refresh_user_fits()
    assert _estimate() == _shipped()

    graph = _add_graph()
    request = WorkingMemoryCalibrationRequest(graph=graph, attempt_id="whole", attempt_steps=2)
    second = asyncio.run(service.create_run("model-a", request, "system"))
    service.refresh_user_fits()
    assert _estimate() == _shipped()
    assert {run.held_back for run in service.get_model_summary("model-a").runs} == {"running"}

    _finish(service, second.item_ids)
    service.refresh_user_fits()
    assert _estimate() == _user_bytes(1024)
    assert {run.held_back for run in service.get_model_summary("model-a").runs} == {None}


def _idle(monkeypatch: pytest.MonkeyPatch, by: datetime.timedelta = ATTEMPT_IDLE_GRACE) -> None:
    """Let the service's clock run on by `by` and a second, as if the client went away between two steps."""
    clock = calibration_default._now
    monkeypatch.setattr(calibration_default, "_now", lambda: clock() + by + datetime.timedelta(seconds=1))


def _queue_attempt(service: WorkingMemoryCalibrationService, attempt_id: str, steps: int) -> list:
    """Every run of the attempt queued, as the client posts them; the first measures the line."""
    request = WorkingMemoryCalibrationRequest(graph=_add_graph(), attempt_id=attempt_id, attempt_steps=steps)
    created = [asyncio.run(service.create_run("model-a", request, "system")) for _ in range(steps)]
    service.record_windows(created[0].run_id, _on_the_line(created[0].run_id))
    _finish(service, created[0].item_ids)
    return created


def test_an_attempt_stopped_after_it_queued_every_run_never_feeds_a_fit(tmp_path: Path, started) -> None:
    """Stop cancels the items the queue still holds: the attempt has all its runs, and still was aborted."""
    service = started(tmp_path)
    _, stopped = _queue_attempt(service, "stopped-late", 2)

    service._services.session_queue.cancel_queue_item(stopped.item_ids[0])
    service.refresh_user_fits()

    assert _estimate() == _shipped()
    assert {run.held_back for run in service.get_model_summary("model-a").runs} == {"aborted_attempt"}


def test_a_step_stopped_after_it_measured_a_size_aborts_its_attempt(tmp_path: Path, started) -> None:
    """Stop while a step runs: its measurement is recorded, then its item is canceled, so the run ends `partial`."""
    service = started(tmp_path)
    _, stopped = _queue_attempt(service, "stopped-mid-step", 2)
    queue = service._services.session_queue
    item = queue.get_queue_item(stopped.item_ids[0])
    assert service.admit(item, stopped.run_id) is None
    service.record_windows(stopped.run_id, [_window(stopped.run_id, 1536, 900 * 1536**2 + 400 * MIB)])
    queue.cancel_queue_item(item.item_id)
    service.observe(item, stopped.run_id)
    service.refresh_user_fits()

    summary = service.get_model_summary("model-a")
    assert {run.status for run in summary.runs} == {"completed", "partial"}
    assert _estimate() == _shipped()
    assert {run.held_back for run in summary.runs} == {"aborted_attempt"}


def test_a_cleared_queue_aborts_the_attempt_whose_step_it_removed(tmp_path: Path, started) -> None:
    """Clear Queue deletes the waiting step's row before the runner saw it: the queue no longer knows it."""
    service = started(tmp_path)
    _queue_attempt(service, "cleared", 2)

    service._services.session_queue.clear("default")
    service.refresh_user_fits()

    assert _estimate() == _shipped()
    assert {run.held_back for run in service.get_model_summary("model-a").runs} == {"aborted_attempt"}


def test_an_attempt_whose_largest_step_failed_still_feeds_a_fit(tmp_path: Path, started) -> None:
    """An out-of-memory error at the largest size ends a run, not the attempt: the finished sizes count."""
    service = started(tmp_path)
    _, largest = _queue_attempt(service, "oom-at-the-top", 2)

    service._services.session_queue.fail_queue_item(largest.item_ids[0], "OutOfMemoryError", "out of memory", "")
    service.refresh_user_fits()

    assert _estimate() == _user_bytes(1024)
    assert {run.held_back for run in service.get_model_summary("model-a").runs} == {None}


def test_a_canceled_run_whose_retry_completed_no_longer_aborts_its_attempt(tmp_path: Path, started) -> None:
    """A run's outcome is that of its items' latest retries (`derive_run_status`), as the runner admitted them."""
    service = started(tmp_path)
    _, stopped = _queue_attempt(service, "resumed", 2)
    queue = service._services.session_queue
    queue.cancel_queue_item(stopped.item_ids[0])
    service.refresh_user_fits()
    assert _estimate() == _shipped()

    [retried] = queue.retry_items_by_id("default", stopped.item_ids).retried_items
    retry = queue.get_queue_item(retried.item_id)
    assert service.admit(retry, stopped.run_id) is None
    queue.complete_queue_item(retry.item_id)
    service.observe(retry, stopped.run_id)
    service.refresh_user_fits()

    assert _estimate() == _user_bytes(1024)


def test_a_retry_of_a_failed_step_canceled_before_it_ran_aborts_its_attempt(tmp_path: Path, started) -> None:
    """Out of memory at the largest size, a retry, then Stop while the retry waits: the runner never saw the retry,
    which the store recorded as the queue committed it, and the run's status is canceled. The fit agrees."""
    service = started(tmp_path)
    _, largest = _queue_attempt(service, "retried-then-stopped", 2)
    queue = service._services.session_queue
    queue.fail_queue_item(largest.item_ids[0], "OutOfMemoryError", "out of memory", "")
    service.refresh_user_fits()
    assert _estimate() == _user_bytes(1024)

    [retried] = queue.retry_items_by_id("default", largest.item_ids).retried_items
    queue.cancel_queue_item(retried.item_id)
    service.refresh_user_fits()

    assert _estimate() == _shipped()
    summary = service.get_model_summary("model-a")
    assert {run.run_id: run.status for run in summary.runs}[largest.run_id] == "canceled"
    assert {run.held_back for run in summary.runs} == {"aborted_attempt"}


def test_a_retry_of_a_judged_failed_step_cleared_before_the_next_refresh_aborts_its_attempt(
    tmp_path: Path, started
) -> None:
    """The failed step was judged (its outcome read from the queue, the attempt feeding the fit), then retried, and a
    queue clear deleted the retry's row before anything refreshed: no read of the queue finds the retry any more. The
    store recorded it as the queue committed it, so the attempt is aborted and its measurements leave the fit."""
    service = started(tmp_path)
    _, largest = _queue_attempt(service, "retried-then-cleared", 2)
    queue = service._services.session_queue
    queue.fail_queue_item(largest.item_ids[0], "OutOfMemoryError", "out of memory", "")
    service.refresh_user_fits()
    assert _estimate() == _user_bytes(1024)

    [retried] = queue.retry_items_by_id("default", largest.item_ids).retried_items
    queue.clear("default")
    service.refresh_user_fits()

    assert queue.get_queue_item_ids("default").item_ids == []
    assert _estimate() == _shipped()
    summary = service.get_model_summary("model-a")
    assert {run.held_back for run in summary.runs} == {"aborted_attempt"}
    sizes = {(size.counted, size.excluded_reason) for line in summary.lines for size in line.sizes}
    assert sizes == {(False, "aborted_attempt")}


def test_a_queue_item_id_that_now_names_another_item_is_not_taken_for_the_runs(
    tmp_path: Path, started, mock_invoker: Invoker
) -> None:
    """A recreated or restored database numbers its items anew while the store stays: the stopped step's id now names
    a finished generation, which says nothing about the step."""
    service = started(tmp_path)
    _, stopped = _queue_attempt(service, "renumbered", 2)
    queue = service._services.session_queue
    queue.cancel_queue_item(stopped.item_ids[0])
    queue.delete_queue_item(stopped.item_ids[0])
    generation = asyncio.run(queue.enqueue_batch("default", Batch(graph=_add_graph(), origin="canvas"), prepend=False))
    [other] = generation.item_ids
    queue.complete_queue_item(other)
    service.stop(mock_invoker)
    path = next((tmp_path / "working_memory").glob("*.json"))
    stored = json.loads(path.read_bytes())
    [renumbered] = [run for run in stored["runs"] if run["run_id"] == stopped.run_id]
    renumbered["queue_item_ids"] = [other]
    path.write_text(json.dumps(stored))

    started(tmp_path).refresh_user_fits()

    assert _estimate() == _shipped()


def test_a_step_with_a_failed_and_a_canceled_item_aborts_its_attempt(tmp_path: Path, started) -> None:
    """A step whose batch data makes two items: one ran out of memory, Stop canceled the other. The run shows as
    failed, yet its attempt was stopped."""
    service = started(tmp_path)
    first = _create(service, attempt_id="two-items", attempt_steps=2)
    service.record_windows(first, _on_the_line(first))
    data = [[BatchDatum(node_path="add", field_name="a", items=[1, 2])]]
    request = WorkingMemoryCalibrationRequest(graph=_add_graph(), data=data, attempt_id="two-items", attempt_steps=2)
    second = asyncio.run(service.create_run("model-a", request, "system"))
    queue = service._services.session_queue
    failed, stopped = second.item_ids
    queue.fail_queue_item(failed, "OutOfMemoryError", "out of memory", "")
    queue.cancel_queue_item(stopped)
    service.refresh_user_fits()

    assert _estimate() == _shipped()
    summary = service.get_model_summary("model-a")
    assert {run.run_id: run.status for run in summary.runs}[second.run_id] == "failed"
    assert {run.held_back for run in summary.runs} == {"aborted_attempt"}


def _stored(tmp_path: Path) -> dict:
    return json.loads(next((tmp_path / "working_memory").glob("*.json")).read_bytes())


def test_removing_a_run_of_a_complete_attempt_discards_only_its_own_measurements(tmp_path: Path, started) -> None:
    """The attempt queued all three runs: removing one leaves the other two judged as part of a complete attempt."""
    service = started(tmp_path)
    _, *rest = _queue_attempt(service, "complete", 3)
    for created in rest:
        _finish(service, created.item_ids)
    service.record_windows(rest[1].run_id, [_window(rest[1].run_id, 2048, 8000 * 2048**2)])
    assert "removed_attempt_steps" not in _stored(tmp_path)

    removed = service.reset_run(rest[1].run_id)

    assert (removed.runs_removed, removed.points_removed) == (1, 1)
    assert _estimate() == _user_bytes(1024)
    assert {run.held_back for run in service.get_model_summary("model-a").runs} == {None}
    assert _stored(tmp_path)["removed_attempt_steps"] == [
        {"user_id": "system", "attempt_id": "complete", "removed": 1, "aborted": False, "attempt_steps": 3}
    ]


def test_each_reset_forgets_what_the_service_kept_of_the_runs_it_removed(tmp_path: Path, started) -> None:
    """The service keeps, for the runs it judged, which file holds each and the outcomes of their items it read from
    the queue. Those last as long as the runs: calibrating and resetting again and again leaves nothing behind."""
    service = started(tmp_path)
    run_ids = [
        _create(service, model_key, attempt_id=f"attempt-{index}", attempt_steps=1)
        for index, model_key in enumerate(("model-a", "model-a", "model-b", "model-c"))
    ]
    items = {run_id: set(_stored_run(tmp_path, run_id)["queue_item_ids"]) for run_id in run_ids}
    service.refresh_user_fits()

    def kept() -> tuple[set[int], set[str]]:
        return set(service._item_outcomes), set(service._run_files)

    def of(*ids: str) -> tuple[set[int], set[str]]:
        return set().union(*(items[run_id] for run_id in ids)), set(ids)

    first, second, third, fourth = run_ids
    assert kept() == of(*run_ids)
    service.reset_run(first)
    assert kept() == of(second, third, fourth)
    service.reset_model("model-b")
    assert kept() == of(second, fourth)
    service.reset_all()
    assert kept() == (set(), set())


def test_removing_a_canceled_run_keeps_its_attempt_aborted(tmp_path: Path, started) -> None:
    service = started(tmp_path)
    _, stopped = _queue_attempt(service, "stopped-then-removed", 2)
    service._services.session_queue.cancel_queue_item(stopped.item_ids[0])

    service.reset_run(stopped.run_id)

    assert _estimate() == _shipped()
    assert {run.held_back for run in service.get_model_summary("model-a").runs} == {"aborted_attempt"}


def test_a_reset_while_an_attempt_runs_discards_the_attempt(tmp_path: Path, started) -> None:
    """The model's runs are removed while the attempt's second step waits: the first step's measurements go with
    the reset, and the attempt, which can no longer measure every size it planned, never feeds a fit."""
    service = started(tmp_path)
    request = WorkingMemoryCalibrationRequest(graph=_add_graph(), attempt_id="reset-while-running", attempt_steps=3)
    first = asyncio.run(service.create_run("model-a", request, "system"))
    _finish(service, first.item_ids)
    second = asyncio.run(service.create_run("model-a", request, "system"))

    service.reset_model("model-a")
    third = asyncio.run(service.create_run("model-a", request, "system"))
    service.record_windows(third.run_id, _on_the_line(third.run_id))
    _finish(service, [*second.item_ids, *third.item_ids])
    service.refresh_user_fits()

    assert _estimate() == _shipped()
    assert [run.held_back for run in service.get_model_summary("model-a").runs] == ["aborted_attempt"]


def test_a_finished_attempts_removal_record_is_dropped_once_none_of_its_runs_is_left(tmp_path: Path, started) -> None:
    """Nothing is left that the record could judge, and no later step of the finished attempt will come."""
    service = started(tmp_path)
    for created in _queue_attempt(service, "finished", 2)[1:]:
        _finish(service, created.item_ids)
    service.reset_run(service.get_model_summary("model-a").runs[0].run_id)
    assert [record["attempt_id"] for record in _stored(tmp_path)["removed_attempt_steps"]] == ["finished"]

    service.reset_model("model-a")

    assert "removed_attempt_steps" not in _stored(tmp_path)


@pytest.mark.parametrize("reset", ["run", "all"])
def test_a_later_run_of_an_attempt_a_reset_removed_while_it_ran_must_plan_the_same_steps(
    tmp_path: Path, started, reset: str
) -> None:
    """The reset left no run of the attempt, yet its later steps may still come: they must plan its count."""
    service = started(tmp_path)
    request = WorkingMemoryCalibrationRequest(graph=_add_graph(), attempt_id="still-posting", attempt_steps=2)
    waiting = asyncio.run(service.create_run("model-a", request, "system"))

    if reset == "run":
        service.reset_run(waiting.run_id)
    else:
        service.reset_all()

    assert _stored(tmp_path)["removed_attempt_steps"] == [
        {"user_id": "system", "attempt_id": "still-posting", "removed": 1, "aborted": True, "attempt_steps": 2}
    ]
    other_count = request.model_copy(update={"attempt_steps": 3})
    with pytest.raises(CalibrationAttemptConflictError):
        asyncio.run(service.create_run("model-a", other_count, "system"))
    asyncio.run(service.create_run("model-a", request, "system"))


def test_a_run_whose_items_were_never_recorded_aborts_its_attempt(
    tmp_path: Path, started, mock_invoker: Invoker
) -> None:
    """An enqueue that raised, a lookup of its items that raised too: the run is stored with no items, and the queue
    holds none with its origin. It measured nothing, and the attempt cannot have measured every size."""
    service = started(tmp_path)
    _, lost = _queue_attempt(service, "items-lost", 2)
    service._services.session_queue.delete_queue_item(lost.item_ids[0])
    service.stop(mock_invoker)
    path = next((tmp_path / "working_memory").glob("*.json"))
    stored = json.loads(path.read_bytes())
    [run] = [run for run in stored["runs"] if run["run_id"] == lost.run_id]
    run["queue_item_ids"] = []
    path.write_text(json.dumps(stored))

    started(tmp_path).refresh_user_fits()

    assert _estimate() == _shipped()


def test_a_retry_queued_between_the_two_queue_reads_holds_its_attempt_back(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stopped step is retried after the read for running items and before the items are read by id: the refresh
    judges the store it read, in which the step ended canceled, so the attempt feeds nothing; the next one finds the
    retry running, and the attempt feeds once it completed."""
    service = started(tmp_path)
    _, stopped = _queue_attempt(service, "retried-between-reads", 2)
    queue = service._services.session_queue
    queue.cancel_queue_item(stopped.item_ids[0])
    read_by_id = queue.get_queue_item_summaries_by_ids
    retries: list[int] = []

    def retry_then_read(*args, **kwargs):
        if not retries:
            retries.extend(item.item_id for item in queue.retry_items_by_id("default", stopped.item_ids).retried_items)
        return read_by_id(*args, **kwargs)

    monkeypatch.setattr(queue, "get_queue_item_summaries_by_ids", retry_then_read)
    service.refresh_user_fits()

    assert retries and _estimate() == _shipped()
    assert {run.held_back for run in service.get_model_summary("model-a").runs} == {"running"}

    retry = queue.get_queue_item(retries[0])
    assert service.admit(retry, stopped.run_id) is None
    queue.complete_queue_item(retry.item_id)
    service.observe(retry, stopped.run_id)
    service.refresh_user_fits()
    assert _estimate() == _user_bytes(1024)


def _settle_as_the_runner(service: WorkingMemoryCalibrationService, run_id: str, item_ids: list[int], status: str):
    """Run each item as the session runner would: admitted, finished in the queue, and observed."""
    queue = service._services.session_queue
    for item_id in item_ids:
        item = queue.get_queue_item(item_id)
        assert service.admit(item, run_id) is None
        if status == "completed":
            queue.complete_queue_item(item_id)
        else:
            queue.fail_queue_item(item_id, "OutOfMemoryError", "out of memory", "")
        service.observe(item, run_id)


def _failed_largest_step(service: WorkingMemoryCalibrationService, attempt_id: str):
    """A two-step attempt whose first step measured the line and whose largest step ran out of memory, as the runner
    saw them: complete, it feeds the fit."""
    request = WorkingMemoryCalibrationRequest(graph=_add_graph(), attempt_id=attempt_id, attempt_steps=2)
    first, largest = (asyncio.run(service.create_run("model-a", request, "system")) for _ in range(2))
    _settle_as_the_runner(service, first.run_id, first.item_ids, "completed")
    service.record_windows(first.run_id, _on_the_line(first.run_id))
    _settle_as_the_runner(service, largest.run_id, largest.item_ids, "failed")
    return largest


def test_a_retry_canceled_and_pruned_before_any_refresh_still_aborts_its_attempt_also_after_a_restart(
    tmp_path: Path, started, mock_invoker: Invoker
) -> None:
    """Out of memory at the largest size, which the runner saw; a retry; Stop; then a prune removes every finished row
    before anything refreshed, and the server restarts. Nothing but the store knows the retry, which it recorded as the
    queue committed it: canceled."""
    service = started(tmp_path)
    queue = service._services.session_queue
    largest = _failed_largest_step(service, "retried-then-pruned")
    service.refresh_user_fits()
    assert _estimate() == _user_bytes(1024)

    [retried] = queue.retry_items_by_id("default", largest.item_ids).retried_items
    queue.cancel_queue_item(retried.item_id)
    queue.prune("default")
    service.stop(mock_invoker)
    restarted = started(tmp_path)
    restarted.refresh_user_fits()

    assert _estimate() == _shipped()
    assert {run.held_back for run in restarted.get_model_summary("model-a").runs} == {"aborted_attempt"}


def _stored_run(tmp_path: Path, run_id: str) -> dict:
    [run] = [run for run in _stored(tmp_path)["runs"] if run["run_id"] == run_id]
    return run


def _refuse_the_record(monkeypatch: pytest.MonkeyPatch, service: WorkingMemoryCalibrationService, refusals: list):
    """Let the store refuse the count-down of a retry's record with each error of `refusals`, one per try."""
    change = service._store.change

    def refuse_the_record(device_class_id, apply, **kwargs):
        if refusals and getattr(apply, "func", None) is calibration_default._record_committed_retries:
            raise refusals.pop(0)
        return change(device_class_id, apply, **kwargs)

    monkeypatch.setattr(service._store, "change", refuse_the_record)


@pytest.mark.parametrize(
    "refusals",
    [
        [CalibrationStoreError("Cannot write the file")],
        [CalibrationStoreLockTimeoutError("Timed out waiting for the lock")] * 2,
    ],
    ids=["unwritable", "lock-timed-out-twice"],
)
def test_a_retry_the_store_could_not_record_keeps_its_attempt_out(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, refusals: list
) -> None:
    """The queue committed the retry, and recording it failed (a server stopped there leaves the same store). The
    retry is canceled and pruned unseen: no record names it, so the count left over keeps the attempt out."""
    service = started(tmp_path)
    queue = service._services.session_queue
    largest = _failed_largest_step(service, "record-failed")
    _refuse_the_record(monkeypatch, service, list(refusals))

    [retried] = queue.retry_items_by_id("default", largest.item_ids).retried_items
    queue.cancel_queue_item(retried.item_id)
    queue.prune("default")
    service.refresh_user_fits()

    assert "Could not record the retries of calibration runs" in caplog.text
    run = _stored_run(tmp_path, largest.run_id)
    assert (run["retries_in_flight"], sorted(run["observed"])) == (1, [str(largest.item_ids[0])])
    assert _estimate() == _shipped()
    assert {run.held_back for run in service.get_model_summary("model-a").runs} == {"aborted_attempt"}


def test_a_record_whose_lock_timed_out_once_is_tried_again(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another process held the file's lock past the timeout once: a count left over for that would keep a healthy
    attempt out for good."""
    service = started(tmp_path)
    queue = service._services.session_queue
    largest = _failed_largest_step(service, "lock-timed-out-once")
    _refuse_the_record(monkeypatch, service, [CalibrationStoreLockTimeoutError("Timed out waiting for the lock")])

    [retried] = queue.retry_items_by_id("default", largest.item_ids).retried_items
    _settle_as_the_runner(service, largest.run_id, [retried.item_id], "completed")
    service.refresh_user_fits()

    run = _stored_run(tmp_path, largest.run_id)
    assert "retries_in_flight" not in run and str(retried.item_id) in run["observed"]
    assert _estimate() == _user_bytes(1024)


def test_a_retry_the_full_queue_had_no_room_for_leaves_no_count(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = started(tmp_path)
    queue = service._services.session_queue
    largest = _failed_largest_step(service, "queue-full")
    monkeypatch.setattr(service._services.configuration, "max_queue_size", 1)
    asyncio.run(queue.enqueue_batch("default", Batch(graph=_add_graph(), origin="canvas"), prepend=False))

    assert queue.retry_items_by_id("default", largest.item_ids).retried_items == []

    service.refresh_user_fits()
    assert "retries_in_flight" not in _stored_run(tmp_path, largest.run_id)
    assert _estimate() == _user_bytes(1024)


def test_a_retry_that_failed_before_its_insertions_ended_rolled_back_and_leaves_no_count(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = started(tmp_path)
    queue = service._services.session_queue
    largest = _failed_largest_step(service, "rolled-back")
    before = queue.get_queue_item_ids("default").item_ids

    def refuse(*args, **kwargs):
        raise OSError("disk I/O error")

    insert_item = SessionQueueQueries.insert_item
    monkeypatch.setattr(SessionQueueQueries, "insert_item", refuse)
    with pytest.raises(OSError):
        queue.retry_items_by_id("default", largest.item_ids)
    monkeypatch.setattr(SessionQueueQueries, "insert_item", insert_item)
    service.refresh_user_fits()

    assert queue.get_queue_item_ids("default").item_ids == before
    run = _stored_run(tmp_path, largest.run_id)
    assert "retries_in_flight" not in run and sorted(run["observed"]) == [str(largest.item_ids[0])]
    assert _estimate() == _user_bytes(1024)


def test_a_retry_whose_commit_failed_after_its_insertions_keeps_its_count(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The connection to MySQL or MariaDB is lost at the commit, which the server carried out: the retry exists, and
    nobody can say so. It is canceled and pruned unseen; the count left over keeps the attempt out."""
    service = started(tmp_path)
    queue = service._services.session_queue
    largest = _failed_largest_step(service, "commit-lost")
    before = set(queue.get_queue_item_ids("default").item_ids)
    run = queue._queries.run

    def commit_then_lose_the_connection(work, *, read_only: bool = False):
        result = run(work, read_only=read_only)
        if work.__name__ == "admit":
            raise DatabaseUnavailableError("Lost connection to server during query")
        return result

    monkeypatch.setattr(queue._queries, "run", commit_then_lose_the_connection)
    with pytest.raises(DatabaseUnavailableError):
        queue.retry_items_by_id("default", largest.item_ids)
    monkeypatch.setattr(queue._queries, "run", run)
    [retry] = set(queue.get_queue_item_ids("default").item_ids) - before
    queue.cancel_queue_item(retry)
    queue.prune("default")
    service.refresh_user_fits()

    assert "failed at its commit, which may have committed" in caplog.text
    stored = _stored_run(tmp_path, largest.run_id)
    assert (stored["retries_in_flight"], sorted(stored["observed"])) == (1, [str(largest.item_ids[0])])
    assert _estimate() == _shipped()
    assert {run.held_back for run in service.get_model_summary("model-a").runs} == {"aborted_attempt"}


def test_a_store_that_cannot_count_a_retry_refuses_it(tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the count, a retry the store then failed to record could go unnoticed: the queue retries nothing."""
    service = started(tmp_path)
    queue = service._services.session_queue
    largest = _failed_largest_step(service, "count-refused")
    before = queue.get_queue_item_ids("default").item_ids

    def refuse(*args, **kwargs):
        raise CalibrationStoreError("Timed out waiting for the lock")

    monkeypatch.setattr(service._store, "change", refuse)
    with pytest.raises(CalibrationStoreError):
        queue.retry_items_by_id("default", largest.item_ids)

    assert queue.get_queue_item_ids("default").item_ids == before


def test_a_retry_of_an_unrelated_item_that_reuses_a_recorded_id_is_not_the_runs(
    tmp_path: Path, started, mock_invoker: Invoker
) -> None:
    """A recreated database gave the id the run recorded to a canvas item, which fails and is retried: its row's
    origin says it is not the run's, and its retry is not recorded."""
    service = started(tmp_path)
    queue = service._services.session_queue
    created = asyncio.run(service.create_run("model-a", WorkingMemoryCalibrationRequest(graph=_add_graph()), "system"))
    [canvas] = asyncio.run(
        queue.enqueue_batch("default", Batch(graph=_add_graph(), origin="canvas"), prepend=False)
    ).item_ids
    service.stop(mock_invoker)
    path = next((tmp_path / "working_memory").glob("*.json"))
    stored = json.loads(path.read_bytes())
    stored["runs"][0]["queue_item_ids"] = [canvas]
    path.write_text(json.dumps(stored))
    service = started(tmp_path)
    queue.fail_queue_item(canvas, "OutOfMemoryError", "out of memory", "")

    assert queue.retry_items_by_id("default", [canvas]).retried_items

    assert _stored(tmp_path)["runs"][0]["observed"] == {}
    assert created.item_ids


def test_a_retry_of_a_retry_is_recorded_on_its_run_standing_for_the_item_it_all_started_from(
    tmp_path: Path, started
) -> None:
    """Out of memory at the largest size; its retry R1 is canceled, and retried as R2, which is canceled too; a prune
    removes every row before anything refreshed. The queue links R2 to the item it all started from, and so does the
    store."""
    service = started(tmp_path)
    queue = service._services.session_queue
    largest = _failed_largest_step(service, "retried-twice")
    [first_retry] = queue.retry_items_by_id("default", largest.item_ids).retried_items
    queue.cancel_queue_item(first_retry.item_id)
    [second_retry] = queue.retry_items_by_id("default", [first_retry.item_id]).retried_items
    queue.cancel_queue_item(second_retry.item_id)
    queue.prune("default")

    service.refresh_user_fits()

    [run] = [run for run in _stored(tmp_path)["runs"] if run["run_id"] == largest.run_id]
    assert {int(key): value["retried_from_item_id"] for key, value in run["observed"].items()} == {
        largest.item_ids[0]: None,
        first_retry.item_id: largest.item_ids[0],
        second_retry.item_id: largest.item_ids[0],
    }
    assert "retries_in_flight" not in run
    assert _estimate() == _shipped()


def test_each_retry_of_a_calibration_item_is_recorded_on_its_run_as_the_queue_commits_it(
    tmp_path: Path, started
) -> None:
    """Items of two runs and a generation, retried in one call: each retry becomes an item of its own run, pending,
    standing for the item it retried, and no count is left. The generation's retry leaves the store as it is."""
    service = started(tmp_path)
    queue = service._services.session_queue
    created = [
        asyncio.run(service.create_run(model, WorkingMemoryCalibrationRequest(graph=_add_graph()), "system"))
        for model in ("model-a", "model-b")
    ]
    [generation] = asyncio.run(
        queue.enqueue_batch("default", Batch(graph=_add_graph(), origin="canvas"), prepend=False)
    ).item_ids
    item_ids = [run.item_ids[0] for run in created] + [generation]
    for item_id in item_ids:
        queue.fail_queue_item(item_id, "OutOfMemoryError", "out of memory", "")

    retried = {
        retry.source_item_id: retry.item_id for retry in queue.retry_items_by_id("default", item_ids).retried_items
    }

    assert len(retried) == 3
    for run in created:
        [item_id] = run.item_ids
        stored = _stored_run(tmp_path, run.run_id)
        assert "retries_in_flight" not in stored
        assert {key: {**value, "observed_at": None} for key, value in stored["observed"].items()} == {
            str(retried[item_id]): {
                "status": "pending",
                "error_message": None,
                "retried_from_item_id": item_id,
                "observed_at": None,
            }
        }


def test_a_retry_the_runner_settled_before_it_was_recorded_keeps_what_the_runner_saw(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Out of memory at the largest size; the runner takes the retry and completes it between the queue's commit and
    the store's record of it. The record must not turn it back into a pending item, which, once pruned, would read
    canceled and abort the attempt."""
    service = started(tmp_path)
    queue = service._services.session_queue
    largest = _failed_largest_step(service, "settled-first")
    committed = service.retries_committed

    def run_it_first(in_flight, retried) -> None:
        retried = list(retried)
        _settle_as_the_runner(service, largest.run_id, [retry.item_id for retry in retried], "completed")
        committed(in_flight, retried)

    monkeypatch.setattr(service, "retries_committed", run_it_first)
    queue.retry_items_by_id("default", largest.item_ids)
    queue.prune("default")
    service.refresh_user_fits()

    assert _estimate() == _user_bytes(1024)


@pytest.mark.parametrize("reset", ["run", "model", "all"])
def test_a_retry_queued_as_a_reset_takes_the_files_lock_still_stops_the_attempt(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch, reset: str
) -> None:
    """The failed step's retry is queued by the time the reset reads the queue, under the file's lock: the reset finds
    the attempt running and stops it for good. Resetting the model is what the UI does."""
    service = started(tmp_path)
    queue = service._services.session_queue
    _, failed = _queue_attempt(service, "retried-while-reset", 2)
    queue.fail_queue_item(failed.item_ids[0], "OutOfMemoryError", "out of memory", "")
    service.refresh_user_fits()
    assert _estimate() == _user_bytes(1024)
    change = service._store.change
    retried: list[int] = []

    def retry_then_change(*args, **kwargs):
        if not retried:
            # Marked first: the retry's own changes of the file pass.
            retried.append(-1)
            retried.extend(item.item_id for item in queue.retry_items_by_id("default", failed.item_ids).retried_items)
        return change(*args, **kwargs)

    monkeypatch.setattr(service._store, "change", retry_then_change)
    if reset == "run":
        service.reset_run(failed.run_id)
    elif reset == "model":
        service.reset_model("model-a")
    else:
        service.reset_all()

    assert retried and _estimate() == _shipped()
    assert _stored(tmp_path).get("removed_attempt_steps") == [
        {
            "user_id": "system",
            "attempt_id": "retried-while-reset",
            "removed": 1 if reset == "run" else 2,
            "aborted": True,
            "attempt_steps": 2,
        }
    ]


def test_two_owners_attempts_with_one_id_are_two_incomplete_attempts(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each owner posts one of two planned steps under the same attempt id: neither attempt queued both, so neither
    feeds a fit; and each owner's count is its own."""
    service = started(tmp_path)
    request = WorkingMemoryCalibrationRequest(graph=_add_graph(), attempt_id="shared-id", attempt_steps=2)
    for owner in ("system", "other-admin"):
        created = asyncio.run(service.create_run("model-a", request, owner))
        service.record_windows(created.run_id, _on_the_line(created.run_id))
        _finish(service, created.item_ids)
        # The other owner's attempt may start once this one went idle between its steps.
        _idle(monkeypatch)

    service.refresh_user_fits()

    assert _estimate() == _shipped()
    assert {run.held_back for run in service.get_model_summary("model-a").runs} == {"aborted_attempt"}
    asyncio.run(service.create_run("model-a", request.model_copy(update={"attempt_steps": 3}), "third-admin"))


@pytest.mark.parametrize(
    "change, refusal",
    [
        ("hash", "MODEL-A changed after this calibration was planned (hash). Calibrate again."),
        ("base", "MODEL-A changed after this calibration was planned (base). Calibrate again."),
        ("removed", "Model model-a was removed after this calibration was planned. Calibrate again."),
        ("format", "MODEL-A changed after this calibration was planned (format). Calibrate again."),
        ("variant", "MODEL-A changed after this calibration was planned (variant). Calibrate again."),
    ],
)
def test_an_item_whose_model_changed_after_planning_fails_at_admission_with_its_code(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch, change: str, refusal: str
) -> None:
    """The record changed while the calibration waited in the queue: the item would measure another model than the
    run was planned with, so it fails before any node runs, and the run's status says why in a code."""
    service = started(tmp_path)
    queue = service._services.session_queue
    created = asyncio.run(service.create_run("model-a", WorkingMemoryCalibrationRequest(graph=_add_graph()), "system"))
    if change == "removed":
        monkeypatch.delitem(MODELS, "model-a")
    elif change == "hash":
        monkeypatch.setattr(MODELS["model-a"], "hash", "blake3:reinstalled")
    elif change == "base":
        monkeypatch.setattr(MODELS["model-a"], "base", BaseModelType.StableDiffusion1)
    else:
        # A model without a format or variant passes as it is (every other test); one set since does not.
        monkeypatch.setattr(
            MODELS["model-a"], change, "gguf_quantized" if change == "format" else "klein_9b", raising=False
        )
    item = queue.get_queue_item(created.item_ids[0])

    assert service.admit(item, created.run_id) == refusal

    # As the runner fails it, and records what the queue then holds.
    queue.fail_queue_item(item.item_id, "WorkingMemoryCalibrationError", refusal, refusal)
    service.observe(item, created.run_id)
    [run] = service.get_status().device_classes[0].runs
    assert (run.status, run.error, run.error_code) == ("failed", refusal, "model_changed")


def test_a_model_changed_between_steps_aborts_its_attempt(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 1 measured the model; then it was reinstalled (a new hash) while steps 2 and 3 waited, and they fail at
    admission. Unlike an out-of-memory failure, that interrupts the attempt: step 1's windows feed no fit."""
    service = started(tmp_path)
    queue = service._services.session_queue
    request = WorkingMemoryCalibrationRequest(graph=_add_graph(), attempt_id="reinstalled", attempt_steps=3)
    first, *later = [asyncio.run(service.create_run("model-a", request, "system")) for _ in range(3)]
    _settle_as_the_runner(service, first.run_id, first.item_ids, "completed")
    service.record_windows(first.run_id, _on_the_line(first.run_id))
    monkeypatch.setattr(MODELS["model-a"], "hash", "blake3:reinstalled")

    for step in later:
        item = queue.get_queue_item(step.item_ids[0])
        refusal = service.admit(item, step.run_id)
        assert refusal == "MODEL-A changed after this calibration was planned (hash). Calibrate again."
        queue.fail_queue_item(item.item_id, "WorkingMemoryCalibrationError", refusal, refusal)
    service.refresh_user_fits()

    assert _estimate() == _shipped()
    assert {run.held_back for run in service.get_model_summary("model-a").runs} == {"aborted_attempt"}


def test_a_later_step_that_would_measure_another_version_of_the_model_is_refused(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The client posts each step once the last one finished. The model was reinstalled in between: the next step
    would measure another model than the first did, so it is refused, and the attempt, short of a step, never feeds."""
    service = started(tmp_path)
    request = WorkingMemoryCalibrationRequest(graph=_add_graph(), attempt_id="posted-late", attempt_steps=2)
    first = asyncio.run(service.create_run("model-a", request, "system"))
    _settle_as_the_runner(service, first.run_id, first.item_ids, "completed")
    service.record_windows(first.run_id, _on_the_line(first.run_id))
    monkeypatch.setattr(MODELS["model-a"], "hash", "blake3:reinstalled")

    with pytest.raises(CalibrationModelChangedError) as refused:
        asyncio.run(service.create_run("model-a", request, "system"))

    assert str(refused.value) == "MODEL-A changed after this calibration was planned (hash). Calibrate again."
    assert _estimate() == _shipped()
    # Another attempt of the model as it is now measures it afresh.
    other = request.model_copy(update={"attempt_id": "afresh", "attempt_steps": 1})
    asyncio.run(service.create_run("model-a", other, "system"))


def _naming_model_b(field: str, model_hash: str) -> Graph:
    """A graph that names model-b as the client compiled it: as a loader's model, or as a LoRA in a collection."""
    named = ModelIdentifierField(
        key="model-b", hash=model_hash, name="MODEL-B", base=BaseModelType.StableDiffusionXL, type=ModelType.Main
    )
    graph = _add_graph()
    if field == "loader":
        graph.add_node(SDXLModelLoaderInvocation(id="loader", model=named))
    else:
        graph.add_node(LoRACollectionLoader(id="loras", loras=[LoRAField(lora=named, weight=1.0)]))
    return graph


@pytest.mark.parametrize("change", ["variant", "format"])
def test_an_item_whose_component_model_changed_variant_or_format_fails_at_admission(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """The graph's identifiers carry no variant or format: the run recorded them for model-b as it was created."""
    service = started(tmp_path)
    monkeypatch.setattr(MODELS["model-b"], "hash", "blake3:planned")
    monkeypatch.setattr(MODELS["model-b"], "variant", "dev", raising=False)
    request = WorkingMemoryCalibrationRequest(graph=_naming_model_b("loader", "blake3:planned"))
    created = asyncio.run(service.create_run("model-a", request, "system"))
    monkeypatch.setattr(
        MODELS["model-b"], change, "klein_9b" if change == "variant" else "gguf_quantized", raising=False
    )
    item = service._services.session_queue.get_queue_item(created.item_ids[0])

    assert service.admit(item, created.run_id) == (
        f"MODEL-B changed after this calibration was planned ({change}). Calibrate again."
    )


def test_a_component_recorded_without_a_hash_is_admitted_on_the_rest_of_its_identity(
    tmp_path: Path, started, mock_invoker: Invoker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runs stored before graph_models recorded hashes: their items are admitted while the rest of the identity holds,
    and refused once it changed."""
    service = started(tmp_path)
    monkeypatch.setattr(MODELS["model-b"], "hash", "blake3:planned")
    monkeypatch.setattr(MODELS["model-b"], "variant", "dev", raising=False)
    request = WorkingMemoryCalibrationRequest(graph=_naming_model_b("loader", "blake3:planned"))
    created = asyncio.run(service.create_run("model-a", request, "system"))
    service.stop(mock_invoker)
    path = next((tmp_path / "working_memory").glob("*.json"))
    stored = json.loads(path.read_bytes())
    del stored["runs"][0]["graph_models"]["model-b"]["hash"]
    path.write_text(json.dumps(stored))
    restarted = started(tmp_path)
    item = restarted._services.session_queue.get_queue_item(created.item_ids[0])

    assert restarted.admit(item, created.run_id) is None
    monkeypatch.setattr(MODELS["model-b"], "variant", "klein_9b")
    assert restarted.admit(item, created.run_id) == (
        "MODEL-B changed after this calibration was planned (variant). Calibrate again."
    )


def test_a_later_step_whose_model_changed_variant_is_refused(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Step 1 measured model-a as FLUX.2 [dev]; the record now says Klein 9B: step 2 would measure another model."""
    service = started(tmp_path)
    monkeypatch.setattr(MODELS["model-a"], "variant", "dev", raising=False)
    request = WorkingMemoryCalibrationRequest(graph=_add_graph(), attempt_id="variant-changed", attempt_steps=2)
    first = asyncio.run(service.create_run("model-a", request, "system"))
    _settle_as_the_runner(service, first.run_id, first.item_ids, "completed")
    monkeypatch.setattr(MODELS["model-a"], "variant", "klein_9b")

    with pytest.raises(CalibrationModelChangedError, match=r"\(variant\)"):
        asyncio.run(service.create_run("model-a", request, "system"))


@pytest.mark.parametrize("field", ["loader", "lora-collection"])
def test_an_item_whose_component_model_changed_after_planning_fails_at_admission(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """The run calibrates model-a, and its graph loads model-b too, which was reinstalled while the item waited."""
    service = started(tmp_path)
    monkeypatch.setattr(MODELS["model-b"], "hash", "blake3:planned")
    request = WorkingMemoryCalibrationRequest(graph=_naming_model_b(field, "blake3:planned"))
    created = asyncio.run(service.create_run("model-a", request, "system"))
    monkeypatch.setattr(MODELS["model-b"], "hash", "blake3:reinstalled")
    queue = service._services.session_queue
    item = queue.get_queue_item(created.item_ids[0])

    refusal = service.admit(item, created.run_id)

    assert refusal == "MODEL-B changed after this calibration was planned (hash). Calibrate again."
    queue.fail_queue_item(item.item_id, "WorkingMemoryCalibrationError", refusal, refusal)
    [run] = service.get_status().device_classes[0].runs
    assert (run.status, run.error_code) == ("failed", "model_changed")


def _step(service: WorkingMemoryCalibrationService, attempt_id: Optional[str], owner: str = "system"):
    """One step of a two-step attempt (or a run without one), queued and run to completion by the runner."""
    request = WorkingMemoryCalibrationRequest(
        graph=_add_graph(), attempt_id=attempt_id, attempt_steps=2 if attempt_id is not None else None
    )
    created = asyncio.run(service.create_run("model-a", request, owner))
    _settle_as_the_runner(service, created.run_id, created.item_ids, "completed")
    return created


def test_a_calibration_between_two_steps_of_an_attempt_refuses_its_next_step(tmp_path: Path, started) -> None:
    """The reviewer's sequence: A's first step finishes, another calibration (an API client's run without an attempt,
    which no attempt blocks) runs to completion, and A's next step is refused: it would measure the GPU after that
    calibration. A stays short and never feeds."""
    service = started(tmp_path)
    first = _step(service, "tab-a")
    service.record_windows(first.run_id, _on_the_line(first.run_id))
    _step(service, None)

    with pytest.raises(CalibrationAttemptInterleavedError) as refused:
        asyncio.run(
            service.create_run(
                "model-a",
                WorkingMemoryCalibrationRequest(graph=_add_graph(), attempt_id="tab-a", attempt_steps=2),
                "system",
            )
        )

    assert (refused.value.code, str(refused.value)) == (
        "calibration_interleaved",
        "Another calibration ran on this GPU after this calibration started, so its measurements can no longer be "
        "combined. Calibrate again.",
    )
    service.refresh_user_fits()
    assert _estimate() == _shipped()
    runs = {run.run_id: run.held_back for run in service.get_model_summary("model-a").runs}
    assert runs[first.run_id] == "aborted_attempt"


def test_another_attempt_waits_while_one_is_between_its_steps(tmp_path: Path, started, monkeypatch) -> None:
    """A's first step finished a moment ago, and its client is about to post the next: B, another administrator's,
    is refused, A's next step is not, and once A finished B may start."""
    service = started(tmp_path)
    _step(service, "tab-a")

    with pytest.raises(CalibrationAttemptBusyError):
        _step(service, "tab-b", "other-admin")
    _step(service, "tab-a")
    _step(service, "tab-b", "other-admin")


def test_an_attempt_idle_past_the_grace_blocks_nothing_and_cannot_go_on_after_another_ran(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A's client went away after its first step. Past `ATTEMPT_IDLE_GRACE` B may run; A's next step, posted after
    all, is refused, and A never feeds."""
    service = started(tmp_path)
    first = _step(service, "tab-a")
    service.record_windows(first.run_id, _on_the_line(first.run_id))
    _idle(monkeypatch, ATTEMPT_IDLE_GRACE - datetime.timedelta(seconds=2))
    with pytest.raises(CalibrationAttemptBusyError):
        _step(service, "tab-b", "other-admin")
    _idle(monkeypatch, datetime.timedelta(seconds=2))

    _step(service, "tab-b", "other-admin")
    with pytest.raises(CalibrationAttemptInterleavedError):
        _step(service, "tab-a")
    service.refresh_user_fits()
    runs = {run.run_id: run.held_back for run in service.get_model_summary("model-a").runs}
    assert runs[first.run_id] == "aborted_attempt"


def test_the_idle_wait_counts_from_the_last_observation_not_the_run_s_creation(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A long step: created long ago, it finished a moment ago. A is open."""
    service = started(tmp_path)
    request = WorkingMemoryCalibrationRequest(graph=_add_graph(), attempt_id="tab-a", attempt_steps=2)
    created = asyncio.run(service.create_run("model-a", request, "system"))
    _idle(monkeypatch)
    _settle_as_the_runner(service, created.run_id, created.item_ids, "completed")

    with pytest.raises(CalibrationAttemptBusyError):
        _step(service, "tab-b", "other-admin")


def test_an_attempt_whose_steps_another_calibration_ran_between_never_feeds(
    tmp_path: Path, started, mock_invoker: Invoker
) -> None:
    """A store that holds such an attempt all the same (its later step posted past the refusal, say by a build without
    it): complete, every step finished, and still aborted."""
    service = started(tmp_path)
    first, _ = _step(service, "tab-a"), _step(service, "tab-a")
    service.record_windows(first.run_id, _on_the_line(first.run_id))
    other = _step(service, None)
    service.stop(mock_invoker)
    path = next((tmp_path / "working_memory").glob("*.json"))
    stored = json.loads(path.read_bytes())
    start = datetime.datetime(2026, 10, 27, 12, tzinfo=datetime.timezone.utc)
    # The plain run between the attempt's two steps.
    for run, offset in zip(
        sorted(stored["runs"], key=lambda run: (run["run_id"] == other.run_id, run["created_at"])),
        (0, 10, 5),
        strict=True,
    ):
        run["created_at"] = (start + datetime.timedelta(seconds=offset)).isoformat()
    path.write_text(json.dumps(stored))

    reread = started(tmp_path)
    reread.refresh_user_fits()

    assert _estimate() == _shipped()
    held = {run.run_id: run.held_back for run in reread.get_model_summary("model-a").runs}
    assert held[first.run_id] == "aborted_attempt"


def test_another_attempt_started_as_the_next_step_of_one_is_posted_is_refused(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A's first step finished; A's client posts the next step as another tab starts B. A's step is held inside its
    enqueue, recorded and not yet queued, while B reaches the attempts' lock: B is refused, A's step is queued."""
    service = started(tmp_path)
    _step(service, "tab-a")
    held = HeldEnqueue(monkeypatch)
    lock = service._file_lock
    arrived = asyncio.Event()

    @asynccontextmanager
    async def watched(name: str, os_lock):
        if name == _ATTEMPTS_LOCK and held.entered.is_set():
            arrived.set()
        async with lock(name, os_lock):
            yield

    monkeypatch.setattr(service, "_file_lock", watched)

    def request(attempt_id: str) -> WorkingMemoryCalibrationRequest:
        return WorkingMemoryCalibrationRequest(graph=_add_graph(), attempt_id=attempt_id, attempt_steps=2)

    async def both() -> list:
        step = asyncio.create_task(service.create_run("model-a", request("tab-a"), "system"))
        assert await asyncio.to_thread(held.entered.wait, 10)
        other = asyncio.create_task(service.create_run("model-a", request("tab-b"), "system"))
        await asyncio.wait_for(arrived.wait(), 10)
        held.release.set()
        return await asyncio.gather(step, other, return_exceptions=True)

    created, refused = asyncio.run(both())

    assert isinstance(refused, CalibrationAttemptBusyError), refused
    assert [run.attempt_id for run in service.get_status().device_classes[0].runs] == ["tab-a", "tab-a"]
    assert created.item_ids


def _steps(attempt_id: str, steps: int) -> WorkingMemoryCalibrationRequest:
    return WorkingMemoryCalibrationRequest(graph=_add_graph(), attempt_id=attempt_id, attempt_steps=steps)


def test_an_owners_new_attempt_replaces_its_own_attempt_that_stopped_between_steps(tmp_path: Path, started) -> None:
    """The submission stopped after step 2 of 3 (a full queue, a lost connection): the summary shows it aborted, and
    its owner's Retry starts a new attempt at once. The old one stays short, and a late step of it is refused."""
    service = started(tmp_path)
    stopped = [asyncio.run(service.create_run("model-a", _steps("first-try", 3), "system")) for _ in range(2)]
    for created in stopped:
        _settle_as_the_runner(service, created.run_id, created.item_ids, "completed")
        service.record_windows(created.run_id, _on_the_line(created.run_id))

    retry = asyncio.run(service.create_run("model-a", _steps("second-try", 1), "system"))
    _settle_as_the_runner(service, retry.run_id, retry.item_ids, "completed")

    with pytest.raises(CalibrationAttemptInterleavedError):
        asyncio.run(service.create_run("model-a", _steps("first-try", 3), "system"))
    service.refresh_user_fits()
    held = {run.run_id: run.held_back for run in service.get_model_summary("model-a").runs}
    assert [held[created.run_id] for created in stopped] == ["aborted_attempt", "aborted_attempt"]
    assert held[retry.run_id] is None


def test_another_owners_attempt_between_its_steps_is_said_to_be_between_its_steps(tmp_path: Path, started) -> None:
    service = started(tmp_path)
    _step(service, "tab-a")

    with pytest.raises(CalibrationAttemptBusyError) as refused:
        _step(service, "tab-b", "other-admin")

    assert str(refused.value) == (
        "Another calibration on this GPU is between its steps. Try again in a moment, or once it has finished."
    )


def test_an_attempt_stopped_between_its_steps_blocks_no_other_owner(tmp_path: Path, started) -> None:
    """Stop canceled the item of step 2 of 3: the attempt is aborted, and another administrator may start at once."""
    service = started(tmp_path)
    queue = service._services.session_queue
    first = asyncio.run(service.create_run("model-a", _steps("stopped", 3), "system"))
    _settle_as_the_runner(service, first.run_id, first.item_ids, "completed")
    second = asyncio.run(service.create_run("model-a", _steps("stopped", 3), "system"))
    queue.cancel_queue_item(second.item_ids[0])

    _step(service, "tab-b", "other-admin")


def test_an_attempt_a_reset_stopped_blocks_no_other_owner(tmp_path: Path, started) -> None:
    """A reset removed step 2 of 3 between the steps: the attempt can no longer measure every size, so it is aborted."""
    service = started(tmp_path)
    for _ in range(2):
        created = asyncio.run(service.create_run("model-a", _steps("reset", 3), "system"))
        _settle_as_the_runner(service, created.run_id, created.item_ids, "completed")
    service.reset_run(created.run_id)

    _step(service, "tab-b", "other-admin")


def test_an_attempt_another_run_cut_off_blocks_no_other_owner(tmp_path: Path, started) -> None:
    """A script's run without an attempt ran after A's first step: A cannot go on, so it holds nothing open."""
    service = started(tmp_path)
    _step(service, "tab-a")
    _step(service, None)

    _step(service, "tab-b", "other-admin")


@pytest.mark.parametrize("change", ["variant", "format", "hash"])
def test_a_later_step_whose_component_model_changed_is_refused(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    """The graph loads model-b too, which changed between the steps: its variant or format, which the graph's
    identifiers do not carry, or its hash, which a graph compiled after the reinstall names as it is now."""
    service = started(tmp_path)
    monkeypatch.setattr(MODELS["model-b"], "hash", "blake3:planned")
    monkeypatch.setattr(MODELS["model-b"], "variant", "dev", raising=False)
    monkeypatch.setattr(MODELS["model-b"], "format", "diffusers", raising=False)
    request = WorkingMemoryCalibrationRequest(
        graph=_naming_model_b("loader", "blake3:planned"), attempt_id="component", attempt_steps=2
    )
    first = asyncio.run(service.create_run("model-a", request, "system"))
    _settle_as_the_runner(service, first.run_id, first.item_ids, "completed")
    changed = {"variant": "klein_9b", "format": "gguf_quantized", "hash": "blake3:reinstalled"}[change]
    monkeypatch.setattr(MODELS["model-b"], change, changed)
    later = request.model_copy(
        update={"graph": _naming_model_b("loader", changed if change == "hash" else "blake3:planned")}
    )

    with pytest.raises(CalibrationModelChangedError) as refused:
        asyncio.run(service.create_run("model-a", later, "system"))

    assert str(refused.value) == f"MODEL-B changed after this calibration was planned ({change}). Calibrate again."


def test_a_bulk_retry_of_generations_touches_no_store_file(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A thousand failed generations retried at once: a retry carries its source's origin, no calibration's."""
    service = started(tmp_path)
    queue = service._services.session_queue
    asyncio.run(service.create_run("model-a", WorkingMemoryCalibrationRequest(graph=_add_graph()), "system"))
    data = [[BatchDatum(node_path="add", field_name="a", items=list(range(1000)))]]
    generations = asyncio.run(
        queue.enqueue_batch("default", Batch(graph=_add_graph(), data=data, origin="canvas"), prepend=False)
    ).item_ids
    for generation in generations:
        queue.fail_queue_item(generation, "OutOfMemoryError", "out of memory", "")
    touched: list[str] = []
    for name in ("read", "change"):
        method = getattr(service._store, name)

        def noted(device_class_id, *args, _method=method, **kwargs):
            touched.append(device_class_id)
            return _method(device_class_id, *args, **kwargs)

        monkeypatch.setattr(service._store, name, noted)

    retried = queue.retry_items_by_id("default", generations).retried_items

    assert len(retried) == 1000 and touched == []


def test_one_step_attempts_started_at_once_queue_only_the_first(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A one-step attempt held inside its enqueue is complete, and its item not queued yet: neither running nor open,
    it would let another administrator's attempt through. Only the attempts' lock keeps that one out until the item is
    queued, and then it is running."""
    service = started(tmp_path)
    queue = service._services.session_queue
    held = HeldEnqueue(monkeypatch)

    async def both() -> list:
        first = asyncio.create_task(service.create_run("model-a", _steps("tab-a", 1), "system"))
        assert await asyncio.to_thread(held.entered.wait, 10)
        second = asyncio.create_task(service.create_run("model-a", _steps("tab-b", 1), "other-admin"))
        # Time for the second creation to pass the first, were nothing to hold it.
        await asyncio.wait({second}, timeout=0.5)
        held.release.set()
        return await asyncio.gather(first, second, return_exceptions=True)

    created, refused = asyncio.run(both())

    assert isinstance(refused, CalibrationAttemptBusyError), refused
    assert queue.get_queue_item_ids("default").item_ids == created.item_ids


def test_two_attempts_started_at_once_queue_only_the_first(
    tmp_path: Path, started, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two clients start attempts together. The first is held inside its enqueue, its run recorded, its items not yet
    queued; the second reaches the attempts' lock meanwhile and waits there. Once the first has queued its items the
    second finds them running and is refused, and nothing of it is queued."""
    service = started(tmp_path)
    queue = service._services.session_queue
    held = HeldEnqueue(monkeypatch)
    lock = service._file_lock
    arrived: list[asyncio.Event] = []

    @asynccontextmanager
    async def watched(name: str, os_lock):
        if name == _ATTEMPTS_LOCK and held.entered.is_set():
            arrived[0].set()
        async with lock(name, os_lock):
            yield

    monkeypatch.setattr(service, "_file_lock", watched)

    def request(attempt_id: str) -> WorkingMemoryCalibrationRequest:
        return WorkingMemoryCalibrationRequest(graph=_add_graph(), attempt_id=attempt_id, attempt_steps=2)

    async def both() -> list:
        arrived.append(asyncio.Event())
        first = asyncio.create_task(service.create_run("model-a", request("first-tab"), "system"))
        assert await asyncio.to_thread(held.entered.wait, 10)
        second = asyncio.create_task(service.create_run("model-a", request("second-tab"), "system"))
        # A ceiling, not a wait: the second creation reaches the lock at once.
        await asyncio.wait_for(arrived[0].wait(), 10)
        held.release.set()
        return await asyncio.gather(first, second, return_exceptions=True)

    created, refused = asyncio.run(both())

    assert isinstance(refused, CalibrationAttemptBusyError), refused
    assert queue.get_queue_item_ids("default").item_ids == created.item_ids
    assert [run.attempt_id for run in service.get_status().device_classes[0].runs] == ["first-tab"]


def test_a_development_builds_removal_records_read_as_each_owners(
    tmp_path: Path, started, mock_invoker: Invoker, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Development builds kept what resets removed by attempt id alone. Such a record reads as the record of each owner
    whose runs carry the id: each owner's attempt queued one step, lost the other to a reset, and is complete. A record
    no stored run carries any more is dropped."""
    service = started(tmp_path)
    request = WorkingMemoryCalibrationRequest(graph=_add_graph(), attempt_id="before", attempt_steps=2)
    for owner in ("system", "other-admin"):
        created = asyncio.run(service.create_run("model-a", request, owner))
        service.record_windows(created.run_id, _on_the_line(created.run_id))
        _finish(service, created.item_ids)
        _idle(monkeypatch)
    service.stop(mock_invoker)
    path = next((tmp_path / "working_memory").glob("*.json"))
    stored = json.loads(path.read_bytes())
    stored["removed_attempt_steps"] = {
        "before": {"removed": 1, "aborted": False, "attempt_steps": 2},
        "forgotten": {"removed": 1, "aborted": True, "attempt_steps": 3},
    }
    path.write_text(json.dumps(stored))

    reread = started(tmp_path)
    reread.refresh_user_fits()
    assert _estimate() == _user_bytes(1024)

    # Any write stores them per owner; a reset of another model removes nothing of theirs.
    reread.reset_model("model-b")
    assert _stored(tmp_path)["removed_attempt_steps"] == [
        {"user_id": owner, "attempt_id": "before", "removed": 1, "aborted": False, "attempt_steps": 2}
        for owner in ("other-admin", "system")
    ]


def test_an_attempt_stored_before_attempt_steps_is_judged_on_the_runs_it_recorded(
    tmp_path: Path, started, mock_invoker: Invoker
) -> None:
    """Only development builds stored such runs; their attempt feeds once its recorded runs finished, as it did then."""
    service = started(tmp_path)
    run_id = _create(service, attempt_id="before", attempt_steps=2)
    service.record_windows(run_id, _on_the_line(run_id))
    service.stop(mock_invoker)
    path = next((tmp_path / "working_memory").glob("*.json"))
    stored = json.loads(path.read_bytes())
    del stored["runs"][0]["attempt_steps"]
    path.write_text(json.dumps(stored))

    started(tmp_path).refresh_user_fits()

    assert _estimate() == _user_bytes(1024)


def test_a_line_measured_at_one_size_is_insufficient(tmp_path: Path, started) -> None:
    service = started(tmp_path)
    run_id = _create(service)
    service.record_windows(run_id, _on_the_line(run_id, sides=(1024,)))

    [line] = service.get_model_summary("model-a").lines

    assert line.state == "insufficient" and "measure more sizes" in (line.problem or "")
    assert line.measured_range is None and [size.user_bytes for size in line.sizes] == [None]


def test_a_stopped_service_no_longer_feeds_estimates(tmp_path: Path, started, mock_invoker: Invoker) -> None:
    service = started(tmp_path)
    run_id = _create(service)
    service.record_windows(run_id, _on_the_line(run_id))
    service.refresh_user_fits()

    service.stop(mock_invoker)

    assert _estimate() == _shipped()
