"""User fits from the calibration store: what estimates read after a session start, and the per-model summary."""

import asyncio
import builtins
import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from invokeai.app.invocations.math import AddInvocation
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.model_records import UnknownModelException
from invokeai.app.services.session_queue.session_queue_common import Batch, BatchDatum
from invokeai.app.services.session_queue.session_queue_default import SessionQueue
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.graph import Graph
from invokeai.app.services.working_memory_calibration.working_memory_calibration_common import (
    CalibrationAttemptConflictError,
    WorkingMemoryCalibrationRequest,
)
from invokeai.app.services.working_memory_calibration.working_memory_calibration_default import (
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
from tests.fixtures.working_memory_calibration import GPU, calibration_service, fake_environment, measured_window

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
    idle = DeviceMemory(process_bytes=(16304 - 300) * MIB, outside_torch_bytes=300 * MIB)
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
    idle = DeviceMemory(process_bytes=(16304 - 300) * MIB, outside_torch_bytes=300 * MIB)
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
    and the run's status, which the queue's items decide, is canceled. The fit agrees."""
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
    assert _stored(tmp_path)["removed_attempt_steps"] == {
        "complete": {"removed": 1, "aborted": False, "attempt_steps": 3}
    }


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
    assert set(_stored(tmp_path)["removed_attempt_steps"]) == {"finished"}

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

    assert _stored(tmp_path)["removed_attempt_steps"] == {
        "still-posting": {"removed": 1, "aborted": True, "attempt_steps": 2}
    }
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
    """The stopped step is retried after the read for running items and before the lineage read: the lineage shows the
    retry waiting, so the attempt is running for that refresh, and the next one judges it again."""
    service = started(tmp_path)
    _, stopped = _queue_attempt(service, "retried-between-reads", 2)
    queue = service._services.session_queue
    queue.cancel_queue_item(stopped.item_ids[0])
    read_lineage = service._live_items
    retries: list[int] = []

    def retry_then_read():
        if not retries:
            retries.extend(item.item_id for item in queue.retry_items_by_id("default", stopped.item_ids).retried_items)
        return read_lineage()

    monkeypatch.setattr(service, "_live_items", retry_then_read)
    service.refresh_user_fits()

    assert retries and _estimate() == _shipped()
    assert {run.held_back for run in service.get_model_summary("model-a").runs} == {"running"}

    retry = queue.get_queue_item(retries[0])
    assert service.admit(retry, stopped.run_id) is None
    queue.complete_queue_item(retry.item_id)
    service.observe(retry, stopped.run_id)
    service.refresh_user_fits()
    assert _estimate() == _user_bytes(1024)


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
