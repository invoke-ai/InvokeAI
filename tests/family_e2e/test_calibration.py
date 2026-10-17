"""Calibration runs against the fake server: the calibrate route instead of enqueue_batch, refusals per case or for
the run, and the store of every server lifetime saved into the run."""

import json
from pathlib import Path

import pytest

from scripts.family_e2e import runner
from scripts.family_e2e.calibration import PROBE_REASON, CalibrationError
from scripts.family_e2e.records import load_records, load_run_info, write_run_info
from tests.family_e2e.test_flow import FOREIGN_PID, MODEL, FakeGpu, isolated_runner, setup  # noqa: F401 - fixtures

DEVICE_CLASS_ID = "0123456789abcdef"


def _export(request: dict, work_dir: Path) -> dict:
    graph = {
        "id": "fake-graph",
        "nodes": {
            "loader": {"id": "loader", "type": "fake_loader", "is_intermediate": True, "model": MODEL},
            "output": {"id": "output", "type": "fake_l2i", "is_intermediate": False, "width": 64, "height": 48},
        },
        "edges": [],
    }
    batch = {"graph": graph, "data": [], "origin": "test", "destination": "test"}
    return {"version": 1, "cases": [{"id": request["cases"][0]["id"], "batch": batch, "components": {}}]}


def _store(run_dir: Path, path: str) -> dict:
    return json.loads((run_dir / path).read_text())


def test_a_calibration_run_measures_every_step_once_and_saves_every_lifetimes_store(
    setup: tuple[Path, Path],  # noqa: F811 - fixture
    isolated_runner: None,  # noqa: F811 - fixture
) -> None:
    manifest_path, _ = setup
    # The first server dies in its second queue item, the first measure.
    (manifest_path.parent / "code" / "crash-at-enqueue").write_text("2")
    gpu = FakeGpu()

    run_dir = runner.new_run(manifest_path, [], [], calibrate=True)
    assert runner.run(run_dir, gpu_factory=lambda _: gpu.gpu(), export=_export) == 0

    records = load_records(run_dir)
    # The second server warms up again and repeats the crashed measure. Foreign load spoils that one, and it is kept:
    # the server keeps every window it measured.
    assert [(r.kind, r.rep, r.attempt) for r in records] == [
        ("warmup", 0, 1),
        ("measure", 0, 1),
        ("warmup", 0, 2),
        ("measure", 0, 2),
        ("measure", 1, 1),
    ]
    crashed, spoiled = records.pop(1), records.pop(2)
    assert crashed.status == "error" and crashed.calibration is None
    assert spoiled.invalid_reasons == [f"foreign GPU load: pid {FOREIGN_PID} (?) 2048 MiB", PROBE_REASON]
    for record in records:
        assert record.status == "completed", record.error
        # The probe alone makes time and VRAM invalid; the residency is known.
        assert record.invalid_reasons == [PROBE_REASON]
        assert (record.valid_timing, record.valid_memory, record.valid_psnr) == (False, False, True)
        assert (run_dir / record.image_path).is_file()

    info = load_run_info(run_dir)
    assert [(e.lifetime, e.source, e.device_class_id) for e in info.calibration_exports] == [
        (1, "store_file", DEVICE_CLASS_ID),
        (2, "api", DEVICE_CLASS_ID),
    ]
    first, second = info.calibration_exports
    assert first.status is None
    assert json.loads((run_dir / second.status).read_text())["current_device_class_id"] == DEVICE_CLASS_ID
    # Each lifetime's store holds the runs its server created; the crashed one's item never ran a node.
    first_store, second_store = _store(run_dir, first.export), _store(run_dir, second.export)
    assert [run["run_id"] for run in first_store["runs"]] == ["run-1", "run-2"]
    assert [p["queue_item_id"] for p in first_store["points"]] == [records[0].queue_item_id]
    later = [records[1], spoiled, records[2]]
    assert [run["run_id"] for run in second_store["runs"]] == [r.calibration.run_id for r in later]
    assert [p["queue_item_id"] for p in second_store["points"]] == [r.queue_item_id for r in later]

    # A runner killed before the last measure: the resumed run's server saves its own store.
    lines = (run_dir / "records.jsonl").read_text().splitlines(keepends=True)
    (run_dir / "records.jsonl").write_text("".join(lines[:4]))
    write_run_info(run_dir, info.model_copy(update={"status": "running"}))
    assert runner.run(run_dir, gpu_factory=lambda _: gpu.gpu(), export=_export) == 0

    resumed = load_records(run_dir)
    assert [(r.kind, r.rep, r.attempt) for r in resumed[4:]] == [("warmup", 0, 3), ("measure", 1, 1)]
    assert runner.remaining_steps(info.plan, resumed) == []
    third = load_run_info(run_dir).calibration_exports[-1]
    assert (third.lifetime, third.source) == (3, "api")
    assert [p["queue_item_id"] for p in _store(run_dir, third.export)["points"]] == [
        r.queue_item_id for r in resumed[4:]
    ]


def _refuse(manifest_path: Path, at: int | None, status: str, detail: str) -> None:
    refusal = {"at": at, "status": status, "detail": detail}
    (manifest_path.parent / "code" / "calibration-refused").write_text(json.dumps(refusal))


def test_a_refused_case_gets_an_error_record_and_the_run_goes_on(
    setup: tuple[Path, Path],  # noqa: F811 - fixture
    isolated_runner: None,  # noqa: F811 - fixture
) -> None:
    manifest_path, _ = setup
    _refuse(manifest_path, 2, "422 Unprocessable Entity", "Node n1 calls a saved workflow.")
    gpu = FakeGpu()
    run_dir = runner.new_run(manifest_path, [], [], calibrate=True)

    # The case's error record fails the report, and the run completes.
    assert runner.run(run_dir, gpu_factory=lambda _: gpu.gpu(), export=_export) == 1

    records = load_records(run_dir)
    assert [(r.kind, r.rep, r.status) for r in records] == [
        ("warmup", 0, "completed"),
        ("measure", 0, "error"),
        ("measure", 1, "completed"),
    ]
    assert "-> 422: " in records[1].error and "saved workflow" in records[1].error
    info = load_run_info(run_dir)
    assert info.status == "completed"
    [export] = info.calibration_exports
    assert export.source == "api"
    assert [p["queue_item_id"] for p in _store(run_dir, export.export)["points"]] == [
        records[0].queue_item_id,
        records[2].queue_item_id,
    ]


@pytest.mark.parametrize(
    "status, detail",
    [
        ("403 Forbidden", "Admin access required"),
        ("409 Conflict", "Calibration runs on a single generation device, and this machine generates on 2."),
        ("429 Too Many Requests", "Too many unacknowledged idempotency keys"),
    ],
)
def test_a_refusal_that_concerns_every_case_fails_the_run_and_keeps_its_windows(
    status: str,
    detail: str,
    setup: tuple[Path, Path],  # noqa: F811 - fixture
    isolated_runner: None,  # noqa: F811 - fixture
) -> None:
    manifest_path, _ = setup
    _refuse(manifest_path, 2, status, detail)
    gpu = FakeGpu()
    run_dir = runner.new_run(manifest_path, [], [], calibrate=True)

    with pytest.raises(CalibrationError, match=rf"refused the calibration run: POST .* -> {status[:3]}: "):
        runner.run(run_dir, gpu_factory=lambda _: gpu.gpu(), export=_export)

    records = load_records(run_dir)
    assert [(r.kind, r.status) for r in records] == [("warmup", "completed")]
    info = load_run_info(run_dir)
    assert info.status == "failed"
    # The scratch root's store is copied before the root goes: the warm-up's windows are kept.
    [export] = info.calibration_exports
    assert export.source == "store_file"
    assert [p["queue_item_id"] for p in _store(run_dir, export.export)["points"]] == [records[0].queue_item_id]


def test_a_server_that_cannot_calibrate_fails_the_run_before_any_step(
    setup: tuple[Path, Path],  # noqa: F811 - fixture
    isolated_runner: None,  # noqa: F811 - fixture
) -> None:
    manifest_path, _ = setup
    reason = "Calibration measures GPU memory, and this machine generates on cpu."
    (manifest_path.parent / "code" / "calibration-unavailable").write_text(reason)
    gpu = FakeGpu()
    run_dir = runner.new_run(manifest_path, [], [], calibrate=True)

    with pytest.raises(CalibrationError, match=f"the server cannot calibrate: {reason}"):
        runner.run(run_dir, gpu_factory=lambda _: gpu.gpu(), export=_export)

    assert load_records(run_dir) == []
    info = load_run_info(run_dir)
    assert (info.status, info.calibration_exports) == ("failed", [])
