import json
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scripts.family_e2e import records as records_module
from scripts.family_e2e.records import (
    RECORDS_FILE,
    RUN_INFO_FILE,
    Environment,
    Memory,
    NodeMemory,
    NodeTiming,
    PlannedStep,
    Record,
    Reference,
    RunInfo,
    Thresholds,
    Timing,
    append_record,
    load_records,
    load_run_info,
    write_run_info,
)

MB = 2**20


def make_environment(**overrides: Any) -> Environment:
    fields: dict[str, Any] = {
        "code_dir": "D:/code/InvokeAI",
        "git_commit": "0123456789abcdef",
        "git_branch": "main",
        "git_dirty": False,
        "python": "D:/code/InvokeAI/.venv/Scripts/python.exe",
        "invokeai_file": "D:/code/InvokeAI/invokeai/__init__.py",
        "app_version": {"version": "7.0.0"},
        "app_deps": {"torch": "2.8.0+cu128", "CUDA": "12.8", "numpy": "1.26.4"},
        "runtime_config": {
            "set_fields": [],
            "config": {"pytorch_cuda_alloc_conf": "backend:cudaMallocAsync", "enable_partial_loading": False},
        },
        "gpu_description": "NVIDIA GeForce RTX 4090",
        "gpu_luid": 0x1234,
        "gpu_driver": "32.0.15.6094",
        "server_cmd": ["python", "-m", "invokeai.app.run_app"],
        "extra_env": {},
    }
    fields.update(overrides)
    return Environment(**fields)


def make_record(
    case_key: str = "zimage-t2i@8x8",
    kind: str = "measure",
    rep: int = 0,
    attempt: int = 0,
    *,
    run_id: str = "20261001-120000-rtx4090",
    arm: str = "default",
    status: str = "completed",
    total_s: float = 10.0,
    per_step_s: float | None = 1.0,
    peak_bytes: int | None = 8000 * MB,
    node_peaks: dict[str, int] | None = None,
    image: bool = True,
    image_size: tuple[int, int] | None = None,
    image_sha256: str = "sha-a",
    graph_hash: str = "graph-1",
    valid_timing: bool = True,
    valid_memory: bool = True,
    valid_psnr: bool = True,
    invalid_reasons: list[str] | None = None,
    error: str | None = None,
) -> Record:
    case_id, size = case_key.split("@")
    width, height = (int(v) for v in size.split("x"))
    record_id = f"{case_key}.{arm}.{kind}.{rep}.{attempt}"
    nodes = node_peaks if node_peaks is not None else {"denoise": peak_bytes or 0}
    image_width, image_height = image_size or (width, height)
    return Record(
        record_id=record_id,
        run_id=run_id,
        arm=arm,
        case_id=case_id,
        case_key=case_key,
        family="z-image",
        width=width,
        height=height,
        steps=8,
        seed=1234,
        kind=kind,  # type: ignore[arg-type]
        rep=rep,
        attempt=attempt,
        queue_item_id=1,
        graph_hash=graph_hash,
        status=status,  # type: ignore[arg-type]
        error=error,
        image_path=f"images/{record_id}.png" if image else None,
        image_sha256=image_sha256 if image else None,
        image_width=image_width if image else None,
        image_height=image_height if image else None,
        timing=Timing(
            total_s=total_s,
            nodes=[
                NodeTiming(node_id=node_id, type="z_image_denoise", t_start=1.0, t_end=total_s) for node_id in nodes
            ],
            model_loads=[],
            per_step_s=per_step_s,
            queue_started_at=None,
            queue_completed_at=None,
        ),
        memory=Memory(
            method="d3dkmt",
            interval_ms=3.0,
            process_baseline_bytes=1000 * MB,
            process_peak_bytes=peak_bytes,
            nodes=[NodeMemory(node_id=n, baseline_bytes=1000 * MB, peak_bytes=p) for n, p in nodes.items()],
            shared_max_delta_bytes=0,
            foreign=[],
            trace=f"traces/{record_id}.csv.gz",
        ),
        residency=[],
        uses_fp8=False,
        valid_timing=valid_timing,
        valid_memory=valid_memory,
        valid_psnr=valid_psnr,
        invalid_reasons=invalid_reasons or [],
    )


def make_run_info(
    run_id: str = "20261001-120000-rtx4090",
    *,
    profile: str = "rtx4090",
    case_keys: tuple[str, ...] = ("zimage-t2i@8x8",),
    arms: tuple[str, ...] = ("default",),
    environment: Environment | None = None,
    graph_hash: str = "graph-1",
    thresholds: Thresholds | None = None,
    status: str = "completed",
) -> RunInfo:
    plan = [
        PlannedStep(case_key=key, case_id=key.split("@")[0], family="z-image", arm=arm, kind=kind, rep=rep)  # type: ignore[arg-type]
        for arm in arms
        for key in case_keys
        for kind, rep in (("warmup", 0), ("measure", 0), ("measure", 1), ("measure", 2))
    ]
    return RunInfo(
        run_id=run_id,
        profile=profile,
        created_at="2026-10-01T12:00:00+00:00",
        finished_at=None,
        status=status,  # type: ignore[arg-type]
        manifest_path="D:/e2e/manifest.rtx4090.yaml",
        manifest_sha256="f" * 64,
        tool_commit="abc",
        environments={arm: environment or make_environment() for arm in arms},
        plan=plan,
        case_thresholds={
            key: thresholds or Thresholds(psnr_min_db=30.0, time_tolerance=0.10, peak_tolerance_mb=256.0)
            for key in case_keys
        },
        graph_hashes=dict.fromkeys(case_keys, graph_hash),
    )


def write_png(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels.astype(np.uint8)).save(path)


def test_records_round_trip_in_append_order(tmp_path: Path) -> None:
    written = [make_record(rep=rep, total_s=10.0 + rep) for rep in range(3)]
    for record in written:
        append_record(tmp_path, record)

    assert load_records(tmp_path) == written


def test_run_without_records_file_has_no_records(tmp_path: Path) -> None:
    assert load_records(tmp_path) == []


def test_truncated_last_line_from_a_killed_writer_is_skipped(tmp_path: Path) -> None:
    complete = [make_record(rep=0), make_record(rep=1)]
    for record in complete:
        append_record(tmp_path, record)
    partial = make_record(rep=2).model_dump_json()[:57]
    with open(tmp_path / RECORDS_FILE, "ab") as f:
        f.write(partial.encode())

    assert load_records(tmp_path) == complete


@pytest.mark.parametrize("malformed_at_end", [False, True], ids=["middle", "terminated-last-line"])
def test_malformed_complete_line_raises(tmp_path: Path, malformed_at_end: bool) -> None:
    good = make_record().model_dump_json()
    lines = [good, '{"record_id": "x"}'] if malformed_at_end else ['{"record_id": "x"}', good]
    (tmp_path / RECORDS_FILE).write_bytes(("\n".join(lines) + "\n").encode())

    with pytest.raises(ValueError, match="line 2" if malformed_at_end else "line 1"):
        load_records(tmp_path)


def test_append_after_a_killed_writer_drops_the_partial_line(tmp_path: Path) -> None:
    append_record(tmp_path, make_record(rep=0))
    with open(tmp_path / RECORDS_FILE, "ab") as f:
        f.write(make_record(rep=1).model_dump_json()[:40].encode())

    append_record(tmp_path, make_record(rep=1, attempt=1))

    assert [(r.rep, r.attempt) for r in load_records(tmp_path)] == [(0, 0), (1, 1)]


def test_append_keeps_a_complete_record_that_lacks_only_its_line_end(tmp_path: Path) -> None:
    (tmp_path / RECORDS_FILE).write_bytes(make_record(rep=0).model_dump_json().encode())

    append_record(tmp_path, make_record(rep=1))

    assert [r.rep for r in load_records(tmp_path)] == [0, 1]


def test_run_info_rewrite_replaces_the_file_and_leaves_no_temp_files(tmp_path: Path) -> None:
    info = make_run_info(status="running")
    write_run_info(tmp_path, info)
    finished = info.model_copy(update={"status": "completed", "finished_at": "2026-10-01T13:00:00+00:00"})

    write_run_info(tmp_path, finished)

    assert load_run_info(tmp_path) == finished
    assert [p.name for p in tmp_path.iterdir()] == [RUN_INFO_FILE]


def test_run_info_and_reference_written_before_gpu_driver_existed_still_load(tmp_path: Path) -> None:
    old = make_run_info().model_dump(mode="json")
    del old["environments"]["default"]["gpu_driver"]
    (tmp_path / RUN_INFO_FILE).write_text(json.dumps(old))
    reference = {
        "profile": "rtx4090",
        "case_key": "zimage-t2i@8x8",
        "source_run_id": old["run_id"],
        "accepted_at": "2026-10-01T13:00:00+00:00",
        "image_sha256": "a" * 64,
        "graph_hash": "graph-1",
        "environment": old["environments"]["default"],
        "total_s_median": 10.0,
        "per_step_s_median": None,
        "process_peak_bytes_median": None,
        "node_peak_bytes_median": {},
    }

    assert load_run_info(tmp_path).environments["default"].gpu_driver is None
    assert Reference.model_validate_json(json.dumps(reference)).environment.gpu_driver is None


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to replace a file another handle holds")
def test_run_info_write_waits_for_a_brief_concurrent_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(records_module, "REPLACE_RETRY_DELAY_S", 0.005)
    write_run_info(tmp_path, make_run_info(status="running"))
    reader = open(tmp_path / RUN_INFO_FILE, "rb")
    timer = threading.Timer(0.02, reader.close)
    timer.start()
    try:
        write_run_info(tmp_path, make_run_info(status="stopped"))
    finally:
        timer.join()

    assert load_run_info(tmp_path).status == "stopped"


@pytest.mark.skipif(sys.platform != "win32", reason="only Windows refuses to replace a file another handle holds")
def test_failed_run_info_write_keeps_the_old_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(records_module, "REPLACE_ATTEMPTS", 2)
    monkeypatch.setattr(records_module, "REPLACE_RETRY_DELAY_S", 0.0)
    write_run_info(tmp_path, make_run_info(status="running"))

    with open(tmp_path / RUN_INFO_FILE, "rb"), pytest.raises(PermissionError):
        write_run_info(tmp_path, make_run_info(status="stopped"))

    assert load_run_info(tmp_path).status == "running"
    assert [p.name for p in tmp_path.iterdir()] == [RUN_INFO_FILE]
