import socket
import sys
from pathlib import Path

import pytest
import yaml

from scripts.family_e2e import runner, server
from scripts.family_e2e.gpu_memory import Adapter, MemoryTrace, ProcessUsage
from scripts.family_e2e.manifest import Manifest
from scripts.family_e2e.records import NodeTiming, PlannedStep, Record, Timing, append_record, load_records
from scripts.family_e2e.runner import (
    FOREIGN_GPU_LOAD,
    SERVER_CRASHED,
    Gpu,
    RunnerError,
    build_memory,
    foreign_processes,
    gpu_lock,
    preflight,
    remaining_steps,
    uses_fp8,
)

MIB = 2**20
GIB = 2**30


def _plan() -> list[PlannedStep]:
    return [
        PlannedStep(case_key="a@64x64", case_id="a", family="f", arm="default", kind=kind, rep=rep)
        for kind, rep in (("warmup", 0), ("measure", 0), ("measure", 1), ("measure", 2))
    ]


def _record(step: PlannedStep, attempt: int, status: str = "completed", reasons: tuple[str, ...] = ()) -> Record:
    return Record(
        record_id=f"{step.case_key}.{step.arm}.{step.kind}.{step.rep}.{attempt}",
        run_id="r",
        arm=step.arm,
        case_id=step.case_id,
        case_key=step.case_key,
        family=step.family,
        width=64,
        height=64,
        steps=4,
        seed=1,
        kind=step.kind,
        rep=step.rep,
        attempt=attempt,
        queue_item_id=attempt,
        graph_hash="h",
        status=status,
        error=None,
        image_path=None,
        image_sha256=None,
        image_width=None,
        image_height=None,
        timing=None,
        memory=None,
        residency=[],
        uses_fp8=False,
        valid_timing=not reasons,
        valid_memory=not reasons,
        valid_psnr=True,
        invalid_reasons=list(reasons),
    )


def test_resume_skips_final_steps_and_tolerates_a_truncated_last_line(tmp_path: Path) -> None:
    warmup, m0, m1, m2 = _plan()
    foreign = (f"{FOREIGN_GPU_LOAD}: pid 4 (chrome.exe) 2048 MiB",)
    append_record(tmp_path, _record(warmup, 1))
    append_record(tmp_path, _record(m0, 1, reasons=foreign))  # a retry was due
    append_record(tmp_path, _record(m1, 1, status="failed"))
    with open(tmp_path / "records.jsonl", "ab") as f:
        f.write(_record(m2, 1).model_dump_json().encode()[:40])  # the runner died while writing
    assert [(s.kind, s.rep) for s in remaining_steps(_plan(), load_records(tmp_path))] == [
        ("measure", 0),
        ("measure", 1),
        ("measure", 2),
    ]

    append_record(tmp_path, _record(m0, 2, reasons=foreign))
    append_record(tmp_path, _record(m0, 3, reasons=foreign))  # retries used up: final, though invalid
    append_record(tmp_path, _record(m1, 2))
    assert [(s.kind, s.rep) for s in remaining_steps(_plan(), load_records(tmp_path))] == [("measure", 2)]


def test_a_case_that_keeps_crashing_the_server_is_given_up() -> None:
    warmup, m0, m1, m2 = _plan()
    crash = (f"{SERVER_CRASHED} (exit code 3): ConnectionError: refused",)
    records = [_record(warmup, 1), _record(m0, 1, status="error", reasons=crash)]
    assert [(s.kind, s.rep) for s in remaining_steps(_plan(), records)] == [
        ("measure", 0),
        ("measure", 1),
        ("measure", 2),
    ]
    # The restarted server repeats the warm-up, which crashes it again: the case stops counting as remaining.
    records.append(_record(warmup, 2, status="error", reasons=crash))
    assert remaining_steps(_plan(), records) == []


def test_foreign_processes_are_large_or_growing_others() -> None:
    before = {
        1: ProcessUsage(dedicated_bytes=8 * GIB, shared_bytes=0),  # the server itself
        2: ProcessUsage(dedicated_bytes=2 * GIB, shared_bytes=0),  # large from the start
        3: ProcessUsage(dedicated_bytes=300 * MIB, shared_bytes=0),  # grows by 400 MiB
        4: ProcessUsage(dedicated_bytes=500 * MIB, shared_bytes=0),  # stable and small
    }
    after = dict(before)
    after[3] = ProcessUsage(dedicated_bytes=700 * MIB, shared_bytes=0)
    after[5] = ProcessUsage(dedicated_bytes=200 * MIB, shared_bytes=0)  # new but small
    found = foreign_processes(before, after, own={1})
    assert [(p.pid, p.dedicated_bytes) for p in found] == [(2, 2 * GIB), (3, 700 * MIB)]


def test_fp8_comes_from_storage_settings_or_runtime_compute() -> None:
    configs = {
        "plain": {"default_settings": {"fp8_storage": False}},
        "fp8": {"default_settings": {"fp8_storage": True}},
    }
    assert not uses_fp8(frozenset({"plain", "lora-without-config"}), configs, fp8_compute=False)
    assert uses_fp8(frozenset({"plain", "fp8"}), configs, fp8_compute=False)
    assert uses_fp8(frozenset({"plain"}), configs, fp8_compute=True)


def test_memory_is_split_per_node_and_shared_growth_is_measured_from_enqueue() -> None:
    t0 = 100.0
    trace = MemoryTrace(
        method="d3dkmt",
        interval_ms=3.0,
        dedicated=((t0 - 0.01, GIB), (t0 + 0.2, 4 * GIB), (t0 + 0.4, 2 * GIB), (t0 + 0.6, 6 * GIB), (t0 + 0.9, GIB)),
        shared=((t0 - 0.05, 100 * MIB), (t0 + 0.5, 500 * MIB), (t0 + 2.0, 900 * MIB)),
    )
    nodes = [
        NodeTiming(node_id="encode", type="e", t_start=0.1, t_end=0.3),
        NodeTiming(node_id="denoise", type="d", t_start=0.5, t_end=None),
    ]
    timing = Timing(
        total_s=1.0, nodes=nodes, model_loads=[], per_step_s=None, queue_started_at=None, queue_completed_at=None
    )
    memory = build_memory(trace, timing, t0, t0 + 1.0, [], "traces/x.csv.gz")
    assert (memory.process_baseline_bytes, memory.process_peak_bytes) == (GIB, 6 * GIB)
    assert [(n.node_id, n.baseline_bytes, n.peak_bytes) for n in memory.nodes] == [
        ("encode", GIB, 4 * GIB),
        ("denoise", 2 * GIB, 6 * GIB),
    ]
    # The sample after the window does not count.
    assert memory.shared_max_delta_bytes == 400 * MIB


def test_second_runner_on_the_same_gpu_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "LOCK_DIR", tmp_path / "locks")
    adapter = Adapter(luid=0xABC, description="Fake GPU")
    with gpu_lock(adapter):
        with pytest.raises(RunnerError, match="another runner"):
            with gpu_lock(adapter):
                pass
    with gpu_lock(adapter):
        pass


def test_preflight_refuses_a_template_with_an_unfinished_model_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(server, "PORT_PROBE_TIMEOUT_S", 0.01)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    template = tmp_path / "template"
    (template / "my-models").mkdir(parents=True)
    (template / "invokeai.yaml").write_text(
        yaml.safe_dump({"schema_version": "4.0.3", "port": free, "models_dir": "my-models"})
    )
    code_dir = tmp_path / "code"
    (code_dir / "scripts").mkdir(parents=True)
    (code_dir / "scripts" / "invokeai-web.py").write_text("")
    manifest = Manifest.model_validate(
        {
            "profile": "box",
            "data_dir": str(tmp_path / "data"),
            "gpu": "Fake",
            "server": {"template_root": str(template), "python": sys.executable, "code_dir": str(code_dir)},
            "defaults": {"prompt": "a fox"},
            "families": [{"family": "f", "cases": [{"id": "a", "model": "m", "steps": 1, "sizes": [[8, 8]]}]}],
        }
    )
    gpu = Gpu(adapter=Adapter(luid=1, description="Fake"), process_usage=dict, sampler=lambda pid: None)

    (template / "my-models" / "tmpinstall_k2j3").mkdir()
    with pytest.raises(RunnerError, match="tmpinstall_k2j3.*finish or cancel the pending model install"):
        preflight(manifest, gpu)
    (template / "my-models" / "tmpinstall_k2j3").rmdir()
    preflight(manifest, gpu)
