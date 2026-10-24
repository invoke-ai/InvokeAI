"""A whole run against a fake server launched the way InvokeAI is launched, with a fake GPU and canned graphs."""

import hashlib
import json
import os
import shutil
import socket
import sqlite3
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from PIL import Image

from scripts.family_e2e import client, runner, server
from scripts.family_e2e.gpu_memory import Adapter, MemoryTrace, ProcessUsage
from scripts.family_e2e.records import ModelIdentity, load_records, load_run_info, write_run_info

GIB = 2**30
CASE_KEY = "fake-cn@64x48"
FOREIGN_PID = 4_000_000
MODEL = {"key": "fake-model", "hash": "h", "name": "Fake", "base": "fake", "type": "main"}


class FakeSampler:
    """Memory that rises from 1 GiB to 3 GiB as the run enqueues its item, and falls back when sampling stops.

    The runner reads its enqueue instant just before the enqueue call, and its finish instant from events the server
    sends only after that call, so the rise lies in the measured window however slow the machine is. A rise placed
    by the clock alone, e.g. halfway between start and stop, misses it when the runner's tail outlasts the fake run.
    """

    def __init__(self) -> None:
        self.dedicated: list[tuple[float, int]] = []

    def start(self) -> None:
        self.dedicated = [(time.perf_counter(), GIB)]

    def enqueued(self) -> None:
        self.dedicated.append((time.perf_counter(), 3 * GIB))

    def stop(self) -> MemoryTrace:
        t1 = time.perf_counter()
        return MemoryTrace(
            method="fake",
            interval_ms=3.0,
            dedicated=(*self.dedicated, (t1, GIB)),
            shared=((self.dedicated[0][0], 0), (t1, 0)),
        )


class FakeGpu:
    """Reports the sampled server process on the adapter once a sampler watched it.

    A foreign process shows up at the end of the fourth sampled run.
    """

    def __init__(self) -> None:
        self.server_pid: int | None = None
        self.adapter = Adapter(luid=0x1234, description="Fake GPU")
        self.samplers = 0
        self.foreign_pending = False
        self.current: FakeSampler | None = None

    def watch_enqueue(self, enqueue: Callable[..., int]) -> Callable[..., int]:
        """Wraps `InvokeClient.enqueue` so the current sampler sees the instant the run submits its item."""

        def enqueue_and_rise(invoke_client: client.InvokeClient, batch: dict) -> int:
            assert self.current is not None
            self.current.enqueued()
            return enqueue(invoke_client, batch)

        return enqueue_and_rise

    def process_usage(self) -> dict[int, ProcessUsage]:
        usage = {} if self.server_pid is None else {self.server_pid: ProcessUsage(dedicated_bytes=GIB, shared_bytes=0)}
        if self.foreign_pending:
            self.foreign_pending = False
            usage[FOREIGN_PID] = ProcessUsage(dedicated_bytes=2 * GIB, shared_bytes=0)
        return usage

    def sampler(self, pid: int) -> FakeSampler:
        self.server_pid = pid
        self.samplers += 1
        self.foreign_pending = self.samplers == 4
        self.current = FakeSampler()
        return self.current

    def gpu(self) -> runner.Gpu:
        return runner.Gpu(adapter=self.adapter, process_usage=self.process_usage, sampler=self.sampler)


def _graph() -> dict:
    return {
        "id": "fake-graph",
        "nodes": {
            "loader": {"id": "loader", "type": "fake_loader", "is_intermediate": True, "model": MODEL},
            "denoise": {"id": "denoise", "type": "fake_denoise", "is_intermediate": True, "steps": 4},
            "output": {"id": "output", "type": "fake_l2i", "is_intermediate": False, "width": 64, "height": 48},
        },
        "edges": [],
    }


@pytest.fixture
def isolated_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(server, "STARTUP_POLL_S", 0.02)
    monkeypatch.setattr(server, "STARTUP_TIMEOUT_S", 60.0)
    monkeypatch.setattr(client, "POLL_S", 0.05)
    monkeypatch.setattr(runner, "LOCK_DIR", tmp_path / "locks")


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, Path]:
    code_dir = tmp_path / "code"
    (code_dir / "invokeai").mkdir(parents=True)
    (code_dir / "invokeai" / "__init__.py").write_text("")
    (code_dir / "scripts").mkdir()
    shutil.copy(Path(__file__).parent / "fake_invokeai_server.py", code_dir / "scripts" / "invokeai-web.py")

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        template_port = s.getsockname()[1]
    template = tmp_path / "template"
    (template / "databases").mkdir(parents=True)
    sqlite3.connect(template / "databases" / "invokeai.db").close()
    (template / "invokeai.yaml").write_text(yaml.safe_dump({"schema_version": "4.0.3", "port": template_port}))

    data_dir = tmp_path / "data"
    (data_dir / "inputs").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(data_dir / "inputs" / "fox.png")
    manifest = {
        "profile": "fake",
        "data_dir": "data",
        "gpu": "Fake GPU",
        "server": {"template_root": "template", "python": sys.executable, "code_dir": "code"},
        "defaults": {"warmup": 1, "repeats": 2, "prompt": "a fox"},
        "families": [
            {
                "family": "fake",
                "cases": [
                    {
                        "id": "fake-cn",
                        "model": "fake-model",
                        "mode": "controlnet",
                        "steps": 4,
                        "sizes": [[64, 48]],
                        "reference_images": ["fox.png"],
                        "control": [{"model": "fake-controlnet", "image": "fox.png"}],
                    }
                ],
            }
        ],
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))
    return manifest_path, data_dir


def test_run_survives_a_server_crash_retries_foreign_load_and_resumes(
    setup: tuple[Path, Path], isolated_runner: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, data_dir = setup
    # The first server dies in its second queue item, the first measure.
    (manifest_path.parent / "code" / "crash-at-enqueue").write_text("2")
    gpu = FakeGpu()
    monkeypatch.setattr(client.InvokeClient, "enqueue", gpu.watch_enqueue(client.InvokeClient.enqueue))
    requests: list[dict] = []

    def export(request: dict, work_dir: Path) -> dict:
        requests.append(request)
        batch = {"graph": _graph(), "data": [], "origin": "test", "destination": "test"}
        return {"version": 1, "cases": [{"id": request["cases"][0]["id"], "batch": batch, "components": {}}]}

    run_dir = runner.new_run(manifest_path, [], [])
    assert runner.run(run_dir, gpu_factory=lambda _: gpu.gpu(), export=export) == 0

    case = requests[0]["cases"][0]
    assert (case["id"], case["width"], case["height"], case["seed"]) == (CASE_KEY, 64, 48, 1234)
    # Reference images go as they are; a control image is resized to the case size first, as the canvas would.
    assert case["reference_images"] == ["upload-1.png"]
    assert case["control"] == [
        {"model_key": "fake-controlnet", "image_name": "upload-2.png", "weight": 1.0, "begin": 0.0, "end": 1.0}
    ]
    with Image.open(run_dir / "inputs" / "fox@64x48.png") as resized:
        assert resized.size == (64, 48)
    assert [model["key"] for model in requests[0]["models"]] == ["fake-model"]

    records = load_records(run_dir)
    assert [(r.kind, r.rep, r.attempt) for r in records] == [
        ("warmup", 0, 1),
        ("measure", 0, 1),
        ("warmup", 0, 2),
        ("measure", 0, 2),
        ("measure", 0, 3),
        ("measure", 1, 1),
    ]
    # The crashed step has its error record; the new server is warmed up again before the step is repeated.
    crashed = records.pop(1)
    assert crashed.status == "error"
    # Without os.waitid (macOS) the code of the unreaped server is not to be had before it is reaped.
    code = "3" if sys.platform == "win32" or hasattr(os, "waitid") else "unknown"
    assert crashed.invalid_reasons[0].startswith(f"server crashed (exit code {code})"), crashed.invalid_reasons
    assert len(requests) == 2  # each server's uploads need their own export
    # Foreign GPU load spoils time and memory but not the image, and the step is repeated.
    spoiled = records.pop(2)
    assert (spoiled.valid_timing, spoiled.valid_memory, spoiled.valid_psnr) == (False, False, True)
    assert spoiled.invalid_reasons == [f"foreign GPU load: pid {FOREIGN_PID} (?) 2048 MiB"]
    assert [(p.pid, p.dedicated_bytes) for p in spoiled.memory.foreign] == [(FOREIGN_PID, 2 * GIB)]
    for record in records:
        assert record.status == "completed", record.error
        assert (record.valid_timing, record.valid_memory, record.valid_psnr) == (True, True, True), (
            record.invalid_reasons
        )
        assert (record.image_width, record.image_height) == (64, 48)
        image = (run_dir / record.image_path).read_bytes()
        assert hashlib.sha256(image).hexdigest() == record.image_sha256
        assert record.memory.process_baseline_bytes == GIB
        assert record.memory.process_peak_bytes == 3 * GIB
        assert record.memory.foreign == []
        assert (run_dir / record.memory.trace).is_file()
        assert [n.node_id for n in record.timing.nodes] == ["loader", "denoise", "output"]
        assert all(0 <= n.t_start <= n.t_end <= record.timing.total_s for n in record.timing.nodes)
        assert record.timing.per_step_s is not None and record.timing.per_step_s > 0
        assert [(r.model, r.submodel, r.percent) for r in record.residency] == [("fake-model", "transformer", 100.0)]
    # Each server's warm-up loads the model; the measures stay valid on the residency seen since the cache flush.
    assert [len(r.timing.model_loads) for r in records] == [1, 1, 0, 0]

    info = load_run_info(run_dir)
    assert info.status == "completed"
    environment = info.environments["default"]
    assert Path(environment.invokeai_file).is_relative_to(manifest_path.parent / "code")
    assert environment.app_version == {"version": "0.0.0-fake"}
    saved = json.loads((run_dir / "graphs" / f"{CASE_KEY}.json").read_text())
    assert all(node["use_cache"] is False for node in saved["graph"]["nodes"].values())
    assert info.graph_hashes[CASE_KEY] == records[0].graph_hash
    # What names the model elsewhere; neither its path nor a source URL that carries a token.
    assert info.case_models == {
        CASE_KEY: [
            ModelIdentity(
                key="fake-model",
                name="Fake",
                base="fake",
                type="main",
                format="checkpoint",
                variant=None,
                hash="blake3:fakehash",
                source=None,
            )
        ]
    }
    assert not (data_dir / "scratch-roots" / f"{info.run_id}-default").exists()
    assert (run_dir / "report.html").is_file()

    # A runner killed before the last measure: the resumed run warms its new server up before that measure.
    lines = (run_dir / "records.jsonl").read_text().splitlines(keepends=True)
    (run_dir / "records.jsonl").write_text("".join(lines[:5]))
    write_run_info(run_dir, info.model_copy(update={"status": "running"}))
    assert runner.run(run_dir, gpu_factory=lambda _: gpu.gpu(), export=export) == 0

    resumed = load_records(run_dir)[5:]
    assert [(r.kind, r.rep, r.attempt) for r in resumed] == [("warmup", 0, 3), ("measure", 1, 1)]
    assert all(r.valid_timing and r.valid_memory for r in resumed)
    assert resumed[0].graph_hash == records[0].graph_hash
    assert runner.remaining_steps(info.plan, load_records(run_dir)) == []
