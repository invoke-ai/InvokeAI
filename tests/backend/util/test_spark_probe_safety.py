import json
import subprocess
from types import SimpleNamespace

import pytest
import torch

from scripts import probe_spark_gpu_allocation as probe
from scripts import report_spark_memory as report


def test_nvidia_cuda_gate_rejects_rocm_and_non_cuda(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        probe._require_nvidia_cuda()

    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.version, "hip", "6.3")
    with pytest.raises(ValueError, match="NVIDIA CUDA"):
        probe._require_nvidia_cuda()


def test_nvidia_cuda_gate_accepts_cuda(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.version, "hip", None)
    probe._require_nvidia_cuda()


def test_maximum_allocation_limit_is_rejected_before_cuda(monkeypatch):
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: pytest.fail("CUDA should not be checked"))
    monkeypatch.setattr(probe.sys, "argv", ["probe", "--max-gb", "65"])
    with pytest.raises(SystemExit):
        probe.main()


def test_report_exposes_effective_host_availability(monkeypatch):
    monkeypatch.setattr(report, "_host_memory_available", lambda: 321)
    monkeypatch.setattr(
        report.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(
            total=1000,
            available=900,
            free=800,
            used=100,
            percent=10,
        ),
    )
    monkeypatch.setattr(report.psutil, "swap_memory", lambda: SimpleNamespace(total=0, used=0, free=0))
    monkeypatch.setattr(report.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(report, "_nvidia_smi", lambda: {})
    result = report.build_report(None)
    assert result["system_memory"]["effective_available_bytes"] == 321


def test_nvidia_smi_keeps_snapshots_when_process_query_times_out(monkeypatch):
    monkeypatch.setattr(report.shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def run(command, **_kwargs):
        arguments = command[1:]
        if "--query-compute-apps=pid,process_name,used_memory" in arguments:
            raise subprocess.TimeoutExpired("nvidia-smi", 5)
        return SimpleNamespace(stdout="snapshot", stderr="", returncode=0)

    monkeypatch.setattr(report.subprocess, "run", run)
    result = report._nvidia_smi()
    assert result["gpu_memory_csv"] == "snapshot"
    assert result["full_output"] == "snapshot"
    assert result["processes_csv"] is None
    assert "timed out" in result["processes_error"]


def test_stop_reason_uses_effective_host_floor_and_windows_error():
    sample = {
        "host_available_bytes": 99,
        "effective_available_bytes": 99,
        "windows_memory_error": None,
        "windows_commit_total_bytes": None,
        "windows_commit_limit_bytes": None,
        "cgroup": None,
    }
    assert probe._stop_reason(sample, 100, 0) == "host_available_floor"
    sample["host_available_bytes"] = 1000
    sample["effective_available_bytes"] = 1000
    sample["windows_memory_error"] = "unavailable"
    assert probe._stop_reason(sample, 100, 1) == "windows_commit_unavailable"

    sample["windows_memory_error"] = None
    sample["windows_commit_total_bytes"] = 1000
    sample["windows_commit_limit_bytes"] = 2000
    assert probe._stop_reason(sample, 100, 1001) == "windows_commit_floor"


def test_windows_memory_converts_performance_counters(monkeypatch):
    monkeypatch.setattr(report.os, "name", "nt")

    class Api:
        @staticmethod
        def GetPerformanceInfo(pointer, _size):
            info = pointer._obj
            info.commit_total = 2
            info.commit_limit = 3
            info.commit_peak = 4
            info.physical_total = 5
            info.physical_available = 6
            info.page_size = 4096
            return 1

    monkeypatch.setattr(report.ctypes, "WinDLL", lambda *_args, **_kwargs: Api(), raising=False)
    result = report._windows_memory()
    assert result == {
        "commit_total_bytes": 8192,
        "commit_limit_bytes": 12288,
        "commit_peak_bytes": 16384,
        "physical_total_bytes": 20480,
        "physical_available_bytes": 24576,
    }


def test_probe_cleanup_preserves_oom_output(monkeypatch):
    chunks = [object()]
    released = []
    monkeypatch.setattr(probe.gc, "collect", lambda: released.append("gc"))
    monkeypatch.setattr(probe.torch.cuda, "empty_cache", lambda: released.append("empty"))
    probe._release_chunks(chunks, torch.device("cuda:0"))
    assert chunks == []
    assert released == ["gc", "empty"]


def test_probe_cleanup_returns_no_error_when_release_succeeds(monkeypatch):
    monkeypatch.setattr(probe.gc, "collect", lambda: None)
    monkeypatch.setattr(probe.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(probe.torch.cuda, "synchronize", lambda _: None)
    assert probe._release_chunks([], torch.device("cuda:0")) is None


def test_probe_reports_after_oom_and_releases_allocations(monkeypatch, capsys):
    monkeypatch.setattr(probe.torch.version, "cuda", "12.8")
    monkeypatch.setattr(probe.torch.version, "hip", None)
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(probe, "device_is_integrated", lambda _: True)
    monkeypatch.setattr(probe.torch.cuda, "get_device_name", lambda _: "test")
    monkeypatch.setattr(probe.torch.cuda, "mem_get_info", lambda _: (1000, 2000))
    monkeypatch.setattr(probe.torch.cuda, "memory_allocated", lambda _: 0)
    monkeypatch.setattr(probe.torch.cuda, "memory_reserved", lambda _: 0)
    monkeypatch.setattr(probe.torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(probe.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        probe.torch, "empty", lambda *args, **kwargs: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError("oom"))
    )
    monkeypatch.setattr(probe, "build_report", lambda label: {"label": label})
    monkeypatch.setattr(probe, "_windows_memory", lambda: None)
    monkeypatch.setattr(probe, "_host_memory_available", lambda: 10**12)
    monkeypatch.setattr(probe, "device_is_integrated", lambda _: True)
    monkeypatch.setattr(probe.sys, "argv", ["probe", "--max-gb", "1", "--min-host-available-gb", "0"])

    assert probe.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["stop_reason"] == "cuda_out_of_memory"
    assert output["cleanup_error"] is None
    assert output["after_cleanup_report"] == {"label": "after-allocation-probe"}


def test_probe_handles_interrupt_during_held_report(monkeypatch, capsys):
    monkeypatch.setattr(probe.torch.version, "cuda", "12.8")
    monkeypatch.setattr(probe.torch.version, "hip", None)
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(probe, "device_is_integrated", lambda _: True)
    monkeypatch.setattr(probe.torch.cuda, "get_device_name", lambda _: "test")
    monkeypatch.setattr(probe.torch.cuda, "mem_get_info", lambda _: (1000, 2000))
    monkeypatch.setattr(probe.torch.cuda, "memory_allocated", lambda _: 0)
    monkeypatch.setattr(probe.torch.cuda, "memory_reserved", lambda _: 0)
    monkeypatch.setattr(probe.torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(probe.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(probe.torch, "empty", lambda *args, **kwargs: torch.tensor([1]))
    monkeypatch.setattr(probe, "_windows_memory", lambda: None)
    monkeypatch.setattr(probe, "_host_memory_available", lambda: 10**12)
    monkeypatch.setattr(probe.sys, "argv", ["probe", "--max-gb", "0.001", "--min-host-available-gb", "0"])

    def report(label):
        if label == "allocation-probe-held":
            raise KeyboardInterrupt
        return {"label": label}

    monkeypatch.setattr(probe, "build_report", report)
    assert probe.main() == 0
    output = json.loads(capsys.readouterr().out)
    assert output["stop_reason"] == "interrupted"
    assert output["after_cleanup_report"] == {"label": "after-allocation-probe"}


def test_probe_handles_interrupt_during_cleanup(monkeypatch, capsys):
    monkeypatch.setattr(probe.torch.version, "cuda", "12.8")
    monkeypatch.setattr(probe.torch.version, "hip", None)
    monkeypatch.setattr(probe.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(probe, "device_is_integrated", lambda _: True)
    monkeypatch.setattr(probe.torch.cuda, "get_device_name", lambda _: "test")
    monkeypatch.setattr(probe.torch.cuda, "mem_get_info", lambda _: (1000, 2000))
    monkeypatch.setattr(probe.torch.cuda, "memory_allocated", lambda _: 0)
    monkeypatch.setattr(probe.torch.cuda, "memory_reserved", lambda _: 0)
    monkeypatch.setattr(probe.torch.cuda, "synchronize", lambda _: None)
    empty_calls = 0

    def empty_cache():
        nonlocal empty_calls
        empty_calls += 1
        if empty_calls == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(probe.torch.cuda, "empty_cache", empty_cache)
    monkeypatch.setattr(probe.torch, "empty", lambda *args, **kwargs: torch.tensor([1]))
    monkeypatch.setattr(probe, "_windows_memory", lambda: None)
    monkeypatch.setattr(probe, "_host_memory_available", lambda: 10**12)
    monkeypatch.setattr(probe.sys, "argv", ["probe", "--max-gb", "0.001", "--min-host-available-gb", "0"])
    monkeypatch.setattr(probe, "build_report", lambda label: {"label": label})

    assert probe.main() == 1
    output = json.loads(capsys.readouterr().out)
    assert output["stop_reason"] == "interrupted"
    assert output["after_cleanup_report"] == {"label": "after-allocation-probe"}
    assert empty_calls == 2
