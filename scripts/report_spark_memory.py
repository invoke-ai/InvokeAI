#!/usr/bin/env python3
"""Report host and CUDA memory counters without loading a model or allocating test tensors."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil
import torch

from invokeai.backend.util.devices import _host_memory_available


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _cgroup_memory() -> dict[str, str | None] | None:
    membership = _read_text(Path("/proc/self/cgroup"))
    if membership is None:
        return None

    for line in membership.splitlines():
        hierarchy, controllers, group = line.split(":", 2)
        if hierarchy == "0" and not controllers:
            root = Path("/sys/fs/cgroup") / group.lstrip("/")
            values = {
                "limit": _read_text(root / "memory.max"),
                "current": _read_text(root / "memory.current"),
                "high": _read_text(root / "memory.high"),
            }
            if any(value is not None for value in values.values()):
                return {"version": "v2", "membership": group, **values}
        if "memory" in controllers.split(","):
            root = Path("/sys/fs/cgroup/memory") / group.lstrip("/")
            values = {
                "limit": _read_text(root / "memory.limit_in_bytes"),
                "current": _read_text(root / "memory.usage_in_bytes"),
            }
            if any(value is not None for value in values.values()):
                return {"version": "v1", "membership": group, **values}
    return None


def _cuda_devices() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []

    from invokeai.backend.util import wddm
    from invokeai.backend.util.devices import device_has_dedicated_vram, device_is_integrated

    devices = []
    for index in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{index}")
        properties = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        local_video_memory = wddm.local_video_memory(device)
        devices.append(
            {
                "device": str(device),
                "name": properties.name,
                "is_integrated": device_is_integrated(device),
                "has_dedicated_vram": device_has_dedicated_vram(device),
                "cuda_total_bytes": total_bytes,
                "cuda_free_bytes": free_bytes,
                "report_process_allocated_bytes": torch.cuda.memory_allocated(device),
                "report_process_reserved_bytes": torch.cuda.memory_reserved(device),
                "wddm_local_budget_and_usage_bytes": list(local_video_memory) if local_video_memory else None,
                "wddm_report_process_local_usage_bytes": wddm.process_local_usage(device),
                "wddm_report_process_paged_bytes": wddm.process_paged_bytes(device),
            }
        )
    return devices


def _windows_memory() -> dict[str, int | str] | None:
    if os.name != "nt":
        return None

    class PerformanceInformation(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("commit_total", ctypes.c_size_t),
            ("commit_limit", ctypes.c_size_t),
            ("commit_peak", ctypes.c_size_t),
            ("physical_total", ctypes.c_size_t),
            ("physical_available", ctypes.c_size_t),
            ("system_cache", ctypes.c_size_t),
            ("kernel_total", ctypes.c_size_t),
            ("kernel_paged", ctypes.c_size_t),
            ("kernel_nonpaged", ctypes.c_size_t),
            ("page_size", ctypes.c_size_t),
            ("handle_count", ctypes.c_uint32),
            ("process_count", ctypes.c_uint32),
            ("thread_count", ctypes.c_uint32),
        ]

    try:
        info = PerformanceInformation()
        info.cb = ctypes.sizeof(info)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_performance_info = psapi.GetPerformanceInfo
        get_performance_info.argtypes = [ctypes.POINTER(PerformanceInformation), ctypes.c_uint32]
        get_performance_info.restype = ctypes.c_int
        if not get_performance_info(ctypes.byref(info), info.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            "commit_total_bytes": info.commit_total * info.page_size,
            "commit_limit_bytes": info.commit_limit * info.page_size,
            "commit_peak_bytes": info.commit_peak * info.page_size,
            "physical_total_bytes": info.physical_total * info.page_size,
            "physical_available_bytes": info.physical_available * info.page_size,
        }
    except (AttributeError, OSError) as exc:
        return {"error": str(exc)}


def _run_nvidia_smi(executable: str, *arguments: str) -> dict[str, str | None]:
    try:
        result = subprocess.run(
            [executable, *arguments],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"output": None, "error": str(exc)}
    return {
        "output": result.stdout.strip() or None,
        "error": result.stderr.strip() or None if result.returncode else None,
    }


def _nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": 0, "gpu_memory_csv": None, "processes_csv": None, "full_output": None}
    gpu = _run_nvidia_smi(
        executable, "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"
    )
    processes = _run_nvidia_smi(
        executable, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"
    )
    full = _run_nvidia_smi(executable)
    return {
        "available": 1,
        "gpu_memory_csv": gpu["output"],
        "gpu_memory_fields": "name,memory.total,memory.used,memory.free (MiB)",
        "processes_csv": processes["output"],
        "full_output": full["output"],
        "gpu_memory_error": gpu["error"],
        "processes_error": processes["error"],
        "full_output_error": full["error"],
    }


def build_report(label: str | None) -> dict[str, Any]:
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "test": "spark_memory_inventory",
        "label": label,
        "host": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "system_memory": {
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "effective_available_bytes": _host_memory_available(),
            "free_bytes": memory.free,
            "used_bytes": memory.used,
            "percent_used": memory.percent,
            "swap_total_bytes": swap.total,
            "swap_used_bytes": swap.used,
            "swap_free_bytes": swap.free,
            "cgroup": _cgroup_memory(),
        },
        "windows_memory": _windows_memory(),
        "report_process_rss_bytes": psutil.Process(os.getpid()).memory_info().rss,
        "cuda_devices": _cuda_devices(),
        "nvidia_smi": _nvidia_smi(),
        "note": "No model tensors are allocated; CUDA queries may create a context. Torch/WDDM process counters describe this reporter; nvidia-smi and Windows commit counters are system-wide.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", help="Snapshot label, e.g. idle, invoke-idle, or llm-loaded")
    parser.add_argument("--output", type=Path, help="Also write JSON to this path")
    args = parser.parse_args()

    report = json.dumps(build_report(args.label), indent=2, sort_keys=True) + "\n"
    print(report, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
