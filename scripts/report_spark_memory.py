#!/usr/bin/env python3
"""Report host and CUDA memory counters without loading a model or allocating test tensors."""

from __future__ import annotations

import argparse
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


def _nvidia_smi() -> dict[str, str | int | None]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": 0, "gpu_memory_csv": None, "error": "nvidia-smi not found"}
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": 1, "gpu_memory_csv": None, "error": str(exc)}
    return {
        "available": 1,
        "gpu_memory_csv": result.stdout.strip() or None,
        "gpu_memory_fields": "name,memory.total,memory.used,memory.free (MiB)",
        "error": result.stderr.strip() or None if result.returncode else None,
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
            "free_bytes": memory.free,
            "used_bytes": memory.used,
            "percent_used": memory.percent,
            "swap_total_bytes": swap.total,
            "swap_used_bytes": swap.used,
            "swap_free_bytes": swap.free,
            "cgroup": _cgroup_memory(),
        },
        "report_process_rss_bytes": psutil.Process(os.getpid()).memory_info().rss,
        "cuda_devices": _cuda_devices(),
        "nvidia_smi": _nvidia_smi(),
        "note": "No model tensors are allocated; CUDA queries may create a context. Process counters describe this reporter, not InvokeAI.",
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
