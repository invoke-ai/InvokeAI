#!/usr/bin/env python3
"""Probe live Spark CUDA allocation in touched 1 MiB chunks, stopping at explicit safety bounds."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch

try:
    from scripts.report_spark_memory import _cgroup_memory, _windows_memory, build_report
except ImportError:
    from report_spark_memory import _cgroup_memory, _windows_memory, build_report

from invokeai.backend.util.devices import _host_memory_available, device_is_integrated

GB = 2**30
MB = 2**20
CHUNK_BYTES = MB
SAMPLE_INTERVAL_CHUNKS = 64


def _require_nvidia_cuda() -> None:
    if torch.version.cuda is None or torch.version.hip is not None:
        raise ValueError("this probe requires NVIDIA CUDA, not ROCm/HIP")


def _release_chunks(chunks: list[torch.Tensor], device: torch.device) -> str | None:
    chunks.clear()
    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    except Exception as exc:
        return f"Could not fully release CUDA allocations: {exc!r}"
    return None


def _sample(device: torch.device, allocated_bytes: int) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    raw_available = psutil.virtual_memory().available
    effective_available = _host_memory_available()
    windows = _windows_memory()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "probe_allocated_bytes": allocated_bytes,
        "cuda_free_bytes": free_bytes,
        "cuda_total_bytes": total_bytes,
        "report_process_allocated_bytes": torch.cuda.memory_allocated(device),
        "report_process_reserved_bytes": torch.cuda.memory_reserved(device),
        "host_available_bytes": raw_available,
        "effective_available_bytes": effective_available,
        "windows_commit_total_bytes": windows.get("commit_total_bytes") if windows else None,
        "windows_commit_limit_bytes": windows.get("commit_limit_bytes") if windows else None,
        "windows_memory_error": windows.get("error") if windows else None,
        "cgroup": _cgroup_memory(),
    }


def _stop_reason(sample: dict[str, Any], min_host_available: int, min_commit_free: int) -> str | None:
    if sample["effective_available_bytes"] <= min_host_available:
        return "host_available_floor"

    if sample["windows_memory_error"] is not None and min_commit_free > 0:
        return "windows_commit_unavailable"

    commit_total = sample["windows_commit_total_bytes"]
    commit_limit = sample["windows_commit_limit_bytes"]
    if commit_total is not None and commit_limit is not None and commit_limit - commit_total <= min_commit_free:
        return "windows_commit_floor"

    cgroup = sample["cgroup"]
    if cgroup is not None:
        limit, current = cgroup.get("limit"), cgroup.get("current")
        if limit not in (None, "max") and current is not None:
            try:
                if int(limit) - int(current) <= min_host_available:
                    return "cgroup_headroom_floor"
            except ValueError:
                pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-gb", required=True, type=float, help="Hard cap on additional allocation (0-128 GiB)")
    parser.add_argument("--min-host-available-gb", type=float, default=16.0)
    parser.add_argument("--min-windows-commit-free-gb", type=float, default=8.0)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Also write JSON to this path")
    args = parser.parse_args()
    if (
        not math.isfinite(args.max_gb)
        or not 0 < args.max_gb <= 128
        or not math.isfinite(args.min_host_available_gb)
        or not 0 <= args.min_host_available_gb
        or not math.isfinite(args.min_windows_commit_free_gb)
        or not 0 <= args.min_windows_commit_free_gb
    ):
        parser.error("--max-gb must be in (0, 128]; memory headroom limits must be nonnegative")
    if args.device_index < 0:
        parser.error("--device-index must be nonnegative")
    try:
        _require_nvidia_cuda()
    except ValueError as exc:
        parser.error(str(exc))
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment")

    device = torch.device(f"cuda:{args.device_index}")
    integrated = device_is_integrated(device)
    if integrated is not True:
        parser.error(f"expected integrated CUDA; is_integrated={integrated!r}")
    try:
        device_name = torch.cuda.get_device_name(device)
    except Exception:
        device_name = None

    max_bytes = int(args.max_gb * GB)
    max_chunks = max_bytes // CHUNK_BYTES
    if max_chunks == 0:
        parser.error("--max-gb must allow at least one 1 MiB allocation")
    min_host_available = int(args.min_host_available_gb * GB)
    min_commit_free = int(args.min_windows_commit_free_gb * GB)
    try:
        initial_report = build_report("before-allocation-probe")
    except Exception as exc:
        initial_report = None
        error: str | None = f"Could not capture initial report: {exc!r}"
    else:
        error = None
    chunks: list[torch.Tensor] = []
    samples: list[dict[str, Any]] = []
    stop_reason: str | None = None
    cleanup_error: str | None = None
    chunk: torch.Tensor | None = None
    started = time.monotonic()

    before_cleanup: dict[str, Any] | None = None
    after_cleanup: dict[str, Any] | None = None
    held_bytes = 0
    try:
        try:
            initial_sample = _sample(device, 0)
            samples.append(initial_sample)
            stop_reason = _stop_reason(initial_sample, min_host_available, min_commit_free)
            for index in range(max_chunks):
                safety_sample = _sample(device, len(chunks) * CHUNK_BYTES)
                stop_reason = _stop_reason(safety_sample, min_host_available, min_commit_free)
                if stop_reason is not None:
                    if not samples or samples[-1] is not safety_sample:
                        samples.append(safety_sample)
                    break
                chunk = torch.empty(CHUNK_BYTES, dtype=torch.uint8, device=device)
                chunk.zero_()
                chunks.append(chunk)
                chunk = None

                if (index + 1) % SAMPLE_INTERVAL_CHUNKS == 0:
                    torch.cuda.synchronize(device)
                    sample = _sample(device, len(chunks) * CHUNK_BYTES)
                    samples.append(sample)
                    stop_reason = _stop_reason(sample, min_host_available, min_commit_free)
                    print(f"Allocated {sample['probe_allocated_bytes'] / GB:.2f} GiB", file=sys.stderr, flush=True)

            if stop_reason is None:
                torch.cuda.synchronize(device)
                sample = _sample(device, len(chunks) * CHUNK_BYTES)
                samples.append(sample)
                stop_reason = _stop_reason(sample, min_host_available, min_commit_free) or "max_additional_reached"
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError as exc:
            stop_reason = "cuda_out_of_memory"
            error = str(exc)
            try:
                samples.append(_sample(device, len(chunks) * CHUNK_BYTES))
            except Exception:
                pass
        except KeyboardInterrupt:
            stop_reason = "interrupted"
        except Exception as exc:
            stop_reason = "probe_error"
            error = repr(exc)

        held_bytes = len(chunks) * CHUNK_BYTES
        chunk = None
        try:
            before_cleanup = build_report("allocation-probe-held")
        except KeyboardInterrupt:
            stop_reason = "interrupted"
        except Exception as exc:
            cleanup_error = f"Could not capture held-allocation report: {exc!r}"
    finally:
        chunk = None
        try:
            cleanup_error = _release_chunks(chunks, device) or cleanup_error
        except KeyboardInterrupt:
            stop_reason = "interrupted"
            cleanup_error = "Interrupted while releasing CUDA allocations"
            chunk = None
            chunks.clear()
            try:
                retry_error = _release_chunks(chunks, device)
            except KeyboardInterrupt:
                retry_error = "Interrupted again while releasing CUDA allocations"
            cleanup_error = retry_error or cleanup_error
        try:
            after_cleanup = build_report("after-allocation-probe")
        except KeyboardInterrupt:
            stop_reason = "interrupted"
            cleanup_error = "Interrupted while capturing cleanup report"
            after_cleanup = None
        except Exception as exc:
            cleanup_error = f"Could not capture cleanup report: {exc!r}"

    report = {
        "test": "spark_gpu_allocation_probe",
        "host": platform.node(),
        "platform": platform.platform(),
        "device": str(device),
        "device_name": device_name,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "chunk_bytes": CHUNK_BYTES,
        "max_additional_bytes": max_bytes,
        "min_host_available_bytes": min_host_available,
        "min_windows_commit_free_bytes": min_commit_free,
        "initial_report": initial_report,
        "samples": samples,
        "allocated_bytes_before_cleanup": held_bytes,
        "stop_reason": stop_reason,
        "error": error,
        "elapsed_seconds": time.monotonic() - started,
        "before_cleanup_report": before_cleanup,
        "after_cleanup_report": after_cleanup,
        "cleanup_error": cleanup_error,
        "note": "This bounded probe touches and holds 1 MiB CUDA allocations. It checks host and commit headroom before each allocation but stores output samples every 64 MiB and at the final stop; allocator rounding and concurrent changes can still move counters between checks. A safety stop gives a lower bound, not the hardware maximum.",
    }
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(output, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    return 1 if stop_reason == "probe_error" or cleanup_error else 0


if __name__ == "__main__":
    sys.exit(main())
