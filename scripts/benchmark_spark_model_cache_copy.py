#!/usr/bin/env python3
"""Measure retained CPU-weight copies versus move-only cache switching on Spark.

This is a bounded synthetic benchmark. It downloads no models and starts no server.
Run it on each Spark target; it is intentionally not a production performance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

MODEL_WIDTH = 8192
MEASURED_SWITCHES = 8
GB = 2**30
MB = 2**20


def _memory_snapshot(torch: Any, psutil: Any, device: Any, TorchDevice: Any, wddm: Any) -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    local_memory = wddm.local_video_memory(device)
    return {
        "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
        "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
        "host_available_bytes": psutil.virtual_memory().available,
        "process_rss_bytes": process.memory_info().rss,
        "memory_available_to_process_bytes": TorchDevice.memory_available_to_process(device),
        "wddm_local_video_memory": list(local_memory) if local_memory is not None else None,
        "wddm_process_local_usage_bytes": wddm.process_local_usage(device),
        "wddm_process_paged_bytes": wddm.process_paged_bytes(device),
    }


def _run_child(repo_root: Path, device_index: int, width: int, switches: int, keep_ram_copy: bool) -> dict[str, Any]:
    sys.path.insert(0, str(repo_root))
    import psutil
    import torch

    from invokeai.backend.model_manager.load.model_cache.model_cache import ModelCache
    from invokeai.backend.util import wddm
    from invokeai.backend.util.devices import TorchDevice, device_has_dedicated_vram, device_is_integrated

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    device = torch.device(f"cuda:{device_index}")
    properties = torch.cuda.get_device_properties(device)
    integrated = device_is_integrated(device)
    dedicated = device_has_dedicated_vram(device)
    if integrated is not True or dedicated is not False:
        raise RuntimeError(f"expected integrated CUDA; is_integrated={integrated!r}, dedicated={dedicated!r}")

    model_bytes = width * width * 4
    preflight = TorchDevice.memory_available_to_process(device)
    minimum_headroom = 3 * model_bytes + 512 * MB
    if preflight < minimum_headroom:
        raise RuntimeError(f"need {minimum_headroom} bytes of headroom; observed {preflight}")

    cache = ModelCache(
        execution_device_working_mem_gb=0.0,
        enable_partial_loading=True,
        keep_ram_copy_of_weights=keep_ram_copy,
        max_ram_cache_size_gb=4.0,
        max_vram_cache_size_gb=(1.5 * model_bytes) / GB,
        execution_device=device,
        storage_device="cpu",
        shared_cpu_weights=None,
        reserve_vram_gb=0.0,
    )
    records: dict[str, Any] = {}
    failures: list[str] = []
    try:
        for index, name in enumerate(("synthetic-a", "synthetic-b")):
            torch.manual_seed(4200 + index)
            model = torch.nn.Linear(width, width, bias=False, dtype=torch.float32).eval()
            cache.put(name, model)
            records[name] = cache.get(name)

        after_put = _memory_snapshot(torch, psutil, device, TorchDevice, wddm)
        input_tensor = torch.randn(1, width, generator=torch.Generator().manual_seed(7)).to(device)
        torch.cuda.reset_peak_memory_stats(device)
        output_hashes: dict[str, str] = {}

        def lock_and_run(name: str) -> tuple[float, dict[str, Any]]:
            record = records[name]
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            settled = cache.lock(record, working_mem_bytes=0)
            try:
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                if not settled:
                    failures.append(f"lock for {name} did not settle")
                with torch.inference_mode():
                    output = record.cached_model.model(input_tensor).detach().cpu()
                torch.cuda.synchronize(device)
                output_hash = hashlib.sha256(output.contiguous().numpy().tobytes()).hexdigest()
                previous_hash = output_hashes.get(name)
                if previous_hash is not None and previous_hash != output_hash:
                    failures.append(f"output changed across switches for {name}")
                output_hashes[name] = output_hash
                target_resident = record.cached_model.cur_vram_bytes()
                other_name = "synthetic-b" if name == "synthetic-a" else "synthetic-a"
                other_resident = records[other_name].cached_model.cur_vram_bytes()
                if target_resident != model_bytes:
                    failures.append(f"{name} resident bytes {target_resident}, expected {model_bytes}")
                if other_resident != 0:
                    failures.append(f"{other_name} stayed resident at {other_resident} bytes")
                snapshot = _memory_snapshot(torch, psutil, device, TorchDevice, wddm)
                snapshot.update(
                    {
                        "target": name,
                        "target_resident_bytes": target_resident,
                        "other_resident_bytes": other_resident,
                        "lock_seconds": elapsed,
                    }
                )
                return elapsed, snapshot
            finally:
                cache.unlock(record)

        # Warm-up makes the measured sequence represent switching, not first CUDA setup.
        lock_and_run("synthetic-a")
        lock_and_run("synthetic-b")
        measurements = [lock_and_run("synthetic-a" if index % 2 == 0 else "synthetic-b") for index in range(switches)]
        after_switch = _memory_snapshot(torch, psutil, device, TorchDevice, wddm)
        copy_flags = {name: record.cached_model.get_cpu_state_dict() is not None for name, record in records.items()}
        expected_flag = keep_ram_copy
        if set(copy_flags.values()) != {expected_flag}:
            failures.append(f"CPU copy flags {copy_flags} did not match requested {expected_flag}")
        times = [item[0] for item in measurements]
        per_model = {
            name: {
                "model_bytes": record.cached_model.total_bytes(),
                "resident_bytes": record.cached_model.cur_vram_bytes(),
                "cpu_state_dict_retained": copy_flags[name],
            }
            for name, record in records.items()
        }
        return {
            "keep_ram_copy": keep_ram_copy,
            "device_name": properties.name,
            "is_integrated": integrated,
            "device_has_dedicated_vram": dedicated,
            "preflight_memory_available_to_process_bytes": preflight,
            "after_put": after_put,
            "after_switch": after_switch,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "switch_lock_seconds": times,
            "median_switch_lock_seconds": statistics.median(times),
            "measurements": [item[1] for item in measurements],
            "models": per_model,
            "failures": failures,
            "output_sha256": output_hashes,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }
    finally:
        cache.shutdown()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _child_main(args: argparse.Namespace, repo_root: Path) -> int:
    report = _run_child(repo_root, args.device_index, args.width, args.switches, args.keep_ram_copy == "true")
    args.report_file.write_text(json.dumps(report, sort_keys=True) + "\n")
    return 0 if not report["failures"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=MODEL_WIDTH)
    parser.add_argument("--switches", type=int, default=MEASURED_SWITCHES)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--keep-ram-copy", choices=("true", "false"), help=argparse.SUPPRESS)
    parser.add_argument("--_report-file", dest="report_file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    if not (repo_root / "invokeai/backend/util/devices.py").is_file():
        parser.error("--repo-root must point to the InvokeAI source checkout")
    if args.width < 256 or args.width > 16384 or args.switches < 4 or args.switches > 32:
        parser.error("--width must be between 256 and 16384 and --switches between 4 and 32")
    if args._child:
        if args.keep_ram_copy is None or args.report_file is None:
            parser.error("child mode requires --keep-ram-copy and --_report-file")
        return _child_main(args, repo_root)

    children: dict[str, dict[str, Any]] = {}
    for mode in ("true", "false"):
        report_handle = tempfile.NamedTemporaryFile(prefix=f"spark-copy-timing-{mode}-", suffix=".json", delete=False)
        report_path = Path(report_handle.name)
        report_handle.close()
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_child",
            "--keep-ram-copy",
            mode,
            "--repo-root",
            str(repo_root),
            "--device-index",
            str(args.device_index),
            "--width",
            str(args.width),
            "--switches",
            str(args.switches),
            "--_report-file",
            str(report_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        try:
            children[mode] = json.loads(report_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"keep_ram_copy={mode} failed ({completed.returncode}):\n{completed.stderr}{completed.stdout}"
            ) from exc
        finally:
            report_path.unlink(missing_ok=True)
        if completed.returncode != 0 or children[mode]["failures"]:
            raise RuntimeError(f"keep_ram_copy={mode} reported failures: {children[mode]['failures']}")

    retained = children["true"]
    move_only = children["false"]
    retained_delta = retained["after_switch"]["host_available_bytes"] - retained["after_put"]["host_available_bytes"]
    move_only_delta = move_only["after_switch"]["host_available_bytes"] - move_only["after_put"]["host_available_bytes"]
    retained_time = retained["median_switch_lock_seconds"]
    move_only_time = move_only["median_switch_lock_seconds"]
    output_hashes_match = retained["output_sha256"] == move_only["output_sha256"]
    failures = [] if output_hashes_match else ["inference output hashes differ between copy modes"]
    report = {
        "test": "spark_model_cache_copy_timing",
        "pass": not failures,
        "failures": failures,
        "repo_root": str(repo_root),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": retained["torch"],
        "cuda_runtime": retained["cuda_runtime"],
        "device": f"cuda:{args.device_index}",
        "device_name": retained["device_name"],
        "is_integrated": retained["is_integrated"],
        "device_has_dedicated_vram": retained["device_has_dedicated_vram"],
        "synthetic_model": {
            "type": "torch.nn.Linear",
            "count": 2,
            "width": args.width,
            "bytes_each": args.width * args.width * 4,
        },
        "switches": args.switches,
        "modes": children,
        "comparison": {
            "retained_copy_host_available_delta_bytes": retained_delta,
            "move_only_host_available_delta_bytes": move_only_delta,
            "estimated_retained_copy_extra_host_bytes": move_only_delta - retained_delta,
            "retained_copy_median_switch_seconds": retained_time,
            "move_only_median_switch_seconds": move_only_time,
            "move_only_over_retained_switch_time_ratio": move_only_time / retained_time if retained_time else None,
            "output_sha256_matches": output_hashes_match,
        },
        "caveat": "Synthetic switching does not measure real model loading, LoRA patching, or generation.",
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
