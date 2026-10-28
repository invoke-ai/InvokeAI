"""The plumbing every fresh-process calibration script shares: one measuring process per point, and the lines it makes.

A calibration script (`scripts/calibrate_z_image_working_memory.py`, `scripts/calibrate_flux2_denoise_working_memory.py`)
plans its points, builds its model and inputs, and summarizes; this module runs the rest. The driving process runs each
point as `<script> --single <request>`, a fresh process (`run_point`), which applies the server's runtime before torch is
imported (`apply_server_runtime`), measures the point with the script's own function and prints one `RESULT_PREFIX`
line (`measure_in_this_process`). The driving process prints a table and appends every measured point to the point file
`scripts/working_memory/build_shipped.py --points` reads (`measure_points`, `file_record`).

A line is a JSON object: `status` (`point`, `invalid`, `skipped` or `failed`), the script's `fit` and `point` (the grid
entry), and for a measured point the fields of a `MeasuredPoint` and `raw` (each window's counter deltas, the prediction
and the runtime facts). Which of a point's windows decides its bytes is `measurement.measured_bytes_and_method`'s. A
point is `in_fit` where it is a fit point and its estimator's line admits its features (`fit.Line.fits`): a fit-grid
forward with a term on top of the line, such as the score matrix a build's SDPA kernel materializes, measures that term
rather than the line, as a check does.

Nothing here imports torch at import time: a measuring process has to configure the allocator first.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn, Protocol

from invokeai.app.util.rocm_aotriton import AOTRITON_EXPERIMENTAL_ENV
from invokeai.app.util.torch_cuda_allocator import ALLOCATOR_CONF_ENV_VARS

if TYPE_CHECKING:
    import torch

    from invokeai.backend.working_memory.measurement import FreshWindow

GIB = 2**30
RESULT_PREFIX = "RESULT "
"""What starts the one line of stdout a measuring process reports its result on."""
POINT_TIMEOUT_S = 1800
"""How long one measuring process may take. Generous: it only keeps a driver that hangs from stalling a whole run."""


class Point(Protocol):
    """A script's grid entry: a frozen dataclass that says whether it is a point of the fit grid."""

    @property
    def fit(self) -> bool: ...


def point_line(
    point: Point,
    *,
    common: Mapping[str, Any],
    provenance: str,
    windows: Sequence["FreshWindow"],
    raw: Mapping[str, Any],
) -> dict[str, Any]:
    """The output line of a measured point: a `point`, or `invalid` with the reasons of its windows.

    `common` holds the `MeasuredPoint` fields the measurement does not decide: estimator, version, platform, features
    and device. `windows` are the point's fresh windows, cold first; `raw` the point's other facts.
    """
    from invokeai.backend.working_memory.measurement import measured_bytes_and_method, point_reasons
    from invokeai.backend.working_memory.spec import import_estimator_modules

    reasons = point_reasons(windows)
    line = {"status": "invalid" if reasons else "point", "fit": point.fit, "point": dataclasses.asdict(point), **common}
    raw = {**raw, **dict(zip(("cold", "warm"), map(dataclasses.asdict, windows), strict=False))}
    if reasons:
        return line | {"reasons": reasons, "raw": raw}
    measured, method = measured_bytes_and_method(windows)
    on_line = import_estimator_modules()[common["estimator"]].line(common["features"])
    return line | {
        "measured_bytes": measured,
        "method": method,
        "provenance": provenance,
        "in_fit": point.fit and on_line is not None and on_line.fits,
        "raw": raw,
    }


def skipped_line(point: Point, common: Mapping[str, Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    """The line of a point predicted not to fit free device memory; `raw` carries the prediction."""
    return {"status": "skipped", "fit": point.fit, "point": dataclasses.asdict(point), **common, "raw": raw}


def failed_line(point: Point, error: str) -> dict[str, Any]:
    return {"status": "failed", "fit": point.fit, "point": dataclasses.asdict(point), "error": error}


def file_record(line: Mapping[str, Any]) -> dict[str, Any] | None:
    """What the point file receives of a line: a measured point, without the status only the table reads. None for a
    forward that is no point: the data builder refuses a point file record that is no `MeasuredPoint`."""
    if line["status"] != "point":
        return None
    return {key: value for key, value in line.items() if key != "status"}


def alloc_conf(facts: Mapping[str, Any]) -> str:
    return ", ".join(f"{var}={value}" for var, value in facts["alloc_conf"].items()) or "no allocator configuration"


def gib(value: int | None) -> str:
    return "" if value is None else f"{value / GIB:.3f}"


def runtime_provenance(facts: Mapping[str, Any]) -> str:
    """The runtime part of a point's provenance: the GPU, the torch build, the OS, the allocator configuration, and on
    ROCm `MIOPEN_FIND_MODE`, the AOTriton switch and the SDPA guard."""
    build = f"torch {facts['torch']}" + (f" (HIP {facts['hip']})" if facts["hip"] else "")
    parts = [
        facts["device"],
        build,
        facts["os"].capitalize(),
        f"{alloc_conf(facts)} (torch reports the {facts['allocator_backend']} allocator)",
    ]
    if facts["hip"]:
        parts += [
            f"MIOPEN_FIND_MODE={facts['miopen_find_mode'] or 'unset'}",
            f"{AOTRITON_EXPERIMENTAL_ENV}={facts['aotriton_experimental'] or 'unset'}",
            f"ROCm SDPA guard {'installed' if facts['rocm_sdpa_guard'] else 'not installed'}",
        ]
    return ", ".join(parts)


def runtime_header(facts: Mapping[str, Any], platform: Mapping[str, str]) -> str:
    rocm = ""
    if facts["hip"]:
        guard = "installed" if facts["rocm_sdpa_guard"] else "NOT installed"
        rocm = f" | {AOTRITON_EXPERIMENTAL_ENV}={facts['aotriton_experimental'] or 'unset'} | SDPA guard {guard}"
    return (
        f"{facts['device']} | torch {facts['torch']} | {'/'.join(platform.values())} | {alloc_conf(facts)}{rocm} | "
        f"estimator from {facts['estimator_module']}"
    )


# The measuring process. Everything below imports torch, so it runs only after `apply_server_runtime`.


def apply_server_runtime(alloc_conf: str, logger_name: str) -> None:
    """Apply the server's runtime (`run_app.apply_server_runtime`) before torch is imported, with the settings of a
    server without a config file and the allocator configuration this run measures: `alloc_conf`, or none for the
    platform's default."""
    from invokeai.app.run_app import apply_server_runtime as apply
    from invokeai.app.services.config.config_default import get_config
    from invokeai.backend.util.logging import InvokeAILogger

    config = get_config().model_copy(update={"pytorch_cuda_alloc_conf": alloc_conf or None})
    apply(config, InvokeAILogger.get_logger(logger_name, config=config))


def runtime_facts(device: "torch.device", os_name: str, estimator_file: str) -> dict[str, Any]:
    """The runtime a point is measured in, as its provenance and `raw` record it. `estimator_file` is the `__file__` of
    the estimator module the point was priced with: the checkout it came from. (A module's file, not the package's:
    `invokeai` can be imported as a namespace package, whose `__file__` is None.)"""
    import torch

    from invokeai.backend.util.attention import rocm_sdpa_chunks_math
    from invokeai.backend.working_memory.counters import backend_for, method_verified

    name = torch.cuda.get_device_name(device)
    if torch.version.hip is not None:
        name = f"{name} ({torch.cuda.get_device_properties(device).gcnArchName.split(':')[0]})"
    backend = backend_for(device)
    return {
        "device": name,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "os": os_name,
        "allocator_backend": torch.cuda.get_allocator_backend(),
        "alloc_conf": {var: os.environ[var] for var in ALLOCATOR_CONF_ENV_VARS if os.environ.get(var)},
        "miopen_find_mode": os.environ.get("MIOPEN_FIND_MODE"),
        "aotriton_experimental": os.environ.get(AOTRITON_EXPERIMENTAL_ENV),
        "rocm_sdpa_guard": rocm_sdpa_chunks_math(device.type),
        "driver_method_verified": backend is not None and method_verified(backend),
        "estimator_module": str(Path(estimator_file).resolve()),
    }


def measure_in_this_process(
    request: str, *, point_type: Callable[..., Point], measure: Callable[[Any, bool], dict[str, Any]], logger_name: str
) -> NoReturn:
    """Measure one point in this fresh process with the script's `measure(point, force)` and print its line.

    Leaves without Python's shutdown, whose HIP teardown can hang on Windows while its DLLs unload (see
    `invokeai/app/util/rocm_integrated_gpu.py`)."""
    spec = json.loads(request)
    try:
        apply_server_runtime(spec["alloc_conf"], logger_name)
        line = measure(point_type(**spec["point"]), spec["force"])
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    print(RESULT_PREFIX + json.dumps(line), flush=True)
    sys.stderr.flush()
    os._exit(0)


# The driving process: measures each point in a fresh process, appends the lines, prints the table.


def run_point(script: Path, point: Point, alloc_conf: str, force: bool) -> dict[str, Any]:
    """The line of `point`, measured by `script --single` in a fresh process."""
    request = json.dumps({"point": dataclasses.asdict(point), "alloc_conf": alloc_conf, "force": force})
    command = [sys.executable, str(script), "--single", request]
    try:
        done = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=POINT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return failed_line(point, f"no result within {POINT_TIMEOUT_S} s")
    for output in done.stdout.splitlines():
        if output.startswith(RESULT_PREFIX):
            return json.loads(output[len(RESULT_PREFIX) :])
    tail = done.stderr.strip().splitlines()[-1:] or ["no output"]
    return failed_line(point, f"exit code {done.returncode}: {tail[0]}")


def row(line: Mapping[str, Any], head: str) -> str:
    """A line's row of the table, after the script's own `head` columns: the demand and its windows, or why not."""
    status = line["status"]
    raw = line.get("raw", {})
    cold, warm = (gib(raw[run]["demand"]) if run in raw else "" for run in ("cold", "warm"))
    if status == "point":
        return f"{head} {gib(line['measured_bytes']):>10} {cold:>8} {warm:>8}  {line['method']}"
    if status == "invalid":
        return f"{head} {'invalid':>10} {cold:>8} {warm:>8}  {', '.join(line['reasons'])}"
    if status == "skipped":
        return (
            f"{head} {'skipped':>10}  needs ~{raw['predicted_bytes'] / GIB:.1f} GiB, {raw['free_bytes'] / GIB:.1f} GiB "
            "free (--force runs it)"
        )
    return f"{head} {'FAILED':>10}  {line['error']}"


def check_alloc_conf(parser: argparse.ArgumentParser, alloc_conf: str) -> None:
    """Refuse `--alloc-conf` under an exported allocator variable: the server keeps an exported value and only warns,
    and a measurement must not run another allocator than asked."""
    exported = [var for var in ALLOCATOR_CONF_ENV_VARS if os.environ.get(var)]
    if alloc_conf and exported:
        parser.error(f"--alloc-conf cannot apply while {', '.join(exported)} is set in the environment")


def measure_points(
    points: Sequence[Point],
    *,
    script: Path,
    out: Path,
    alloc_conf: str,
    force: bool,
    logger_name: str,
    columns: str,
    head: Callable[[Mapping[str, Any]], str],
) -> list[dict[str, Any]]:
    """Measure `points` one fresh process each, append the measured ones to `out` and print their rows under the
    script's `columns`, each row starting with `head(line)`. Returns the lines."""
    from invokeai.app.util.rocm_integrated_gpu import hide_integrated_gpus_on_rocm_windows
    from invokeai.backend.util.logging import InvokeAILogger

    # The runtime's integrated-GPU check, once for the whole run, before torch is imported anywhere: the measuring
    # processes inherit the HIP_VISIBLE_DEVICES it sets instead of each asking HIP in a child process of its own.
    hide_integrated_gpus_on_rocm_windows(InvokeAILogger.get_logger(logger_name))

    print(f"{len(points)} forwards, one fresh process each")
    lines: list[dict[str, Any]] = []
    with out.open("a", encoding="utf-8") as file:
        for point in points:
            line = run_point(script, point, alloc_conf, force)
            if "raw" in line and not any("raw" in earlier for earlier in lines):
                print(runtime_header(line["raw"]["runtime"], line["platform"]))
            if not lines:
                print(f"{columns} {'demand GiB':>10} {'cold':>8} {'warm':>8}")
            record = file_record(line)
            if record is not None:
                file.write(json.dumps(record) + "\n")
                file.flush()
            print(row(line, head(line)), flush=True)
            lines.append(line)
    return lines


def print_counts(lines: Sequence[Mapping[str, Any]], out: Path) -> None:
    statuses = [line["status"] for line in lines]
    counts = ", ".join(f"{statuses.count(status)} {status}" for status in ("point", "invalid", "skipped", "failed"))
    print(f"{counts}; the points appended to {out}")
