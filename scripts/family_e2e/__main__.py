"""Command line of the family E2E tool: `python -m scripts.family_e2e <command>`.

See docs/src/content/docs/development/Guides/family-e2e.mdx.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.family_e2e.bundle import write_bundle
from scripts.family_e2e.calibration import PROBE_REASON, CalibrationError, read_store_files
from scripts.family_e2e.detach import RUNNER_LOG, DetachError, spawn_runner
from scripts.family_e2e.gpu_memory import UnsupportedPlatformError
from scripts.family_e2e.graphs import ExportError
from scripts.family_e2e.manifest import DEFAULT_ARM, Manifest, ManifestError, load_manifest
from scripts.family_e2e.metrics import MB
from scripts.family_e2e.records import RUN_INFO_FILE, load_records, load_run_info, run_id_profile, write_run_info
from scripts.family_e2e.references import AcceptError, accept
from scripts.family_e2e.report import write_report
from scripts.family_e2e.runner import (
    SERVER_ROLES,
    RunnerError,
    RunStopped,
    finish_run,
    gpu_lock,
    is_alive,
    kill_recorded,
    load_run_manifest,
    new_run,
    open_gpu,
    preflight,
    read_heartbeat,
    remaining_steps,
    run,
    run_lock,
    save_calibration,
    scratch_root,
    wait_for_runner,
)
from scripts.family_e2e.server import ServerError, remove_tree

logger = logging.getLogger("family_e2e")
# Windows refuses to open heartbeat.json for a moment while the runner replaces it.
STOP_HEARTBEAT_WAIT_S = 1.0
STOP_HEARTBEAT_POLL_S = 0.02
EXPECTED_ERRORS = (
    ManifestError,
    RunnerError,
    ServerError,
    ExportError,
    AcceptError,
    DetachError,
    UnsupportedPlatformError,
    CalibrationError,
    # load_records names the malformed line of a damaged records.jsonl.
    ValueError,
)


class UsageError(ValueError):
    pass


def _data_dir(run_dir: Path) -> Path:
    # <data_dir>/runs/<run_id>
    return run_dir.parent.parent


def _resolve_run(arg: str | None, manifest_path: Path | None) -> Path:
    """A run directory, a run id of the manifest's data dir, or (no argument) the manifest profile's latest run."""
    if arg is not None and Path(arg).is_dir():
        return Path(arg).resolve()
    if manifest_path is None:
        raise UsageError("pass a run directory, or --manifest to look up run ids and the latest run")
    manifest = load_manifest(manifest_path)
    runs_dir = manifest.data_dir / "runs"
    if arg is not None:
        if not (runs_dir / arg / RUN_INFO_FILE).is_file():
            raise UsageError(f"no run {arg} in {runs_dir}")
        return runs_dir / arg
    runs = sorted(
        d.name
        for d in runs_dir.glob("*")
        if run_id_profile(d.name) == manifest.profile and (d / RUN_INFO_FILE).is_file()
    )
    if not runs:
        raise UsageError(f"no runs of profile {manifest.profile} in {runs_dir}")
    return runs_dir / runs[-1]


def _check_startable(manifest: Manifest) -> None:
    gpu = open_gpu(manifest.gpu)
    with gpu_lock(gpu.adapter):
        preflight(manifest, gpu)


def _print_started(run_dir: Path, pid: int) -> None:
    print(f"run     {run_dir.name}")
    print(f"dir     {run_dir}")
    print(f"runner  pid {pid}, log {run_dir / RUNNER_LOG}")
    print(f"report  {run_dir / 'report.html'} (when finished)")
    print(f"status  python -m scripts.family_e2e status {run_dir}")


def _start_runner(run_dir: Path) -> int:
    """Starts the runner in the background and returns its pid once it registered: from then on `stop` finds it."""
    previous = read_heartbeat(run_dir).get("runner")
    return int(wait_for_runner(run_dir, spawn_runner(run_dir), previous)[0])


def cmd_start(args: argparse.Namespace) -> int:
    _check_startable(load_manifest(args.manifest))
    run_dir = new_run(args.manifest, args.family, args.case, calibrate=args.calibrate)
    _print_started(run_dir, _start_runner(run_dir))
    return 0


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def cmd_run(args: argparse.Namespace) -> int:
    # A plain kill (SIGTERM) would end the runner without its cleanup, and on Linux the server, in a session of its
    # own, would keep running. As an interrupt it stops the server, removes the scratch root and marks the run stopped.
    signal.signal(signal.SIGTERM, _interrupt)
    try:
        return run(args.run_dir.resolve())
    except RunStopped as e:
        logger.info("%s", e)
        return 130
    except KeyboardInterrupt:
        return 130
    except Exception:
        logger.exception("the runner failed")
        return 2


def cmd_resume(args: argparse.Namespace) -> int:
    run_dir = _resolve_run(args.run, args.manifest)
    if is_alive(read_heartbeat(run_dir).get("runner")):
        raise UsageError(f"the runner of {run_dir.name} is still running")
    _check_startable(load_run_manifest(load_run_info(run_dir)))
    with run_lock(run_dir):
        info = load_run_info(run_dir)
        write_run_info(run_dir, info.model_copy(update={"status": "running", "finished_at": None}))
    _print_started(run_dir, _start_runner(run_dir))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    run_dir = _resolve_run(args.run, args.manifest)
    info = load_run_info(run_dir)
    records = load_records(run_dir)
    left = remaining_steps(info.plan, records)
    heartbeat = read_heartbeat(run_dir)
    alive = is_alive(heartbeat.get("runner"))
    print(f"run       {info.run_id} ({info.status})")
    print(f"dir       {run_dir}")
    print(f"progress  {len(info.plan) - len(left)}/{len(info.plan)} steps final, {len(records)} records")
    # Every step of a calibration run carries the probe's reason; only the others make a record invalid there.
    invalid = sum(1 for r in records if set(r.invalid_reasons) - {PROBE_REASON})
    print(f"invalid   {invalid} records with invalid reasons")
    print(f"errors    {sum(1 for r in records if r.status != 'completed')} records not completed")
    if info.calibrate:
        print(
            f"calibrate time and VRAM not rated; {len(info.calibration_exports)} store exports in {run_dir / 'calibration'}"
        )
    if heartbeat:
        updated = datetime.fromisoformat(heartbeat["updated_at"])
        age = (datetime.now(timezone.utc) - updated).total_seconds()
        state = "alive" if alive else "gone"
        print(f"runner    {state}, step: {heartbeat.get('step')}, heartbeat {age:.0f}s ago")
    print(f"log       {run_dir / RUNNER_LOG}")
    if (run_dir / "report.html").is_file():
        print(f"report    {run_dir / 'report.html'}")
    if info.status == "running" and not alive:
        print(f"the runner is gone; continue with: python -m scripts.family_e2e resume {run_dir}")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    run_dir = _resolve_run(args.run, args.manifest)
    # Marked before the heartbeat is read, under the run lock a runner registers under: a runner registered before is
    # in the heartbeat and is killed, one that registers after finds the run stopped and leaves it untouched.
    was_running = finish_run(run_dir, "stopped")
    heartbeat = read_heartbeat(run_dir)
    # A running run has a registered runner, so an empty read is most likely one that met the runner's replace.
    deadline = time.monotonic() + STOP_HEARTBEAT_WAIT_S
    while was_running and not heartbeat and time.monotonic() < deadline:
        time.sleep(STOP_HEARTBEAT_POLL_S)
        heartbeat = read_heartbeat(run_dir)
    # The runner first: it would otherwise notice its server dying and record a crash.
    killed, survivors = kill_recorded(heartbeat, ("runner", *SERVER_ROLES))
    if survivors:
        # What still runs may still write to its scratch root and the calibration store in it.
        raise RunnerError(
            f"the run's processes survived the kill: {survivors}. {run_dir.name} is marked stopped; its scratch roots "
            "stay as they are. End them, then `stop` again"
        )
    for role in killed:
        print(f"killed {role} (pid {int(heartbeat[role][0])}) and its descendants")
    info = load_run_info(run_dir)
    unsaved = []
    for arm in sorted({DEFAULT_ARM, *info.environments}):
        root = scratch_root(_data_dir(run_dir), info.run_id, arm)
        if info.calibrate:
            # The killed server left its calibration store in the scratch root; the root goes next.
            try:
                exports = save_calibration(run_dir, read_store_files(root))
            except OSError as e:
                # The root holds the only copy of the lifetime's windows; `resume` saves it first.
                unsaved.append(f"could not save the calibration store ({e}); kept the scratch root {root} with it")
                continue
            for export in exports:
                print(f"saved the calibration store {run_dir / export.export}")
        # Nothing of the run is alive any more, so its scratch roots are unused.
        for leftover in remove_tree(root):
            print(f"could not remove {leftover}")
    print(f"{info.run_id} stopped; continue with: python -m scripts.family_e2e resume {run_dir}")
    for message in unsaved:
        print(f"error: {message}. Fix the cause and `resume`, which saves it first", file=sys.stderr)
    return 2 if unsaved else 0


def cmd_accept(args: argparse.Namespace) -> int:
    run_dir = _resolve_run(args.run, args.manifest)
    for reference in accept(_data_dir(run_dir), run_dir, args.case or None):
        print(f"accepted {reference.case_key} from {reference.source_run_id}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = _resolve_run(args.run, args.manifest)
    result = write_report(_data_dir(run_dir), run_dir)
    print(f"report   {result.report_html}")
    print(f"summary  {result.summary_md}")
    print(
        f"{result.failures} failures, {result.outliers} outliers, {result.invalid} invalid, "
        f"{result.changed} changed as expected, {result.baseline} baseline"
    )
    return 1 if result.failures else 0


def cmd_bundle(args: argparse.Namespace) -> int:
    run_dir = _resolve_run(args.run, args.manifest)
    if is_alive(read_heartbeat(run_dir).get("runner")):
        raise UsageError(f"the runner of {run_dir.name} is still running; bundle the run once it finished or stopped")
    output = (args.output or run_dir / f"{run_dir.name}.zip").resolve()
    size = write_bundle(_data_dir(run_dir), run_dir, output, args.manifest)
    print(f"bundle  {output}")
    print(f"size    {size / MB:.1f} MB")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.family_e2e", description="End-to-end runs per model family."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start", help="create a run and start its runner in the background")
    start.add_argument("--manifest", type=Path, required=True)
    start.add_argument("--family", action="append", default=[], help="only this family (repeatable)")
    start.add_argument("--case", action="append", default=[], help="only this case id (repeatable)")
    start.add_argument(
        "--calibrate",
        action="store_true",
        help="measure working memory: submit every step through the server's calibrate route and save its exports",
    )
    start.set_defaults(handler=cmd_start)

    runner = commands.add_parser("run", help="run the remaining steps of a run in the foreground (what start launches)")
    runner.add_argument("--run-dir", type=Path, required=True)
    runner.add_argument("--log-file", type=Path, help="append stdout and stderr to this file")
    runner.set_defaults(handler=cmd_run)

    for name, handler, help_text in (
        ("resume", cmd_resume, "start the runner of a stopped or crashed run again"),
        ("status", cmd_status, "show progress and whether the runner is alive"),
        ("stop", cmd_stop, "kill the runner and its server"),
        ("accept", cmd_accept, "make a run's measured results the references of its profile"),
        ("report", cmd_report, "write report.html and summary.md of a run again"),
        ("bundle", cmd_bundle, "pack a run into a zip without this machine's paths, to attach elsewhere"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("run", nargs="?", help="run directory or run id; default: the profile's latest run")
        command.add_argument("--manifest", type=Path, help="resolves run ids and the latest run")
        if name == "accept":
            command.add_argument("--case", action="append", default=[], help="only this case key (repeatable)")
        if name == "bundle":
            command.add_argument("--output", type=Path, help="the zip to write; default: <run dir>/<run id>.zip")
        command.set_defaults(handler=handler)
    return parser


def _redirect_stdio(path: Path) -> None:
    # A background runner has no usable stdio; tracebacks of a crash must land in its log too.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    sys.stdout = open(1, "w", encoding="utf-8", buffering=1, closefd=False)
    sys.stderr = open(2, "w", encoding="utf-8", buffering=1, closefd=False)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if getattr(args, "log_file", None):
        _redirect_stdio(args.log_file)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    try:
        return args.handler(args)
    except (UsageError, *EXPECTED_ERRORS) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
