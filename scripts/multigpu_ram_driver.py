#!/usr/bin/env python
"""Driver to exercise the multi-GPU shared-RAM model cache under real, concurrent generations.

It repeatedly enqueues N batches at once (so the multi-GPU session processor runs them in parallel
across devices), polls the queue until each round drains, and samples the InvokeAI server process's
RAM (RSS) the whole time. It then reports:

  - baseline (idle) RSS,
  - peak RSS during generation (this is the text/reference-encode spike you care about), and
  - idle RSS after each round -> a leak verdict (does RAM return to baseline, or creep up?).

This automates the two manual checks from the test plan:
  #1 "dual concurrent encode RAM"  -> run with --rounds 1 --pairs <#gpus> and read the peak.
  #5 "leak check over many gens"   -> run with --rounds 25+ and read the idle drift.

------------------------------------------------------------------------------------------------
Getting a batch file
------------------------------------------------------------------------------------------------
The script needs the exact body InvokeAI's UI sends to enqueue a generation. Easiest way to capture
it:
  1. Open InvokeAI in the browser with devtools -> Network open.
  2. Click Invoke once.
  3. Find the POST to `.../queue/default/enqueue_batch`, copy its JSON request body, save to a file
     (e.g. batch.json). It looks like {"prepend": false, "batch": {"graph": {...}, "runs": 1}}.

The script bust the node cache by default (sets use_cache=false on every node and randomizes any
"seed" fields) so every submission actually runs the model instead of returning a cached result.

------------------------------------------------------------------------------------------------
Examples
------------------------------------------------------------------------------------------------
  # Headline dual-GPU encode RAM (2 GPUs -> 2 concurrent jobs), one round:
  python scripts/multigpu_ram_driver.py --graph batch.json --pairs 2 --rounds 1

  # Leak soak: 30 rounds of 2 concurrent jobs, save timeline for plotting:
  python scripts/multigpu_ram_driver.py --graph batch.json --pairs 2 --rounds 30 --csv ram.csv

  # If PID auto-detection fails, point it at the server explicitly:
  python scripts/multigpu_ram_driver.py --graph batch.json --pid 12345
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

import psutil


# --------------------------------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# --------------------------------------------------------------------------------------------------
def _request(method: str, url: str, body: dict | None = None, timeout: float = 60.0) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method=method
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise SystemExit(f"HTTP {e.code} on {method} {url}\n{detail}") from e
    except urllib.error.URLError as e:
        raise SystemExit(f"Could not reach {url}: {e.reason}. Is the server running?") from e


def enqueue(base: str, queue_id: str, body: dict) -> dict:
    return _request("POST", f"{base}/api/v1/queue/{queue_id}/enqueue_batch", body)


def queue_counts(base: str, queue_id: str) -> tuple[int, int]:
    """Return (pending, in_progress), searching the response defensively for those keys."""
    resp = _request("GET", f"{base}/api/v1/queue/{queue_id}/status")
    # The status payload nests the queue counts under "queue"; fall back to top-level.
    node = resp.get("queue", resp) if isinstance(resp, dict) else {}
    return int(node.get("pending", 0)), int(node.get("in_progress", 0))


# --------------------------------------------------------------------------------------------------
# Batch preparation
# --------------------------------------------------------------------------------------------------
def normalize_body(loaded: dict) -> dict:
    """Accept either the full {"prepend":..., "batch": {...}} body or a bare Batch ({"graph":...})."""
    if "batch" in loaded:
        return copy.deepcopy(loaded)
    if "graph" in loaded:
        return {"prepend": False, "batch": copy.deepcopy(loaded)}
    raise SystemExit("Batch file must contain either a top-level 'batch' or 'graph' key.")


def bust_cache(body: dict, mutate_seed: bool, disable_cache: bool) -> dict:
    """Return a copy of the body with the node cache busted so the submission really computes."""
    body = copy.deepcopy(body)
    nodes = body.get("batch", {}).get("graph", {}).get("nodes", {})
    if not isinstance(nodes, dict):
        return body
    for node in nodes.values():
        if not isinstance(node, dict):
            continue
        if disable_cache:
            node["use_cache"] = False
        if mutate_seed and "seed" in node:
            node["seed"] = random.randint(0, 2**31 - 1)
    return body


# --------------------------------------------------------------------------------------------------
# Process discovery + RSS sampling
# --------------------------------------------------------------------------------------------------
def find_server_pid(port: int) -> int:
    """Best-effort: find the PID listening on `port`, else a process whose cmdline looks like the server."""
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr and conn.laddr.port == port and conn.pid:
            return conn.pid
    needles = ("invokeai-web", "invokeai.app.run_app", "invokeai_web", "uvicorn")
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmd = " ".join(proc.info.get("cmdline") or [])
        if any(n in cmd for n in needles):
            return proc.info["pid"]
    raise SystemExit(
        f"Could not auto-detect the InvokeAI server PID on port {port}. Pass --pid explicitly."
    )


def tree_rss(proc: psutil.Process, use_uss: bool) -> int:
    """RSS (or USS) of the process and its children, in bytes."""
    procs = [proc] + proc.children(recursive=True)
    total = 0
    for p in procs:
        try:
            if use_uss:
                total += p.memory_full_info().uss
            else:
                total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


@dataclass
class Sampler:
    proc: psutil.Process
    hz: float
    use_uss: bool
    samples: list[tuple[float, int]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        period = 1.0 / self.hz
        while not self._stop.is_set():
            self.samples.append((time.monotonic(), tree_rss(self.proc, self.use_uss)))
            time.sleep(period)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def current(self) -> int:
        return self.samples[-1][1] if self.samples else tree_rss(self.proc, self.use_uss)

    def peak_between(self, t0: float, t1: float) -> int:
        vals = [rss for t, rss in self.samples if t0 <= t <= t1]
        return max(vals) if vals else 0


# --------------------------------------------------------------------------------------------------
# Round loop
# --------------------------------------------------------------------------------------------------
GB = 1024**3


def gb(n: int) -> float:
    return n / GB


def wait_drained(base: str, queue_id: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pending, in_progress = queue_counts(base, queue_id)
        if pending == 0 and in_progress == 0:
            return
        time.sleep(0.5)
    raise SystemExit(f"Queue did not drain within {timeout}s. Aborting.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph", required=True, help="Path to a captured enqueue_batch body (JSON).")
    ap.add_argument("--url", default="http://127.0.0.1:9090", help="Server base URL.")
    ap.add_argument("--queue-id", default="default")
    ap.add_argument("--pairs", type=int, default=2, help="Concurrent batches per round (>= #GPUs).")
    ap.add_argument("--rounds", type=int, default=1, help="Number of rounds (use 25+ for leak soak).")
    ap.add_argument("--pid", type=int, default=None, help="Server PID (auto-detected if omitted).")
    ap.add_argument("--hz", type=float, default=10.0, help="RSS sampling rate.")
    ap.add_argument("--uss", action="store_true", help="Sample USS instead of RSS (more accurate, slower).")
    ap.add_argument("--settle", type=float, default=4.0, help="Seconds to wait after each round for RAM to release.")
    ap.add_argument("--timeout", type=float, default=1800.0, help="Per-round drain timeout (s).")
    ap.add_argument("--warmup", action="store_true", help="Run one un-measured round first (loads models from disk).")
    ap.add_argument("--keep-cache", action="store_true", help="Do NOT set use_cache=false on nodes.")
    ap.add_argument("--no-seed-mutate", action="store_true", help="Do NOT randomize node 'seed' fields.")
    ap.add_argument("--csv", default=None, help="Write the full (t, rss_gb) timeline here.")
    args = ap.parse_args()

    with open(args.graph) as f:
        body = normalize_body(json.load(f))

    base = args.url.rstrip("/")
    port = urllib.parse.urlparse(base).port or 9090
    pid = args.pid or find_server_pid(port)
    proc = psutil.Process(pid)
    print(f"Server PID {pid}: {' '.join(proc.cmdline()[:3])} ...")
    print(f"Metric: {'USS' if args.uss else 'RSS'} (process tree)  |  pairs/round={args.pairs}  rounds={args.rounds}")

    def submit_round() -> tuple[float, float]:
        t0 = time.monotonic()
        for _ in range(args.pairs):
            prepared = bust_cache(body, mutate_seed=not args.no_seed_mutate, disable_cache=not args.keep_cache)
            res = enqueue(base, args.queue_id, prepared)
            if res.get("enqueued", 0) < 1:
                raise SystemExit(f"Enqueue returned nothing useful: {res}")
        wait_drained(base, args.queue_id, args.timeout)
        return t0, time.monotonic()

    sampler = Sampler(proc=proc, hz=args.hz, use_uss=args.uss)
    sampler.start()
    try:
        if args.warmup:
            print("Warmup round (not measured)...")
            submit_round()
            time.sleep(args.settle)

        time.sleep(2.0)  # settle before baseline
        baseline = sampler.current()
        print(f"\nBaseline idle {('USS' if args.uss else 'RSS')}: {gb(baseline):.2f} GB\n")
        print(f"{'round':>5} {'peak_GB':>9} {'Δpeak_GB':>9} {'idle_after_GB':>14} {'Δidle_GB':>9}")

        idle_after_first = None
        overall_peak = baseline
        for r in range(1, args.rounds + 1):
            t0, t1 = submit_round()
            peak = sampler.peak_between(t0, t1)
            overall_peak = max(overall_peak, peak)
            time.sleep(args.settle)
            idle_after = sampler.current()
            if idle_after_first is None:
                idle_after_first = idle_after
            print(
                f"{r:>5} {gb(peak):>9.2f} {gb(peak - baseline):>9.2f} "
                f"{gb(idle_after):>14.2f} {gb(idle_after - baseline):>9.2f}"
            )
    finally:
        sampler.stop()

    # Summary
    idle_drift = (sampler.current() - (idle_after_first or baseline))
    print("\n--- Summary ---")
    print(f"Baseline idle:        {gb(baseline):.2f} GB")
    print(f"Overall peak:         {gb(overall_peak):.2f} GB  (Δ {gb(overall_peak - baseline):+.2f} GB over baseline)")
    print(f"Idle drift (leak):    {gb(idle_drift):+.2f} GB across {args.rounds} rounds")
    verdict = "LIKELY LEAK" if idle_drift > 0.5 * GB else "no leak detected"
    print(f"Leak verdict:         {verdict} (threshold 0.50 GB)")
    print("Interpretation: peak Δ should be ~1x the encoder size (not Nx). Idle drift should be ~0.")

    if args.csv:
        t_start = sampler.samples[0][0] if sampler.samples else 0.0
        with open(args.csv, "w") as f:
            f.write("t_seconds,rss_gb\n")
            for t, rss in sampler.samples:
                f.write(f"{t - t_start:.3f},{gb(rss):.4f}\n")
        print(f"\nTimeline written to {args.csv} ({len(sampler.samples)} samples).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
