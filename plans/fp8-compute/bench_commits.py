"""Benchmark commit/config variants of a real Krea-2 graph through the actual server.

Per variant: checkout, start a server on a private port against the user's data root but with its
OWN config file (their invokeai.yaml is never touched), enqueue the graph N times, parse the
per-node timings out of the server log, shut down. The working branch is always restored.

Three things this gets wrong if you are not careful, all learned the hard way:

* `uv run` is a wrapper — terminating it orphans the actual server, which then keeps serving the
  port and every later enqueue silently goes to the WRONG commit's code. Launch the venv python
  directly so the subprocess *is* the server, and verify the port it actually bound.
* Config keys that a commit predates are a hard validation error, so the config has to be built
  from the keys that commit's `config_default.py` really defines.
* Identical graphs hit the invocation cache, so runs 2..N report ~0.001s per node. `node_cache_size: 0`
  disables it; without that you measure the cache, not the model.
"""

import argparse
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(r"D:\Entwicklung\InvokeAI")
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
DATA_ROOT = Path(r"D:\Entwicklung\InvokeAI_data2")
DB = DATA_ROOT / "databases" / "invokeai.db"
SCRATCH = Path(__file__).parent
PORT = 9393
BASE_URL = f"http://127.0.0.1:{PORT}"

# Mirrors the user's live settings. Emitted only where the commit defines the key.
WANTED_CONFIG = {
    "host": "127.0.0.1",
    "port": PORT,
    "max_cache_ram_gb": 40,
    "device_working_mem_gb": 3,
    "fp8_compute": True,
    "pytorch_cuda_alloc_conf": '"backend:cudaMallocAsync"',
    "node_cache_size": 0,  # must be 0 or repeated identical graphs measure the cache
}

TIMING_RE = re.compile(r"^\s*(\w+)\s+(\d+)\s+([\d.]+)s\s", re.M)
GRAPH_TOTAL_RE = re.compile(r"TOTAL GRAPH EXECUTION TIME:\s+([\d.]+)s")
RUNNING_RE = re.compile(r"Invoke running on http://[\d.]+:(\d+)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout


def read_log(path: Path) -> str:
    return ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace").replace("\x00", ""))


def commit_defines(commit: str, key: str) -> bool:
    src = git("show", f"{commit}:invokeai/app/services/config/config_default.py")
    return bool(re.search(rf"^\s{{4}}{re.escape(key)}\s*:", src, re.M))


def build_config(
    commit: str, hints: bool | None, partial: bool | None, fp8: bool | None = None
) -> tuple[str, list[str]]:
    lines, skipped = ["schema_version: 4.0.3"], []
    extras = {"fp8_compute_full_precision_hints": hints, "enable_partial_loading": partial}
    if fp8 is not None:
        extras["fp8_compute"] = fp8
    for key, value in {**WANTED_CONFIG, **{k: v for k, v in extras.items() if v is not None}}.items():
        if commit_defines(commit, key):
            lines.append(f"{key}: {json.dumps(value) if isinstance(value, bool) else value}")
        else:
            skipped.append(key)
    return "\n".join(lines) + "\n", skipped


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill the server and everything it spawned.

    `.venv\\Scripts\\python.exe` re-execs into the real interpreter, so the process we hold a handle
    to is only a launcher: `proc.terminate()` leaves the actual server alive, still holding the port.
    The next variant then either aborts or — worse — silently talks to the previous commit's code.
    `taskkill /T` takes the whole tree.
    """
    if proc.poll() is not None:
        return
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def port_free() -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) != 0


def load_graph(item_id: int) -> dict:
    con = sqlite3.connect(DB)
    (sess,) = con.execute("SELECT session FROM session_queue WHERE item_id = ?", (item_id,)).fetchone()
    con.close()
    graph = json.loads(sess)["graph"]
    graph.pop("id", None)
    return graph


def wait_for_server(proc: subprocess.Popen, log_path: Path, timeout: int = 420) -> bool:
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"    server exited early (code {proc.returncode}):")
            for line in read_log(log_path).splitlines()[-8:]:
                print(f"      {line}")
            return False
        try:
            with urllib.request.urlopen(f"{BASE_URL}/api/v1/app/version", timeout=2) as r:
                if r.status != 200:
                    continue
        except Exception:
            time.sleep(2)
            continue
        # It answers on our port -- but make sure it is OUR process, not a leftover.
        bound = RUNNING_RE.search(read_log(log_path))
        if not bound:
            time.sleep(1)
            continue
        if int(bound.group(1)) != PORT:
            print(f"    ABORT: server bound port {bound.group(1)}, not {PORT} — something else holds it")
            return False
        return True
    print("    timeout waiting for server")
    return False


def enqueue(graph: dict, runs: int) -> bool:
    import urllib.request

    body = json.dumps({"batch": {"graph": graph, "runs": runs}, "prepend": False}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/api/v1/queue/default/enqueue_batch", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            print(f"    enqueued {json.loads(r.read()).get('enqueued')} item(s)")
        return True
    except Exception as e:
        print(f"    enqueue failed: {e}")
        return False


def wait_for_queue(timeout: int = 1800) -> None:
    """Poll until the queue drains.

    A status poll that times out does NOT mean the run failed. Loading a scaled-fp8 checkpoint with
    `fp8_compute: false` dequantizes to bf16 and re-quantizes on the CPU — minutes of blocking work
    during which the server's event loop does not service HTTP at all. Treat a failed poll as "still
    busy" and keep waiting; only the outer deadline may end the wait.
    """
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/api/v1/queue/default/status", timeout=30) as r:
                q = json.loads(r.read())["queue"]
        except Exception:
            time.sleep(5)
            continue
        if q["pending"] == 0 and q["in_progress"] == 0:
            if q.get("failed"):
                print(f"    WARNING: {q['failed']} failed item(s) in queue")
            return
        time.sleep(3)
    print("    timeout waiting for queue to drain")


def parse_timings(log_text: str) -> list[dict]:
    out = []
    for block in log_text.split("Graph stats:")[1:]:
        block = block.split("RAM cache statistics")[0]
        nodes = {name: float(sec) for name, _calls, sec in TIMING_RE.findall(block)}
        if nodes:
            total = GRAPH_TOTAL_RE.search(block)
            nodes["_total"] = float(total.group(1)) if total else 0.0
            out.append(nodes)
    return out


WORKTREE = "WORKTREE"  # pseudo-commit: measure the current working tree, uncommitted changes included


def bench(
    commit: str, hints: bool | None, partial: bool | None, graph: dict, runs: int, tag: str, fp8: bool | None = None
) -> list[dict]:
    print(f"\n=== {tag}")
    if not port_free():
        print(f"    ABORT: port {PORT} is already in use")
        return []
    if commit == WORKTREE:
        print("    using the current working tree (no checkout)")
    elif subprocess.run(["git", "checkout", "--quiet", commit], cwd=REPO).returncode != 0:
        print("    checkout failed")
        return []

    cfg_text, skipped = build_config("HEAD" if commit == WORKTREE else commit, hints, partial, fp8)
    # Branch names contain '/', which would turn the filename into a path into a missing directory.
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)
    cfg = SCRATCH / f"bench_{slug}.yaml"
    cfg.write_text(cfg_text, encoding="utf-8")
    if skipped:
        print(f"    keys not defined at this commit (omitted): {', '.join(skipped)}")

    log_path = SCRATCH / f"server_{slug}.log"
    env = {**os.environ, "PYTHONUTF8": "1", "INVOKEAI_MEMORY_TRACE": "0"}
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [str(PYTHON), "scripts/invokeai-web.py", "--root", str(DATA_ROOT), "--config", str(cfg)],
            cwd=REPO,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            if not wait_for_server(proc, log_path):
                return []
            print("    server up")
            if enqueue(graph, runs):
                wait_for_queue()
        finally:
            kill_tree(proc)
    for _ in range(20):  # let the port actually free up before the next variant
        if port_free():
            break
        time.sleep(1)

    timings = parse_timings(read_log(log_path))
    print(f"    {len(timings)} graph(s) measured")
    return timings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        action="append",
        required=True,
        help="commit[:hints] where hints is true/false; omit for 'key absent'",
    )
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--item", type=int, default=602)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument(
        "--fp8", choices=["true", "false"], default="true", help="fp8_compute for every variant (default true)"
    )
    args = ap.parse_args()
    fp8 = args.fp8 == "true"

    orig = git("branch", "--show-current").strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--", "invokeai", "tests"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
    )
    specs = [(s.partition(":")[0], s.partition(":")[2]) for s in args.variant]
    needs_checkout = any(c != WORKTREE for c, _ in specs)
    # A WORKTREE variant measures uncommitted changes, so the tree is *expected* to be dirty. Any
    # other variant needs a checkout, which would fail or clobber — stash across it and always pop.
    stashed = False
    if dirty and needs_checkout:
        stashed = "No local changes" not in git(
            "stash", "push", "-u", "-m", "bench: uncommitted work", "--", "invokeai", "tests"
        )
        print("stashed uncommitted work for the checkout variants" if stashed else "nothing to stash")
    elif dirty:
        print("measuring the current working tree (uncommitted changes included)")
    print(f"original branch: {orig}\nrunning {len(args.variant)} variant(s) x {args.runs} run(s)")

    graph = load_graph(args.item)
    results: list[tuple[str, str, list[dict]]] = []
    try:
        for spec in args.variant:
            commit, _, rest = spec.partition(":")
            h, _, p = rest.partition(":")
            flag = {"true": True, "false": False, "": None}
            if commit == WORKTREE and stashed:
                git("stash", "pop")
                stashed = False
            subject = (
                "current working tree" if commit == WORKTREE else git("log", "-1", "--format=%s", commit).strip()[:40]
            )
            tag = f"{commit[:8]}" + (f" hints={h}" if h else "") + (f" partial={p}" if p else "") + f" fp8={args.fp8}"
            results.append((tag, subject, bench(commit, flag[h], flag[p], graph, args.runs, tag, fp8)))
    finally:
        subprocess.run(["git", "checkout", "--quiet", orig], cwd=REPO)
        if stashed:
            git("stash", "pop")
        print(f"\nrestored branch: {git('branch', '--show-current').strip()}")
        left = subprocess.run(["git", "stash", "list"], cwd=REPO, capture_output=True, text=True).stdout
        if "bench: uncommitted work" in left:
            print("WARNING: a bench stash is still present — run `git stash pop`")

    (SCRATCH / "bench_results.json").write_text(
        json.dumps([{"tag": t, "subject": s, "runs": r} for t, s, r in results], indent=2), encoding="utf-8"
    )

    print(f"\n{'variant':<22} {'run':>4} {'s/it':>8} {'denoise':>9} {'encoder':>9} {'l2i':>7} {'total':>8}")
    print("-" * 72)
    for tag, subject, runs_ in results:
        if not runs_:
            print(f"{tag:<22} {'FEHLGESCHLAGEN':>4}")
            continue
        for i, t in enumerate(runs_):
            dn = t.get("krea2_denoise", 0.0)
            mark = "cold" if i == 0 else str(i)
            print(
                f"{tag if i == 0 else '':<22} {mark:>4} {dn / args.steps:8.3f} {dn:9.2f} "
                f"{t.get('krea2_text_encoder', 0):9.2f} {t.get('qwen_image_l2i', 0):7.2f} {t.get('_total', 0):8.2f}"
            )
        warm = [t.get("krea2_denoise", 0.0) / args.steps for t in runs_[1:]]
        if warm:
            print(f"{'':<22} {'MEAN':>4} {sum(warm) / len(warm):8.3f}   (warm runs only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
