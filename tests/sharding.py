"""Splits a pytest run into disjoint shards by test file, for CI jobs that cannot hold the whole suite.

Loaded by tests/conftest.py. Without `--shard` the plugin leaves the selection unchanged.

    pytest --shard=2/3 --shard-durations=tests/shard_durations.json --shard-record=out.json

The partition is a function of the set of collected files and of the durations file alone. Every xdist
worker and every shard therefore computes the same one, whatever order it collected in: files with a recorded
duration go longest-first to the currently lightest shard, other files (and files recorded as 0 s) to a shard
picked by a hash of the path. Anything that narrows the collection (`-k`, `--lf`, paths) changes the partition,
so reproduce a CI shard locally with CI's exact selection arguments: its markers and no narrowing.

`--shard-record` writes the measured per-file durations, only after a session that exited with status 0.
Refreshing tests/shard_durations.json from those recordings is described in
docs/src/content/docs/development/Guides/tests.mdx ("Windows ARM64 in CI").
"""

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import pytest

DURATIONS_VERSION = 1
ROOT = Path(__file__).parents[1]
DURATIONS_PATH = Path(__file__).with_name("shard_durations.json")

_PLAN = pytest.StashKey[tuple[int, int, dict[str, float]]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("sharding")
    group.addoption("--shard", metavar="I/N", help="run shard I of N (1-based) of the collected test files")
    group.addoption("--shard-durations", metavar="PATH", help="JSON file of per-file durations that balances shards")
    group.addoption("--shard-record", metavar="PATH", help="write the measured per-file durations to this JSON file")


def _read_durations(path: Path, label: str = "--shard-durations") -> dict[str, float]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["version"] != DURATIONS_VERSION:
            raise ValueError(f"version {data['version']!r}, expected {DURATIONS_VERSION}")
        files = data["files"]
        if not isinstance(files, dict) or not all(
            type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in files.values()
        ):
            raise ValueError("'files' must map paths to finite, non-negative seconds")
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise pytest.UsageError(f"{label} {path}: {e!r}") from e
    return {str(k): float(v) for k, v in files.items()}


def pytest_configure(config: pytest.Config) -> None:
    record = config.getoption("shard_record")
    # Under xdist the controller receives every worker's reports; the workers record nothing.
    if record is not None and not hasattr(config, "workerinput"):
        config.pluginmanager.register(_Recorder(Path(record)))
    durations_path = config.getoption("shard_durations")
    durations = _read_durations(Path(durations_path)) if durations_path else {}
    spec = config.getoption("shard")
    if spec is None:
        return
    try:
        index_text, count_text = spec.split("/")
        index, count = int(index_text), int(count_text)
        if not 1 <= index <= count:
            raise ValueError
    except ValueError:
        raise pytest.UsageError(f"--shard must be I/N with 1 <= I <= N, got {spec!r}") from None
    config.stash[_PLAN] = (index - 1, count, durations)


def assign_shards(files: set[str], durations: dict[str, float], count: int) -> dict[str, int]:
    """Map each file to a 0-based shard."""
    loads = [0.0] * count
    assignment: dict[str, int] = {}
    for path in sorted((f for f in files if durations.get(f, 0) > 0), key=lambda f: (-durations[f], f)):
        shard = min(range(count), key=lambda i: (loads[i], i))
        assignment[path] = shard
        loads[shard] += durations[path]
    for path in files - assignment.keys():
        assignment[path] = int(hashlib.sha256(path.encode()).hexdigest(), 16) % count
    return assignment


# Last, so that `-m` and other deselection has run and only the files that would run are balanced.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _PLAN not in config.stash:
        return
    shard, count, durations = config.stash[_PLAN]
    assignment = assign_shards({item.nodeid.split("::", 1)[0] for item in items}, durations, count)
    kept = [item for item in items if assignment[item.nodeid.split("::", 1)[0]] == shard]
    removed = [item for item in items if assignment[item.nodeid.split("::", 1)[0]] != shard]
    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept


def pytest_report_header(config: pytest.Config) -> str | None:
    if _PLAN not in config.stash:
        return None
    shard, count, durations = config.stash[_PLAN]
    return f"shard {shard + 1}/{count} ({len(durations)} recorded durations)"


def _write_durations(path: Path, files: dict[str, float]) -> None:
    document = {"version": DURATIONS_VERSION, "files": dict(sorted(files.items()))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


class _Recorder:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._seconds: defaultdict[str, float] = defaultdict(float)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self._seconds[report.nodeid.split("::", 1)[0]] += report.duration

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        # A crashed worker records 0 s for its file and leaves the unrun files out, so only a clean session counts.
        if exitstatus != pytest.ExitCode.OK:
            session.config.get_terminal_writer().line(f"shard recording skipped: exit status {int(exitstatus)}")
            return
        _write_durations(self._path, {path: round(seconds, 1) for path, seconds in self._seconds.items()})


def main(argv: list[str]) -> None:
    """Update the durations file with recordings. Run it from the repository root: `tests` only imports from there."""
    if not argv:
        sys.exit("usage: python -m tests.sharding RECORDING.json...")
    try:
        existing = _read_durations(DURATIONS_PATH) if DURATIONS_PATH.exists() else {}
        recorded: dict[str, float] = {}
        for recording in argv:
            recorded |= _read_durations(Path(recording), "recording")
    except pytest.UsageError as e:
        sys.exit(str(e))
    merged = {path: seconds for path, seconds in (existing | recorded).items() if (ROOT / path).is_file()}
    updated = len(merged.keys() & recorded.keys())
    _write_durations(DURATIONS_PATH, merged)
    dropped = len(existing | recorded) - len(merged)
    print(f"{DURATIONS_PATH}: {updated} updated, {len(merged) - updated} kept, {dropped} dropped")


if __name__ == "__main__":
    main(sys.argv[1:])
