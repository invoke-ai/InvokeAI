"""Calibration runs: each step goes through the server's working-memory calibrate route instead of enqueue_batch.

The server measures every model operation of such a queue item with its working-memory probe and keeps the windows in
its calibration store, one file per device class under `<db_dir>/working_memory/` of the scratch root. The scratch root
goes at the end of each server lifetime, so every lifetime saves its store into the run's `calibration/` first: the
export route's bytes and the status while the server still runs, or the store file itself when the server died or was
stopped. `scripts/working_memory/build_shipped.py` turns the exports into shipped working-memory data.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

from scripts.family_e2e.client import ApiError, InvokeClient
from scripts.family_e2e.records import (
    CalibrationExport,
    CalibrationSubmission,
    RunInfo,
    write_bytes_atomic,
    write_run_info,
)
from scripts.family_e2e.server import SCRATCH_PATHS

CALIBRATION_DIR = "calibration"
STATUS_PATH = "/api/v2/models/working_memory/status"
EXPORT_PATH = "/api/v2/models/working_memory/export"
# The store's directory under the root's db_dir (invokeai/app/api/dependencies.py) and its file names
# (working_memory_calibration_store.py). Copied, not imported: the tool must not import invokeai, which imports torch.
STORE_DIR = "working_memory"
STORE_FILE = re.compile(r"^[0-9a-f]{16}\.json$")
PROBE_REASON = "calibration run: the probe synchronizes and empties the allocator cache around every model operation"


CASE_REFUSALS = (404, 422)
"""Calibrate refusals that concern one case (an unknown model, a graph the run cannot measure); the step gets an error
record and the run goes on. Any other error response would refuse every later step too."""


class CalibrationError(RuntimeError):
    """The server cannot run a calibration run, or refused one."""


class CaseRefused(CalibrationError):
    """The server refused one case's calibration run (`CASE_REFUSALS`)."""


def check_available(client: InvokeClient) -> None:
    status = client.get(STATUS_PATH)
    if not status.get("available"):
        raise CalibrationError(f"the server cannot calibrate: {status.get('unavailable_reason')}")


def submit(client: InvokeClient, model_key: str, batch: dict[str, Any]) -> tuple[int, CalibrationSubmission]:
    """Starts a calibration run of the batch's graph and data; returns its one queue item and the run.

    A refusal raises `CaseRefused` for a `CASE_REFUSALS` status and CalibrationError for any other error response.
    Each server lifetime has a fresh queue and store and the tool submits each attempt once, so it sends no retry key.
    """
    body: dict[str, Any] = {"graph": batch["graph"]}
    if batch.get("data"):
        body["data"] = batch["data"]
    try:
        created = client.post(f"/api/v2/models/i/{quote(model_key, safe='')}/working_memory/calibrate", body)
    except ApiError as e:
        refused = CaseRefused if e.status_code in CASE_REFUSALS else CalibrationError
        raise refused(f"the server refused the calibration run: {e}") from e
    if len(created["item_ids"]) != 1:
        raise CalibrationError(f"expected one queue item from the calibration run, got {len(created['item_ids'])}")
    submission = CalibrationSubmission(run_id=created["run_id"], device_class_id=created["device_class_id"])
    return created["item_ids"][0], submission


Stores = list[tuple[str, bytes]]
"""(device class id, store bytes) per saved store."""


def save(run_dir: Path, info: RunInfo, stores: Stores, status: bytes | None) -> list[CalibrationExport]:
    """Writes one lifetime's stores into the run's `calibration/` and lists them in `info`, which it writes to run.json.
    Stores from the export routes come with the status; store files copied from a scratch root come without one. Hold
    the run lock with `info` read under it: `runner.save_calibration` does."""
    source: Literal["api", "store_file"] = "store_file" if status is None else "api"
    lifetime = max((e.lifetime for e in info.calibration_exports), default=0) + 1
    (run_dir / CALIBRATION_DIR).mkdir(exist_ok=True)
    status_path = None
    if status is not None:
        status_path = f"{CALIBRATION_DIR}/{info.run_id}-{lifetime}-status.json"
        write_bytes_atomic(run_dir / status_path, status)
    saved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    exports = []
    for device_class_id, raw in stores:
        path = f"{CALIBRATION_DIR}/{info.run_id}-{lifetime}-{device_class_id}.json"
        write_bytes_atomic(run_dir / path, raw)
        exports.append(
            CalibrationExport(
                lifetime=lifetime,
                device_class_id=device_class_id,
                export=path,
                status=status_path,
                source=source,
                saved_at=saved_at,
            )
        )
    info.calibration_exports.extend(exports)
    write_run_info(run_dir, info)
    return exports


def fetch_exports(client: InvokeClient, device_class_ids: Iterable[str]) -> tuple[Stores, bytes]:
    """The store of every device class the lifetime's runs were created for, and the status, from the export routes."""
    stores = [
        (device_class_id, client.get_bytes(EXPORT_PATH, device_class_id=device_class_id))
        for device_class_id in sorted(set(device_class_ids))
    ]
    return stores, client.get_bytes(STATUS_PATH)


def read_store_files(scratch_root: Path) -> Stores:
    """The store files of a scratch root whose server is gone; none when it has none."""
    store_dir = scratch_root / SCRATCH_PATHS["db_dir"] / STORE_DIR
    files = sorted(p for p in store_dir.glob("*.json") if STORE_FILE.match(p.name)) if store_dir.is_dir() else []
    return [(path.stem, path.read_bytes()) for path in files]
