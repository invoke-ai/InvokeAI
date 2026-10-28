"""Accepted references: the image and medians a later run is compared against.

References change only through `accept`, an explicit user command; nothing else in the tool writes them.
"""

from __future__ import annotations

import hashlib
import logging
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scripts.family_e2e.manifest import DEFAULT_ARM
from scripts.family_e2e.metrics import measured_values, median
from scripts.family_e2e.records import Record, Reference, load_records, load_run_info, write_bytes_atomic

REFERENCE_FILE = "reference.json"
REFERENCE_IMAGE = "image.png"

logger = logging.getLogger(__name__)


class AcceptError(ValueError):
    pass


def reference_dir(data_dir: Path, profile: str, case_key: str) -> Path:
    return data_dir / "references" / profile / case_key


def load_reference(data_dir: Path, profile: str, case_key: str) -> tuple[Reference, Path] | None:
    """Returns the reference and the path of its image (which may be missing), or None if none was accepted."""
    directory = reference_dir(data_dir, profile, case_key)
    try:
        content = (directory / REFERENCE_FILE).read_bytes()
    except FileNotFoundError:
        return None
    return Reference.model_validate_json(content), directory / REFERENCE_IMAGE


def accept(data_dir: Path, run_dir: Path, case_keys: list[str] | None) -> list[Reference]:
    """Makes the measured results of `run_dir` the references of its profile, for `case_keys` or all its cases.

    Every requested case is checked before anything is written; any problem raises AcceptError and changes nothing.
    """
    info = load_run_info(run_dir)
    records = load_records(run_dir)
    arms = sorted(set(info.environments) | {r.arm for r in records})
    if arms != [DEFAULT_ARM]:
        raise AcceptError(f"run {info.run_id} has arms {', '.join(arms)}; references come only from normal runs")
    environment = info.environments.get(DEFAULT_ARM)
    if environment is None:
        raise AcceptError(f"run {info.run_id} has no recorded environment")

    by_case: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_case[record.case_key].append(record)
    if case_keys is None:
        planned = [step.case_key for step in info.plan]
        case_keys = list(dict.fromkeys(planned + list(by_case)))
    if not case_keys:
        raise AcceptError(f"run {info.run_id} has no cases")

    accepted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    staged: list[tuple[Reference, bytes]] = []
    problems: list[str] = []
    for case_key in dict.fromkeys(case_keys):
        measured = [r for r in by_case.get(case_key, []) if r.kind == "measure" and r.status == "completed"]
        candidates = [r for r in measured if r.valid_psnr and r.image_path]
        if not candidates:
            problems.append(f"{case_key}: no completed measured record with an image valid for PSNR")
            continue
        source = candidates[0]
        if (source.image_width, source.image_height) != (source.width, source.height):
            problems.append(
                f"{case_key}: image {source.image_width}x{source.image_height} is not the requested "
                f"{source.width}x{source.height}"
            )
            continue
        assert source.image_path is not None
        try:
            image = (run_dir / source.image_path).read_bytes()
        except OSError as e:
            problems.append(f"{case_key}: cannot read {source.image_path}: {e}")
            continue
        distinct = {r.image_sha256 for r in candidates}
        if len(distinct) > 1:
            logger.warning(
                "%s: the %d measured images are not identical (%d distinct); the reference uses %s",
                case_key,
                len(candidates),
                len(distinct),
                source.record_id,
            )
        values = measured_values(measured)
        process_peak = median(values.process_peak_bytes)
        reference = Reference(
            profile=info.profile,
            case_key=case_key,
            source_run_id=info.run_id,
            accepted_at=accepted_at,
            image_sha256=hashlib.sha256(image).hexdigest(),
            graph_hash=source.graph_hash,
            environment=environment,
            total_s_median=median(values.total_s),
            per_step_s_median=median(values.per_step_s),
            process_peak_bytes_median=None if process_peak is None else round(process_peak),
            node_peak_bytes_median={
                node_id: round(statistics.median(peaks)) for node_id, peaks in values.node_peak_bytes.items()
            },
        )
        staged.append((reference, image))
    if problems:
        raise AcceptError(f"nothing accepted from run {info.run_id}:\n" + "\n".join(problems))

    for reference, image in staged:
        directory = reference_dir(data_dir, reference.profile, reference.case_key)
        directory.mkdir(parents=True, exist_ok=True)
        write_bytes_atomic(directory / REFERENCE_IMAGE, image)
        write_bytes_atomic(directory / REFERENCE_FILE, reference.model_dump_json(indent=2).encode("utf-8"))
        logger.info("accepted %s from run %s", reference.case_key, reference.source_run_id)
    return [reference for reference, _ in staged]
