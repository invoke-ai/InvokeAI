"""Background measurement of intermediate file sizes.

Media written before size accounting existed, or whose writer could not stat its file, has a NULL
size. The measurer fills those gaps in bounded batches on the intermediates worker thread while it
has no cleanup to run, and only on request, so an instance that never opens the manager never pays
for a scan.
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, Callable, Optional, cast

from invokeai.app.services.intermediates.intermediates_records_sqlite import IntermediatesRecordsSqlite, MediaKind

if TYPE_CHECKING:
    from invokeai.app.services.invocation_services import InvocationServices

MEASURE_BATCH_SIZE = 200
# A row is written before its file; measuring it a moment later would record a missing file.
MEASURE_MIN_AGE_SECONDS = 10


class IntermediatesSizeMeasurer:
    def __init__(
        self,
        records: IntermediatesRecordsSqlite,
        services: Callable[[], "InvocationServices"],
        stop: threading.Event,
        logger: logging.Logger,
    ) -> None:
        self._records = records
        self._services = services
        self._stop = stop
        self._logger = logger
        self._requested = threading.Event()
        self._due_at: Optional[float] = None

    def reset(self) -> None:
        self._requested.clear()
        self._due_at = None

    def request(self) -> None:
        self._requested.set()

    def is_due(self) -> bool:
        return self._requested.is_set() or (self._due_at is not None and time.monotonic() >= self._due_at)

    def measure_once(self) -> None:
        """Measures a bounded batch of unmeasured intermediates; clears the request when none remain."""
        self.reset()
        services = self._services()
        image_moves = getattr(services, "image_moves", None)
        if image_moves is not None and image_moves.is_maintenance_active():
            # Files are being relocated; a measurement now could record a missing file as empty.
            return
        try:
            remaining = False
            retry_young = False
            for kind, files, records in (
                ("image", services.image_files, services.image_records),
                ("video", services.video_files, services.video_records),
            ):
                pending = self._records.next_unmeasured(
                    cast(MediaKind, kind), MEASURE_BATCH_SIZE, min_age_seconds=MEASURE_MIN_AGE_SECONDS
                )
                sizes: dict[str, int] = {}
                unmeasurable: list[str] = []
                for name, subfolder in pending:
                    if self._stop.is_set():
                        return
                    try:
                        size = files.get_file_size_bytes(name, subfolder)
                    except Exception as error:
                        self._logger.warning(f"Could not measure {kind} {name}; skipping it until restart: {error}")
                        unmeasurable.append(name)
                        continue
                    # A missing file occupies nothing; recording 0 is a measurement, not a guess, and
                    # keeps the row from being re-measured forever. The writer's own measurement
                    # still replaces it if the file lands afterwards.
                    sizes[name] = size if size is not None else 0
                self._records.mark_unmeasurable(cast(MediaKind, kind), unmeasurable)
                # One transaction per batch: a library of hundreds of thousands of intermediates is
                # measured in a few thousand commits rather than one per file.
                records.set_file_sizes_bytes(sizes)
                if len(pending) == MEASURE_BATCH_SIZE:
                    remaining = True
                elif self._records.next_unmeasured(cast(MediaKind, kind), 1):
                    # All older rows were measured or skipped; a new row still needs its file to be
                    # written before measuring it. Wake after the minimum age has elapsed.
                    retry_young = True
            if remaining:
                # Batches hold the database lock only for their reads and one write; the file stats
                # between them leave it free for generation and saves.
                self._requested.set()
            elif retry_young:
                self._due_at = time.monotonic() + MEASURE_MIN_AGE_SECONDS
        except Exception as error:
            self._logger.warning(f"Measuring intermediate file sizes failed; will retry on demand: {error}")
