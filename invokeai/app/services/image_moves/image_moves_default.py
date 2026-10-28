import os
import tempfile
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence, cast

from PIL import Image, UnidentifiedImageError

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries.image_moves import MoveItem
from invokeai.app.util.thumbnails import make_thumbnail

MoveJobState = Literal["planned", "moving", "moved", "committed", "error"]
ImageMoveBackgroundOperation = Literal["move_all", "recovery"]


@dataclass(frozen=True)
class PlannedImageMove:
    image_name: str
    old_subfolder: str
    new_subfolder: str
    is_intermediate: bool
    old_path: Path
    new_path: Path
    old_thumbnail_path: Path
    new_thumbnail_path: Path


@dataclass(frozen=True)
class ImageMoveJob:
    id: int
    state: MoveJobState
    error_message: str | None


@dataclass(frozen=True)
class ImageMoveResult:
    planned: int = 0
    committed: int = 0
    errors: int = 0


@dataclass(frozen=True)
class ImageMoveBackgroundStatus:
    is_running: bool
    operation: ImageMoveBackgroundOperation | None
    active_job_id: int | None
    latest_job: ImageMoveJob | None
    last_error: str | None
    needs_move_count: int


class ImageMoveJobAlreadyRunning(Exception):
    pass


class ImageMoveQueueActive(Exception):
    pass


class UnreadableImageError(Exception):
    pass


def _journaled(move: PlannedImageMove) -> MoveItem:
    return MoveItem(
        image_name=move.image_name,
        old_subfolder=move.old_subfolder,
        new_subfolder=move.new_subfolder,
        is_intermediate=move.is_intermediate,
        old_path=str(move.old_path),
        new_path=str(move.new_path),
        old_thumbnail_path=str(move.old_thumbnail_path),
        new_thumbnail_path=str(move.new_thumbnail_path),
    )


class ImageMoveService:
    def __init__(
        self,
        database: Database,
        image_files: ImageFileStorageBase,
        config: InvokeAIAppConfig,
        logger,
    ) -> None:
        self._queries = database.queries
        self.image_files = image_files
        self._config = config
        self._logger = logger
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="image-move")
        self._future_lock = threading.Lock()
        self._future: Future | None = None
        self._future_operation: ImageMoveBackgroundOperation | None = None
        self._gallery_maintenance_reserved = False
        self._last_background_error: str | None = None
        # Serializes the move service's relocate-and-repoint units against the image delete
        # units in ImageService. See image_mutation_lock() for the interleaving it prevents.
        self._image_mutation_lock = threading.RLock()
        self._invoker = None
        self._session_queue = None

    def start(self, invoker) -> None:
        self._invoker = invoker
        self._session_queue = getattr(invoker.services, "session_queue", None)
        result = self.startup_recovery()
        if result.committed > 0 or result.errors > 0:
            self._logger.info(
                "Image move startup recovery completed: committed=%s, errors=%s",
                result.committed,
                result.errors,
            )

    def set_session_queue(self, session_queue) -> None:
        self._session_queue = session_queue

    def stop(self, *args, **kwargs) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    @contextmanager
    def image_mutation_lock(self, *, blocking: bool = True) -> Iterator[bool]:
        """Serializes image delete units against subfolder relocation units.

        An image delete reads an image's subfolder, deletes its record, then purges its files
        at that subfolder. A move unit does the opposite: it relocates the files and repoints
        the record. If the two interleave, the delete purges the path its snapshot named while
        the files sit at the new one — permanent orphans, unrecoverable because the record is
        gone and a clean purge drops the journal (JPPhoto, PR #9361). ``ImageService`` holds
        this lock across each delete and copy unit, and the move service holds it across each
        plan-relocate-repoint cycle below. Session workers hold it across the maintenance check
        and queue claim, so maintenance either observes a claimed item or prevents the claim.

        It is a reentrant lock because both sides run their units to completion in one thread;
        nothing inside a unit may block on another thread that needs this lock. It is also
        process-local: two Invoke processes sharing one output folder and database are not
        serialized by it — the same limitation the route guard has, which the delete journal's
        startup recovery re-check papers over for deletes.

        Nonblocking callers receive ``False`` when another unit holds the lock and do not own
        it. Queue workers use that path so shutdown can wake a worker even during a long scan.
        """
        acquired = self._image_mutation_lock.acquire(blocking=blocking)
        try:
            yield acquired
        finally:
            if acquired:
                self._image_mutation_lock.release()

    @contextmanager
    def reserve_gallery_maintenance(self) -> Iterator[None]:
        """Excludes image moves and queue claims while gallery maintenance runs.

        This process-local reservation is visible before waiting for the mutation lock, so
        session workers cannot claim new work while this operation waits for a claim already
        in flight. The future lock is released before acquiring the mutation lock; workers take
        the locks in the opposite order when they atomically check maintenance state and dequeue.
        """
        with self._future_lock:
            self._refresh_finished_future_locked()
            if (
                self._gallery_maintenance_reserved
                or self._future_operation is not None
                or (self._future is not None and not self._future.done())
            ):
                raise ImageMoveJobAlreadyRunning("An image storage maintenance operation is already running")
            if self._queries.image_moves.active_job_id() is not None:
                raise ImageMoveJobAlreadyRunning("An image move job is already active")
            self._gallery_maintenance_reserved = True

        try:
            with self.image_mutation_lock():
                self._assert_no_active_queue_work()
                yield
        finally:
            # Release only after the mutation lock is gone. Queue workers acquire that lock
            # before consulting this flag, so this order cannot deadlock with a worker claim.
            with self._future_lock:
                self._gallery_maintenance_reserved = False

    def start_background_move_all(self) -> ImageMoveBackgroundStatus:
        return self._start_background_operation("move_all", self.move_all_images, require_idle_queue=True)

    def start_background_recovery(self) -> ImageMoveBackgroundStatus:
        return self._start_background_operation("recovery", self.startup_recovery)

    def get_background_status(self) -> ImageMoveBackgroundStatus:
        with self._future_lock:
            self._refresh_finished_future_locked()
            return self._build_background_status_locked()

    def is_maintenance_active(self) -> bool:
        with self._future_lock:
            self._refresh_finished_future_locked()
            is_running = self._future is not None and not self._future.done()
            operation_reserved = self._future_operation is not None
            gallery_maintenance_reserved = self._gallery_maintenance_reserved
        return (
            gallery_maintenance_reserved
            or operation_reserved
            or is_running
            or self._queries.image_moves.active_job_id() is not None
        )

    def _assert_no_active_queue_work(self) -> None:
        session_queue = self._session_queue
        if session_queue is None and self._invoker is not None:
            session_queue = getattr(self._invoker.services, "session_queue", None)
        if session_queue is None:
            return
        if session_queue.has_active_queue_work():
            raise ImageMoveQueueActive("Cannot start image move while queue work is active")

    def _start_background_operation(
        self,
        operation: ImageMoveBackgroundOperation,
        target,
        require_idle_queue: bool = False,
    ) -> ImageMoveBackgroundStatus:
        with self._future_lock:
            self._refresh_finished_future_locked()
            if self._gallery_maintenance_reserved:
                raise ImageMoveJobAlreadyRunning("An image storage maintenance operation is already running")
            if self._future_operation is not None or (self._future is not None and not self._future.done()):
                raise ImageMoveJobAlreadyRunning("An image move job is already running")
            active_job_id = self._queries.image_moves.active_job_id()
            if operation != "recovery" and active_job_id is not None:
                raise ImageMoveJobAlreadyRunning("An image move job is already active")
            self._last_background_error = None
            self._future_operation = operation

        try:
            if require_idle_queue:
                self._assert_no_active_queue_work()
            future = self._executor.submit(self._run_background_operation, operation, target)
        except Exception:
            with self._future_lock:
                self._future_operation = None
            raise

        with self._future_lock:
            self._future = future
            return self._build_background_status_locked()

    def _run_background_operation(self, operation: ImageMoveBackgroundOperation, target) -> None:
        try:
            target()
        except Exception as e:
            self._record_background_error(str(e))
            self._logger.exception("Image move background operation failed: %s", operation)

    def _record_background_error(self, message: str) -> None:
        with self._future_lock:
            self._last_background_error = message
        active_job_id = self._queries.image_moves.active_job_id()
        if active_job_id is not None:
            try:
                self.record_job_error_message(active_job_id, message)
            except Exception:
                self._logger.exception("Failed to record image move background error on active job")

    def _refresh_finished_future_locked(self) -> None:
        if self._future is None or not self._future.done():
            return
        self._future = None
        self._future_operation = None

    def _build_background_status_locked(self) -> ImageMoveBackgroundStatus:
        latest_job = self.get_latest_job()
        return ImageMoveBackgroundStatus(
            is_running=self._future is not None and not self._future.done(),
            operation=self._future_operation,
            active_job_id=self._queries.image_moves.active_job_id(),
            latest_job=latest_job,
            last_error=self._last_background_error,
            needs_move_count=self.count_images_needing_move(),
        )

    def move_all_images(self) -> ImageMoveResult:
        recovered = self.startup_recovery()
        last_image_name = ""
        planned = 0
        committed = recovered.committed
        errors = recovered.errors

        while True:
            # The whole batch cycle — plan, journal the job, relocate files, repoint the
            # records — holds the image-mutation lock. A delete interleaved inside the cycle
            # would purge the path its snapshot named while this job's relocate-and-repoint
            # landed mid-flight, stranding files at the new subfolder with no record and no
            # journal (JPPhoto, PR #9361). Planning and creating the job are inside the lock
            # too, so an item whose record a delete removes is never planned in the first
            # place instead of failing the job after the fact.
            with self.image_mutation_lock():
                moves, plan_errors = self._plan_batch(
                    last_image_name=last_image_name, limit=100, record_missing_errors=True
                )
                errors += plan_errors
                if not moves:
                    next_name = self._queries.image_moves.next_image_name(last_image_name)
                    if next_name is None:
                        break
                    last_image_name = next_name
                    continue

                job_id = self.create_move_job(moves)
                planned += len(moves)
                try:
                    self.perform_filesystem_moves(job_id)
                    committed += self.commit_database_updates(job_id)
                    errors += self._queries.image_moves.error_count(job_id)
                except Exception as e:
                    errors += 1
                    self.record_job_error_message(job_id, str(e))
                    raise
            last_image_name = moves[-1].image_name

        return ImageMoveResult(planned=planned, committed=committed, errors=errors)

    def startup_recovery(self) -> ImageMoveResult:
        job_ids = self._queries.image_moves.recoverable_job_ids()

        committed = 0
        errors = 0
        for job_id in job_ids:
            try:
                # Same unit as a live batch: finishing an interrupted relocation must not
                # interleave with a concurrent delete, which would otherwise purge the path
                # its snapshot named while the files land at the new one (JPPhoto, PR #9361).
                with self.image_mutation_lock():
                    self.complete_partial_filesystem_moves(job_id)
                    self.cleanup_empty_source_dirs(job_id)
                    committed += self.commit_database_updates(job_id)
            except Exception as e:
                if self._is_unrecoverable_error(e):
                    self.mark_job_unrecoverable(job_id, str(e))
                    errors += max(1, self._queries.image_moves.error_count(job_id))
                else:
                    errors += 1
                    self.record_job_error_message(job_id, str(e))
            else:
                errors += self._queries.image_moves.error_count(job_id)
        return ImageMoveResult(committed=committed, errors=errors)

    def plan_batch(self, last_image_name: str, limit: int) -> list[PlannedImageMove]:
        moves, _errors = self._plan_batch(last_image_name=last_image_name, limit=limit, record_missing_errors=False)
        return moves

    def count_images_needing_move(self) -> int:
        # Runs over every image on each status poll, so it calls the subfolder rule directly.
        count = 0
        for image_name, subfolder, category, is_intermediate, created_at in self._queries.image_moves.placements():
            new_subfolder = self._get_new_subfolder(
                image_name=image_name,
                image_category=ImageCategory(category),
                is_intermediate=is_intermediate,
                created_at=created_at,
            )
            if new_subfolder != subfolder:
                count += 1
        return count

    def _plan_batch(
        self, last_image_name: str, limit: int, record_missing_errors: bool
    ) -> tuple[list[PlannedImageMove], int]:
        moves: list[PlannedImageMove] = []
        for image_name, subfolder, category, is_intermediate, created_at in self._queries.image_moves.placements_after(
            last_image_name, limit
        ):
            new_subfolder = self._get_new_subfolder(
                image_name=image_name,
                image_category=ImageCategory(category),
                is_intermediate=is_intermediate,
                created_at=created_at,
            )
            if new_subfolder != subfolder:
                moves.append(self._planned_move(image_name, subfolder, new_subfolder, is_intermediate))
        errors = 0
        if record_missing_errors:
            moves, errors = self._record_missing_source_errors(moves)
        self.preflight_moves(moves)
        return moves, errors

    def create_move_job(self, moves: Sequence[PlannedImageMove]) -> int:
        if not moves:
            raise ValueError("Cannot create an image move job with no items")
        job_id = self._queries.image_moves.create_job([_journaled(move) for move in moves])
        if job_id is None:
            raise ValueError("Cannot create image move job while another active image move job exists")
        return job_id

    def create_error_move_job(self, move: PlannedImageMove, message: str) -> int:
        return self._queries.image_moves.create_failed_job(_journaled(move), message)

    def preflight_moves(self, moves: Sequence[PlannedImageMove]) -> None:
        destinations: set[Path] = set()
        thumbnail_destinations: set[Path] = set()
        for move in moves:
            if not move.old_path.exists():
                if not move.is_intermediate:
                    raise FileNotFoundError(f"Source image does not exist: {move.old_path}")
                continue
            if move.new_path.exists():
                raise FileExistsError(f"Destination image already exists: {move.new_path}")
            if move.old_path == move.new_path:
                raise ValueError(f"Old and new paths are identical for {move.image_name}")
            if move.new_path in destinations:
                raise ValueError(f"Duplicate destination path: {move.new_path}")
            destinations.add(move.new_path)
            if move.new_thumbnail_path in thumbnail_destinations:
                raise ValueError(f"Duplicate destination thumbnail path: {move.new_thumbnail_path}")
            thumbnail_destinations.add(move.new_thumbnail_path)
            if self._queries.image_moves.has_active_job_for_image(move.image_name):
                raise ValueError(f"Image {move.image_name} already has an active image move job")
            self._assert_same_filesystem(move.old_path, move.new_path)
            if move.old_thumbnail_path.exists():
                if move.new_thumbnail_path.exists():
                    raise FileExistsError(f"Destination thumbnail already exists: {move.new_thumbnail_path}")
                self._assert_same_filesystem(move.old_thumbnail_path, move.new_thumbnail_path)

    def _record_missing_source_errors(self, moves: Sequence[PlannedImageMove]) -> tuple[list[PlannedImageMove], int]:
        remaining_moves: list[PlannedImageMove] = []
        errors = 0
        for move in moves:
            if move.old_path.exists() or move.is_intermediate:
                remaining_moves.append(move)
                continue
            message = f"Source image does not exist: {move.old_path}"
            self.create_error_move_job(move, message)
            self._logger.error(message)
            errors += 1
        return remaining_moves, errors

    def perform_filesystem_moves(self, job_id: int) -> None:
        self._queries.image_moves.set_job_state(job_id, "moving")
        self.complete_partial_filesystem_moves(job_id)
        self.cleanup_empty_source_dirs(job_id)
        self._queries.image_moves.set_job_state(job_id, "moved")

    def complete_partial_filesystem_moves(self, job_id: int) -> None:
        items = self._get_items(job_id, include_terminal=False)
        if not items:
            if self._get_items(job_id):
                return
            raise RuntimeError(f"Image move job {job_id} has no items")
        for item in items:
            try:
                self._complete_partial_filesystem_move(job_id, item)
            except Exception as e:
                if not self._is_unrecoverable_error(e):
                    raise
                self._reconcile_destination_subfolder(item)
                self.mark_item_unrecoverable(job_id, item.image_name, f"{item.image_name}: {e}")
                self._logger.error("Image move skipped unrecoverable item %s: %s", item.image_name, e)

    def _complete_partial_filesystem_move(self, job_id: int, item: PlannedImageMove) -> None:
        old_path = self.image_files.get_path(item.image_name, image_subfolder=item.old_subfolder)
        new_path = self.image_files.get_path(item.image_name, image_subfolder=item.new_subfolder)
        old_thumbnail_path = self.image_files.get_path(
            item.image_name, thumbnail=True, image_subfolder=item.old_subfolder
        )
        new_thumbnail_path = self.image_files.get_path(
            item.image_name, thumbnail=True, image_subfolder=item.new_subfolder
        )
        old_exists = old_path.exists()
        new_exists = new_path.exists()
        if old_exists and new_exists:
            raise RuntimeError(f"Both old and new image files exist for {item.image_name}")
        if not old_exists and not new_exists:
            if item.is_intermediate:
                self._mark_missing_intermediate_moved(
                    job_id=job_id,
                    image_name=item.image_name,
                    old_path=old_path,
                    new_path=new_path,
                    old_thumbnail_path=old_thumbnail_path,
                    new_thumbnail_path=new_thumbnail_path,
                )
                return
            raise RuntimeError(f"Neither old nor new image file exists for {item.image_name}")

        old_thumbnail_exists = old_thumbnail_path.exists()
        new_thumbnail_exists = new_thumbnail_path.exists()
        if (
            old_exists
            and not new_exists
            and (
                (not old_thumbnail_exists and not new_thumbnail_exists)
                or (old_thumbnail_exists and new_thumbnail_exists)
            )
        ):
            # Generate the thumbnail while the source is still available. If this fails,
            # leave the source untouched so transient failures can be retried and corrupt
            # images can be repaired or removed by the operator.
            self._regenerate_thumbnail(old_path, new_thumbnail_path)

        if not old_exists and new_exists and not old_thumbnail_exists and not new_thumbnail_exists:
            self._regenerate_thumbnail(new_path, new_thumbnail_path)

        if old_exists:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old_path, new_path)
            self._fsync_file(new_path)
            self._fsync_dir(new_path.parent)
            self._fsync_dir(old_path.parent)

        old_thumbnail_exists = old_thumbnail_path.exists()
        new_thumbnail_exists = new_thumbnail_path.exists()
        if old_thumbnail_exists and new_thumbnail_exists:
            old_thumbnail_path.unlink()
            self._fsync_dir(old_thumbnail_path.parent)
        elif old_thumbnail_exists and not new_thumbnail_exists:
            new_thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(old_thumbnail_path, new_thumbnail_path)
            self._fsync_file(new_thumbnail_path)
            self._fsync_dir(new_thumbnail_path.parent)
            self._fsync_dir(old_thumbnail_path.parent)
        elif not new_thumbnail_exists:
            self._regenerate_thumbnail(new_path, new_thumbnail_path)

        self.image_files.evict_cache_paths([old_path, new_path, old_thumbnail_path, new_thumbnail_path])
        self.mark_item_moved(job_id, item.image_name)

    def _reconcile_destination_subfolder(self, item: PlannedImageMove) -> None:
        old_path = self.image_files.get_path(item.image_name, image_subfolder=item.old_subfolder)
        new_path = self.image_files.get_path(item.image_name, image_subfolder=item.new_subfolder)
        if old_path.exists() or not new_path.exists():
            return
        self._queries.image_moves.repoint_image(item.image_name, item.old_subfolder, item.new_subfolder)

    def cleanup_empty_source_dirs(self, job_id: int) -> None:
        for item in self._get_items(job_id):
            self._remove_empty_parents(
                self.image_files.get_path(item.image_name, image_subfolder=item.old_subfolder).parent,
                self.image_files.image_root,
            )
            self._remove_empty_parents(
                self.image_files.get_path(item.image_name, thumbnail=True, image_subfolder=item.old_subfolder).parent,
                self.image_files.thumbnail_root,
            )

    def commit_database_updates(self, job_id: int) -> int:
        def commit(q: Queries) -> int:
            moved_count = q.image_moves.repoint_moved_images(job_id)
            if q.image_moves.invalid_move_count(job_id):
                # Raised inside the transaction, so that no record is repointed.
                raise RuntimeError(f"Image move job {job_id} failed commit validation")
            error_messages = q.image_moves.error_messages(job_id)
            error_message = None
            if error_messages:
                error_message = "\n".join(message for message in error_messages if message) or (
                    "One or more image move items could not be completed"
                )
            q.image_moves.finish_job(job_id, error_message=error_message)
            return moved_count

        return self._queries.run(commit)

    def mark_item_moved(self, job_id: int, image_name: str) -> None:
        self._queries.image_moves.mark_item_moved(job_id, image_name)

    def record_job_error_message(self, job_id: int, message: str) -> None:
        self._queries.image_moves.set_job_error_message(job_id, message)

    def mark_item_unrecoverable(self, job_id: int, image_name: str, message: str) -> None:
        self._queries.image_moves.fail_item(job_id, image_name, message)

    def mark_job_unrecoverable(self, job_id: int, message: str) -> None:
        self._queries.image_moves.fail_job(job_id, message)

    def get_job(self, job_id: int) -> ImageMoveJob:
        job = self._queries.image_moves.job(job_id)
        if job is None:
            raise ValueError(f"Image move job not found: {job_id}")
        return ImageMoveJob(id=job.id, state=cast(MoveJobState, job.state), error_message=job.error_message)

    def get_latest_job(self) -> ImageMoveJob | None:
        job = self._queries.image_moves.latest_job()
        if job is None:
            return None
        return ImageMoveJob(id=job.id, state=cast(MoveJobState, job.state), error_message=job.error_message)

    def _get_new_subfolder(
        self, image_name: str, image_category: ImageCategory, is_intermediate: bool, created_at: str | datetime
    ) -> str:
        strategy = self._config.image_subfolder_strategy
        if strategy == "flat":
            return ""
        if strategy == "type":
            return "intermediate" if is_intermediate else image_category.value
        if strategy == "hash":
            return image_name[:2]
        if strategy == "date":
            timestamp = created_at if isinstance(created_at, datetime) else datetime.fromisoformat(created_at)
            return f"{timestamp.year}/{timestamp.month:02d}/{timestamp.day:02d}"
        raise ValueError(f"Unknown image subfolder strategy: {strategy}")

    def _get_items(self, job_id: int, include_terminal: bool = True) -> list[PlannedImageMove]:
        return [
            self._planned_move(image_name, old_subfolder, new_subfolder, is_intermediate)
            for image_name, old_subfolder, new_subfolder, is_intermediate in self._queries.image_moves.items(
                job_id, unfinished_only=not include_terminal
            )
        ]

    def _planned_move(
        self, image_name: str, old_subfolder: str, new_subfolder: str, is_intermediate: bool
    ) -> PlannedImageMove:
        return PlannedImageMove(
            image_name=image_name,
            old_subfolder=old_subfolder,
            new_subfolder=new_subfolder,
            is_intermediate=is_intermediate,
            old_path=self.image_files.get_path(image_name, image_subfolder=old_subfolder),
            new_path=self.image_files.get_path(image_name, image_subfolder=new_subfolder),
            old_thumbnail_path=self.image_files.get_path(image_name, thumbnail=True, image_subfolder=old_subfolder),
            new_thumbnail_path=self.image_files.get_path(image_name, thumbnail=True, image_subfolder=new_subfolder),
        )

    def _regenerate_thumbnail(self, image_path: Path, thumbnail_path: Path) -> None:
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with Image.open(image_path) as image:
                thumbnail = make_thumbnail(image)
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as e:
            raise UnreadableImageError(f"Unable to decode image {image_path}: {e}") from e
        with tempfile.NamedTemporaryFile(
            dir=thumbnail_path.parent, prefix=f".{thumbnail_path.name}.", suffix=".tmp", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            thumbnail.save(temp_path, format="WEBP")
            self._fsync_file(temp_path)
            os.replace(temp_path, thumbnail_path)
            self._fsync_file(thumbnail_path)
            self._fsync_dir(thumbnail_path.parent)
        finally:
            temp_path.unlink(missing_ok=True)

    def _mark_missing_intermediate_moved(
        self,
        job_id: int,
        image_name: str,
        old_path: Path,
        new_path: Path,
        old_thumbnail_path: Path,
        new_thumbnail_path: Path,
    ) -> None:
        for path in (old_thumbnail_path, new_thumbnail_path):
            if path.exists():
                path.unlink()
                self._fsync_dir(path.parent)
        self.image_files.evict_cache_paths([old_path, new_path, old_thumbnail_path, new_thumbnail_path])
        self.mark_item_moved(job_id, image_name)

    def _remove_empty_parents(self, start: Path, root: Path) -> None:
        root = root.resolve()
        current = start.resolve()
        while current != root and current.is_relative_to(root):
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent

    def _assert_same_filesystem(self, source: Path, destination: Path) -> None:
        source_parent = source.parent
        destination_parent = self._nearest_existing_parent(destination.parent)
        if source_parent.stat().st_dev != destination_parent.stat().st_dev:
            raise ValueError(f"Cross-filesystem image move is not supported: {source} -> {destination}")

    def _nearest_existing_parent(self, path: Path) -> Path:
        current = path
        while not current.exists():
            if current.parent == current:
                raise FileNotFoundError(f"No existing parent found for {path}")
            current = current.parent
        return current

    def _fsync_file(self, path: Path) -> None:
        try:
            with path.open("rb") as file:
                os.fsync(file.fileno())
        except OSError as e:
            self._logger.debug("Unable to fsync file: %s: %s", path, e)

    def _fsync_dir(self, path: Path) -> None:
        try:
            dir_fd = os.open(path, os.O_RDONLY)
        except OSError as e:
            self._logger.debug("Unable to open directory for fsync: %s: %s", path, e)
            return
        try:
            os.fsync(dir_fd)
        except OSError as e:
            self._logger.debug("Unable to fsync directory: %s: %s", path, e)
        finally:
            try:
                os.close(dir_fd)
            except OSError as e:
                self._logger.debug("Unable to close directory fsync handle: %s: %s", path, e)

    def _is_unrecoverable_error(self, error: Exception) -> bool:
        if isinstance(error, UnreadableImageError):
            return True
        return isinstance(error, RuntimeError) and (
            str(error).startswith("Both old and new image files exist")
            or str(error).startswith("Neither old nor new image file exists")
            or str(error).startswith("Image move job")
            and "has no items" in str(error)
        )
