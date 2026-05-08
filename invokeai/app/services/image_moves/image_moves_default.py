import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence, cast

from PIL import Image

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.image_files.image_files_base import ImageFileStorageBase
from invokeai.app.services.image_records.image_records_common import ImageCategory
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.util.thumbnails import make_thumbnail

MoveJobState = Literal["planned", "moving", "moved", "committed", "error"]
MoveItemState = Literal["planned", "moved", "committed", "error"]


@dataclass(frozen=True)
class PlannedImageMove:
    image_name: str
    old_subfolder: str
    new_subfolder: str
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


class ImageMoveService:
    def __init__(
        self,
        db: SqliteDatabase,
        image_files: ImageFileStorageBase,
        config: InvokeAIAppConfig,
        logger,
    ) -> None:
        self._db = db
        self.image_files = image_files
        self._config = config
        self._logger = logger

    def move_all_images(self) -> ImageMoveResult:
        recovered = self.startup_recovery()
        last_image_name = ""
        planned = 0
        committed = recovered.committed
        errors = recovered.errors

        while True:
            moves = self.plan_batch(last_image_name=last_image_name, limit=100)
            if not moves:
                next_name = self._next_image_name(last_image_name)
                if next_name is None:
                    break
                last_image_name = next_name
                continue

            job_id = self.create_move_job(moves)
            planned += len(moves)
            try:
                self.perform_filesystem_moves(job_id)
                self.commit_database_updates(job_id)
                committed += len(moves)
            except Exception as e:
                errors += 1
                self.record_job_error_message(job_id, str(e))
                raise
            last_image_name = moves[-1].image_name

        return ImageMoveResult(planned=planned, committed=committed, errors=errors)

    def startup_recovery(self) -> ImageMoveResult:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT id FROM image_subfolder_move_jobs
                WHERE state IN ('planned', 'moving', 'moved')
                ORDER BY id;
                """
            )
            job_ids = [cast(int, row[0]) for row in cursor.fetchall()]

        committed = 0
        errors = 0
        for job_id in job_ids:
            try:
                self.complete_partial_filesystem_moves(job_id)
                self.commit_database_updates(job_id)
                committed += len(self._get_items(job_id))
            except Exception as e:
                errors += 1
                if self._is_unrecoverable_error(e):
                    self.mark_job_unrecoverable(job_id, str(e))
                else:
                    self.record_job_error_message(job_id, str(e))
        return ImageMoveResult(committed=committed, errors=errors)

    def plan_batch(self, last_image_name: str, limit: int) -> list[PlannedImageMove]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT image_name, image_subfolder, image_category, is_intermediate, created_at
                FROM images
                WHERE image_name > ?
                  AND deleted_at IS NULL
                ORDER BY image_name
                LIMIT ?;
                """,
                (last_image_name, limit),
            )
            rows = cursor.fetchall()

        moves: list[PlannedImageMove] = []
        for row in rows:
            image_name = cast(str, row["image_name"])
            old_subfolder = cast(str, row["image_subfolder"] or "")
            new_subfolder = self._get_new_subfolder(
                image_name=image_name,
                image_category=ImageCategory(row["image_category"]),
                is_intermediate=bool(row["is_intermediate"]),
                created_at=row["created_at"],
            )
            if new_subfolder == old_subfolder:
                continue
            moves.append(
                PlannedImageMove(
                    image_name=image_name,
                    old_subfolder=old_subfolder,
                    new_subfolder=new_subfolder,
                    old_path=self.image_files.get_path(image_name, image_subfolder=old_subfolder),
                    new_path=self.image_files.get_path(image_name, image_subfolder=new_subfolder),
                    old_thumbnail_path=self.image_files.get_path(
                        image_name, thumbnail=True, image_subfolder=old_subfolder
                    ),
                    new_thumbnail_path=self.image_files.get_path(
                        image_name, thumbnail=True, image_subfolder=new_subfolder
                    ),
                )
            )
        self.preflight_moves(moves)
        return moves

    def create_move_job(self, moves: Sequence[PlannedImageMove]) -> int:
        if not moves:
            raise ValueError("Cannot create an image move job with no items")
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT 1
                FROM image_subfolder_move_jobs
                WHERE state NOT IN ('committed', 'error')
                LIMIT 1;
                """
            )
            if cursor.fetchone() is not None:
                raise ValueError("Cannot create image move job while another active image move job exists")
            cursor.execute("INSERT INTO image_subfolder_move_jobs (state) VALUES ('planned');")
            job_id = cast(int, cursor.lastrowid)
            cursor.executemany(
                """--sql
                INSERT INTO image_subfolder_move_items (
                    job_id,
                    image_name,
                    old_subfolder,
                    new_subfolder,
                    old_path,
                    new_path,
                    old_thumbnail_path,
                    new_thumbnail_path,
                    state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'planned');
                """,
                [
                    (
                        job_id,
                        move.image_name,
                        move.old_subfolder,
                        move.new_subfolder,
                        str(move.old_path),
                        str(move.new_path),
                        str(move.old_thumbnail_path),
                        str(move.new_thumbnail_path),
                    )
                    for move in moves
                ],
            )
            return job_id

    def preflight_moves(self, moves: Sequence[PlannedImageMove]) -> None:
        destinations: set[Path] = set()
        thumbnail_destinations: set[Path] = set()
        for move in moves:
            if not move.old_path.exists():
                raise FileNotFoundError(f"Source image does not exist: {move.old_path}")
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
            if self._has_active_job_for_image(move.image_name):
                raise ValueError(f"Image {move.image_name} already has an active image move job")
            self._assert_same_filesystem(move.old_path, move.new_path)
            if move.old_thumbnail_path.exists():
                if move.new_thumbnail_path.exists():
                    raise FileExistsError(f"Destination thumbnail already exists: {move.new_thumbnail_path}")
                self._assert_same_filesystem(move.old_thumbnail_path, move.new_thumbnail_path)

    def perform_filesystem_moves(self, job_id: int) -> None:
        self._set_job_state(job_id, "moving")
        self.complete_partial_filesystem_moves(job_id)
        self.cleanup_empty_source_dirs(job_id)
        self._set_job_state(job_id, "moved")

    def complete_partial_filesystem_moves(self, job_id: int) -> None:
        items = self._get_items(job_id)
        if not items:
            raise ValueError(f"Image move job {job_id} has no items")
        for item in items:
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
                raise RuntimeError(f"Neither old nor new image file exists for {item.image_name}")
            if old_exists:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(old_path, new_path)
                self._fsync_file(new_path)
                self._fsync_dir(new_path.parent)
                self._fsync_dir(old_path.parent)

            old_thumbnail_exists = old_thumbnail_path.exists()
            new_thumbnail_exists = new_thumbnail_path.exists()
            if old_thumbnail_exists and new_thumbnail_exists:
                self._regenerate_thumbnail(new_path, new_thumbnail_path)
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

    def commit_database_updates(self, job_id: int) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                UPDATE images
                SET image_subfolder = (
                    SELECT item.new_subfolder
                    FROM image_subfolder_move_items AS item
                    WHERE item.job_id = ?
                      AND item.image_name = images.image_name
                )
                WHERE image_name IN (
                    SELECT image_name
                    FROM image_subfolder_move_items
                    WHERE job_id = ?
                      AND state = 'moved'
                )
                AND image_subfolder = (
                    SELECT item.old_subfolder
                    FROM image_subfolder_move_items AS item
                    WHERE item.job_id = ?
                      AND item.image_name = images.image_name
                );
                """,
                (job_id, job_id, job_id),
            )
            cursor.execute(
                """--sql
                SELECT COUNT(*)
                FROM image_subfolder_move_items AS item
                LEFT JOIN images ON images.image_name = item.image_name
                WHERE item.job_id = ?
                  AND (
                    images.image_name IS NULL
                    OR images.deleted_at IS NOT NULL
                    OR images.image_subfolder != item.new_subfolder
                  );
                """,
                (job_id,),
            )
            invalid_count = cast(int, cursor.fetchone()[0])
            if invalid_count:
                raise RuntimeError(f"Image move job {job_id} failed commit validation")
            cursor.execute(
                "UPDATE image_subfolder_move_items SET state = 'committed' WHERE job_id = ?;",
                (job_id,),
            )
            cursor.execute(
                "UPDATE image_subfolder_move_jobs SET state = 'committed', error_message = NULL WHERE id = ?;",
                (job_id,),
            )

    def mark_item_moved(self, job_id: int, image_name: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                "UPDATE image_subfolder_move_items SET state = 'moved' WHERE job_id = ? AND image_name = ?;",
                (job_id, image_name),
            )

    def record_job_error_message(self, job_id: int, message: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                "UPDATE image_subfolder_move_jobs SET error_message = ? WHERE id = ?;",
                (message, job_id),
            )

    def mark_job_unrecoverable(self, job_id: int, message: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                "UPDATE image_subfolder_move_jobs SET state = 'error', error_message = ? WHERE id = ?;",
                (message, job_id),
            )
            cursor.execute(
                "UPDATE image_subfolder_move_items SET state = 'error', error_message = ? WHERE job_id = ?;",
                (message, job_id),
            )

    def get_job(self, job_id: int) -> ImageMoveJob:
        with self._db.transaction() as cursor:
            cursor.execute(
                "SELECT id, state, error_message FROM image_subfolder_move_jobs WHERE id = ?;",
                (job_id,),
            )
            row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Image move job not found: {job_id}")
        return ImageMoveJob(
            id=cast(int, row["id"]), state=cast(MoveJobState, row["state"]), error_message=row["error_message"]
        )

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

    def _get_items(self, job_id: int) -> list[PlannedImageMove]:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT image_name, old_subfolder, new_subfolder
                FROM image_subfolder_move_items
                WHERE job_id = ?
                ORDER BY image_name;
                """,
                (job_id,),
            )
            rows = cursor.fetchall()
        return [
            PlannedImageMove(
                image_name=row["image_name"],
                old_subfolder=row["old_subfolder"],
                new_subfolder=row["new_subfolder"],
                old_path=self.image_files.get_path(row["image_name"], image_subfolder=row["old_subfolder"]),
                new_path=self.image_files.get_path(row["image_name"], image_subfolder=row["new_subfolder"]),
                old_thumbnail_path=self.image_files.get_path(
                    row["image_name"], thumbnail=True, image_subfolder=row["old_subfolder"]
                ),
                new_thumbnail_path=self.image_files.get_path(
                    row["image_name"], thumbnail=True, image_subfolder=row["new_subfolder"]
                ),
            )
            for row in rows
        ]

    def _set_job_state(self, job_id: int, state: MoveJobState) -> None:
        with self._db.transaction() as cursor:
            cursor.execute("UPDATE image_subfolder_move_jobs SET state = ? WHERE id = ?;", (state, job_id))

    def _has_active_job_for_image(self, image_name: str) -> bool:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT 1
                FROM image_subfolder_move_items AS item
                JOIN image_subfolder_move_jobs AS job ON job.id = item.job_id
                WHERE item.image_name = ?
                  AND job.state NOT IN ('committed', 'error')
                LIMIT 1;
                """,
                (image_name,),
            )
            return cursor.fetchone() is not None

    def _next_image_name(self, last_image_name: str) -> str | None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT image_name FROM images
                WHERE image_name > ?
                  AND deleted_at IS NULL
                ORDER BY image_name
                LIMIT 1;
                """,
                (last_image_name,),
            )
            row = cursor.fetchone()
        return None if row is None else cast(str, row[0])

    def _regenerate_thumbnail(self, image_path: Path, thumbnail_path: Path) -> None:
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            thumbnail = make_thumbnail(image)
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
        with path.open("rb") as file:
            os.fsync(file.fileno())

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
            os.close(dir_fd)

    def _is_unrecoverable_error(self, error: Exception) -> bool:
        return isinstance(error, RuntimeError) and (
            str(error).startswith("Both old and new image files exist")
            or str(error).startswith("Neither old nor new image file exists")
            or str(error).startswith("Image move job")
            and "has no items" in str(error)
        )
