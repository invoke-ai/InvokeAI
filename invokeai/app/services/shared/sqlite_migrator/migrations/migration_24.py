import json
import sqlite3
from logging import Logger
from pathlib import Path
from typing import NamedTuple

from pydantic import ValidationError

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
from invokeai.backend.model_manager.configs.factory import AnyModelConfigValidator


class NormalizeResult(NamedTuple):
    new_relative_path: str | None
    rollback_ops: list[tuple[Path, Path]]


class Migration24Callback:
    def __init__(self, app_config: InvokeAIAppConfig, logger: Logger) -> None:
        self._app_config = app_config
        self._logger = logger
        self._models_dir = app_config.models_path.resolve()

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        # Grab all model records
        cursor.execute("SELECT id, config FROM models;")
        rows = cursor.fetchall()

        for model_id, config_json in rows:
            try:
                config = AnyModelConfigValidator.validate_json(config_json)
            except ValidationError:
                # This could happen if the config schema changed in a way that makes old configs invalid. Unlikely
                # for users, more likely for devs testing out migration paths.
                self._logger.warning("Skipping model %s: invalid config schema", model_id)
                continue
            except json.JSONDecodeError:
                # This should never happen, as we use pydantic to serialize the config to JSON.
                self._logger.warning("Skipping model %s: invalid config JSON", model_id)
                continue

            # We'll use a savepoint so we can roll back the database update if something goes wrong, and a simple
            # rollback of file operations if needed.
            cursor.execute("SAVEPOINT migrate_model")
            try:
                new_relative_path, rollback_ops = self._normalize_model_storage(
                    key=config.key,
                    path_value=config.path,
                )
            except Exception as err:
                self._logger.error("Error normalizing model %s: %s", config.key, err)
                cursor.execute("ROLLBACK TO SAVEPOINT migrate_model")
                cursor.execute("RELEASE SAVEPOINT migrate_model")
                continue

            if new_relative_path is None:
                cursor.execute("RELEASE SAVEPOINT migrate_model")
                continue

            config.path = new_relative_path
            try:
                cursor.execute(
                    "UPDATE models SET config = ? WHERE id = ?;",
                    (config.model_dump_json(), model_id),
                )
            except Exception as err:
                self._logger.error("Database update failed for model %s: %s", config.key, err)
                cursor.execute("ROLLBACK TO SAVEPOINT migrate_model")
                cursor.execute("RELEASE SAVEPOINT migrate_model")
                self._rollback_file_ops(rollback_ops)
                continue

            cursor.execute("RELEASE SAVEPOINT migrate_model")

        self._prune_empty_directories()

    def _normalize_model_storage(self, key: str, path_value: str) -> NormalizeResult:
        models_dir = self._models_dir
        stored_path = Path(path_value)

        relative_path: Path | None
        if stored_path.is_absolute():
            # If the stored path is absolute, we need to check if it's inside the models directory, which means it is
            # an Invoke-managed model. If it's outside, it is user-managed we leave it alone.
            try:
                relative_path = stored_path.resolve().relative_to(models_dir)
            except ValueError:
                self._logger.info("Leaving user-managed model %s at %s", key, stored_path)
                return NormalizeResult(new_relative_path=None, rollback_ops=[])
        else:
            # Relative paths are always relative to the models directory and thus Invoke-managed.
            relative_path = stored_path

        # If the relative path is empty, assume something is wrong. Warn and skip.
        if not relative_path.parts:
            self._logger.warning("Skipping model %s: empty relative path", key)
            return NormalizeResult(new_relative_path=None, rollback_ops=[])

        # Sanity check: the path is relative. It should be present in the models directory.
        absolute_path = (models_dir / relative_path).resolve()
        if not absolute_path.exists():
            self._logger.warning(
                "Skipping model %s: expected model files at %s but nothing was found",
                key,
                absolute_path,
            )
            return NormalizeResult(new_relative_path=None, rollback_ops=[])

        if relative_path.parts[0] == key:
            # Already normalized. Still ensure the stored path is relative.
            normalized_path = relative_path.as_posix()
            # If the stored path is already the normalized path, no change is needed.
            new_relative_path = normalized_path if stored_path.as_posix() != normalized_path else None
            return NormalizeResult(new_relative_path=new_relative_path, rollback_ops=[])

        # We'll store the file operations we perform so we can roll them back if needed.
        rollback_ops: list[tuple[Path, Path]] = []

        # Destination directory is models_dir/<key> - a flat directory structure.
        destination_dir = models_dir / key

        try:
            if absolute_path.is_file():
                destination_dir.mkdir(parents=True, exist_ok=True)
                dest_file = destination_dir / absolute_path.name
                # This really shouldn't happen.
                if dest_file.exists():
                    self._logger.warning(
                        "Destination for model %s already exists at %s; skipping move",
                        key,
                        dest_file,
                    )
                    return NormalizeResult(new_relative_path=None, rollback_ops=[])

                self._logger.info("Moving model file %s -> %s", absolute_path, dest_file)

                # `Path.rename()` effectively moves the file or directory.
                absolute_path.rename(dest_file)
                rollback_ops.append((dest_file, absolute_path))

                return NormalizeResult(
                    new_relative_path=(Path(key) / dest_file.name).as_posix(),
                    rollback_ops=rollback_ops,
                )

            if absolute_path.is_dir():
                dest_path = destination_dir
                # This really shouldn't happen.
                if dest_path.exists():
                    self._logger.warning(
                        "Destination directory %s already exists for model %s; skipping",
                        dest_path,
                        key,
                    )
                    return NormalizeResult(new_relative_path=None, rollback_ops=[])

                self._logger.info("Moving model directory %s -> %s", absolute_path, dest_path)

                # `Path.rename()` effectively moves the file or directory.
                absolute_path.rename(dest_path)
                rollback_ops.append((dest_path, absolute_path))

                return NormalizeResult(
                    new_relative_path=Path(key).as_posix(),
                    rollback_ops=rollback_ops,
                )

            # Maybe a broken symlink or something else weird?
            self._logger.warning("Skipping model %s: path %s is neither a file nor directory", key, absolute_path)
            return NormalizeResult(new_relative_path=None, rollback_ops=[])
        except Exception:
            self._rollback_file_ops(rollback_ops)
            raise

    def _rollback_file_ops(self, rollback_ops: list[tuple[Path, Path]]) -> None:
        # This is a super-simple rollback that just reverses the move operations we performed.
        for source, destination in reversed(rollback_ops):
            try:
                if source.exists():
                    source.rename(destination)
            except Exception as err:
                self._logger.error("Failed to rollback move %s -> %s: %s", source, destination, err)

    def _prune_empty_directories(self) -> None:
        # These directories are system directories we want to keep even if empty. Technically, the app should not
        # have any problems if these are removed, creating them as needed, but it's cleaner to just leave them alone.
        keep_names = {"model_images", ".download_cache"}
        keep_dirs = {self._models_dir / name for name in keep_names}
        removed_dirs: set[Path] = set()

        # Walk the models directory tree from the bottom up, removing empty directories. We sort by path length
        # descending to ensure we visit children before parents.
        for directory in sorted(self._models_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if not directory.is_dir():
                continue
            if directory == self._models_dir:
                continue
            if any(directory == keep or keep in directory.parents for keep in keep_dirs):
                continue

            try:
                next(directory.iterdir())
            except StopIteration:
                try:
                    directory.rmdir()
                    removed_dirs.add(directory)
                    self._logger.debug("Removed empty directory %s", directory)
                except OSError:
                    # Directory not empty (or some other error) - bail out.
                    self._logger.warning("Failed to prune directory %s - not empty?", directory)
                    continue
            except OSError:
                continue

        self._logger.info("Pruned %d empty directories under %s", len(removed_dirs), self._models_dir)


def build_migration_24(app_config: InvokeAIAppConfig, logger: Logger) -> Migration:
    """Builds the migration object for migrating from version 23 to version 24.

    This migration normalizes on-disk model storage so that each model lives within
    a directory named by its key inside the Invoke-managed models directory, and
    updates database records to reference the new relative paths.

    This migration behaves a bit differently than others. Because it involves FS operations, if we rolled the
    DB back on any failure, we could leave the FS out of sync with the DB. Instead, we use savepoints
    to roll back individual model updates on failure, and we roll back any FS operations we performed
    for that model.

    If a model cannot be migrated for any reason (invalid config, missing files, FS errors, DB errors), we log a
    warning and skip it, leaving it in its original state and location. The model will still work, but it will be in
    the "wrong" location on disk.
    """

    return Migration(
        from_version=23,
        to_version=24,
        callback=Migration24Callback(app_config=app_config, logger=logger),
    )
