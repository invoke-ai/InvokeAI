"""Service for finding and removing orphaned model files.

Orphaned models are files in the models directory that are not referenced
in the database models table.
"""

import json
import shutil
from pathlib import Path
from typing import Set

from pydantic import BaseModel, Field

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class OrphanedModelInfo(BaseModel):
    """Information about an orphaned model directory."""

    path: str = Field(description="Relative path to the orphaned directory from models root")
    absolute_path: str = Field(description="Absolute path to the orphaned directory")
    files: list[str] = Field(description="List of model files in this directory")
    size_bytes: int = Field(description="Total size of all files in bytes")


class OrphanedModelsService:
    """Service for finding and removing orphaned model files."""

    # Common model file extensions
    MODEL_EXTENSIONS = {
        ".safetensors",
        ".ckpt",
        ".pt",
        ".pth",
        ".bin",
        ".onnx",
        ".gguf",
    }

    # Directories to skip during scan
    SKIP_DIRS = {
        ".download_cache",
        ".convert_cache",
        "__pycache__",
        ".git",
    }

    def __init__(self, config: InvokeAIAppConfig, db: SqliteDatabase):
        """Initialize the service.

        Args:
            config: Application configuration containing models path
            db: Database connection for querying registered models
        """
        self._config = config
        self._db = db

    def find_orphaned_models(self) -> list[OrphanedModelInfo]:
        """Find all orphaned model directories.

        Returns:
            List of OrphanedModelInfo objects describing orphaned directories
        """
        models_path = self._config.models_path

        # Get all model directories registered in the database
        db_model_directories = self._get_registered_model_directories(models_path)

        # Find all model files on disk
        disk_model_files = self._get_all_model_files(models_path)

        # Find orphaned files (files not under any registered model directory)
        orphaned_files = set()
        for disk_file in disk_model_files:
            is_under_model_dir = False
            for model_dir in db_model_directories:
                try:
                    # Check if disk_file is under model_dir
                    disk_file.relative_to(model_dir)
                    is_under_model_dir = True
                    break
                except ValueError:
                    # Not under this model directory, continue checking
                    continue

            if not is_under_model_dir:
                orphaned_files.add(disk_file)

        # Group orphaned files by their top-level directory
        orphaned_dirs_map: dict[Path, list[Path]] = {}
        for orphaned_file in orphaned_files:
            # Get the top-level directory relative to models_path
            try:
                rel_path = orphaned_file.relative_to(models_path)
                if rel_path.parts:
                    top_level_dir = models_path / rel_path.parts[0]
                    if top_level_dir not in orphaned_dirs_map:
                        orphaned_dirs_map[top_level_dir] = []
                    orphaned_dirs_map[top_level_dir].append(orphaned_file)
            except ValueError:
                # File is outside models_path, skip it
                continue

        # Convert to OrphanedModelInfo objects
        result = []
        for dir_path, files in orphaned_dirs_map.items():
            # Calculate total size
            total_size = sum(f.stat().st_size for f in files if f.exists())

            # Get relative file paths
            file_names = [str(f.relative_to(dir_path)) for f in files]

            result.append(
                OrphanedModelInfo(
                    path=str(dir_path.relative_to(models_path)),
                    absolute_path=str(dir_path),
                    files=file_names,
                    size_bytes=total_size,
                )
            )

        return result

    def delete_orphaned_models(self, orphaned_paths: list[str]) -> dict[str, str]:
        """Delete the specified orphaned model directories.

        Args:
            orphaned_paths: List of relative paths to delete (relative to models root)

        Returns:
            Dictionary mapping paths to status messages ("deleted" or error message)
        """
        models_path = self._config.models_path
        results = {}

        for rel_path in orphaned_paths:
            try:
                full_path = models_path / rel_path
                if not full_path.exists():
                    results[rel_path] = "error: path does not exist"
                    continue

                # Safety check: ensure path is under models directory
                try:
                    full_path.relative_to(models_path)
                except ValueError:
                    results[rel_path] = "error: path is not under models directory"
                    continue

                # Delete the directory
                shutil.rmtree(full_path)
                results[rel_path] = "deleted"

            except Exception as e:
                results[rel_path] = f"error: {str(e)}"

        return results

    def _get_registered_model_directories(self, models_dir: Path) -> Set[Path]:
        """Get the set of all model directories from the database."""
        model_directories = set()

        with self._db.transaction() as cursor:
            cursor.execute("SELECT config FROM models")
            rows = cursor.fetchall()

            for row in rows:
                try:
                    config = json.loads(row[0])
                    if "path" in config and config["path"]:
                        path_str = config["path"]
                        path = Path(path_str)

                        # If the path is relative, resolve it relative to models_dir
                        if not path.is_absolute():
                            full_path = (models_dir / path).resolve()
                        else:
                            full_path = path.resolve()

                        # Extract the top-level directory under models_dir
                        try:
                            rel_path = full_path.relative_to(models_dir)
                            if rel_path.parts:
                                top_level_dir = models_dir / rel_path.parts[0]
                                model_directories.add(top_level_dir.resolve())
                        except ValueError:
                            # Path is not relative to models_dir
                            model_directories.add(full_path)

                except (json.JSONDecodeError, KeyError, TypeError):
                    # Skip invalid model configs
                    continue

        return model_directories

    def _get_all_model_files(self, models_path: Path) -> Set[Path]:
        """Get all model files in the models directory."""
        model_files = set()

        for item in models_path.rglob("*"):
            # Skip directories we don't want to scan
            if any(skip_dir in item.parts for skip_dir in self.SKIP_DIRS):
                continue

            if item.is_file() and item.suffix.lower() in self.MODEL_EXTENSIONS:
                model_files.add(item.resolve())

        return model_files
