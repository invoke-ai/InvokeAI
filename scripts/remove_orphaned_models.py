#!/usr/bin/env python
"""Script to remove orphaned model files from INVOKEAI_ROOT directory.

Orphaned models are ones that appear in the INVOKEAI_ROOT/models directory,
but which are not referenced in the database `models` table.
"""

import argparse
import datetime
import json
import locale
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Set

import yaml


class ConfigMapper:
    """Configuration loader for InvokeAI paths."""

    YAML_FILENAME = "invokeai.yaml"
    DATABASE_FILENAME = "invokeai.db"
    DEFAULT_DB_DIR = "databases"
    DEFAULT_MODELS_DIR = "models"

    def __init__(self):
        self.database_path = None
        self.database_backup_dir = None
        self.models_path = None

    def load(self, root_path: Path) -> bool:
        """Load configuration from root directory."""
        yaml_path = root_path / self.YAML_FILENAME
        if not yaml_path.exists():
            print(f"Unable to find {self.YAML_FILENAME} at {yaml_path}!")
            return False

        db_dir, models_dir = self._load_paths_from_yaml_file(yaml_path)

        if db_dir is None:
            db_dir = self.DEFAULT_DB_DIR
            print(f"The {self.YAML_FILENAME} file was found but is missing the db_dir setting! Defaulting to {db_dir}")

        if models_dir is None:
            models_dir = self.DEFAULT_MODELS_DIR
            print(
                f"The {self.YAML_FILENAME} file was found but is missing the models_dir setting! Defaulting to {models_dir}"
            )

        # Set database path
        if os.path.isabs(db_dir):
            self.database_path = Path(db_dir) / self.DATABASE_FILENAME
        else:
            self.database_path = root_path / db_dir / self.DATABASE_FILENAME

        self.database_backup_dir = self.database_path.parent / "backup"

        # Set models path
        if os.path.isabs(models_dir):
            self.models_path = Path(models_dir)
        else:
            self.models_path = root_path / models_dir

        db_exists = self.database_path.exists()
        models_exists = self.models_path.exists()

        print(f"Found {self.YAML_FILENAME} file at {yaml_path}:")
        print(f"  Database : {self.database_path} - {'Exists!' if db_exists else 'Not Found!'}")
        print(f"  Models   : {self.models_path} - {'Exists!' if models_exists else 'Not Found!'}")

        if db_exists and models_exists:
            return True
        else:
            print(
                "\nOne or more paths specified in invokeai.yaml do not exist. Please inspect/correct the configuration."
            )
            return False

    def _load_paths_from_yaml_file(self, yaml_path: Path):
        """Load paths from YAML configuration file."""
        try:
            with open(yaml_path, "rt", encoding=locale.getpreferredencoding()) as file:
                yamlinfo = yaml.safe_load(file)
                db_dir = yamlinfo.get("InvokeAI", {}).get("Paths", {}).get("db_dir", None)
                models_dir = yamlinfo.get("InvokeAI", {}).get("Paths", {}).get("models_dir", None)
                return db_dir, models_dir
        except Exception as e:
            print(f"Failed to load paths from yaml file! {yaml_path}! Error: {e}")
            return None, None


class DatabaseMapper:
    """Class to abstract database functionality."""

    def __init__(self, database_path: Path, database_backup_dir: Path):
        self.database_path = database_path
        self.database_backup_dir = database_backup_dir
        self.connection = None
        self.cursor = None

    def backup(self, timestamp_string: str):
        """Take a backup of the database."""
        if not self.database_backup_dir.exists():
            print(f"Database backup directory {self.database_backup_dir} does not exist -> creating...", end="")
            self.database_backup_dir.mkdir(parents=True, exist_ok=True)
            print("Done!")

        database_backup_path = self.database_backup_dir / f"backup-{timestamp_string}-invokeai.db"
        print(f"Making DB Backup at {database_backup_path}...", end="")
        shutil.copy2(self.database_path, database_backup_path)
        print("Done!")

    def connect(self):
        """Open connection to the database."""
        self.connection = sqlite3.connect(str(self.database_path))
        self.cursor = self.connection.cursor()

    def get_all_model_directories(self, models_dir: Path) -> Set[Path]:
        """Get the set of all model directories from the database.

        A model directory is the top-level directory under models/ that contains
        the model files. If the path in the database is just a directory, that's
        the model directory. If it's a file path, we extract the first directory
        component.

        Args:
            models_dir: The root models directory path. Relative paths from the database
                        will be resolved relative to this directory.

        Returns:
            Set of absolute Path objects for model directories.
        """
        sql_get_models = "SELECT config FROM models"
        self.cursor.execute(sql_get_models)
        rows = self.cursor.fetchall()
        model_directories = set()
        for row in rows:
            try:
                config = json.loads(row[0])
                if "path" in config and config["path"]:
                    path_str = config["path"]
                    # Convert to Path object
                    path = Path(path_str)

                    # If the path is relative, resolve it relative to models_dir
                    # If it's absolute, use it as-is
                    if not path.is_absolute():
                        full_path = (models_dir / path).resolve()
                    else:
                        full_path = path.resolve()

                    # Extract the top-level directory under models_dir
                    # This handles both cases:
                    # 1. path is "model-id" -> model-id is the directory
                    # 2. path is "model-id/file.safetensors" -> model-id is the directory
                    try:
                        # Get the relative path from models_dir
                        rel_path = full_path.relative_to(models_dir)
                        # Get the first component (top-level directory)
                        if rel_path.parts:
                            top_level_dir = models_dir / rel_path.parts[0]
                            model_directories.add(top_level_dir.resolve())
                    except ValueError:
                        # Path is not relative to models_dir, use the path itself
                        # This handles absolute paths outside models_dir
                        model_directories.add(full_path)

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Warning: Failed to parse model config: {e}")
                continue
        return model_directories

    def disconnect(self):
        """Disconnect from the database."""
        if self.cursor is not None:
            self.cursor.close()
        if self.connection is not None:
            self.connection.close()


class ModelFileMapper:
    """Class to handle model file system operations."""

    # Common model file extensions
    MODEL_EXTENSIONS = {
        ".safetensors",
        ".ckpt",
        ".pt",
        ".pth",
        ".bin",
        ".onnx",
    }

    # Directories to skip during scan
    SKIP_DIRS = {
        ".download_cache",
        ".convert_cache",
        "__pycache__",
        ".git",
    }

    def __init__(self, models_path: Path):
        self.models_path = models_path

    def get_all_model_directories(self) -> Set[Path]:
        """
        Get all directories in the models path that contain model files.
        Returns a set of directory paths that contain at least one model file.
        """
        model_dirs = set()

        for item in self.models_path.rglob("*"):
            # Skip directories we don't want to scan
            if any(skip_dir in item.parts for skip_dir in self.SKIP_DIRS):
                continue

            if item.is_file() and item.suffix.lower() in self.MODEL_EXTENSIONS:
                # Add the parent directory of the model file
                model_dirs.add(item.parent)

        return model_dirs

    def get_all_model_files(self) -> Set[Path]:
        """Get all model files in the models directory."""
        model_files = set()

        for item in self.models_path.rglob("*"):
            # Skip directories we don't want to scan
            if any(skip_dir in item.parts for skip_dir in self.SKIP_DIRS):
                continue

            if item.is_file() and item.suffix.lower() in self.MODEL_EXTENSIONS:
                model_files.add(item.resolve())

        return model_files

    def remove_file(self, file_path: Path):
        """Remove a single model file."""
        try:
            file_path.unlink()
            print(f"  Deleted file: {file_path}")
        except Exception as e:
            print(f"  Error deleting {file_path}: {e}")

    def remove_directory_if_empty(self, directory: Path):
        """Remove a directory if it's empty (after removing files)."""
        try:
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
                print(f"  Deleted empty directory: {directory}")
        except Exception as e:
            print(f"  Error removing directory {directory}: {e}")


class OrphanedModelsApp:
    """Main application class for removing orphaned model files."""

    def __init__(self, delete_without_confirm: bool = False):
        self.delete_without_confirm = delete_without_confirm
        self.orphaned_count = 0

    def find_orphaned_files_by_directory(
        self, file_mapper: ModelFileMapper, db_mapper: DatabaseMapper, models_path: Path
    ) -> dict[Path, list[Path]]:
        """Find orphaned files grouped by their parent directory.

        A file is orphaned if it's NOT under any model directory registered in the database.
        Model directories are extracted from the database paths - if a path is
        'model-id/file.safetensors', then 'model-id' is the model directory and ALL files
        under it belong to that model.
        """
        print("\nScanning models directory for orphaned models...")

        # Get all model files on disk
        disk_model_files = file_mapper.get_all_model_files()
        print(f"Found {len(disk_model_files)} model directories on disk")

        # Get all model directories from database
        db_model_directories = db_mapper.get_all_model_directories(models_path)
        print(f"Found {len(db_model_directories)} model directories in database")

        # Find orphaned files (files on disk but not under any registered model directory)
        orphaned_files = set()
        for disk_file in disk_model_files:
            # Check if this file is under any registered model directory
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

        # Group orphaned files by their parent directory
        orphaned_dirs = {}
        for orphaned_file in orphaned_files:
            parent = orphaned_file.parent
            if parent not in orphaned_dirs:
                orphaned_dirs[parent] = []
            orphaned_dirs[parent].append(orphaned_file)

        return orphaned_dirs

    def ask_to_continue(self) -> bool:
        """Ask user whether they want to continue with the operation."""
        while True:
            try:
                input_choice = input("\nDo you wish to delete these models? (Y or N) [N]: ")
                # Default to 'N' if user presses Enter without input
                if input_choice.strip() == "":
                    return False
                if str.lower(input_choice) == "y":
                    return True
                if str.lower(input_choice) == "n":
                    return False
                print("Please enter Y or N")
            except (KeyboardInterrupt, EOFError):
                return False

    def remove_orphaned_models(self, config: ConfigMapper, file_mapper: ModelFileMapper, db_mapper: DatabaseMapper):
        """Remove orphaned model directories."""
        print("\n" + "=" * 80)
        print("= Remove Orphaned Model Files")
        print("=" * 80)
        print("\nThis operation will find model files in the models directory that are not")
        print("referenced in the database and remove them.")
        print()
        print(f"Database File Path : {config.database_path}")
        print(f"Models Directory   : {config.models_path}")
        print()
        print("Notes:")
        print("- A database backup will be created before any changes")
        print("- Model files not referenced in the database will be permanently deleted")
        print("- This operation cannot be undone (except by restoring the deleted files)")
        print()

        # Connect to database and find orphaned files
        db_mapper.connect()
        try:
            orphaned_dirs = self.find_orphaned_files_by_directory(file_mapper, db_mapper, config.models_path)

            if not orphaned_dirs:
                print("\nNo orphaned model files found!")
                return

            print(f"\nFound {len(orphaned_dirs)} directories with orphaned model files:")
            print()

            for directory, files in sorted(orphaned_dirs.items()):
                print(f"Directory: {directory}")
                for file in sorted(files):
                    print(f"  - {file.name}")
                print()

            self.orphaned_count = sum(len(files) for files in orphaned_dirs.values())
            print(f"Total orphans: {self.orphaned_count}")

            # Ask for confirmation unless --delete flag is used
            if not self.delete_without_confirm:
                if not self.ask_to_continue():
                    print("\nOperation cancelled by user.")
                    self.orphaned_count = 0  # Reset count since no files were removed
                    return

            # Create database backup with timestamp
            timestamp_string = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
            db_mapper.backup(timestamp_string)

            # Delete the orphaned files
            print("\nDeleting orphaned model files...")
            for directory, files in sorted(orphaned_dirs.items()):
                for file in sorted(files):
                    file_mapper.remove_file(file)
                # After removing files, clean up the directory if it's now empty
                file_mapper.remove_directory_if_empty(directory)

        finally:
            db_mapper.disconnect()

    def main(self, root_path: Path):
        """Main entry point."""
        print("\n" + "=" * 80)
        print("Orphaned Model Files Cleanup for InvokeAI")
        print("=" * 80 + "\n")

        config_mapper = ConfigMapper()
        if not config_mapper.load(root_path):
            print("\nInvalid configuration...exiting.\n")
            return 1

        file_mapper = ModelFileMapper(config_mapper.models_path)
        db_mapper = DatabaseMapper(config_mapper.database_path, config_mapper.database_backup_dir)

        try:
            self.remove_orphaned_models(config_mapper, file_mapper, db_mapper)
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            return 1
        except Exception as e:
            print(f"\n\nError during operation: {e}")
            import traceback

            traceback.print_exc()
            return 1

        print("\n" + "=" * 80)
        print("= Operation Complete")
        print("=" * 80)
        print(f"\nOrphaned model files removed: {self.orphaned_count}")
        print()

        return 0


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Remove orphaned model files from InvokeAI installation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script finds and removes model files that exist in the models directory
but are not referenced in the InvokeAI database. This can happen if:
- Models were manually deleted from the database
- The database was reset but model files were kept
- Files were manually copied into the models directory

By default, the script will list orphaned files and ask for confirmation
before deleting them.
""",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=os.environ.get("INVOKEAI_ROOT", "."),
        help="InvokeAI root directory (default: $INVOKEAI_ROOT or current directory)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete orphan model files without asking for confirmation",
    )
    args = parser.parse_args()

    # Resolve the root path
    root_path = Path(args.root).resolve()
    if not root_path.exists():
        print(f"Error: Root directory does not exist: {root_path}")
        return 1

    app = OrphanedModelsApp(delete_without_confirm=args.delete)
    return app.main(root_path)


if __name__ == "__main__":
    exit(main())
