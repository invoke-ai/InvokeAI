# pylint: disable=line-too-long
# pylint: disable=broad-exception-caught
# pylint: disable=missing-function-docstring
"""Script to peform db maintenance and outputs directory management."""

import argparse
import datetime
import enum
import glob
import os
import shutil
from pathlib import Path
from typing import Optional

import PIL
import PIL.ImageOps
import PIL.PngImagePlugin

from invokeai.app.services.config.config_default import InvokeAIAppConfig, load_config_from_root
from invokeai.app.services.image_records.image_records_default import ImageRecordStorage
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.startup import open_migrated_database, redacted_database_url
from invokeai.backend.util.logging import InvokeAILogger


class ConfigMapper:
    """The paths of an Invoke AI install, from its config."""

    TIMESTAMP_STRING = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    config: Optional[InvokeAIAppConfig] = None
    database_path = None
    database_backup_dir = None
    outputs_path = None
    archive_path = None
    thumbnails_path = None
    thumbnails_archive_path = None

    def load(self, root: Path = Path(".")):
        """Read the install's config and validate its paths."""
        root = root.resolve()
        self.config = load_config_from_root(root)
        if not self.config.config_file_path.exists():
            print(f"Unable to find invokeai.yaml at {self.config.config_file_path}!")
            return False

        self.database_path = str(self.config.db_path)
        self.database_backup_dir = os.path.join(os.path.dirname(self.database_path), "backup")
        self.outputs_path = str(self.config.outputs_path / "images")
        self.archive_path = str(self.config.outputs_path / "images-archive")
        self.thumbnails_path = os.path.join(self.outputs_path, "thumbnails")
        self.thumbnails_archive_path = os.path.join(self.archive_path, "thumbnails")

        # A server database is checked when it is opened.
        db_exists = bool(self.config.db_url) or os.path.exists(self.database_path)
        outdir_exists = os.path.exists(self.outputs_path)
        database = redacted_database_url(self.config.db_url) or self.database_path

        text = f"Found invokeai.yaml file at {self.config.config_file_path}:"
        text += f"\n  Database : {database} - {'Exists!' if db_exists else 'Not Found!'}"
        text += f"\n  Outputs  : {self.outputs_path}- {'Exists!' if outdir_exists else 'Not Found!'}"
        print(text)

        if db_exists and outdir_exists:
            return True
        print(
            "\nOne or more paths specified in invokeai.yaml do not exist. Please inspect/correct the configuration "
            "and run the script with --root set to an Invoke AI root directory."
        )
        return False


class MaintenanceStats:
    """DTO for tracking work progress."""

    def __init__(self):  # noqa D107
        pass

    time_start = datetime.datetime.utcnow()
    count_orphaned_db_entries_cleaned = 0
    count_orphaned_disk_files_cleaned = 0
    count_orphaned_thumbnails_cleaned = 0
    count_thumbnails_regenerated = 0
    count_errors = 0

    @staticmethod
    def get_elapsed_time_string():
        """Get a friendly time string for the  time elapsed since processing start."""
        time_now = datetime.datetime.utcnow()
        total_seconds = (time_now - MaintenanceStats.time_start).total_seconds()
        hours = int((total_seconds) / 3600)
        minutes = int(((total_seconds) % 3600) / 60)
        seconds = total_seconds % 60
        out_str = f"{hours} hour(s) -" if hours > 0 else ""
        out_str += f"{minutes} minute(s) -" if minutes > 0 else ""
        out_str += f"{seconds:.2f} second(s)"
        return out_str


class DatabaseMapper:
    """The script's work on the database, through the app's image records."""

    def __init__(self, config: InvokeAIAppConfig, database_backup_dir):  # noqa D107
        self.config = config
        self.database_backup_dir = database_backup_dir
        self.database: Optional[Database] = None
        self.records: Optional[ImageRecordStorage] = None
        self.backed_up = False

    def backup(self, timestamp_string):
        """Take a backup of the database, once per run: a later one would hold what the run changed already."""
        if self.backed_up:
            return
        self.connect()
        assert self.database is not None
        if self.database.dialect_name != "sqlite":
            print("The database is a server database: take a backup of it with the server's tools before going on.")
            self.backed_up = True
            return
        if not os.path.exists(self.database_backup_dir):
            print(f"Database backup directory {self.database_backup_dir} does not exist -> creating...", end="")
            os.makedirs(self.database_backup_dir)
            print("Done!")
        database_backup_path = os.path.join(self.database_backup_dir, f"backup-{timestamp_string}-invokeai.db")
        print(f"Making DB Backup at {database_backup_path}...", end="")
        self.database.backup(Path(database_backup_path))
        self.backed_up = True
        print("Done!")

    def connect(self):
        """Open the database, once."""
        if self.database is None:
            logger = InvokeAILogger.get_logger("gallery_maintenance")
            self.database = open_migrated_database(self.config, logger)
            self.records = ImageRecordStorage(self.database)

    def get_all_image_files(self):
        """Get the full list of image file names from the database."""
        assert self.database is not None
        names, _starred = self.database.queries.images.names(
            starred_first=False,
            descending=False,
            image_origin=None,
            categories=None,
            is_intermediate=None,
            board_id=None,
            search_term=None,
            created_from=None,
            created_to=None,
            user_id=None,
            is_admin=True,
        )
        return names

    def remove_image_file_record(self, filename: str):
        """Remove an image file reference from the database by filename."""
        assert self.records is not None
        self.records.delete(filename)

    def does_image_exist(self, image_filename):
        """Check database if a image name already exists and return a boolean."""
        assert self.records is not None
        return self.records.exists(image_filename)

    def disconnect(self):
        """Close the database."""
        if self.database is not None:
            self.database.dispose()
            self.database = None
            self.records = None


class PhysicalFileMapper:
    """Containing class for script functionality."""

    def __init__(self, outputs_path, thumbnails_path, archive_path, thumbnails_archive_path):  # noqa D107
        self.outputs_path = outputs_path
        self.archive_path = archive_path
        self.thumbnails_path = thumbnails_path
        self.thumbnails_archive_path = thumbnails_archive_path

    def create_archive_directories(self):
        """Create the directory for archiving orphaned image files."""
        if not os.path.exists(self.archive_path):
            print(f"Image archive directory ({self.archive_path}) does not exist -> creating...", end="")
            os.makedirs(self.archive_path)
            print("Created!")
        if not os.path.exists(self.thumbnails_archive_path):
            print(
                f"Image thumbnails archive directory ({self.thumbnails_archive_path}) does not exist -> creating...",
                end="",
            )
            os.makedirs(self.thumbnails_archive_path)
            print("Created!")

    def get_image_path_for_image_name(self, image_filename):  # noqa D102
        return os.path.join(self.outputs_path, image_filename)

    def image_file_exists(self, image_filename):  # noqa D102
        return os.path.exists(self.get_image_path_for_image_name(image_filename))

    def get_thumbnail_path_for_image(self, image_filename):  # noqa D102
        return os.path.join(self.thumbnails_path, os.path.splitext(image_filename)[0]) + ".webp"

    def get_image_name_from_thumbnail_path(self, thumbnail_path):  # noqa D102
        return os.path.splitext(os.path.basename(thumbnail_path))[0] + ".png"

    def thumbnail_exists_for_filename(self, image_filename):  # noqa D102
        return os.path.exists(self.get_thumbnail_path_for_image(image_filename))

    def archive_image(self, image_filename):  # noqa D102
        if self.image_file_exists(image_filename):
            image_path = self.get_image_path_for_image_name(image_filename)
            shutil.move(image_path, self.archive_path)

    def archive_thumbnail_by_image_filename(self, image_filename):  # noqa D102
        if self.thumbnail_exists_for_filename(image_filename):
            thumbnail_path = self.get_thumbnail_path_for_image(image_filename)
            shutil.move(thumbnail_path, self.thumbnails_archive_path)

    def get_all_png_filenames_in_directory(self, directory_path):  # noqa D102
        filepaths = glob.glob(directory_path + "/*.png", recursive=False)
        filenames = []
        for filepath in filepaths:
            filenames.append(os.path.basename(filepath))
        return filenames

    def get_all_image_filenames_recursive(self):  # noqa D102
        """Return the set of all image filenames found anywhere under outputs_path.

        When an image_subfolder_strategy is configured, images are written to
        subdirectories of outputs/images (e.g. by date, type or hash). This walks
        the entire tree so that images stored in subfolders are not mistaken for
        missing files. Thumbnails are stored as .webp and so are excluded by the
        .png glob; the images-archive directory is a sibling of outputs/images and
        is therefore not traversed.
        """
        filepaths = glob.glob(os.path.join(self.outputs_path, "**", "*.png"), recursive=True)
        return {os.path.basename(filepath) for filepath in filepaths}

    def get_all_thumbnails_with_full_path(self, thumbnails_directory):  # noqa D102
        return glob.glob(thumbnails_directory + "/*.webp", recursive=False)

    def generate_thumbnail_for_image_name(self, image_filename):  # noqa D102
        # create thumbnail
        file_path = self.get_image_path_for_image_name(image_filename)
        thumb_path = self.get_thumbnail_path_for_image(image_filename)
        thumb_size = 256, 256
        with PIL.Image.open(file_path) as source_image:
            source_image.thumbnail(thumb_size)
            source_image.save(thumb_path, "webp")


class MaintenanceOperation(str, enum.Enum):
    """Enum class for operations."""

    Ask = "ask"
    CleanOrphanedDbEntries = "clean"
    CleanOrphanedDiskFiles = "archive"
    ReGenerateThumbnails = "thumbnails"
    All = "all"


class InvokeAIDatabaseMaintenanceApp:
    """Main processor class for the application."""

    _operation: MaintenanceOperation
    _headless: bool = False
    __stats: MaintenanceStats = MaintenanceStats()

    def __init__(self, operation: MaintenanceOperation = MaintenanceOperation.Ask, root: Path = Path(".")):
        """Initialize maintenance app."""
        self._operation = MaintenanceOperation(operation)
        self._headless = operation != MaintenanceOperation.Ask
        self._root = root

    def ask_for_operation(self) -> MaintenanceOperation:
        """Ask user to choose the operation to perform."""
        while True:
            print()
            print("It is recommennded to run these operations as ordered below to avoid additional")
            print("work being performed that will be discarded in a subsequent step.")
            print()
            print("Select maintenance operation:")
            print()
            print("1) Clean Orphaned Database Image Entries")
            print("     Cleans entries in the database where the matching file was removed from")
            print("     the outputs directory.")
            print("2) Archive Orphaned Image Files")
            print("     Files found in the outputs directory without an entry in the database are")
            print("     moved to an archive directory.")
            print("3) Re-Generate Missing Thumbnail Files")
            print("     For files found in the outputs directory, re-generate a thumbnail if it")
            print("     not found in the thumbnails directory.")
            print()
            print("(CTRL-C to quit)")

            try:
                input_option = int(input("Specify desired operation number (1-3): "))

                operations = [
                    MaintenanceOperation.CleanOrphanedDbEntries,
                    MaintenanceOperation.CleanOrphanedDiskFiles,
                    MaintenanceOperation.ReGenerateThumbnails,
                ]
                return operations[input_option - 1]
            except (IndexError, ValueError):
                print("\nInvalid selection!")

    def ask_to_continue(self) -> bool:
        """Ask user whether they want to continue with the operation."""
        while True:
            input_choice = input("Do you wish to continue? (Y or N)? ")
            if str.lower(input_choice) == "y":
                return True
            if str.lower(input_choice) == "n":
                return False

    def clean_orphaned_db_entries(
        self, config: ConfigMapper, file_mapper: PhysicalFileMapper, db_mapper: DatabaseMapper
    ):
        """Clean dangling database entries that no longer point to a file in outputs."""
        if self._headless:
            print(f"Removing database references to images that no longer exist in {config.outputs_path}...")
        else:
            print()
            print("===============================================================================")
            print("= Clean Orphaned Database Entries")
            print()
            print("Perform this operation if you have removed files from the outputs/images")
            print("directory but the database was never updated. You may see this as empty imaages")
            print("in the app gallery, or images that only show an enlarged version of the")
            print("thumbnail.")
            print()
            print(f"Database File Path               : {config.database_path}")
            print(f"Database backup will be taken at : {config.database_backup_dir}")
            print(f"Outputs/Images Directory         : {config.outputs_path}")
            print(f"Outputs/Images Archive Directory : {config.archive_path}")

            print("\nNotes about this operation:")
            print("- This operation will find database image file entries that do not exist in the")
            print("  outputs/images dir and remove those entries from the database.")
            print("- This operation will target all image types including intermediate files.")
            print("- If a thumbnail still exists in outputs/images/thumbnails matching the")
            print("  orphaned entry, it will be moved to the archive directory.")
            print()

            if not self.ask_to_continue():
                raise KeyboardInterrupt

        file_mapper.create_archive_directories()
        db_mapper.backup(config.TIMESTAMP_STRING)
        db_files = db_mapper.get_all_image_files()
        # Build an index of every image present anywhere under outputs/images, including any
        # subfolders created by an image_subfolder_strategy. A db entry is only orphaned if its
        # file is absent from this entire tree, not just the top-level outputs/images directory.
        disk_image_filenames = file_mapper.get_all_image_filenames_recursive()
        for db_file in db_files:
            try:
                if db_file not in disk_image_filenames:
                    print(f"Found orphaned image db entry {db_file}. Cleaning ...", end="")
                    db_mapper.remove_image_file_record(db_file)
                    print("Cleaned!")
                    if file_mapper.thumbnail_exists_for_filename(db_file):
                        print("A thumbnail was found, archiving ...", end="")
                        file_mapper.archive_thumbnail_by_image_filename(db_file)
                        print("Archived!")
                    self.__stats.count_orphaned_db_entries_cleaned += 1
            except Exception as ex:
                print("An error occurred cleaning db entry, error was:")
                print(ex)
                self.__stats.count_errors += 1

    def clean_orphaned_disk_files(
        self, config: ConfigMapper, file_mapper: PhysicalFileMapper, db_mapper: DatabaseMapper
    ):
        """Archive image files that no longer have entries in the database."""
        if self._headless:
            print(f"Archiving orphaned image files to {config.archive_path}...")
        else:
            print()
            print("===============================================================================")
            print("= Clean Orphaned Disk Files")
            print()
            print("Perform this operation if you have files that were copied into the outputs")
            print("directory which are not referenced by the database. This can happen if you")
            print("upgraded to a version with a fresh database, but re-used the outputs directory")
            print("and now new images are mixed with the files not in the db. The script will")
            print("archive these files so you can choose to delete them or re-import using the")
            print("official import script.")
            print()
            print(f"Database File Path               : {config.database_path}")
            print(f"Database backup will be taken at : {config.database_backup_dir}")
            print(f"Outputs/Images Directory         : {config.outputs_path}")
            print(f"Outputs/Images Archive Directory : {config.archive_path}")

            print("\nNotes about this operation:")
            print("- This operation will find image files not referenced by the database and move to an")
            print("  archive directory.")
            print("- This operation will target all image types including intermediate references.")
            print("- The matching thumbnail will also be archived.")
            print("- Any remaining orphaned thumbnails will also be archived.")

            if not self.ask_to_continue():
                raise KeyboardInterrupt

            print()

        file_mapper.create_archive_directories()
        db_mapper.backup(config.TIMESTAMP_STRING)
        phys_files = file_mapper.get_all_png_filenames_in_directory(config.outputs_path)
        for phys_file in phys_files:
            try:
                if not db_mapper.does_image_exist(phys_file):
                    print(f"Found orphaned file {phys_file}, archiving...", end="")
                    file_mapper.archive_image(phys_file)
                    print("Archived!")
                    if file_mapper.thumbnail_exists_for_filename(phys_file):
                        print("Related thumbnail exists, archiving...", end="")
                        file_mapper.archive_thumbnail_by_image_filename(phys_file)
                        print("Archived!")
                    else:
                        print("No matching thumbnail existed to be cleaned.")
                    self.__stats.count_orphaned_disk_files_cleaned += 1
            except Exception as ex:
                print("Error found trying to archive file or thumbnail, error was:")
                print(ex)
                self.__stats.count_errors += 1

        thumb_filepaths = file_mapper.get_all_thumbnails_with_full_path(config.thumbnails_path)
        # archive any remaining orphaned thumbnails
        for thumb_filepath in thumb_filepaths:
            try:
                thumb_src_image_name = file_mapper.get_image_name_from_thumbnail_path(thumb_filepath)
                if not file_mapper.image_file_exists(thumb_src_image_name):
                    print(f"Found orphaned thumbnail {thumb_filepath}, archiving...", end="")
                    file_mapper.archive_thumbnail_by_image_filename(thumb_src_image_name)
                    print("Archived!")
                    self.__stats.count_orphaned_thumbnails_cleaned += 1
            except Exception as ex:
                print("Error found trying to archive thumbnail, error was:")
                print(ex)
                self.__stats.count_errors += 1

    def regenerate_thumbnails(self, config: ConfigMapper, file_mapper: PhysicalFileMapper, *args):
        """Create missing thumbnails for any valid general images both in the db and on disk."""
        if self._headless:
            print("Regenerating missing image thumbnails...")
        else:
            print()
            print("===============================================================================")
            print("= Regenerate Thumbnails")
            print()
            print("This operation will find files that have no matching thumbnail on disk")
            print("and regenerate those thumbnail files.")
            print("NOTE: It is STRONGLY recommended that the user first clean/archive orphaned")
            print("  disk files from the previous menu to avoid wasting time regenerating")
            print("  thumbnails for orphaned files.")

            print()
            print(f"Outputs/Images Directory         : {config.outputs_path}")
            print(f"Outputs/Images Directory         : {config.thumbnails_path}")

            print("\nNotes about this operation:")
            print("- This operation will find image files both referenced in the db and on disk")
            print("  that do not have a matching thumbnail on disk and re-generate the thumbnail")
            print("  file.")

            if not self.ask_to_continue():
                raise KeyboardInterrupt

            print()

        phys_files = file_mapper.get_all_png_filenames_in_directory(config.outputs_path)
        for phys_file in phys_files:
            try:
                if not file_mapper.thumbnail_exists_for_filename(phys_file):
                    print(f"Found file without thumbnail {phys_file}...Regenerating Thumbnail...", end="")
                    file_mapper.generate_thumbnail_for_image_name(phys_file)
                    print("Done!")
                    self.__stats.count_thumbnails_regenerated += 1
            except Exception as ex:
                print("Error found trying to regenerate thumbnail, error was:")
                print(ex)
                self.__stats.count_errors += 1

    def main(self):  # noqa D107
        print("\n===============================================================================")
        print("Database and outputs Maintenance for Invoke AI 3.0.0 +")
        print("===============================================================================\n")

        config_mapper = ConfigMapper()
        if not config_mapper.load(self._root):
            print("\nInvalid configuration...exiting.\n")
            return
        assert config_mapper.config is not None

        file_mapper = PhysicalFileMapper(
            config_mapper.outputs_path,
            config_mapper.thumbnails_path,
            config_mapper.archive_path,
            config_mapper.thumbnails_archive_path,
        )
        db_mapper = DatabaseMapper(config_mapper.config, config_mapper.database_backup_dir)

        op = self._operation
        operations_to_perform = []

        if op == MaintenanceOperation.Ask:
            op = self.ask_for_operation()

        if op in [MaintenanceOperation.CleanOrphanedDbEntries, MaintenanceOperation.All]:
            operations_to_perform.append(self.clean_orphaned_db_entries)
        if op in [MaintenanceOperation.CleanOrphanedDiskFiles, MaintenanceOperation.All]:
            operations_to_perform.append(self.clean_orphaned_disk_files)
        if op in [MaintenanceOperation.ReGenerateThumbnails, MaintenanceOperation.All]:
            operations_to_perform.append(self.regenerate_thumbnails)

        try:
            for operation in operations_to_perform:
                operation(config_mapper, file_mapper, db_mapper)
        finally:
            db_mapper.disconnect()

        print("\n===============================================================================")
        print(f"= Maintenance Complete - Elapsed Time: {MaintenanceStats.get_elapsed_time_string()}")
        print()
        print(f"Orphaned db entries cleaned             : {self.__stats.count_orphaned_db_entries_cleaned}")
        print(f"Orphaned disk files archived            : {self.__stats.count_orphaned_disk_files_cleaned}")
        print(f"Orphaned thumbnail files archived       : {self.__stats.count_orphaned_thumbnails_cleaned}")
        print(f"Thumbnails regenerated                  : {self.__stats.count_thumbnails_regenerated}")
        print(f"Errors during operation                 : {self.__stats.count_errors}")

        print()


def main():  # noqa D107
    parser = argparse.ArgumentParser(
        description="InvokeAI image database maintenance utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Operations:
  ask         Choose operation from a menu [default]
  all         Run all maintenance operations
  clean       Clean database of dangling entries
  archive     Archive orphaned image files
  thumbnails  Regenerate missing image thumbnails
""",
    )
    parser.add_argument("--root", default=".", type=Path, help="InvokeAI root directory")
    parser.add_argument(
        "--operation", default="ask", choices=[x.value for x in MaintenanceOperation], help="Operation to perform."
    )
    args = parser.parse_args()
    if not args.root.is_dir():
        print(f"Invalid root directory '{args.root}'.")
        return
    try:
        app = InvokeAIDatabaseMaintenanceApp(args.operation, args.root)
        app.main()
    except KeyboardInterrupt:
        print("\n\nUser cancelled execution.")


if __name__ == "__main__":
    main()
