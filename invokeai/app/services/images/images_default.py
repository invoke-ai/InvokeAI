from collections.abc import Iterator
from contextlib import contextmanager
from typing import Optional

from PIL.Image import Image as PILImageType

from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.image_files.image_files_common import (
    ImageFileDeleteException,
    ImageFileNotFoundException,
    ImageFileSaveException,
)
from invokeai.app.services.image_files.image_subfolder_strategy import create_subfolder_strategy
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ImageNamesResult,
    ImageRecord,
    ImageRecordChanges,
    ImageRecordDeleteException,
    ImageRecordNotFoundException,
    ImageRecordSaveException,
    InvalidImageCategoryException,
    InvalidOriginException,
    ResourceOrigin,
)
from invokeai.app.services.images.images_base import ImageServiceABC
from invokeai.app.services.images.images_common import ImageDTO, image_record_to_dto
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.bulk_media_delete import StagedMediaDeleteAdapter, delete_media_by_names
from invokeai.app.services.shared.intermediate_delete import (
    IntermediateDeleteGuard,
    IntermediateDeleteResult,
    JournaledDeleteAdapter,
    delete_journaled_intermediates,
)
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection


class ImageService(ImageServiceABC):
    __invoker: Invoker

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

    @contextmanager
    def _image_mutation_lock(self) -> Iterator[None]:
        """Holds the image-mutation lock across a delete unit.

        Every delete here reads an image's subfolder and purges its files at that path after
        the record is gone. A concurrent subfolder move that relocates files and repoints the
        record mid-unit would leave the purge sweeping a path the files already left: permanent
        orphans, unrecoverable because the record is gone and a clean purge drops the journal
        (JPPhoto, PR #9361). The move service takes the same lock around each of its
        plan-relocate-repoint cycles, so neither unit can observe the other half-done. With no
        move service configured there is nothing to coordinate with. The lock is process-local:
        two Invoke processes sharing one output folder and database are not serialized by it.
        """
        image_moves = getattr(self.__invoker.services, "image_moves", None)
        if image_moves is None:
            yield
            return
        with image_moves.image_mutation_lock():
            yield

    def create(
        self,
        image: PILImageType,
        image_origin: ResourceOrigin,
        image_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        board_id: Optional[str] = None,
        is_intermediate: Optional[bool] = False,
        metadata: Optional[str] = None,
        workflow: Optional[str] = None,
        graph: Optional[str] = None,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> ImageDTO:
        if image_origin not in ResourceOrigin:
            raise InvalidOriginException

        if image_category not in ImageCategory:
            raise InvalidImageCategoryException

        image_name = self.__invoker.services.names.create_image_name()

        # Compute subfolder based on configured strategy
        strategy_name = self.__invoker.services.configuration.image_subfolder_strategy
        strategy = create_subfolder_strategy(strategy_name)
        image_subfolder = strategy.get_subfolder(image_name, image_category, is_intermediate or False)

        (width, height) = image.size

        try:
            # The mutation lock spans the record write through the file write: the moment the
            # record commits, the row is visible to the move service's planner, and a relocation
            # that landed before the files did would leave the record naming a subfolder the
            # files were never written to. Under the date strategy the two can disagree by a day
            # (the strategy reads the local clock, the move target comes from the record's UTC
            # created_at), so this is not a rare window. The failure cleanup below takes the
            # lock again; the reentrant lock makes that safe.
            with self._image_mutation_lock():
                # TODO: Consider using a transaction here to ensure consistency between storage and database
                self.__invoker.services.image_records.save(
                    # Non-nullable fields
                    image_name=image_name,
                    image_origin=image_origin,
                    image_category=image_category,
                    width=width,
                    height=height,
                    has_workflow=workflow is not None or graph is not None,
                    # Meta fields
                    is_intermediate=is_intermediate,
                    # Nullable fields
                    node_id=node_id,
                    metadata=metadata,
                    session_id=session_id,
                    user_id=user_id,
                    image_subfolder=image_subfolder,
                    project_id=project_id,
                )
                if board_id is not None:
                    try:
                        self.__invoker.services.board_image_records.add_image_to_board(
                            board_id=board_id, image_name=image_name
                        )
                    except Exception as e:
                        self.__invoker.services.logger.warning(f"Failed to add image to board {board_id}: {str(e)}")
                self.__invoker.services.image_files.save(
                    image_name=image_name,
                    image=image,
                    metadata=metadata,
                    workflow=workflow,
                    graph=graph,
                    image_subfolder=image_subfolder,
                )
                self._record_file_size(image_name, image_subfolder)
            image_dto = self.get_dto(image_name)

            self._on_changed(image_dto)
            return image_dto
        except ImageRecordSaveException:
            self.__invoker.services.logger.error("Failed to save image record")
            raise
        except ImageFileSaveException:
            self.__invoker.services.logger.error("Failed to save image file")
            self.__clean_up_failed_save(image_name, image_subfolder)
            raise
        except Exception as e:
            self.__invoker.services.logger.error(f"Problem saving image record and file: {str(e)}")
            raise e

    def _record_file_size(self, image_name: str, image_subfolder: str) -> None:
        """Measures the files just written so storage accounting never has to stat them again.

        Best effort: the image exists whether or not its size is recorded, and an unmeasured row
        reads as unknown rather than zero until the intermediates backfill measures it.
        """
        try:
            size = self.__invoker.services.image_files.get_file_size_bytes(image_name, image_subfolder=image_subfolder)
            self.__invoker.services.image_records.set_file_size_bytes(image_name, size)
        except Exception as e:
            self.__invoker.services.logger.warning(f"Failed to record the file size of image {image_name}: {e}")

    def __clean_up_failed_save(self, image_name: str, image_subfolder: str) -> None:
        """Removes the half-created image left by a failed save, record first.

        Record-then-files is the order every delete path uses, and it is load-bearing rather than
        cosmetic: a concurrent deleter that has to roll back decides whether to restore an image's
        files by asking whether its record is still there. Purging files while the record survives
        would tell that deleter to put them back, stranding them once this cleanup finally removes
        the record. The journal covers the window in between — and, like every other delete path,
        the record is only removed once that journal is durable. If it cannot be, the half-created
        image is left whole: its record marks the files for a later delete to find, whereas a record
        removed without a journal would leave them orphaned for good.
        """
        with self._image_mutation_lock():
            # A subfolder move that ran while the file save was failing may have relocated the
            # just-saved record, and partial files can be left at either path. Journal both the
            # captured subfolder and the one the record names now. A record that is already gone
            # (a concurrent delete won the race) or a faulting record store leaves us only the
            # captured subfolder, which is the best path known either way.
            delete_subfolders = [image_subfolder]
            try:
                record = self.__invoker.services.image_records.get(image_name)
                if record.image_subfolder != image_subfolder:
                    delete_subfolders.append(record.image_subfolder)
            except Exception as lookup_error:
                # Best effort: a record-store fault here must not mask the ImageFileSaveException
                # this cleanup is running for. We may then purge only the captured subfolder, so
                # say so — if the record had been relocated, its files need a manual sweep.
                self.__invoker.services.logger.warning(
                    f"Could not confirm the subfolder of {image_name} during save-failure cleanup: {str(lookup_error)}"
                )
            try:
                token = self.__invoker.services.image_files.begin_delete(
                    [(image_name, subfolder) for subfolder in delete_subfolders]
                )
            except Exception as cleanup_error:
                # No durable journal, no record deletion: the record is what keeps these files
                # findable. Deleting it and then purging blind would fail at the same journal step
                # and leave whatever survived the save as an orphan nothing can recover.
                self.__invoker.services.logger.error(
                    f"Failed to journal the cleanup of {image_name} after a save failure; leaving the image in place "
                    f"for a later delete: {str(cleanup_error)}"
                )
                return
            try:
                # Deleting the record also removes any board association through the database foreign
                # key cascade.
                self.__invoker.services.image_records.delete(image_name)
            except Exception as cleanup_error:
                self.__invoker.services.logger.error(
                    f"Failed to clean up image record after save failure: {str(cleanup_error)}"
                )
                # The record survived, so the image is still referenced; its files must stay with it.
                try:
                    self.__invoker.services.image_files.abandon_delete(token)
                except Exception as journal_error:
                    self.__invoker.services.logger.error(
                        f"Failed to discard the delete journal for {image_name}: {str(journal_error)}"
                    )
                return
            try:
                self.__invoker.services.image_files.commit_delete(token)
            except Exception as cleanup_error:
                self.__invoker.services.logger.error(
                    f"Failed to clean up image files after save failure: {str(cleanup_error)}"
                )

    def copy(
        self,
        source_image_name: str,
        board_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ImageDTO:
        """Duplicate an existing image under a new identity, optionally onto a board.

        New identity, same picture: the record is cloned and the file copied byte for byte, so
        embedded metadata, workflow and graph travel as file chunks rather than being re-derived.
        The copy is never intermediate, and starring is not copied.

        Board attachment is checked *before* the file is written, where the video twin writes first
        and withdraws on failure. The orders differ because `create` — which the video path reuses —
        swallows a failed attachment, so there it can only be detected after the fact.

        Nothing partial survives a failure: the unwind covers everything after the record exists,
        including reading the DTO back, and removes the file as well as the row.
        """
        try:
            record = self.__invoker.services.image_records.get(source_image_name)
            metadata = self.__invoker.services.image_records.get_metadata(source_image_name)

            image_name = self.__invoker.services.names.create_image_name()
            strategy_name = self.__invoker.services.configuration.image_subfolder_strategy
            strategy = create_subfolder_strategy(strategy_name)
            image_subfolder = strategy.get_subfolder(image_name, record.image_category, False)

            record_saved = False
            file_copied = False
            try:
                self.__invoker.services.image_records.save(
                    image_name=image_name,
                    image_origin=record.image_origin,
                    image_category=record.image_category,
                    width=record.width,
                    height=record.height,
                    has_workflow=record.has_workflow,
                    is_intermediate=False,
                    metadata=metadata.model_dump_json() if metadata is not None else None,
                    user_id=user_id,
                    image_subfolder=image_subfolder,
                )
                record_saved = True

                if board_id is not None:
                    # Deliberately fatal, unlike `create`. There the alternative to a board is
                    # losing a freshly generated image; here the caller asked for a copy *on a
                    # board*, and a copy that silently landed uncategorized would be reported as a
                    # success that the caller then remaps its document onto.
                    self.__invoker.services.board_image_records.add_image_to_board(
                        board_id=board_id, image_name=image_name
                    )

                self.__invoker.services.image_files.copy(
                    source_image_name=source_image_name,
                    image_name=image_name,
                    source_subfolder=record.image_subfolder or "",
                    image_subfolder=image_subfolder,
                )
                file_copied = True
                self._record_file_size(image_name, image_subfolder)

                image_dto = self.get_dto(image_name)
                self._on_changed(image_dto)
                return image_dto
            except Exception:
                # Unwind whatever exists, newest first. The board membership goes with the record
                # via the FK.
                if file_copied:
                    try:
                        self.__invoker.services.image_files.delete(image_name, image_subfolder=image_subfolder)
                    except Exception as cleanup_error:
                        self.__invoker.services.logger.error(
                            f"Failed to roll back the file for copy {image_name}: {cleanup_error}"
                        )
                if record_saved:
                    try:
                        self.__invoker.services.image_records.delete(image_name)
                    except Exception as cleanup_error:
                        self.__invoker.services.logger.error(
                            f"Failed to roll back the record for copy {image_name}: {cleanup_error}"
                        )
                raise
        except Exception:
            self.__invoker.services.logger.error(f"Failed to copy image {source_image_name}", exc_info=True)
            raise

    def update(
        self,
        image_name: str,
        changes: ImageRecordChanges,
    ) -> ImageDTO:
        try:
            self.__invoker.services.image_records.update(image_name, changes)
            image_dto = self.get_dto(image_name)
            self._on_changed(image_dto)
            return image_dto
        except ImageRecordSaveException:
            self.__invoker.services.logger.error("Failed to update image record")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem updating image record")
            raise e

    def get_pil_image(self, image_name: str) -> PILImageType:
        try:
            record = self.__invoker.services.image_records.get(image_name)
            return self.__invoker.services.image_files.get(image_name, image_subfolder=record.image_subfolder)
        except ImageRecordNotFoundException:
            self.__invoker.services.logger.debug(f"Image record not found: {image_name}")
            raise
        except ImageFileNotFoundException:
            self.__invoker.services.logger.error("Failed to get image file")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image file")
            raise e

    def get_record(self, image_name: str) -> ImageRecord:
        try:
            return self.__invoker.services.image_records.get(image_name)
        except ImageRecordNotFoundException:
            self.__invoker.services.logger.debug(f"Image record not found: {image_name}")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image record")
            raise e

    def get_dto(self, image_name: str) -> ImageDTO:
        try:
            image_record = self.__invoker.services.image_records.get(image_name)

            image_dto = image_record_to_dto(
                image_record=image_record,
                image_url=self.__invoker.services.urls.get_image_url(image_name),
                thumbnail_url=self.__invoker.services.urls.get_image_url(image_name, True),
                board_id=self.__invoker.services.board_image_records.get_board_for_image(image_name),
            )

            return image_dto
        except ImageRecordNotFoundException:
            self.__invoker.services.logger.debug(f"Image record not found: {image_name}")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image DTO")
            raise e

    def get_metadata(self, image_name: str) -> Optional[MetadataField]:
        try:
            return self.__invoker.services.image_records.get_metadata(image_name)
        except ImageRecordNotFoundException:
            self.__invoker.services.logger.debug(f"Image record not found: {image_name}")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image metadata")
            raise e

    def get_workflow(self, image_name: str) -> Optional[str]:
        try:
            record = self.__invoker.services.image_records.get(image_name)
            return self.__invoker.services.image_files.get_workflow(image_name, image_subfolder=record.image_subfolder)
        except ImageRecordNotFoundException:
            self.__invoker.services.logger.debug(f"Image record not found: {image_name}")
            raise
        except ImageFileNotFoundException:
            self.__invoker.services.logger.error("Image file not found")
            raise
        except Exception:
            self.__invoker.services.logger.error("Problem getting image workflow")
            raise

    def get_graph(self, image_name: str) -> Optional[str]:
        try:
            record = self.__invoker.services.image_records.get(image_name)
            return self.__invoker.services.image_files.get_graph(image_name, image_subfolder=record.image_subfolder)
        except ImageRecordNotFoundException:
            self.__invoker.services.logger.debug(f"Image record not found: {image_name}")
            raise
        except ImageFileNotFoundException:
            self.__invoker.services.logger.error("Image file not found")
            raise
        except Exception:
            self.__invoker.services.logger.error("Problem getting image graph")
            raise

    def get_path(self, image_name: str, thumbnail: bool = False) -> str:
        try:
            record = self.__invoker.services.image_records.get(image_name)
            return str(
                self.__invoker.services.image_files.get_path(
                    image_name, thumbnail, image_subfolder=record.image_subfolder
                )
            )
        except ImageRecordNotFoundException:
            self.__invoker.services.logger.debug(f"Image record not found: {image_name}")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image path")
            raise e

    def validate_path(self, path: str) -> bool:
        try:
            return self.__invoker.services.image_files.validate_path(path)
        except Exception as e:
            self.__invoker.services.logger.error("Problem validating image path")
            raise e

    def get_url(self, image_name: str, thumbnail: bool = False) -> str:
        try:
            return self.__invoker.services.urls.get_image_url(image_name, thumbnail)
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image path")
            raise e

    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        image_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> OffsetPaginatedResults[ImageDTO]:
        try:
            results = self.__invoker.services.image_records.get_many(
                offset=offset,
                limit=limit,
                starred_first=starred_first,
                order_dir=order_dir,
                image_origin=image_origin,
                categories=categories,
                is_intermediate=is_intermediate,
                board_id=board_id,
                search_term=search_term,
                created_from=created_from,
                created_to=created_to,
                user_id=user_id,
                is_admin=is_admin,
            )

            image_dtos = [
                image_record_to_dto(
                    image_record=r,
                    image_url=self.__invoker.services.urls.get_image_url(r.image_name),
                    thumbnail_url=self.__invoker.services.urls.get_image_url(r.image_name, True),
                    board_id=self.__invoker.services.board_image_records.get_board_for_image(r.image_name),
                )
                for r in results.items
            ]

            return OffsetPaginatedResults[ImageDTO](
                items=image_dtos,
                offset=results.offset,
                limit=results.limit,
                total=results.total,
            )
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting paginated image DTOs")
            raise e

    def delete(self, image_name: str):
        # Record first, files second, with a durable journal spanning the two. Deleting the record
        # first means a database failure leaves the image completely intact, and the only state
        # that can outlive this call is a file nothing references — which the journal lets startup
        # recovery find and purge. Nothing is ever moved aside and put back, so a concurrent
        # deleter of the same image cannot resurrect files whose record has already been removed.
        # The mutation lock spans the record read through the purge so a subfolder move cannot
        # relocate the files between the two and leave the purge sweeping an abandoned path.
        with self._image_mutation_lock():
            try:
                record = self.__invoker.services.image_records.get(image_name)
                token = self.__invoker.services.image_files.begin_delete([(image_name, record.image_subfolder)])
                try:
                    self.__invoker.services.image_records.delete(image_name)
                except Exception:
                    # The image is still live: drop the journal and leave its files alone.
                    try:
                        self.__invoker.services.image_files.abandon_delete(token)
                    except Exception as cleanup_error:
                        self.__invoker.services.logger.error(
                            f"Failed to discard the delete journal for {image_name}: {cleanup_error}"
                        )
                    raise
                try:
                    self.__invoker.services.image_files.commit_delete(token)
                except Exception as cleanup_error:
                    # The record is committed as gone, so the delete succeeded. The journal stays
                    # behind and startup recovery purges the leftover files.
                    self.__invoker.services.logger.error(f"Failed to purge deleted image files: {cleanup_error}")
                self._on_deleted(image_name)
            except ImageRecordNotFoundException:
                # Already deleted by another request; nothing here failed, so nothing to log.
                raise
            except ImageRecordDeleteException:
                self.__invoker.services.logger.error("Failed to delete image record")
                raise
            except ImageFileDeleteException:
                self.__invoker.services.logger.error("Failed to delete image file")
                raise
            except Exception as e:
                self.__invoker.services.logger.error("Problem deleting image record and file")
                raise e

    def delete_images_on_board(self, board_id: str, user_id: Optional[str] = None) -> tuple[list[str], list[str]]:
        # The mutation lock spans the enumeration through the purges and rollbacks so a subfolder
        # move cannot relocate files between a record read and its stage or commit.
        with self._image_mutation_lock():
            # When ``user_id`` is set the lookup filters to images owned by that user so the
            # cascade doesn't destroy other users' contributions to a public/shared board.
            image_names = self.__invoker.services.board_image_records.get_all_board_image_names_for_board(
                board_id,
                categories=None,
                is_intermediate=None,
                user_id=user_id,
            )
            return self.delete_images_by_names(image_names)

    def delete_images_by_names(self, image_names: list[str]) -> tuple[list[str], list[str]]:
        """Delete exactly these images, returning ``(deleted, failed)``.

        Split from ``delete_images_on_board`` so a caller that must decide whether the board may go
        *before* destroying anything can enumerate first and delete second. Records whose file
        delete fails keep their record on purpose and come back as failures.
        """
        # The mutation lock spans the per-image record reads through the purges and rollbacks so a
        # subfolder move cannot relocate files between a record read and its stage or commit.
        with self._image_mutation_lock():
            try:
                records = self.__invoker.services.image_records
                files = self.__invoker.services.image_files
                return delete_media_by_names(
                    image_names,
                    StagedMediaDeleteAdapter(
                        kind="image",
                        stage=lambda name: files.stage_delete(name, image_subfolder=records.get(name).image_subfolder),
                        delete_records=records.delete_many,
                        rollback=files.rollback_delete,
                        commit=files.commit_delete,
                        notify_deleted=self._on_deleted,
                        log_error=self.__invoker.services.logger.error,
                    ),
                )
            except ImageRecordDeleteException:
                self.__invoker.services.logger.error("Failed to delete image records")
                raise
            except ImageFileDeleteException:
                self.__invoker.services.logger.error("Failed to delete image files")
                raise
            except Exception as e:
                self.__invoker.services.logger.error(f"Problem deleting image records and files: {str(e)}")
                raise e

    def delete_intermediates_by_names(
        self, image_names: list[str], guard: Optional[IntermediateDeleteGuard] = None
    ) -> IntermediateDeleteResult:
        with self._image_mutation_lock():
            try:
                subfolders = self.__invoker.services.image_records.get_subfolders(image_names)
                if not subfolders:
                    return IntermediateDeleteResult()
                files = self.__invoker.services.image_files
                records = self.__invoker.services.image_records
                return delete_journaled_intermediates(
                    subfolders,
                    guard,
                    JournaledDeleteAdapter(
                        kind="image",
                        begin=files.begin_delete,
                        delete_records=lambda names, g: records.delete_intermediates_by_names(names, guard=g),
                        abandon=files.abandon_delete,
                        commit=lambda token, names: files.commit_delete(token, image_names=names),
                        notify_deleted=self._on_deleted,
                        log_error=self.__invoker.services.logger.error,
                    ),
                )
            except ImageRecordDeleteException:
                self.__invoker.services.logger.error("Failed to delete image records")
                raise
            except Exception as e:
                self.__invoker.services.logger.error("Problem deleting intermediate image records and files")
                raise e

    def get_image_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        image_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> ImageNamesResult:
        try:
            return self.__invoker.services.image_records.get_image_names(
                starred_first=starred_first,
                order_dir=order_dir,
                image_origin=image_origin,
                categories=categories,
                is_intermediate=is_intermediate,
                board_id=board_id,
                search_term=search_term,
                created_from=created_from,
                created_to=created_to,
                user_id=user_id,
                is_admin=is_admin,
            )
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image names")
            raise e
