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
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection


class ImageService(ImageServiceABC):
    __invoker: Invoker

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

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

    def __clean_up_failed_save(self, image_name: str, image_subfolder: str) -> None:
        """Removes the half-created image left by a failed save, record first.

        Record-then-files is the order every delete path uses, and it is load-bearing rather than
        cosmetic: a concurrent deleter that has to roll back decides whether to restore an image's
        files by asking whether its record is still there. Purging files while the record survives
        would tell that deleter to put them back, stranding them once this cleanup finally removes
        the record. The journal covers the window in between.
        """
        token: object | None = None
        try:
            token = self.__invoker.services.image_files.begin_delete([(image_name, image_subfolder)])
        except Exception as cleanup_error:
            self.__invoker.services.logger.error(
                f"Failed to journal the cleanup of {image_name} after a save failure: {str(cleanup_error)}"
            )
        try:
            # Deleting the record also removes any board association through the database foreign
            # key cascade.
            self.__invoker.services.image_records.delete(image_name)
        except Exception as cleanup_error:
            self.__invoker.services.logger.error(
                f"Failed to clean up image record after save failure: {str(cleanup_error)}"
            )
            # The record survived, so the image is still referenced; its files must stay with it.
            if token is not None:
                try:
                    self.__invoker.services.image_files.abandon_delete(token)
                except Exception as journal_error:
                    self.__invoker.services.logger.error(
                        f"Failed to discard the delete journal for {image_name}: {str(journal_error)}"
                    )
            return
        try:
            if token is None:
                self.__invoker.services.image_files.delete(image_name, image_subfolder=image_subfolder)
            else:
                self.__invoker.services.image_files.commit_delete(token)
        except Exception as cleanup_error:
            self.__invoker.services.logger.error(
                f"Failed to clean up image files after save failure: {str(cleanup_error)}"
            )

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
            self.__invoker.services.logger.error("Image record not found")
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
            self.__invoker.services.logger.error("Image record not found")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image DTO")
            raise e

    def get_metadata(self, image_name: str) -> Optional[MetadataField]:
        try:
            return self.__invoker.services.image_records.get_metadata(image_name)
        except ImageRecordNotFoundException:
            self.__invoker.services.logger.error("Image record not found")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image metadata")
            raise e

    def get_workflow(self, image_name: str) -> Optional[str]:
        try:
            record = self.__invoker.services.image_records.get(image_name)
            return self.__invoker.services.image_files.get_workflow(image_name, image_subfolder=record.image_subfolder)
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
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> OffsetPaginatedResults[ImageDTO]:
        try:
            results = self.__invoker.services.image_records.get_many(
                offset,
                limit,
                starred_first,
                order_dir,
                image_origin,
                categories,
                is_intermediate,
                board_id,
                search_term,
                user_id,
                is_admin,
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
        try:
            # When ``user_id`` is set the lookup filters to images owned by that user so the
            # cascade doesn't destroy other users' contributions to a public/shared board.
            image_names = self.__invoker.services.board_image_records.get_all_board_image_names_for_board(
                board_id,
                categories=None,
                is_intermediate=None,
                user_id=user_id,
            )
            deleted_image_names: list[str] = []
            failed_image_names: list[str] = []
            staged_deletes: list[tuple[str, object]] = []
            for image_name in image_names:
                try:
                    record = self.__invoker.services.image_records.get(image_name)
                    token = self.__invoker.services.image_files.stage_delete(
                        image_name, image_subfolder=record.image_subfolder
                    )
                    staged_deletes.append((image_name, token))
                    deleted_image_names.append(image_name)
                except Exception as e:
                    failed_image_names.append(image_name)
                    self.__invoker.services.logger.error(
                        f"Failed to delete image file {image_name}; keeping record: {str(e)}"
                    )
            try:
                self.__invoker.services.image_records.delete_many(deleted_image_names)
            except Exception:
                for image_name, token in staged_deletes:
                    try:
                        self.__invoker.services.image_files.rollback_delete(token)
                    except Exception as rollback_error:
                        self.__invoker.services.logger.error(
                            f"Failed to restore staged image files for {image_name}: {rollback_error}"
                        )
                raise
            for _, token in staged_deletes:
                try:
                    self.__invoker.services.image_files.commit_delete(token)
                except Exception as cleanup_error:
                    self.__invoker.services.logger.error(f"Failed to purge staged image files: {cleanup_error}")
            for image_name in deleted_image_names:
                self._on_deleted(image_name)
            return deleted_image_names, failed_image_names
        except ImageRecordDeleteException:
            self.__invoker.services.logger.error("Failed to delete image records")
            raise
        except ImageFileDeleteException:
            self.__invoker.services.logger.error("Failed to delete image files")
            raise
        except Exception as e:
            self.__invoker.services.logger.error(f"Problem deleting image records and files: {str(e)}")
            raise e

    def delete_intermediates(self) -> int:
        # Records first, files second, with a durable journal spanning the two. An earlier revision
        # staged every file, then conditionally deleted the records, then restored the files of any
        # image that had been promoted out of intermediate status mid-operation. That restore is
        # unfixably racy: while a promoted image's files sit in a staging directory, a concurrent
        # single-image or board delete can stage-empty (it finds no files to move) and then remove
        # the record; the restore then puts the files back with no record referencing them and no
        # journal to recover from — permanent orphans (JPPhoto, PR #9361).
        #
        # Deleting the records first removes that hazard: the conditional DELETE is atomic and
        # reports exactly which rows it removed, and only the files of already-deleted rows are
        # touched. A promoted image is never deleted and its files are never moved, so a concurrent
        # delete of it operates on real files in the output folder and stays consistent. The
        # journal covers the window the reordering opens: if this process dies, or the filesystem
        # fails, between the commit and the purge, startup recovery finishes the purge for every
        # journalled image whose record is gone.
        try:
            image_name_subfolder_pairs = self.__invoker.services.image_records.get_intermediates()
            if not image_name_subfolder_pairs:
                return 0
            subfolders = dict(image_name_subfolder_pairs)
            token = self.__invoker.services.image_files.begin_delete(list(subfolders.items()))
            try:
                # Conditional on the row still being an intermediate: an image promoted between the
                # snapshot above and this call keeps both its record and its files. Returns exactly
                # the names this call removed (already-absent and promoted rows are excluded).
                deleted_image_names = self.__invoker.services.image_records.delete_intermediates_by_names(
                    list(subfolders.keys())
                )
            except Exception:
                try:
                    self.__invoker.services.image_files.abandon_delete(token)
                except Exception as cleanup_error:
                    self.__invoker.services.logger.error(
                        f"Failed to discard the intermediates delete journal: {cleanup_error}"
                    )
                raise
            try:
                # Only the names whose records this call removed are purged; a promoted image keeps
                # its files. The journal still lists it, which is harmless — recovery re-checks
                # every entry against the record store and skips the ones that survived.
                self.__invoker.services.image_files.commit_delete(token, image_names=deleted_image_names)
            except Exception as cleanup_error:
                # The records are committed as gone, so the deletion succeeded. A file that could
                # not be purged keeps its journal entry and is retried at the next startup; it must
                # neither fail the operation nor undo the committed deletions.
                self.__invoker.services.logger.error(f"Failed to purge intermediate image files: {cleanup_error}")
            for image_name in deleted_image_names:
                self._on_deleted(image_name)
            return len(deleted_image_names)
        except ImageRecordDeleteException:
            self.__invoker.services.logger.error("Failed to delete image records")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem deleting intermediate image records and files")
            raise e

    def get_intermediates_count(self, user_id: Optional[str] = None) -> int:
        try:
            return self.__invoker.services.image_records.get_intermediates_count(user_id=user_id)
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting intermediates count")
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
                user_id=user_id,
                is_admin=is_admin,
            )
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image names")
            raise e
