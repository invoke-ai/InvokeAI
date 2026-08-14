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
            try:
                self.__invoker.services.image_files.delete(image_name, image_subfolder=image_subfolder)
            except Exception as cleanup_error:
                self.__invoker.services.logger.error(
                    f"Failed to clean up image files after save failure: {str(cleanup_error)}"
                )
            try:
                # Deleting the record also removes any board association through the database
                # foreign key cascade. Both cleanup operations are attempted independently.
                self.__invoker.services.image_records.delete(image_name)
            except Exception as cleanup_error:
                self.__invoker.services.logger.error(
                    f"Failed to clean up image record after save failure: {str(cleanup_error)}"
                )
            raise
        except Exception as e:
            self.__invoker.services.logger.error(f"Problem saving image record and file: {str(e)}")
            raise e

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
        # Stage the file deletion first so a database failure can be rolled back by
        # restoring the files, keeping the record and files consistent either way.
        token: object | None = None
        record_deleted = False
        try:
            record = self.__invoker.services.image_records.get(image_name)
            token = self.__invoker.services.image_files.stage_delete(image_name, image_subfolder=record.image_subfolder)
            self.__invoker.services.image_records.delete(image_name)
            record_deleted = True
            try:
                self.__invoker.services.image_files.commit_delete(token)
            except Exception as cleanup_error:
                # The record is gone; a failed purge only leaves a staging directory
                # behind, which startup recovery will clean up. Not a delete failure.
                self.__invoker.services.logger.error(f"Failed to purge staged image files: {cleanup_error}")
            self._on_deleted(image_name)
        except ImageRecordDeleteException:
            if token is not None:
                try:
                    self.__invoker.services.image_files.rollback_delete(token)
                except Exception as rollback_error:
                    self.__invoker.services.logger.error(
                        f"Failed to restore staged image files for {image_name}: {rollback_error}"
                    )
            self.__invoker.services.logger.error("Failed to delete image record")
            raise
        except ImageFileDeleteException:
            self.__invoker.services.logger.error("Failed to delete image file")
            raise
        except Exception as e:
            if token is not None and not record_deleted:
                try:
                    self.__invoker.services.image_files.rollback_delete(token)
                except Exception as rollback_error:
                    self.__invoker.services.logger.error(
                        f"Failed to restore staged image files for {image_name}: {rollback_error}"
                    )
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
        # Records first, files second. An earlier revision staged every file, then conditionally
        # deleted the records, then restored the files of any image that had been promoted out of
        # intermediate status mid-operation. That restore is unfixably racy: while a promoted
        # image's files sit in our staging directory, a concurrent single-image or board delete can
        # stage-empty (it finds no files to move) and then remove the record; our restore then puts
        # the files back with no record referencing them and no staging dir to recover from —
        # permanent orphans (JPPhoto, PR #9361).
        #
        # Deleting the records first removes that hazard entirely: the conditional DELETE is atomic
        # and tells us exactly which rows it removed, and we only ever touch the files of rows that
        # are already gone. A promoted image is never deleted and its files are never staged, so a
        # concurrent delete of it operates on real files in the output folder and stays consistent.
        try:
            image_name_subfolder_pairs = self.__invoker.services.image_records.get_intermediates()
            subfolders = dict(image_name_subfolder_pairs)
            # Conditional on the row still being an intermediate: an image promoted between the
            # snapshot above and this call keeps both its record and its files. Returns exactly the
            # names this call removed (already-absent and promoted rows are excluded).
            deleted_image_names = self.__invoker.services.image_records.delete_intermediates_by_names(
                list(subfolders.keys())
            )
            # The records are committed as gone; purge each file best-effort. A filesystem failure
            # here orphans that file (nothing references it) but must neither abort the remaining
            # purges nor undo the committed deletions, so failures are logged and skipped rather
            # than raised.
            for image_name in deleted_image_names:
                try:
                    self.__invoker.services.image_files.delete(
                        image_name, image_subfolder=subfolders.get(image_name, "")
                    )
                except Exception as cleanup_error:
                    self.__invoker.services.logger.error(
                        f"Failed to purge intermediate image files for {image_name}: {cleanup_error}"
                    )
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
