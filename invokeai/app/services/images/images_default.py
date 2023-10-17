from typing import Optional

from PIL.Image import Image as PILImageType

from invokeai.app.invocations.metadata import ImageMetadata
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.util.metadata import get_metadata_graph_from_raw_session

from ..image_files.image_files_common import (
    ImageFileDeleteException,
    ImageFileNotFoundException,
    ImageFileSaveException,
)
from ..image_records.image_records_common import (
    ImageCategory,
    ImageRecord,
    ImageRecordChanges,
    ImageRecordDeleteException,
    ImageRecordNotFoundException,
    ImageRecordSaveException,
    InvalidImageCategoryException,
    InvalidOriginException,
    ResourceOrigin,
)
from .images_base import ImageServiceABC
from .images_common import ImageDTO, image_record_to_dto


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
        metadata: Optional[dict] = None,
        workflow: Optional[str] = None,
    ) -> ImageDTO:
        if image_origin not in ResourceOrigin:
            raise InvalidOriginException

        if image_category not in ImageCategory:
            raise InvalidImageCategoryException

        image_name = self.__invoker.services.names.create_image_name()

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
                # Meta fields
                is_intermediate=is_intermediate,
                # Nullable fields
                node_id=node_id,
                metadata=metadata,
                session_id=session_id,
            )
            if board_id is not None:
                self.__invoker.services.board_image_records.add_image_to_board(board_id=board_id, image_name=image_name)
            self.__invoker.services.image_files.save(
                image_name=image_name, image=image, metadata=metadata, workflow=workflow
            )
            image_dto = self.get_dto(image_name)

            self._on_changed(image_dto)
            return image_dto
        except ImageRecordSaveException:
            self.__invoker.services.logger.error("Failed to save image record")
            raise
        except ImageFileSaveException:
            self.__invoker.services.logger.error("Failed to save image file")
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
            return self.__invoker.services.image_files.get(image_name)
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
                image_record,
                self.__invoker.services.urls.get_image_url(image_name),
                self.__invoker.services.urls.get_image_url(image_name, True),
                self.__invoker.services.board_image_records.get_board_for_image(image_name),
            )

            return image_dto
        except ImageRecordNotFoundException:
            self.__invoker.services.logger.error("Image record not found")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image DTO")
            raise e

    def get_metadata(self, image_name: str) -> ImageMetadata:
        try:
            image_record = self.__invoker.services.image_records.get(image_name)
            metadata = self.__invoker.services.image_records.get_metadata(image_name)

            if not image_record.session_id:
                return ImageMetadata(metadata=metadata)

            session_raw = self.__invoker.services.graph_execution_manager.get_raw(image_record.session_id)
            graph = None

            if session_raw:
                try:
                    graph = get_metadata_graph_from_raw_session(session_raw)
                except Exception as e:
                    self.__invoker.services.logger.warn(f"Failed to parse session graph: {e}")
                    graph = None

            return ImageMetadata(graph=graph, metadata=metadata)
        except ImageRecordNotFoundException:
            self.__invoker.services.logger.error("Image record not found")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image DTO")
            raise e

    def get_path(self, image_name: str, thumbnail: bool = False) -> str:
        try:
            return str(self.__invoker.services.image_files.get_path(image_name, thumbnail))
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
        image_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
    ) -> OffsetPaginatedResults[ImageDTO]:
        try:
            results = self.__invoker.services.image_records.get_many(
                offset,
                limit,
                image_origin,
                categories,
                is_intermediate,
                board_id,
            )

            image_dtos = list(
                map(
                    lambda r: image_record_to_dto(
                        r,
                        self.__invoker.services.urls.get_image_url(r.image_name),
                        self.__invoker.services.urls.get_image_url(r.image_name, True),
                        self.__invoker.services.board_image_records.get_board_for_image(r.image_name),
                    ),
                    results.items,
                )
            )

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
        try:
            self.__invoker.services.image_files.delete(image_name)
            self.__invoker.services.image_records.delete(image_name)
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

    def delete_images_on_board(self, board_id: str):
        try:
            image_names = self.__invoker.services.board_image_records.get_all_board_image_names_for_board(board_id)
            for image_name in image_names:
                self.__invoker.services.image_files.delete(image_name)
            self.__invoker.services.image_records.delete_many(image_names)
            for image_name in image_names:
                self._on_deleted(image_name)
        except ImageRecordDeleteException:
            self.__invoker.services.logger.error("Failed to delete image records")
            raise
        except ImageFileDeleteException:
            self.__invoker.services.logger.error("Failed to delete image files")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem deleting image records and files")
            raise e

    def delete_intermediates(self) -> int:
        try:
            image_names = self.__invoker.services.image_records.delete_intermediates()
            count = len(image_names)
            for image_name in image_names:
                self.__invoker.services.image_files.delete(image_name)
                self._on_deleted(image_name)
            return count
        except ImageRecordDeleteException:
            self.__invoker.services.logger.error("Failed to delete image records")
            raise
        except ImageFileDeleteException:
            self.__invoker.services.logger.error("Failed to delete image files")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem deleting image records and files")
            raise e

    def get_intermediates_count(self) -> int:
        try:
            return self.__invoker.services.image_records.get_intermediates_count()
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting intermediates count")
            raise e
