from typing import Optional, Union
import uuid
from PIL.Image import Image as PILImageType
from invokeai.app.models.image import ImageCategory, ImageType
from invokeai.app.models.metadata import (
    GeneratedImageOrLatentsMetadata,
    UploadedImageOrLatentsMetadata,
)
from invokeai.app.services.image_record_storage import (
    ImageRecordStorageBase,
)
from invokeai.app.services.models.image_record import (
    ImageRecord,
    ImageDTO,
    image_record_to_dto,
)
from invokeai.app.services.image_file_storage import ImageFileStorageBase
from invokeai.app.services.item_storage import PaginatedResults
from invokeai.app.services.metadata import MetadataServiceBase
from invokeai.app.services.urls import UrlServiceBase
from invokeai.app.util.misc import get_iso_timestamp


class ImageServiceDependencies:
    """Service dependencies for the ImageManagementService."""

    records: ImageRecordStorageBase
    files: ImageFileStorageBase
    metadata: MetadataServiceBase
    urls: UrlServiceBase

    def __init__(
        self,
        image_record_storage: ImageRecordStorageBase,
        image_file_storage: ImageFileStorageBase,
        metadata: MetadataServiceBase,
        url: UrlServiceBase,
    ):
        self.records = image_record_storage
        self.files = image_file_storage
        self.metadata = metadata
        self.urls = url


class ImageService:
    """High-level service for image management."""

    _services: ImageServiceDependencies

    def __init__(
        self,
        image_record_storage: ImageRecordStorageBase,
        image_file_storage: ImageFileStorageBase,
        metadata: MetadataServiceBase,
        url: UrlServiceBase,
    ):
        self._services = ImageServiceDependencies(
            image_record_storage=image_record_storage,
            image_file_storage=image_file_storage,
            metadata=metadata,
            url=url,
        )

    def _create_image_name(
        self,
        image_type: ImageType,
        image_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Creates an image name."""
        uuid_str = str(uuid.uuid4())

        if node_id is not None and session_id is not None:
            return f"{image_type.value}_{image_category.value}_{session_id}_{node_id}_{uuid_str}.png"

        return f"{image_type.value}_{image_category.value}_{uuid_str}.png"

    def create(
        self,
        image: PILImageType,
        image_type: ImageType,
        image_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[
            Union[GeneratedImageOrLatentsMetadata, UploadedImageOrLatentsMetadata]
        ] = None,
    ) -> ImageDTO:
        """Creates an image, storing the file and its metadata."""
        image_name = self._create_image_name(
            image_type=image_type,
            image_category=image_category,
            node_id=node_id,
            session_id=session_id,
        )

        timestamp = get_iso_timestamp()

        try:
            # TODO: Consider using a transaction here to ensure consistency between storage and database
            self._services.files.save(
                image_type=image_type,
                image_name=image_name,
                image=image,
                metadata=metadata,
            )

            self._services.records.save(
                image_name=image_name,
                image_type=image_type,
                image_category=image_category,
                node_id=node_id,
                session_id=session_id,
                metadata=metadata,
                created_at=timestamp,
            )

            image_url = self._services.urls.get_image_url(image_type, image_name)
            thumbnail_url = self._services.urls.get_thumbnail_url(image_type, image_name)

            return ImageDTO(
                image_name=image_name,
                image_type=image_type,
                image_category=image_category,
                node_id=node_id,
                session_id=session_id,
                metadata=metadata,
                created_at=timestamp,
                image_url=image_url,
                thumbnail_url=thumbnail_url,
            )
        except ImageRecordStorageBase.ImageRecordSaveException:
            # TODO: log this
            raise
        except ImageFileStorageBase.ImageFileSaveException:
            # TODO: log this
            raise

    def get_pil_image(self, image_type: ImageType, image_name: str) -> PILImageType:
        """Gets an image as a PIL image."""
        try:
            return self._services.files.get(image_type, image_name)
        except ImageFileStorageBase.ImageFileNotFoundException:
            # TODO: log this
            raise

    def get_record(self, image_type: ImageType, image_name: str) -> ImageRecord:
        """Gets an image record."""
        try:
            return self._services.records.get(image_type, image_name)
        except ImageRecordStorageBase.ImageRecordNotFoundException:
            # TODO: log this
            raise

    def get_dto(self, image_type: ImageType, image_name: str) -> ImageDTO:
        """Gets an image DTO."""
        try:
            image_record = self._services.records.get(image_type, image_name)

            image_dto = image_record_to_dto(
                image_record,
                self._services.urls.get_image_url(image_type, image_name),
                self._services.urls.get_thumbnail_url(image_type, image_name),
            )

            return image_dto
        except ImageRecordStorageBase.ImageRecordNotFoundException:
            # TODO: log this
            raise

    def delete(self, image_type: ImageType, image_name: str):
        """Deletes an image."""
        # TODO: Consider using a transaction here to ensure consistency between storage and database
        try:
            self._services.files.delete(image_type, image_name)
            self._services.records.delete(image_type, image_name)
        except ImageRecordStorageBase.ImageRecordDeleteException:
            # TODO: log this
            raise
        except ImageFileStorageBase.ImageFileDeleteException:
            # TODO: log this
            raise

    def get_many(
        self,
        image_type: ImageType,
        image_category: ImageCategory,
        page: int = 0,
        per_page: int = 10,
    ) -> PaginatedResults[ImageDTO]:
        """Gets a paginated list of image DTOs."""
        try:
            results = self._services.records.get_many(
                image_type,
                image_category,
                page,
                per_page,
            )

            image_dtos = list(
                map(
                    lambda r: image_record_to_dto(
                        r,
                        self._services.urls.get_image_url(image_type, r.image_name),
                        self._services.urls.get_thumbnail_url(image_type, r.image_name),
                    ),
                    results.items,
                )
            )

            return PaginatedResults[ImageDTO](
                items=image_dtos,
                page=results.page,
                pages=results.pages,
                per_page=results.per_page,
                total=results.total,
            )
        except Exception as e:
            raise e

    def add_tag(self, image_type: ImageType, image_id: str, tag: str) -> None:
        """Adds a tag to an image."""
        raise NotImplementedError("The 'add_tag' method is not implemented yet.")

    def remove_tag(self, image_type: ImageType, image_id: str, tag: str) -> None:
        """Removes a tag from an image."""
        raise NotImplementedError("The 'remove_tag' method is not implemented yet.")

    def favorite(self, image_type: ImageType, image_id: str) -> None:
        """Favorites an image."""
        raise NotImplementedError("The 'favorite' method is not implemented yet.")

    def unfavorite(self, image_type: ImageType, image_id: str) -> None:
        """Unfavorites an image."""
        raise NotImplementedError("The 'unfavorite' method is not implemented yet.")
