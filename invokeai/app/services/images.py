from typing import Union
import uuid
from PIL.Image import Image as PILImageType
from invokeai.app.models.image import ImageCategory, ImageType
from invokeai.app.models.metadata import (
    GeneratedImageOrLatentsMetadata,
    UploadedImageOrLatentsMetadata,
)
from invokeai.app.services.image_db import (
    ImageRecordServiceBase,
)
from invokeai.app.services.models.image_record import ImageRecord
from invokeai.app.services.image_storage import ImageStorageBase
from invokeai.app.services.item_storage import PaginatedResults
from invokeai.app.services.metadata import MetadataServiceBase
from invokeai.app.services.urls import UrlServiceBase
from invokeai.app.util.misc import get_iso_timestamp


class ImageServiceDependencies:
    """Service dependencies for the ImageManagementService."""

    db: ImageRecordServiceBase
    storage: ImageStorageBase
    metadata: MetadataServiceBase
    urls: UrlServiceBase

    def __init__(
        self,
        image_db_service: ImageRecordServiceBase,
        image_storage_service: ImageStorageBase,
        image_metadata_service: MetadataServiceBase,
        url_service: UrlServiceBase,
    ):
        self.db = image_db_service
        self.storage = image_storage_service
        self.metadata = image_metadata_service
        self.url = url_service


class ImageService:
    """High-level service for image management."""

    _services: ImageServiceDependencies

    def __init__(
        self,
        image_db_service: ImageRecordServiceBase,
        image_storage_service: ImageStorageBase,
        image_metadata_service: MetadataServiceBase,
        url_service: UrlServiceBase,
    ):
        self._services = ImageServiceDependencies(
            image_db_service=image_db_service,
            image_storage_service=image_storage_service,
            image_metadata_service=image_metadata_service,
            url_service=url_service,
        )

    def _create_image_name(
        self,
        image_type: ImageType,
        image_category: ImageCategory,
        node_id: Union[str, None],
        session_id: Union[str, None],
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
        node_id: Union[str, None],
        session_id: Union[str, None],
        metadata: Union[
            GeneratedImageOrLatentsMetadata, UploadedImageOrLatentsMetadata, None
        ],
    ) -> ImageRecord:
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
            self._services.storage.save(
                image_type=image_type,
                image_name=image_name,
                image=image,
                metadata=metadata,
            )

            self._services.db.save(
                image_name=image_name,
                image_type=image_type,
                image_category=image_category,
                node_id=node_id,
                session_id=session_id,
                metadata=metadata,
                created_at=timestamp,
            )

            image_url = self._services.url.get_image_url(
                image_type=image_type, image_name=image_name
            )

            thumbnail_url = self._services.url.get_thumbnail_url(
                image_type=image_type, image_name=image_name
            )

            return ImageRecord(
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
        except ImageRecordServiceBase.ImageRecordSaveException:
            # TODO: log this
            raise
        except ImageStorageBase.ImageFileSaveException:
            # TODO: log this
            raise

    def get_pil_image(self, image_type: ImageType, image_name: str) -> PILImageType:
        """Gets an image as a PIL image."""
        try:
            pil_image = self._services.storage.get(
                image_type=image_type, image_name=image_name
            )
            return pil_image
        except ImageStorageBase.ImageFileNotFoundException:
            # TODO: log this
            raise

    def get_record(self, image_type: ImageType, image_name: str) -> ImageRecord:
        """Gets an image record."""
        try:
            image_record = self._services.db.get(
                image_type=image_type, image_name=image_name
            )
            return image_record
        except ImageRecordServiceBase.ImageRecordNotFoundException:
            # TODO: log this
            raise

    def delete(self, image_type: ImageType, image_name: str):
        """Deletes an image."""
        # TODO: Consider using a transaction here to ensure consistency between storage and database
        try:
            self._services.storage.delete(image_type=image_type, image_name=image_name)
            self._services.db.delete(image_type=image_type, image_name=image_name)
        except ImageRecordServiceBase.ImageRecordDeleteException:
            # TODO: log this
            raise
        except ImageStorageBase.ImageFileDeleteException:
            # TODO: log this
            raise

    def get_many(
        self,
        image_type: ImageType,
        image_category: ImageCategory,
        page: int = 0,
        per_page: int = 10,
    ) -> PaginatedResults[ImageRecord]:
        """Gets a paginated list of image records."""
        try:
            results = self._services.db.get_many(
                image_type=image_type,
                image_category=image_category,
                page=page,
                per_page=per_page,
            )

            for r in results.items:
                r.image_url = self._services.url.get_image_url(
                    image_type=image_type, image_name=r.image_name
                )

                r.thumbnail_url = self._services.url.get_thumbnail_url(
                    image_type=image_type, image_name=r.image_name
                )

            return results
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
