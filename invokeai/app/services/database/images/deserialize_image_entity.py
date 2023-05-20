from invokeai.app.models.metadata import (
    GeneratedImageOrLatentsMetadata,
    UploadedImageOrLatentsMetadata,
)
from invokeai.app.models.image import ImageCategory, ImageType
from invokeai.app.services.database.images.models import ImageEntity


def deserialize_image_entity(image: dict) -> ImageEntity:
    """Deserializes an image entity from the database."""

    image_type = ImageType(image["image_type"])

    if image_type is ImageType.UPLOAD:
        return ImageEntity(
            id=image["id"],
            image_type=ImageType.UPLOAD,
            image_category=ImageCategory(image["image_category"]),
            created_at=image["created_at"],
            metadata=UploadedImageOrLatentsMetadata.parse_raw(image["metadata"]),
        )

    if image_type is ImageType.INTERMEDIATE:
        return ImageEntity(
            id=image["id"],
            session_id=image["session_id"],
            node_id=image["node_id"],
            image_type=ImageType.INTERMEDIATE,
            image_category=ImageCategory(image["image_category"]),
            created_at=image["created_at"],
            metadata=GeneratedImageOrLatentsMetadata.parse_raw(image["metadata"]),
        )

    return ImageEntity(
        id=image["id"],
        session_id=image["session_id"],
        node_id=image["node_id"],
        image_type=ImageType.RESULT,
        image_category=ImageCategory(image["image_category"]),
        created_at=image["created_at"],
        metadata=GeneratedImageOrLatentsMetadata.parse_raw(image["metadata"]),
    )
