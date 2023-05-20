from invokeai.app.models.metadata import (
    GeneratedImageOrLatentsMetadata,
    UploadedImageOrLatentsMetadata,
)
from invokeai.app.models.resources import ImageKind, ResourceOrigin
from invokeai.app.services.database.images.models import ImageEntity


def deserialize_image_entity(image: dict) -> ImageEntity:
    """Deserializes an image entity from the database."""

    origin = ResourceOrigin(image["origin"])

    if origin is ResourceOrigin.UPLOADS:
        return ImageEntity(
            id=image["id"],
            origin=ResourceOrigin.UPLOADS,
            image_kind=ImageKind(image["image_kind"]),
            created_at=image["created_at"],
            metadata=UploadedImageOrLatentsMetadata.parse_raw(image["metadata"]),
        )

    if origin is ResourceOrigin.INTERMEDIATES:
        return ImageEntity(
            id=image["id"],
            session_id=image["session_id"],
            node_id=image["node_id"],
            origin=ResourceOrigin.INTERMEDIATES,
            image_kind=ImageKind(image["image_kind"]),
            created_at=image["created_at"],
            metadata=GeneratedImageOrLatentsMetadata.parse_raw(image["metadata"]),
        )

    return ImageEntity(
        id=image["id"],
        session_id=image["session_id"],
        node_id=image["node_id"],
        origin=ResourceOrigin.RESULTS,
        image_kind=ImageKind(image["image_kind"]),
        created_at=image["created_at"],
        metadata=GeneratedImageOrLatentsMetadata.parse_raw(image["metadata"]),
    )
