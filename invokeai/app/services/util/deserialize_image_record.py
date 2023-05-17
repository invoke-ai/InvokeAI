from invokeai.app.models.metadata import (
    GeneratedImageOrLatentsMetadata,
    UploadedImageOrLatentsMetadata,
)
from invokeai.app.models.image import ImageCategory, ImageType
from invokeai.app.services.models.image_record import ImageRecord
from invokeai.app.util.misc import get_iso_timestamp


def deserialize_image_record(image: dict) -> ImageRecord:
    """Deserializes an image record."""

    # All values *should* be present, except `session_id` and `node_id`, but provide some defaults just in case

    image_type = ImageType(image.get("image_type", ImageType.RESULT.value))
    raw_metadata = image.get("metadata", {})

    if image_type == ImageType.UPLOAD:
        metadata = UploadedImageOrLatentsMetadata.parse_obj(raw_metadata)
    else:
        metadata = GeneratedImageOrLatentsMetadata.parse_obj(raw_metadata)

    return ImageRecord(
        image_name=image.get("id", "unknown"),
        session_id=image.get("session_id", None),
        node_id=image.get("node_id", None),
        metadata=metadata,
        image_type=image_type,
        image_category=ImageCategory(
            image.get("image_category", ImageCategory.IMAGE.value)
        ),
        created_at=image.get("created_at", get_iso_timestamp()),
    )
