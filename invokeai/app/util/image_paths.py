from typing import Optional
from invokeai.app.models.image import ImageType
from invokeai.app.util.thumbnails import get_thumbnail_name


def build_image_path(image_type: ImageType, image_name: str, is_thumbnail: Optional[bool] = False) -> str:
    """Gets path to access an image"""
    if is_thumbnail is None:
        return f"api/v1/images/{image_type}/{image_name}"
    else:
        thumbnail_name = get_thumbnail_name(image_name)
        return f"api/v1/images/{image_type}/thumbnails/{thumbnail_name}"
