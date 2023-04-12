from pydantic import BaseModel, Field

from invokeai.app.models.image import ImageType
from invokeai.app.models.metadata import ImageMetadata


class ImageResponse(BaseModel):
    """The response type for images"""

    image_type: ImageType = Field(description="The type of the image")
    image_name: str = Field(description="The name of the image")
    image_url: str = Field(description="The url of the image")
    thumbnail_url: str = Field(description="The url of the image's thumbnail")
    metadata: ImageMetadata = Field(description="The image's metadata")
