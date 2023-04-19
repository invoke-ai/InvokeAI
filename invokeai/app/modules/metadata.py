import json
from typing import Any, Dict, Literal, Optional
from PIL import Image, PngImagePlugin
from pydantic import (
    BaseModel,
    Extra,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    root_validator,
)

from invokeai.app.models.image import ImageType


class MetadataImageField(BaseModel):
    """A non-nullable version of ImageField"""

    image_type: Literal[tuple([t.value for t in ImageType])]  # type: ignore
    image_name: StrictStr


class MetadataLatentsField(BaseModel):
    """A non-nullable version of LatentsField"""

    latents_name: StrictStr


# Union of all valid metadata field types - use mostly strict types
NodeMetadataFieldTypes = (
    StrictStr | StrictInt | float | StrictBool  # we want to cast ints to floats here
)


class NodeMetadataField(BaseModel):
    """Helper class used as a hack for arbitrary metadata field keys."""

    __root__: Dict[StrictStr, NodeMetadataFieldTypes]


# `extra=Extra.allow` allows this to model any potential node with `id` and `type` fields
class NodeMetadata(BaseModel, extra=Extra.allow):
    """Node metadata model, used for validation of metadata."""

    @root_validator
    def validate_node_metadata(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Parses the node metadata, ignoring invalid values"""
        parsed: dict[str, Any] = {}

        # Conditionally build the parsed metadata, silently skipping invalid values
        for name, value in values.items():
            # explicitly parse `id` and `type` as strings
            if name == "id":
                if type(value) is not str:
                    continue
                parsed[name] = value
            elif name == "type":
                if type(value) is not str:
                    continue
                parsed[name] = value
            else:
                try:
                    if type(value) is dict:
                        # we only allow certain dicts, else just ignore the value entirely
                        if "image_name" in value or "image_type" in value:
                            # parse as an ImageField
                            parsed[name] = MetadataImageField.parse_obj(value)
                        elif "latents_name" in value:
                            # this is a LatentsField
                            parsed[name] = MetadataLatentsField.parse_obj(value)
                    else:
                        # hack to get parse and validate arbitrary keys
                        NodeMetadataField.parse_obj({name: value})
                        parsed[name] = value
                except ValidationError:
                    # TODO: do we want to somehow alert when metadata is not fully valid?
                    continue
        return parsed


class InvokeAIMetadata(BaseModel):
    session_id: Optional[StrictStr] = Field(
        description="The session in which this image was created"
    )
    node: Optional[NodeMetadata] = Field(description="The node that created this image")

    @root_validator(pre=True)
    def validate_invokeai_metadata(cls, values: dict[str, Any]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        # Conditionally build the parsed metadata, silently skipping invalid values
        for name, value in values.items():
            if name == "session_id":
                if type(value) is not str:
                    continue
                parsed[name] = value
            elif name == "node":
                try:
                    p = NodeMetadata.parse_obj(value)
                    # check for empty NodeMetadata object
                    if len(p.dict().items()) == 0:
                        continue
                except ValidationError:
                    continue
                parsed[name] = value

        return parsed


class ImageMetadata(BaseModel):
    """An image's metadata. Used only in HTTP responses."""

    created: int = Field(description="The creation timestamp of the image")
    width: int = Field(description="The width of the image in pixels")
    height: int = Field(description="The height of the image in pixels")
    mode: str = Field(description="The color mode of the image")
    invokeai: Optional[InvokeAIMetadata] = Field(
        description="The image's InvokeAI-specific metadata"
    )


class MetadataModule:
    """Handles loading metadata from images and parsing it."""

    # TODO: Support parsing old format metadata **hurk**

    @staticmethod
    def _load_metadata(image: Image.Image, key="invokeai") -> Any:
        """Loads a specific info entry from a PIL Image."""

        raw_metadata = image.info.get(key)

        # metadata should always be a dict
        if type(raw_metadata) is not str:
            return None

        loaded_metadata = json.loads(raw_metadata)

        return loaded_metadata

    @staticmethod
    def _parse_invokeai_metadata(
        metadata: Any,
    ) -> InvokeAIMetadata | None:
        """Parses an object as InvokeAI metadata."""
        if type(metadata) is not dict:
            return None

        parsed_metadata = InvokeAIMetadata.parse_obj(metadata)

        return parsed_metadata

    @staticmethod
    def get_metadata(image: Image.Image) -> InvokeAIMetadata | None:
        """Gets the InvokeAI metadata from a PIL Image, skipping invalid values"""
        loaded_metadata = MetadataModule._load_metadata(image)
        parsed_metadata = MetadataModule._parse_invokeai_metadata(loaded_metadata)

        return parsed_metadata

    @staticmethod
    def build_metadata(
        session_id: StrictStr, invocation: BaseModel
    ) -> InvokeAIMetadata:
        """Builds an InvokeAIMetadata object"""
        metadata = InvokeAIMetadata(
            session_id=session_id, node=NodeMetadata(**invocation.dict())
        )

        return metadata

    @staticmethod
    def build_png_info(metadata: InvokeAIMetadata | None):
        png_info = PngImagePlugin.PngInfo()

        if metadata is not None:
            png_info.add_text("invokeai", metadata.json())

        return png_info
