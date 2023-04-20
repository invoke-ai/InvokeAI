import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypedDict
from PIL import Image, PngImagePlugin
from pydantic import BaseModel

from invokeai.app.models.image import ImageType, is_image_type


class MetadataImageField(TypedDict):
    """Pydantic-less ImageField, used for metadata parsing."""

    image_type: ImageType
    image_name: str


class MetadataLatentsField(TypedDict):
    """Pydantic-less LatentsField, used for metadata parsing."""

    latents_name: str


NodeMetadata = Dict[
    str, str | int | float | bool | MetadataImageField | MetadataLatentsField
]


class InvokeAIMetadata(TypedDict, total=False):
    """InvokeAI-specific metadata format."""

    session_id: Optional[str]
    node: Optional[NodeMetadata]


def build_pnginfo(metadata: InvokeAIMetadata | None) -> PngImagePlugin.PngInfo:
    pnginfo = PngImagePlugin.PngInfo()

    if metadata is not None:
        pnginfo.add_text("invokeai", json.dumps(metadata))

    return pnginfo


def parse_image_field(image_field: dict[str, Any]) -> dict[str, Any] | None:
    """Parses an object as a MetadataImageField"""

    # Must be a dict
    if type(image_field) is not dict:
        return None

    # An ImageField must have both `image_name` and `image_type`
    if not ("image_name" in image_field and "image_type" in image_field):
        return None

    # An ImageField's `image_type` must be one of the allowed values
    if not is_image_type(image_field["image_type"]):
        return None

    # An ImageField's `image_name` must be a string
    if type(image_field["image_name"]) is not str:
        return None

    parsed = {
        "image_type": image_field["image_type"],
        "image_name": image_field["image_name"],
    }

    return parsed


def parse_latents_field(latents_field: dict[str, Any]) -> dict[str, Any] | None:
    """Parses an object as a MetadataLatentsField"""

    # Must be a dict
    if type(latents_field) is not dict:
        return None

    # A LatentsField must have a `latents_name`
    if not ("latents_name" in latents_field):
        return None

    # A LatentsField's `latents_name` must be a string
    if type(latents_field["latents_name"]) is not str:
        return None

    parsed = {
        "latents_name": latents_field["latents_name"],
    }

    return parsed


def parse_node_metadata(node_metadata: Any) -> NodeMetadata | None:
    """Parses node metadata, silently skipping invalid entries"""

    # Must be a dict
    if type(node_metadata) is not dict:
        return None

    # Must have attributes
    if len(node_metadata.items()) == 0:
        return None

    parsed: dict[str, Any] = {}

    # Conditionally build the parsed metadata, silently skipping invalid values
    for name, value in node_metadata.items():
        value_type = type(value)

        # explicitly parse `id` and `type` as strings
        if name == "id":
            if value_type is not str:
                continue
            parsed[name] = value
            continue

        if name == "type":
            if value_type is not str:
                continue
            parsed[name] = value
            continue

        # we only allow ImageField and ImageType as dicts
        if value_type is dict:
            if "image_name" in value or "image_type" in value:
                # parse as an ImageField
                image_field = parse_image_field(value)
                if image_field is not None:
                    parsed[name] = image_field
                continue

            if "latents_name" in value:
                # parse as a LatentsField
                latents_field = parse_latents_field(value)
                if latents_field is not None:
                    parsed[name] = latents_field
                continue

        # other allowed primitive values
        if (
            value_type is str
            or value_type is int
            or value_type is float
            or value_type is bool
        ):
            parsed[name] = value
            continue
    return parsed


def parse_invokeai_metadata(
    invokeai_metadata: dict[str, Any]
) -> InvokeAIMetadata | None:
    """Parse the InvokeAI metadata format, silently skipping invalid entries"""

    # Must be a dict
    if type(invokeai_metadata) is not dict:
        return None

    # Must have attributes
    if len(invokeai_metadata.items()) == 0:
        return None

    parsed: InvokeAIMetadata = {}

    for name, value in invokeai_metadata.items():
        if name == "session_id":
            if type(value) is str:
                parsed[name] = value
            continue

        if name == "node":
            node_metadata = parse_node_metadata(value)
            if node_metadata is not None:
                parsed[name] = node_metadata
            continue
    return parsed


class MetadataServiceBase(ABC):
    @abstractmethod
    def get_metadata(self, image: Image.Image) -> InvokeAIMetadata | None:
        """Gets the InvokeAI metadata from a PIL Image, skipping invalid values"""
        pass

    @abstractmethod
    def build_metadata(
        self, session_id: str, node: BaseModel
    ) -> InvokeAIMetadata | None:
        """Builds an InvokeAIMetadata object"""
        pass


class PngMetadataService(MetadataServiceBase):
    """Handles loading metadata from images and parsing it."""

    # TODO: Support parsing old format metadata **hurk**
    # TODO: Use `InvocationsUnion` to **validate** metadata as representing a fully-functioning node
    def _load_metadata(self, image: Image.Image, key="invokeai") -> Any:
        """Loads a specific info entry from a PIL Image."""

        raw_metadata = image.info.get(key)

        # metadata should always be a dict
        if type(raw_metadata) is not str:
            return None

        loaded_metadata = json.loads(raw_metadata)

        return loaded_metadata

    def _parse_invokeai_metadata(
        self,
        metadata: Any,
    ) -> InvokeAIMetadata | None:
        """Parses an object as InvokeAI metadata."""
        if type(metadata) is not dict:
            return None

        parsed_metadata = parse_invokeai_metadata(metadata)

        return parsed_metadata

    def get_metadata(self, image: Image.Image) -> InvokeAIMetadata | None:
        loaded_metadata = self._load_metadata(image)
        parsed_metadata = self._parse_invokeai_metadata(loaded_metadata)

        return parsed_metadata

    def build_metadata(self, session_id: str, node: BaseModel) -> InvokeAIMetadata:
        metadata = InvokeAIMetadata(session_id=session_id, node=node.dict())

        return metadata
