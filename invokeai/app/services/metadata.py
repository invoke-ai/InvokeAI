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


class MetadataColorField(TypedDict):
    """Pydantic-less ColorField, used for metadata parsing"""
    r: int
    g: int
    b: int
    a: int



# TODO: This is a placeholder for `InvocationsUnion` pending resolution of circular imports
NodeMetadata = Dict[
    str, None | str | int | float | bool | MetadataImageField | MetadataLatentsField | MetadataColorField
]


class InvokeAIMetadata(TypedDict, total=False):
    """InvokeAI-specific metadata format."""

    session_id: Optional[str]
    node: Optional[NodeMetadata]


def build_invokeai_metadata_pnginfo(
    metadata: InvokeAIMetadata | None,
) -> PngImagePlugin.PngInfo:
    """Builds a PngInfo object with key `"invokeai"` and value `metadata`"""
    pnginfo = PngImagePlugin.PngInfo()

    if metadata is not None:
        pnginfo.add_text("invokeai", json.dumps(metadata))

    return pnginfo


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
    """Handles loading and building metadata for images."""

    # TODO: Use `InvocationsUnion` to **validate** metadata as representing a fully-functioning node
    def _load_metadata(self, image: Image.Image) -> dict | None:
        """Loads a specific info entry from a PIL Image."""

        try:
            info = image.info.get("invokeai")

            if type(info) is not str:
                return None

            loaded_metadata = json.loads(info)

            if type(loaded_metadata) is not dict:
                return None

            if len(loaded_metadata.items()) == 0:
                return None

            return loaded_metadata
        except:
            return None

    def get_metadata(self, image: Image.Image) -> dict | None:
        """Retrieves an image's metadata as a dict"""
        loaded_metadata = self._load_metadata(image)

        return loaded_metadata

    def build_metadata(self, session_id: str, node: BaseModel) -> InvokeAIMetadata:
        metadata = InvokeAIMetadata(session_id=session_id, node=node.dict())

        return metadata
