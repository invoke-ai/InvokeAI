from abc import ABC, abstractmethod
from enum import Enum, EnumMeta
import uuid


class ResourceType(str, Enum, metaclass=EnumMeta):
    """Enum for resource types."""

    IMAGE = "image"
    LATENT = "latent"


class NameServiceBase(ABC):
    """Low-level service responsible for naming resources (images, latents, etc)."""

    # TODO: Add customizable naming schemes
    @abstractmethod
    def create_image_name(self) -> str:
        """Creates a name for an image."""
        pass


class SimpleNameService(NameServiceBase):
    """Creates image names from UUIDs."""

    # TODO: Add customizable naming schemes
    def create_image_name(self) -> str:
        uuid_str = str(uuid.uuid4())
        filename = f"{uuid_str}.png"
        return filename
