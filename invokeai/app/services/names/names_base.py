from abc import ABC, abstractmethod


class NameServiceBase(ABC):
    """Low-level service responsible for naming resources (images, latents, etc)."""

    # TODO: Add customizable naming schemes
    @abstractmethod
    def create_image_name(self) -> str:
        """Creates a name for an image."""
        pass
