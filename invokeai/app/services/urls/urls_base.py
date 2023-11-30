from abc import ABC, abstractmethod


class UrlServiceBase(ABC):
    """Responsible for building URLs for resources."""

    @abstractmethod
    def get_image_url(self, image_name: str, thumbnail: bool = False) -> str:
        """Gets the URL for an image or thumbnail."""
        pass
