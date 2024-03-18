from abc import ABC, abstractmethod


class UrlServiceBase(ABC):
    """Responsible for building URLs for resources."""

    @abstractmethod
    def get_image_url(self, image_name: str, thumbnail: bool = False) -> str:
        """Gets the URL for an image or thumbnail."""
        pass

    @abstractmethod
    def get_model_image_url(self, model_key: str) -> str:
        """Gets the URL for a model image"""
        pass
