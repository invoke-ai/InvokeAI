from abc import ABC, abstractmethod

from invokeai.app.services.image_index.image_index_common import ImageIndexStatus


class ImageIndexServiceBase(ABC):
    """Background service that keeps the semantic image index up to date.

    When enabled and an embedding model is available, the service embeds every
    eligible gallery image (non-intermediate, `general` category) on a worker
    thread: a backfill pass covers images that existed before the service
    started, and image-service callbacks cover images created afterwards.
    """

    @property
    @abstractmethod
    def model_id(self) -> str | None:
        """Content hash of the active embedding model, or None if the indexer is not running."""
        pass

    @abstractmethod
    def get_status(self) -> ImageIndexStatus | None:
        """Get index progress counts, or None if the indexer is not running."""
        pass
