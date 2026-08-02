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

    @abstractmethod
    def request_projection(self, user_id: str, all_images: bool = False) -> bool:
        """Ask the worker to (re)compute a user's image map projection.

        Requests are deduplicated per user; the projection runs after any
        pending embedding work, and an `image_map_projection_ready` event is
        emitted when the cache is updated.

        Args:
            user_id: The user whose projection cache to update.
            all_images: Compute over every embedded image (admin scope)
                rather than the user's accessible set.

        Returns:
            True if the request was accepted (or already pending); False if
            the indexer is not running.
        """
        pass
