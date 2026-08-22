from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from PIL import Image

from invokeai.app.services.image_index.image_index_common import ImageIndexStatus


class TextSearchUnavailableError(Exception):
    """The configured embedding model has no usable text tower installed."""


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
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text query into the image embedding space (L2-normalized).

        Slow on first call (loads the model's text tower); cheap afterwards.
        Call off the event loop.

        Raises:
            TextSearchUnavailableError: The indexer is not running, or the
                configured model's directory has no text encoder (some
                vision-only CLIP installs, e.g. for IP-Adapter, ship only the
                image tower).
        """
        pass

    @abstractmethod
    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Embed one image with the index's vision encoder (L2-normalized, (D,)).

        Used for one-off similarity queries (assets and external images that
        are not in the index). Call off the event loop — this may load the
        model.

        Raises:
            RuntimeError: The indexer is not running.
        """
        pass

    @abstractmethod
    def get_accessible_embeddings(self, user_id: str | None) -> tuple[list[str], np.ndarray]:
        """Names + L2-normalized embedding matrix of the user's accessible images.

        Served from a small LRU keyed by the accessible-set scope hash (which
        self-invalidates on any set change). Rows align with names. Pass
        user_id=None for the admin scope. Call off the event loop.
        """
        pass

    @abstractmethod
    def get_vocab_embeddings(self) -> tuple[list[str], np.ndarray]:
        """Get the cluster-labeling vocabulary and its phrase embeddings.

        Slow on the first call per model (the whole vocabulary is embedded
        through the text tower, then disk-cached); cheap afterwards. Call off
        the event loop.

        Raises:
            TextSearchUnavailableError: No usable text encoder is installed.
        """
        pass

    @abstractmethod
    def search_similar(self, user_id: str | None, query_embedding: np.ndarray, limit: int) -> list[tuple[str, float]]:
        """Rank the user's accessible embedded images by cosine similarity.

        Embeddings are L2-normalized, so similarity is a dot product. Pass
        user_id=None for the admin scope. Returns (image_name, score) pairs,
        best first. Call off the event loop — the accessible embedding matrix
        may be read from the database on a cache miss.
        """
        pass

    @abstractmethod
    def request_projection(
        self,
        user_id: str,
        all_images: bool = False,
        failed_scope: Optional[str] = None,
        user_initiated: bool = False,
    ) -> bool:
        """Ask the worker to (re)compute a user's image map projection.

        Requests are deduplicated per user; the projection runs after any
        pending embedding work, and an `image_map_projection_ready` event is
        emitted when the cache is updated.

        Args:
            user_id: The user whose projection cache to update.
            all_images: Compute over every embedded image (admin scope)
                rather than the user's accessible set.
            failed_scope: The scope hash of a cached projection the caller
                believes to be a failed fit. The request is refused once that
                scope's single retry has been used, so a caller that asks on
                every poll cannot drive an endless request/event cycle.
            user_initiated: This request came from a person rather than from
                polling, so a spent retry budget is cleared and the failed
                scope gets another fit. Callers driven by a timer, an event, or
                a staleness check must leave this False.

        Returns:
            True if the request was accepted (or already pending); False if
            the indexer is not running.
        """
        pass
