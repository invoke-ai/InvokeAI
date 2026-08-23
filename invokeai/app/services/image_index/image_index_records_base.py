from abc import ABC, abstractmethod

import numpy as np

from invokeai.app.services.image_index.image_index_common import ImageIndexStatus, ProjectionRecord


class ImageIndexRecordsBase(ABC):
    """Storage for the semantic image index.

    Embeddings are global: one row per (image_name, model_id), shared by every
    user who can access the image. Projections are per-user caches over the
    set of images that user can access.

    Only "gallery" images are indexed: non-intermediate images in the
    `general` category.

    Every method here is its own unit of work and must not be called from inside another
    service's open transaction. `SqliteDatabase.transaction()` commits and rolls back the whole
    shared connection rather than using savepoints, so a nested call either commits the outer
    transaction's work early (on success) or discards it (on failure). No caller in the tree
    nests these today; keep it that way.
    """

    @abstractmethod
    def upsert_embedding(self, image_name: str, model_id: str, embedding: np.ndarray) -> None:
        """Insert or replace the embedding for an image under the given model.

        A no-op if the image no longer exists (it may be deleted between being
        scheduled for embedding and the write landing).

        All embeddings stored under one model_id must share the same dim;
        `get_embeddings` fails on a result set with inconsistent dims.

        Raises ValueError on a vector that could not be a normalized embedding — empty, all
        zero, non-floating dtype, or containing NaN/inf — since any of those poisons every
        batch it later appears in.
        """
        pass

    @abstractmethod
    def get_embeddings(self, image_names: list[str], model_id: str) -> tuple[list[str], np.ndarray]:
        """Fetch embeddings for the given images.

        Duplicate input names are deduplicated, preserving first-seen order.

        Returns:
            A tuple of (found_names, matrix) where matrix has shape
            (len(found_names), dim) and rows align with found_names. Images
            without a stored embedding are silently omitted.

            When nothing matches, the matrix is empty with shape (0, 0): no row was read, so
            there is no dim to report. Callers that need the dim must get it elsewhere rather
            than from `matrix.shape[1]`.
        """
        pass

    @abstractmethod
    def delete_embedding(self, image_name: str) -> None:
        """Delete all stored embeddings for an image (across all models)."""
        pass

    @abstractmethod
    def delete_embeddings_for_other_models(self, model_id: str) -> int:
        """Delete embeddings computed by any model other than the given one.

        Used when the configured embedding model changes: rows from the
        previous model are dead weight.

        Returns:
            The number of rows deleted.
        """
        pass

    @abstractmethod
    def list_unembedded_image_names(self, model_id: str, limit: int) -> list[str]:
        """List eligible images that have no embedding under the given model, oldest first."""
        pass

    @abstractmethod
    def count_index_status(self, model_id: str) -> ImageIndexStatus:
        """Count eligible images and how many of them are embedded under the given model."""
        pass

    @abstractmethod
    def list_accessible_embedded_images(self, user_id: str | None, model_id: str) -> list[str]:
        """List embedded images the user can access, sorted by image name.

        A user can access their own unboarded images, images on boards they own, images on
        shared or public boards, and images on boards individually shared with them
        (shared_boards). Pass user_id=None for the admin scope.

        Images on an archived board are excluded from every scope, including the admin one,
        matching the gallery's "all" listing. So the admin scope is every embedded image that
        is not archived, not literally every embedded image.

        The sorted result is the input to the projection scope hash, so the
        ordering here must stay stable.
        """
        pass

    @abstractmethod
    def get_custom_vocab_terms(self) -> list[str]:
        """List the supplementary cluster-labeling vocabulary terms, sorted by term.

        The list is server-wide (see the vocabulary table's migration) and is
        merged with the bundled vocabulary at embedding-build time. The sorted
        order feeds the vocabulary fingerprint, so it must stay stable.
        """
        pass

    @abstractmethod
    def set_custom_vocab_terms(self, terms: list[str]) -> None:
        """Replace the supplementary vocabulary with the given terms, atomically.

        Callers pass normalized terms (`normalize_custom_vocab_terms`); this
        stores what it is given. Replacement rather than per-term mutation:
        the vocabulary is a single small list edited as a whole, and replace
        semantics make the API idempotent.
        """
        pass

    @abstractmethod
    def get_projection(self, user_id: str, model_id: str) -> ProjectionRecord | None:
        """Get the user's cached projection, or None if one was never computed."""
        pass

    @abstractmethod
    def set_projection(
        self,
        user_id: str,
        model_id: str,
        scope_hash: str,
        params: str,
        image_names: list[str],
        coords: np.ndarray,
    ) -> None:
        """Insert or replace the user's cached projection.

        A no-op if the user no longer exists.

        Args:
            user_id: The user the projection was computed for.
            model_id: Content hash of the embedding model.
            scope_hash: Fingerprint of the image set the projection covers.
            params: JSON of the projection parameters.
            image_names: Image names, row-aligned with coords.
            coords: float32 array of shape (len(image_names), 2).
        """
        pass

    @abstractmethod
    def delete_projection(self, user_id: str, model_id: str) -> None:
        """Delete the user's cached projection. Idempotent."""
        pass
