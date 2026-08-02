"""Common types and helpers for the semantic image index services."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

EMBEDDING_DTYPE = np.float32


class ImageIndexStatus(BaseModel):
    """Progress of the embedding index for one embedding model."""

    total: int = Field(description="Number of gallery images eligible for indexing")
    embedded: int = Field(description="Number of eligible images that have an embedding")

    @property
    def pending(self) -> int:
        return max(0, self.total - self.embedded)


class ProjectionRecord(BaseModel):
    """A cached 2D projection of a user's accessible images."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str = Field(description="The user the projection was computed for")
    model_id: str = Field(description="Content hash of the embedding model")
    scope_hash: str = Field(description="Fingerprint of the image set the projection covers")
    params: str = Field(description="JSON of the projection parameters")
    point_count: int = Field(description="Number of projected points")
    image_names: list[str] = Field(description="Image names, row-aligned with coords")
    coords: np.ndarray = Field(description="float32 array of shape (point_count, 2)")
    created_at: str = Field(description="When the projection was first computed")
    updated_at: str = Field(description="When the projection was last recomputed")


def embedding_to_blob(embedding: np.ndarray) -> bytes:
    """Serialize a 1-D embedding vector to bytes for BLOB storage."""
    if embedding.ndim != 1:
        raise ValueError(f"Expected a 1-D embedding, got shape {embedding.shape}")
    return np.ascontiguousarray(embedding, dtype=EMBEDDING_DTYPE).tobytes()


def blob_to_embedding(blob: bytes, dim: int) -> np.ndarray:
    """Deserialize an embedding BLOB, validating its length against the stored dim.

    Returns a read-only view over the blob; copy before mutating.
    """
    if len(blob) != dim * EMBEDDING_DTYPE().itemsize:
        raise ValueError(
            f"Embedding blob is {len(blob)} bytes; expected {dim * EMBEDDING_DTYPE().itemsize} for dim {dim}"
        )
    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE)


def coords_to_blob(coords: np.ndarray) -> bytes:
    """Serialize an (N, 2) coordinate array to bytes for BLOB storage."""
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coords of shape (N, 2), got {coords.shape}")
    return np.ascontiguousarray(coords, dtype=EMBEDDING_DTYPE).tobytes()


def blob_to_coords(blob: bytes, point_count: int) -> np.ndarray:
    """Deserialize a coordinate BLOB, validating its length against the stored point count."""
    expected = point_count * 2 * EMBEDDING_DTYPE().itemsize
    if len(blob) != expected:
        raise ValueError(f"Coords blob is {len(blob)} bytes; expected {expected} for {point_count} points")
    # Copy so callers get a writable array rather than a read-only buffer view.
    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE).reshape(point_count, 2).copy()
