"""Common types and helpers for the semantic image index services."""

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

EMBEDDING_DTYPE = np.float32


class ImageIndexStatus(BaseModel):
    """Progress of the embedding index for one embedding model."""

    total: int = Field(description="Number of gallery images eligible for indexing")
    embedded: int = Field(description="Number of eligible images that have an embedding")
    failed: int = Field(
        default=0,
        description="Eligible images that repeatedly failed to embed; excluded from pending so it can drain",
    )

    @property
    def pending(self) -> int:
        # Excluding failures matters: consumers treat pending == 0 as "the
        # index is settled", and a count that can never drain would wedge
        # them (and show an indexing spinner forever).
        return max(0, self.total - self.embedded - self.failed)


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
    """Serialize a 1-D embedding vector to bytes for BLOB storage.

    The vector is stored as float32; a float64 input is narrowed. Callers are responsible for
    L2-normalizing — this only rejects vectors that could not have been normalized, because a
    non-finite value silently poisons every similarity and projection computation that later
    touches the same batch, with no way to attribute it after the fact.
    """
    if embedding.ndim != 1:
        raise ValueError(f"Expected a 1-D embedding, got shape {embedding.shape}")
    if embedding.shape[0] == 0:
        # A zero-dim row would be stored with dim=0 and then fail every batch it appears in,
        # because `get_embeddings` requires one consistent dim across the result set.
        raise ValueError("Refusing to store a zero-length embedding")
    if not np.issubdtype(embedding.dtype, np.floating):
        # Checked before the cast, which would otherwise raise TypeError (not the documented
        # ValueError) on a structured dtype, and would silently discard the imaginary part of a
        # complex one.
        raise ValueError(f"Expected a floating-point embedding, got dtype {embedding.dtype}")
    with np.errstate(all="ignore"):
        # Narrowing can overflow to inf or underflow to zero; both are reported below as
        # ValueError. Suppressing every flag here keeps that true even if some caller has set
        # `np.seterr(all="raise")` process-wide, which would otherwise surface as
        # FloatingPointError and break this function's documented contract.
        narrowed = np.ascontiguousarray(embedding, dtype=EMBEDDING_DTYPE)
    if not np.isfinite(narrowed).all():
        # Also catches a float64 magnitude that overflows to inf when narrowed to float32.
        raise ValueError("Embedding contains NaN or infinite values")
    if not narrowed.any():
        # All-zero cannot be an L2-normalized vector. It arrives either from an encoder failure
        # or from float64 components that underflowed to zero when narrowed, and it produces
        # NaN in every cosine similarity it takes part in.
        raise ValueError("Refusing to store an all-zero embedding; it cannot be L2-normalized")
    return narrowed.tobytes()


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
