"""Common types and helpers for the semantic image index services."""

from typing import Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

EMBEDDING_DTYPE = np.float32
EMBEDDING_STORAGE_DTYPE = np.dtype("<f2")

MediaKind = Literal["image", "video"]


class IndexedItem(NamedTuple):
    """One indexable gallery item.

    Images and videos are separate namespaces — separate tables, separate access joins,
    separate URLs — so an indexed item is addressed by both. Names are server-assigned and
    unique across both namespaces in practice, but the kind is what tells a client which
    endpoint resolves the name, and the index never has to guess it from an extension.

    A NamedTuple so it can key the worker's pending/failure bookkeeping and be compared and
    sorted without ceremony.
    """

    kind: MediaKind
    name: str


class ImageIndexStatus(BaseModel):
    """Progress of the embedding index for one embedding model.

    Counts cover both media kinds: an indexed gallery is its images plus its videos.
    """

    total: int = Field(description="Number of gallery items (images and videos) eligible for indexing")
    embedded: int = Field(description="Number of eligible items that have an embedding")
    failed: int = Field(
        default=0,
        description="Eligible items that repeatedly failed to embed; excluded from pending so it can drain",
    )

    @property
    def pending(self) -> int:
        # Excluding failures matters: consumers treat pending == 0 as "the
        # index is settled", and a count that can never drain would wedge
        # them (and show an indexing spinner forever).
        return max(0, self.total - self.embedded - self.failed)


class ProjectionRecord(BaseModel):
    """A cached 2D projection of a user's accessible gallery items."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str = Field(description="The user the projection was computed for")
    model_id: str = Field(description="Content hash of the embedding model")
    scope_hash: str = Field(description="Fingerprint of the item set the projection covers")
    params: str = Field(description="JSON of the projection parameters")
    point_count: int = Field(description="Number of projected points")
    items: list[IndexedItem] = Field(description="Indexed items, row-aligned with coords")
    coords: np.ndarray = Field(description="float32 array of shape (point_count, 2)")
    created_at: str = Field(description="When the projection was first computed")
    updated_at: str = Field(description="When the projection was last recomputed")


def embedding_to_blob(embedding: np.ndarray) -> bytes:
    """Serialize a 1-D embedding vector as contiguous little-endian float16 bytes.

    Callers are responsible for L2-normalizing. Narrowing can overflow or underflow, so validate
    the stored representation as well as the input.
    """
    if embedding.ndim != 1:
        raise ValueError(f"Expected a 1-D embedding, got shape {embedding.shape}")
    if embedding.shape[0] == 0:
        raise ValueError("Refusing to store a zero-length embedding")
    if not np.issubdtype(embedding.dtype, np.floating):
        raise ValueError(f"Expected a floating-point embedding, got dtype {embedding.dtype}")
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains NaN or infinite values")

    with np.errstate(all="ignore"):
        narrowed = np.ascontiguousarray(embedding, dtype=EMBEDDING_STORAGE_DTYPE)
    if not np.isfinite(narrowed).all():
        raise ValueError("Embedding contains NaN or infinite values after narrowing to float16")
    if not narrowed.any():
        raise ValueError("Refusing to store an all-zero embedding; it cannot be L2-normalized")
    return narrowed.tobytes()


def blob_to_embedding(blob: bytes, dim: int, encoding: str) -> np.ndarray:
    """Deserialize an explicitly encoded embedding BLOB, validating its dimension and contents.

    Returns a read-only float32 vector. Legacy float32 rows retain their stored values. Float16
    rows decode to float32 and are renormalized after quantization.
    """
    if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)) or dim <= 0:
        raise ValueError(f"Embedding dimension must be a positive integer, got {dim!r}")
    if encoding == "float32":
        dtype = EMBEDDING_DTYPE
    elif encoding == "float16":
        dtype = EMBEDDING_STORAGE_DTYPE
    else:
        raise ValueError(f"Unsupported embedding encoding: {encoding!r}")

    expected = int(dim) * np.dtype(dtype).itemsize
    if len(blob) != expected:
        raise ValueError(f"Embedding blob is {len(blob)} bytes; expected {expected} for {encoding} dimension {dim}")

    decoded = np.frombuffer(blob, dtype=dtype)
    if encoding == "float16":
        decoded = decoded.astype(EMBEDDING_DTYPE)
    if not np.isfinite(decoded).all():
        raise ValueError("Embedding contains NaN or infinite values")
    if not decoded.any():
        raise ValueError("Refusing to read an all-zero embedding; it cannot be L2-normalized")

    if encoding == "float16":
        values64 = decoded.astype(np.float64)
        norm = np.sqrt(np.sum(values64 * values64, dtype=np.float64))
        decoded = (values64 / norm).astype(EMBEDDING_DTYPE)
    decoded.setflags(write=False)
    return decoded


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
