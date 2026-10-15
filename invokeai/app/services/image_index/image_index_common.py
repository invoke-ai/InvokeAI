"""Common types and helpers for the semantic image index services."""

from collections.abc import Sequence
from typing import Literal, NamedTuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

EMBEDDING_DTYPE = np.float32
EMBEDDING_STORAGE_DTYPE = np.dtype("<f2")
_ENCODING_DTYPES = {"float32": np.dtype(EMBEDDING_DTYPE), "float16": EMBEDDING_STORAGE_DTYPE}
_DECODE_CHUNK_ROWS = 4096

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


def blobs_to_embeddings(blobs: Sequence[bytes], encodings: Sequence[str], dim: int) -> np.ndarray:
    """Deserialize explicitly encoded embedding BLOBs into one float32 matrix, validating every row.

    Rows keep the order of `blobs`. Legacy float32 rows retain their stored values. Float16 rows
    are renormalized in float64 after quantization, then narrowed to float32.
    """
    if isinstance(dim, (bool, np.bool_)) or not isinstance(dim, (int, np.integer)) or dim <= 0:
        raise ValueError(f"Embedding dimension must be a positive integer, got {dim!r}")
    dim = int(dim)
    if len(blobs) != len(encodings):
        raise ValueError(f"Got {len(blobs)} embedding blobs but {len(encodings)} encodings")
    unsupported = set(encodings) - _ENCODING_DTYPES.keys()
    if unsupported:
        raise ValueError(f"Unsupported embedding encoding: {min(unsupported)!r}")

    # Each encoding is decoded as one buffer, then validated and renormalized in row chunks that
    # bound the float64 temporaries: per-row decoding dominated reads of a full gallery, which
    # happen whenever the accessible item set changes.
    matrix = np.empty((len(blobs), dim), dtype=EMBEDDING_DTYPE)
    for encoding, dtype in _ENCODING_DTYPES.items():
        rows = [index for index, row_encoding in enumerate(encodings) if row_encoding == encoding]
        if not rows:
            continue
        expected = dim * dtype.itemsize
        for index in rows:
            if len(blobs[index]) != expected:
                raise ValueError(
                    f"Embedding blob is {len(blobs[index])} bytes; expected {expected} for {encoding} dimension {dim}"
                )
        decoded = np.frombuffer(b"".join(blobs[index] for index in rows), dtype=dtype).reshape(len(rows), dim)
        for start in range(0, len(rows), _DECODE_CHUNK_ROWS):
            stop = start + _DECODE_CHUNK_ROWS
            chunk = decoded[start:stop]
            if encoding == "float16":
                # Exact widening, done before validating because numpy's float16 kernels are slow.
                chunk = chunk.astype(np.float64)
            if not np.isfinite(chunk).all():
                raise ValueError("Embedding contains NaN or infinite values")
            if not chunk.any(axis=1).all():
                raise ValueError("Refusing to read an all-zero embedding; it cannot be L2-normalized")
            if encoding == "float16":
                chunk /= np.sqrt(np.einsum("ij,ij->i", chunk, chunk))[:, None]
            if len(rows) == len(blobs):
                matrix[start:stop] = chunk
            else:
                matrix[rows[start:stop]] = chunk
    return matrix


def blob_to_embedding(blob: bytes, dim: int, encoding: str) -> np.ndarray:
    """Deserialize one explicitly encoded embedding BLOB; see `blobs_to_embeddings`.

    Returns a read-only float32 vector.
    """
    vector = blobs_to_embeddings([blob], [encoding], dim)[0]
    vector.setflags(write=False)
    return vector


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
