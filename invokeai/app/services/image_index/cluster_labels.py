"""Vocabulary-based automatic cluster labeling, ported from PhotoMapAI.

Each vocabulary phrase is embedded through the index's text encoder with a
prompt-template ensemble (the standard zero-shot CLIP recipe: encode every
template, L2-normalize, mean-pool, re-normalize). A cluster's label is the
vocabulary phrase most similar to the L2-normalized centroid of its members'
image embeddings.
"""

import hashlib
from pathlib import Path

import numpy as np

from invokeai.app.services.image_index.image_index_common import EMBEDDING_DTYPE

# Photo-heavy with a few non-photo templates so AI-generated drawings and
# paintings score reasonably too (PhotoMapAI's ensemble).
PROMPT_TEMPLATES: tuple[str, ...] = (
    "a photo of a {}",
    "a photo of the {}",
    "a picture of {}",
    "an image of {}",
    "a drawing of {}",
    "a painting of {}",
    "{}",
)

_VOCAB_PATH = Path(__file__).parent / "cluster_vocab.txt"

# Bounds on the admin-maintained supplementary vocabulary. The term cap keeps a
# rebuild after an edit to seconds (each term costs len(PROMPT_TEMPLATES)
# encoder strings); the length cap keeps every templated phrase well inside
# CLIP's 77-token context so no term is silently truncated by the tokenizer.
MAX_CUSTOM_VOCAB_TERMS = 500
MAX_CUSTOM_VOCAB_TERM_LENGTH = 64


def normalize_custom_vocab_terms(raw: list[str]) -> list[str]:
    """Normalize user-supplied vocabulary terms for storage and embedding.

    Whitespace is collapsed, terms are lowercased to match the bundled
    vocabulary's convention, empties are dropped, and duplicates are removed
    (first occurrence wins, order preserved).

    Raises:
        ValueError: A term exceeds MAX_CUSTOM_VOCAB_TERM_LENGTH after
            normalization. Raised (naming the term) rather than truncated:
            a silently shortened phrase would embed as something the user
            never wrote.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for entry in raw:
        # Control characters are dropped before the whitespace collapse: they
        # cannot be part of a meaningful label, and NUL specifically would make
        # the fingerprint's NUL-joined phrase stream ambiguous — two different
        # term lists could hash identically and serve each other's cached
        # embeddings.
        cleaned = "".join(ch for ch in entry if ch.isprintable() or ch.isspace())
        term = " ".join(cleaned.split()).lower()
        if not term:
            continue
        if len(term) > MAX_CUSTOM_VOCAB_TERM_LENGTH:
            raise ValueError(f"Vocabulary term is longer than {MAX_CUSTOM_VOCAB_TERM_LENGTH} characters: '{term[:80]}'")
        if term in seen:
            continue
        seen.add(term)
        terms.append(term)
    return terms


def load_vocabulary() -> list[str]:
    """Load the bundled labeling vocabulary (deduplicated, order-preserving)."""
    phrases: list[str] = []
    seen: set[str] = set()
    for line in _VOCAB_PATH.read_text(encoding="utf-8").splitlines():
        phrase = line.strip()
        if not phrase or phrase.startswith("#"):
            continue
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        phrases.append(phrase)
    return phrases


# Bump when the text-embedding recipe changes (tokenization/padding/pooling):
# cached phrase embeddings computed under an old recipe must be discarded.
# v3 also retires caches written before the text encoder was probed at load
# time: a tokenizer-less model dir produced a cache in which every phrase holds
# the same vector, and nothing about the file itself says so.
_EMBEDDING_RECIPE = "pad-per-model-v3"


def vocab_fingerprint(phrases: list[str]) -> str:
    """Content hash of the vocabulary + templates + embedding recipe."""
    digest = hashlib.sha256()
    digest.update(_EMBEDDING_RECIPE.encode("utf-8"))
    digest.update(b"\x00")
    for template in PROMPT_TEMPLATES:
        digest.update(template.encode("utf-8"))
        digest.update(b"\x00")
    for phrase in phrases:
        digest.update(phrase.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def ensemble_phrase_embeddings(embed_texts, phrases: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed phrases through every prompt template and mean-pool.

    Args:
        embed_texts: Callable[[list[str]], np.ndarray] returning (N, D)
            L2-normalized embeddings.
        phrases: Vocabulary phrases.
        batch_size: Phrases per encoder call (each batch encodes
            batch_size * len(PROMPT_TEMPLATES) strings).

    Returns:
        (len(phrases), D) float32, L2-normalized.
    """
    vectors: list[np.ndarray] = []
    for start in range(0, len(phrases), batch_size):
        batch = phrases[start : start + batch_size]
        texts = [template.format(phrase) for phrase in batch for template in PROMPT_TEMPLATES]
        embedded = np.asarray(embed_texts(texts), dtype=EMBEDDING_DTYPE)
        per_phrase = embedded.reshape(len(batch), len(PROMPT_TEMPLATES), -1).mean(axis=1)
        vectors.append(per_phrase)
    matrix = np.concatenate(vectors) if vectors else np.empty((0, 0), dtype=EMBEDDING_DTYPE)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(EMBEDDING_DTYPE)


def label_clusters(
    cluster_labels: np.ndarray,
    image_embeddings: np.ndarray,
    vocabulary: list[str],
    vocab_embeddings: np.ndarray,
    top_k: int = 3,
) -> dict[int, dict]:
    """Top-k vocabulary labels for every non-noise cluster.

    Args:
        cluster_labels: (N,) DBSCAN labels, row-aligned with image_embeddings.
        image_embeddings: (N, D) L2-normalized image embeddings.
        vocabulary: Phrases, row-aligned with vocab_embeddings.
        vocab_embeddings: (V, D) L2-normalized phrase embeddings.
        top_k: Number of candidate labels per cluster.

    Returns:
        {cluster_id: {"label": str, "alternates": [str], "score": float}};
        noise (-1) is omitted.
    """
    results: dict[int, dict] = {}
    for cluster_id in sorted({int(label) for label in cluster_labels if label >= 0}):
        members = image_embeddings[cluster_labels == cluster_id]
        if members.shape[0] == 0:
            continue
        centroid = members.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            continue
        centroid = centroid / norm
        scores = vocab_embeddings @ centroid.astype(vocab_embeddings.dtype)
        count = min(top_k, len(vocabulary))
        top = np.argpartition(-scores, count - 1)[:count]
        top = top[np.argsort(-scores[top])]
        results[cluster_id] = {
            "alternates": [vocabulary[index] for index in top[1:]],
            "label": vocabulary[top[0]],
            "score": float(scores[top[0]]),
        }
    return results
