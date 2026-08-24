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


# Weight of the corpus-mean term in cluster label ranking (see label_clusters).
# 0 is raw cosine (hub phrases win everywhere); 1 is full mean-centering (no
# floor — content-orthogonal phrases can win on corpus anti-alignment, and two
# clusters rank by exact negations of each other). 0.75 demotes hub phrases
# even when clusters share a large common component, while a phrase must still
# beat the winner's 0.25 * raw-cosine residue to displace a strong content
# match.
_CORPUS_CONTRAST_WEIGHT = 0.75


def label_clusters(
    cluster_labels: np.ndarray,
    image_embeddings: np.ndarray,
    vocabulary: list[str],
    vocab_embeddings: np.ndarray,
    top_k: int = 3,
) -> dict[int, dict]:
    """Top-k vocabulary labels for every non-noise cluster.

    Phrases are ranked by ``cos(phrase, centroid) - W * cos(phrase, corpus
    mean)``, not by raw cosine alone. Raw cosine has a hubness problem: generic
    phrases (color words especially) sit close to the mean direction of any
    image collection, and averaging a cluster pulls its centroid toward that
    same mean — so hub phrases win everywhere regardless of what the cluster
    is about. The corpus term cancels that shared component. It is a weighted
    blend rather than full mean-centering because pure contrast has no floor:
    with the centroid's own cosine fully cancelled, a phrase orthogonal to the
    cluster's content can win on corpus anti-alignment alone (with exactly two
    clusters the centered directions are exact negations of each other, so one
    cluster is always labeled by whatever anti-correlates with the other). The
    ``1 - W`` residue of raw cosine keeps the winner tied to what the cluster
    actually contains, and makes degenerate contrast safe by construction: a
    centroid equal to the corpus mean scores every phrase at ``(1 - W) * raw``,
    which ranks identically to raw cosine — no fallback threshold needed.

    Args:
        cluster_labels: (N,) DBSCAN labels, row-aligned with image_embeddings.
        image_embeddings: (N, D) L2-normalized image embeddings.
        vocabulary: Phrases, row-aligned with vocab_embeddings.
        vocab_embeddings: (V, D) L2-normalized phrase embeddings.
        top_k: Number of candidate labels per cluster.

    Returns:
        {cluster_id: {"label": str, "alternates": [str], "score": float}};
        noise (-1) is omitted. "score" is the raw cosine between the chosen
        phrase and the unit centroid, so it stays comparable across clusters.
    """
    results: dict[int, dict] = {}
    clustered = image_embeddings[cluster_labels >= 0]
    corpus_scores: np.ndarray | None = None
    if clustered.shape[0]:
        corpus_mean = clustered.mean(axis=0)
        corpus_norm = np.linalg.norm(corpus_mean)
        if corpus_norm > 0:
            corpus_scores = vocab_embeddings @ (corpus_mean / corpus_norm).astype(vocab_embeddings.dtype)
    for cluster_id in sorted({int(label) for label in cluster_labels if label >= 0}):
        members = image_embeddings[cluster_labels == cluster_id]
        if members.shape[0] == 0:
            continue
        centroid = members.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            continue
        unit_centroid = (centroid / norm).astype(vocab_embeddings.dtype)
        raw_scores = vocab_embeddings @ unit_centroid
        if corpus_scores is None:
            scores = raw_scores
        else:
            scores = raw_scores - _CORPUS_CONTRAST_WEIGHT * corpus_scores
        count = min(top_k, len(vocabulary))
        top = np.argpartition(-scores, count - 1)[:count]
        top = top[np.argsort(-scores[top])]
        results[cluster_id] = {
            "alternates": [vocabulary[index] for index in top[1:]],
            "label": vocabulary[top[0]],
            "score": float(raw_scores[top[0]]),
        }
    return results
