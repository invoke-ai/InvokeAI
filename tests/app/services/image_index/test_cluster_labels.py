"""Tests for the vocabulary-based cluster labeling math."""

import numpy as np

from invokeai.app.services.image_index.cluster_labels import (
    PROMPT_TEMPLATES,
    ensemble_phrase_embeddings,
    label_clusters,
    load_vocabulary,
    vocab_fingerprint,
)


def test_load_vocabulary_is_nonempty_and_deduplicated() -> None:
    vocabulary = load_vocabulary()
    assert len(vocabulary) > 500
    assert not any(phrase.startswith("#") for phrase in vocabulary)
    assert len({phrase.casefold() for phrase in vocabulary}) == len(vocabulary)


def test_vocab_fingerprint_changes_with_content() -> None:
    assert vocab_fingerprint(["a", "b"]) != vocab_fingerprint(["a", "c"])
    assert vocab_fingerprint(["a", "b"]) == vocab_fingerprint(["a", "b"])


def test_ensemble_phrase_embeddings_pools_templates_and_normalizes() -> None:
    calls: list[list[str]] = []

    def fake_embed(texts: list[str]) -> np.ndarray:
        calls.append(texts)
        # Deterministic per-string vectors.
        rng = np.random.default_rng(len(texts))
        return rng.standard_normal((len(texts), 8)).astype(np.float32)

    matrix = ensemble_phrase_embeddings(fake_embed, ["cat", "dog", "barn"], batch_size=2)

    assert matrix.shape == (3, 8)
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5)
    # Batch of 2 phrases + batch of 1, each with every template expanded.
    assert [len(texts) for texts in calls] == [2 * len(PROMPT_TEMPLATES), 1 * len(PROMPT_TEMPLATES)]
    assert calls[0][0] == PROMPT_TEMPLATES[0].format("cat")


def test_label_clusters_picks_nearest_phrase_per_cluster() -> None:
    vocab = ["x-things", "y-things", "z-things"]
    vocab_embeddings = np.eye(3, dtype=np.float32)
    # Cluster 0 along x, cluster 1 along y, one noise point along z.
    image_embeddings = np.array([[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0.1, 0.9, 0], [0, 0, 1]], dtype=np.float32)
    cluster_labels = np.array([0, 0, 1, 1, -1])

    labels = label_clusters(cluster_labels, image_embeddings, vocab, vocab_embeddings, top_k=2)

    assert set(labels) == {0, 1}
    assert labels[0]["label"] == "x-things"
    assert labels[1]["label"] == "y-things"
    assert labels[0]["alternates"] == ["y-things"]
    assert 0 < labels[0]["score"] <= 1.0
