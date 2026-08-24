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
    # The corpus-contrast term demotes the other cluster's phrase below the
    # neutral one.
    assert labels[0]["alternates"] == ["z-things"]
    assert 0 < labels[0]["score"] <= 1.0


def _unit(v: list[float]) -> np.ndarray:
    arr = np.array(v, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_label_clusters_demotes_hub_phrases() -> None:
    # A "hub" phrase aligned with the shared direction of every cluster wins
    # raw cosine everywhere; the corpus-contrast term must surface the
    # distinctive axis.
    vocab = ["hub-phrase", "x-things", "y-things"]
    vocab_embeddings = np.stack([_unit([1, 1, 1]), _unit([0, 1, 0]), _unit([0, 0, 1])])
    # Two clusters sharing a dominant common component, differing on axes 1/2.
    image_embeddings = np.stack([_unit([1, 0.3, 0]), _unit([1, 0.4, 0]), _unit([1, 0, 0.3]), _unit([1, 0, 0.4])])
    cluster_labels = np.array([0, 0, 1, 1])

    # Sanity: without the contrast term, the hub phrase beats both distinctive
    # phrases.
    raw = vocab_embeddings @ _unit(list(image_embeddings[:2].mean(axis=0)))
    assert raw[0] > raw[1]

    labels = label_clusters(cluster_labels, image_embeddings, vocab, vocab_embeddings, top_k=1)

    assert labels[0]["label"] == "x-things"
    assert labels[1]["label"] == "y-things"


def test_label_clusters_single_cluster_ranks_by_raw_cosine() -> None:
    # With one cluster the centroid equals the corpus mean, so the contrast
    # term subtracts the same multiple of every phrase's raw score — the
    # ranking must degrade to plain raw cosine, not to amplified noise.
    vocab = ["x-things", "y-things"]
    vocab_embeddings = np.eye(2, dtype=np.float32)
    image_embeddings = np.array([[1, 0], [0.9, 0.1]], dtype=np.float32)
    cluster_labels = np.array([0, 0])

    labels = label_clusters(cluster_labels, image_embeddings, vocab, vocab_embeddings, top_k=1)

    assert labels[0]["label"] == "x-things"
    assert 0 < labels[0]["score"] <= 1.0


def test_label_clusters_keeps_shared_content_label_for_similar_clusters() -> None:
    # Two clusters of near-identical content: pure mean-centering would rank
    # each by the exact negation of the other (junk phrases win); the blended
    # score must keep the phrase that describes both.
    vocab = ["cat", "junk-up", "junk-down"]
    vocab_embeddings = np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, -1, 0])])
    image_embeddings = np.stack([_unit([1, 0.05, 0]), _unit([1, 0.06, 0]), _unit([1, -0.05, 0]), _unit([1, -0.06, 0])])
    cluster_labels = np.array([0, 0, 1, 1])

    labels = label_clusters(cluster_labels, image_embeddings, vocab, vocab_embeddings, top_k=1)

    assert labels[0]["label"] == "cat"
    assert labels[1]["label"] == "cat"


def test_label_clusters_dominant_cluster_keeps_own_content_label() -> None:
    # A cluster holding most of the corpus sits nearly on the corpus mean, so
    # pure centering would label it by whatever anti-correlates with the small
    # clusters; the blend must keep its own content phrase on top.
    vocab = ["dog", "anti-minority"]
    vocab_embeddings = np.stack([_unit([1, 0, 0]), _unit([0.5, -0.6, -0.6])])
    dominant = [_unit([1, 0.05, 0]) if i % 2 else _unit([1, -0.05, 0]) for i in range(8)]
    image_embeddings = np.stack(dominant + [_unit([0.3, 1, 0]), _unit([0.3, 0, 1])])
    cluster_labels = np.array([0] * 8 + [1, 2])

    labels = label_clusters(cluster_labels, image_embeddings, vocab, vocab_embeddings, top_k=1)

    assert labels[0]["label"] == "dog"
