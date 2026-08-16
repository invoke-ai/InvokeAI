"""Tests for the pure projection/clustering functions (UMAP, DBSCAN, scope hashing)."""

import json

import numpy as np

from invokeai.app.services.image_index.projection import (
    DEFAULT_CLUSTER_EPS,
    adaptive_cluster_eps,
    compute_clusters,
    compute_umap,
    projection_params,
    resolve_cluster_eps,
    scope_hash,
)


def _blobs(n_per_blob: int = 30, dim: int = 64, n_blobs: int = 3) -> np.ndarray:
    """Well-separated gaussian blobs in embedding space, L2-normalized."""
    rng = np.random.default_rng(7)
    centers = rng.standard_normal((n_blobs, dim)) * 10
    vectors = np.concatenate([center + rng.standard_normal((n_per_blob, dim)) * 0.1 for center in centers]).astype(
        np.float32
    )
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_umap_shapes_and_finiteness() -> None:
    embeddings = _blobs()
    coords = compute_umap(embeddings)
    assert coords.shape == (embeddings.shape[0], 2)
    assert coords.dtype == np.float32
    assert np.isfinite(coords).all()


def test_umap_recovers_separated_blobs_as_clusters() -> None:
    coords = compute_umap(_blobs())
    # Normalize coordinate scale so a fixed eps is meaningful.
    span = np.ptp(coords, axis=0).max()
    labels = compute_clusters(coords, eps=span * 0.05, min_samples=10)
    non_noise = {label for label in labels if label != -1}
    assert len(non_noise) >= 2


def test_umap_is_deterministic_with_seed() -> None:
    embeddings = _blobs(n_per_blob=15)
    assert np.array_equal(compute_umap(embeddings, seed=42), compute_umap(embeddings, seed=42))


def test_umap_degenerate_inputs() -> None:
    assert compute_umap(np.empty((0, 8), dtype=np.float32)).shape == (0, 2)
    single = compute_umap(np.ones((1, 8), dtype=np.float32))
    assert single.shape == (1, 2)
    assert np.isfinite(single).all()


def test_umap_tiny_galleries_fall_back_to_pca() -> None:
    # umap-learn cannot fit 2 or 3 points; these must still produce a map
    # rather than an exception (which would loop the projection worker).
    rng = np.random.default_rng(3)
    for n in (2, 3):
        coords = compute_umap(rng.standard_normal((n, 16)).astype(np.float32))
        assert coords.shape == (n, 2)
        assert np.isfinite(coords).all()
    # Distinct points must land at distinct coordinates.
    two = compute_umap(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
    assert not np.allclose(two[0], two[1])


def test_compute_clusters_empty_and_noise() -> None:
    assert compute_clusters(np.empty((0, 2), dtype=np.float32)).shape == (0,)
    # Points far apart with min_samples > 1 are all noise.
    scattered = np.array([[0, 0], [100, 100], [-100, 50]], dtype=np.float32)
    assert set(compute_clusters(scattered, eps=0.5, min_samples=2)) == {-1}


def test_compute_clusters_adaptive_default_scales_to_small_galleries() -> None:
    # Small-N UMAP output is widely spread (span ~30 for a few dozen points),
    # so a fixed eps=0.2 labels everything noise; the adaptive default must
    # recover three well-separated 12-point blobs without an explicit eps.
    rng = np.random.default_rng(7)
    centers = np.array([[0.0, 0.0], [30.0, 0.0], [0.0, 30.0]])
    coords = np.vstack([center + rng.normal(scale=0.4, size=(12, 2)) for center in centers]).astype(np.float32)

    labels = compute_clusters(coords)

    assert set(labels) == {0, 1, 2}
    for blob in range(3):
        assert len(set(labels[blob * 12 : (blob + 1) * 12])) == 1


def test_compute_clusters_adaptive_default_leaves_structureless_maps_unclustered() -> None:
    # A uniform grid has no density structure; the span clamp keeps the
    # adaptive eps below the grid spacing so nothing chains into a cluster.
    xs, ys = np.meshgrid(np.arange(8.0), np.arange(8.0))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)

    assert set(compute_clusters(coords)) == {-1}


def test_adaptive_cluster_eps_degenerate_inputs() -> None:
    assert adaptive_cluster_eps(np.zeros((1, 2), dtype=np.float32)) == DEFAULT_CLUSTER_EPS
    coincident = np.zeros((20, 2), dtype=np.float32)
    assert adaptive_cluster_eps(coincident) == DEFAULT_CLUSTER_EPS
    # Coincident points still cluster under the fallback eps.
    assert set(compute_clusters(coincident)) == {0}


def test_eps_shrinks_to_fit_neighbor_pair_budget(monkeypatch) -> None:
    # A dense blob on a wide map: the adaptive eps (set by the sparse
    # background) would materialize a quadratic number of neighbor pairs
    # inside the blob. The budget must shrink eps rather than let DBSCAN
    # allocate them.
    import invokeai.app.services.image_index.projection as projection

    rng = np.random.default_rng(3)
    dense = rng.normal(scale=0.05, size=(120, 2))
    sparse = rng.uniform(-50.0, 50.0, size=(120, 2))
    coords = np.vstack([dense, sparse]).astype(np.float32)

    unbudgeted = resolve_cluster_eps(coords)
    monkeypatch.setattr(projection, "MAX_NEIGHBOR_PAIRS", 5_000)
    budgeted = resolve_cluster_eps(coords)

    assert budgeted < unbudgeted
    # And the budgeted eps is what the default clustering path actually uses:
    # identical labels when passed back explicitly.
    np.testing.assert_array_equal(compute_clusters(coords), compute_clusters(coords, eps=budgeted))


def test_resolve_cluster_eps_is_idempotent() -> None:
    # /points reports the resolved eps so a later request can reproduce the
    # clustering; resolving an already-resolved eps must not change it.
    rng = np.random.default_rng(5)
    coords = rng.normal(size=(200, 2)).astype(np.float32) * 10
    resolved = resolve_cluster_eps(coords)
    assert resolve_cluster_eps(coords, eps=resolved) == resolved


def test_compute_clusters_clamps_eps_to_span_fraction() -> None:
    # Span ~1: an eps near the span would make every point a neighbor of
    # every other (the sklearn DBSCAN memory blowup); the clamp keeps distant
    # pairs from ever merging.
    coords = np.array([[0, 0], [0.01, 0], [1, 1], [1.01, 1]], dtype=np.float32)
    labels = compute_clusters(coords, eps=2.0, min_samples=2)
    assert labels[0] == labels[1] != labels[2] == labels[3]


def test_compute_clusters_skips_huge_point_sets() -> None:
    from invokeai.app.services.image_index.projection import MAX_CLUSTERED_POINTS

    coords = np.zeros((MAX_CLUSTERED_POINTS + 1, 2), dtype=np.float32)
    labels = compute_clusters(coords, eps=0.2, min_samples=2)
    assert set(labels) == {-1}


def test_scope_hash_is_order_insensitive_and_discriminating() -> None:
    assert scope_hash("m", ["a", "b"]) == scope_hash("m", ["b", "a"])
    assert scope_hash("m", ["a", "b"]) != scope_hash("m", ["a", "c"])
    assert scope_hash("m", ["a", "b"]) != scope_hash("m2", ["a", "b"])
    assert scope_hash("m", []) != scope_hash("m2", [])
    # Concatenation ambiguity: ["ab"] must differ from ["a", "b"].
    assert scope_hash("m", ["ab"]) != scope_hash("m", ["a", "b"])


def test_projection_params_is_stable_json() -> None:
    params = json.loads(projection_params(n_points=100))
    assert params["n_neighbors"] == 15
    assert json.loads(projection_params(n_points=5))["n_neighbors"] == 4
    assert json.loads(projection_params(n_points=1))["n_neighbors"] == 1
    assert projection_params(n_points=100) == projection_params(n_points=100)


def test_cluster_at_eps_clusters_at_exactly_the_eps_it_is_given(monkeypatch) -> None:
    """The router resolves eps once so it can report the effective value, then clusters.

    Routing that value back through compute_clusters would resolve it a SECOND time —
    re-running the k-distance fit and the whole KD-tree budget shrink (measured at ~65%
    of a 50k-point request), and re-applying the 0.01 floor, which re-inflates a
    budget-shrunk eps onto a different shrink grid so the number reported to the client
    is not the number DBSCAN used.
    """
    import invokeai.app.services.image_index.projection as projection

    calls = {"resolve": 0}
    real_resolve = projection.resolve_cluster_eps

    def counting_resolve(*args, **kwargs):
        calls["resolve"] += 1
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(projection, "resolve_cluster_eps", counting_resolve)

    captured: dict[str, float] = {}
    import sklearn.cluster

    real_dbscan = sklearn.cluster.DBSCAN

    def recording_dbscan(*args, **kwargs):
        captured["eps"] = kwargs["eps"]
        return real_dbscan(*args, **kwargs)

    monkeypatch.setattr(sklearn.cluster, "DBSCAN", recording_dbscan)

    coords = _blob_coords()
    resolved = real_resolve(coords, None, 2)

    projection.cluster_at_eps(coords, resolved, 2)
    assert calls["resolve"] == 0, "clustering at an already-resolved eps must not resolve again"
    assert captured["eps"] == resolved, "DBSCAN must run at exactly the eps the caller reports"

    # compute_clusters keeps resolving for callers that pass a raw/None eps.
    projection.compute_clusters(coords, eps=None, min_samples=2)
    assert calls["resolve"] == 1


def _blob_coords() -> np.ndarray:
    rng = np.random.default_rng(3)
    return np.concatenate([rng.standard_normal((20, 2)) * 0.1 + offset for offset in ([0, 0], [5, 5])]).astype(
        np.float32
    )
