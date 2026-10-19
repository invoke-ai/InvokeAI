"""Tests for the pure projection/clustering functions (UMAP, DBSCAN, scope hashing)."""

import json

import numpy as np

from invokeai.app.services.image_index.image_index_common import IndexedItem
from invokeai.app.services.image_index.projection import (
    CANDIDATE_QUANTILES,
    DEFAULT_CLUSTER_EPS,
    DEFAULT_CLUSTER_MIN_SAMPLES,
    MAX_CLUSTERED_POINTS,
    MAX_EPS_SPAN_FRACTION,
    MAX_TOP_CLUSTER_SHARE,
    adaptive_cluster_eps,
    cluster_with_diagnostics,
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

    unbudgeted = resolve_cluster_eps(coords).resolved_eps
    monkeypatch.setattr(projection, "MAX_NEIGHBOR_PAIRS", 2_000)
    budgeted = resolve_cluster_eps(coords).resolved_eps

    assert budgeted < unbudgeted
    # And the budgeted eps is what the default clustering path actually uses:
    # identical labels when passed back explicitly.
    np.testing.assert_array_equal(compute_clusters(coords), compute_clusters(coords, eps=budgeted))


def test_resolve_cluster_eps_is_idempotent() -> None:
    # /points reports the resolved eps so a later request can reproduce the
    # clustering; resolving an already-resolved eps must not change it.
    rng = np.random.default_rng(5)
    coords = rng.normal(size=(200, 2)).astype(np.float32) * 10
    resolved = resolve_cluster_eps(coords).resolved_eps
    assert resolve_cluster_eps(coords, eps=resolved).resolved_eps == resolved


def test_the_quantile_scan_leaves_most_of_a_structured_map_clustered() -> None:
    """The reason the scan exists: a median k-distance leaves ~30% noise.

    Roughly half the points are non-core under the median by construction, so
    a map with real structure still reads as mostly unclustered. The scan
    climbs past the median instead, stopping before one cluster swallows it.
    """
    rng = np.random.default_rng(1)
    centers = rng.uniform(-40.0, 40.0, size=(60, 2))
    coords = np.concatenate([c + rng.normal(scale=0.35, size=(250, 2)) for c in centers]).astype(np.float32)

    _, scanned = cluster_with_diagnostics(coords)
    median_eps = float(np.percentile(_k_distances_for(coords, DEFAULT_CLUSTER_MIN_SAMPLES), 50))
    _, at_median = cluster_with_diagnostics(coords, eps=median_eps)

    assert scanned.unclustered / scanned.n_points < 0.10
    assert at_median.unclustered > scanned.unclustered * 3, "the median is the baseline the scan improves on"
    # And it stops short of a blob: no cluster owns the map.
    assert scanned.largest_cluster / scanned.n_points <= MAX_TOP_CLUSTER_SHARE


def test_the_quantile_scan_refuses_to_let_one_cluster_swallow_the_map() -> None:
    # Two dense blobs with nothing between them: past a point every candidate
    # merges them, and the scan has to stop before that.
    rng = np.random.default_rng(2)
    coords = np.concatenate([rng.normal(scale=0.4, size=(400, 2)) + c for c in ([0.0, 0.0], [1.6, 0.0])]).astype(
        np.float32
    )

    _, diagnostics = cluster_with_diagnostics(coords)

    assert diagnostics.cluster_count >= 2, "the two blobs must not be merged into one"
    assert diagnostics.largest_cluster / diagnostics.n_points <= MAX_TOP_CLUSTER_SHARE + 0.25


def test_the_scan_stops_at_the_first_candidate_that_blobs() -> None:
    """It walks the candidates; it must not search them.

    The obvious optimisation — the passing candidates are a prefix, so binary
    search the boundary — is wrong. Raising eps grows every core-point
    cluster, but a border point can be claimed by a different cluster that
    has just come into reach, so the largest cluster can LOSE points and the
    top share is not monotone. A search reads a pass after a failure and
    keeps climbing, choosing a looser eps than this rule on some maps only.
    """
    coords = _non_monotone_share_map()
    distances = _k_distances_for(coords, DEFAULT_CLUSTER_MIN_SAMPLES)
    candidates = [float(np.percentile(distances, q)) for q in CANDIDATE_QUANTILES]
    shares = [_top_cluster_share_for(coords, eps) for eps in candidates]

    # The fixture earns its keep only while it is genuinely non-monotone.
    assert any(later < earlier for earlier, later in zip(shares, shares[1:], strict=False)), shares
    assert shares[0] <= MAX_TOP_CLUSTER_SHARE < shares[1], shares
    assert shares[2] <= MAX_TOP_CLUSTER_SHARE, "a later candidate passes again, which is the trap"

    assert adaptive_cluster_eps(coords) == candidates[0]


def _k_distances_for(coords: np.ndarray, min_samples: int) -> np.ndarray:
    from invokeai.app.services.image_index import projection

    distances = projection._k_distances(coords, min_samples)
    assert distances is not None

    return distances


def _top_cluster_share_for(coords: np.ndarray, eps: float) -> float:
    from invokeai.app.services.image_index import projection

    return projection._top_cluster_share(coords, eps, DEFAULT_CLUSTER_MIN_SAMPLES)


def _non_monotone_share_map() -> np.ndarray:
    """A map whose largest-cluster share dips as eps grows.

    One long backbone with five border points hanging off it, plus five
    "thief" chains placed so that the next candidate up reaches those border
    points first and takes them off the backbone. The rings are filler that
    shapes the k-distance distribution so the candidates land where they must.
    """

    def ring(centre_x: float, count: int, diameter: float) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        return np.stack([centre_x + (diameter / 2) * np.cos(angles), (diameter / 2) * np.sin(angles)], axis=1)

    height, perp, thief_gap, thief_step = (
        0.8556190490722656,
        0.4038790583610535,
        1.0640721321105957,
        0.11982051849365234,
    )
    backbone = np.stack([np.arange(96) * 0.1, np.zeros(96)], axis=1)
    anchors, borders, thieves = [], [], []
    for index in range(5):
        x = (95 * 0.1) * (index + 1) / 6
        anchors.append([x, perp])
        borders.append([x, perp + height])
        thieves.append(np.stack([np.full(12, x), perp + height + thief_gap + np.arange(12) * thief_step], axis=1))

    filler, centre = [], 500.0
    for _ in range(4):
        filler.append(ring(centre, 12, 0.04))
        centre += 60.0
    for diameter, repeats in ((1.0, 4), (1.15, 2), (1.6, 4), (2.0, 4), (2.6, 4)):
        for _ in range(repeats):
            filler.append(ring(centre, 11, diameter))
            centre += 60.0

    return np.vstack(thieves + filler + [backbone, np.array(anchors), np.array(borders)]).astype(np.float32)


def test_a_derived_eps_is_clamped_to_the_span_fraction() -> None:
    # 20 scattered points: the median k-distance is half the map wide, which
    # says there is no density structure to find. Clustering at that radius
    # would merge everything, so the heuristic's own answer is bounded.
    rng = np.random.default_rng(4)
    coords = rng.uniform(0.0, 1.0, size=(20, 2)).astype(np.float32)
    resolution = resolve_cluster_eps(coords, None, 10)

    assert resolution.adaptive_eps is not None
    assert resolution.adaptive_eps > resolution.coord_span * MAX_EPS_SPAN_FRACTION
    assert resolution.span_clamped_eps == resolution.coord_span * MAX_EPS_SPAN_FRACTION


def test_a_requested_eps_is_not_clamped_to_the_span_fraction() -> None:
    """The clustering-strength control has to mean what it says.

    Clamping a number the user typed would silently retune it — on a small
    map (span 2, clamp 0.1) that overrides most of the range the control
    offers, and the value reported back would not be the one they chose. The
    pair budget still applies to both, because that one is a memory bound.
    """
    rng = np.random.default_rng(4)
    coords = rng.uniform(0.0, 1.0, size=(20, 2)).astype(np.float32)
    resolution = resolve_cluster_eps(coords, 0.5, 10)

    assert 0.5 > resolution.coord_span * MAX_EPS_SPAN_FRACTION, "the clamp would bind if it were applied"
    assert resolution.span_clamped_eps == 0.5
    assert resolution.resolved_eps == 0.5
    # And it reaches DBSCAN: at 0.5 on a span-0.88 map everything is one cluster.
    assert set(compute_clusters(coords, eps=0.5, min_samples=2)) == {0}


def test_compute_clusters_skips_huge_point_sets() -> None:
    coords = np.zeros((MAX_CLUSTERED_POINTS + 1, 2), dtype=np.float32)
    labels = compute_clusters(coords, eps=0.2, min_samples=2)
    assert set(labels) == {-1}


def test_scope_hash_is_order_insensitive_and_discriminating() -> None:
    a, b, c = IndexedItem("image", "a"), IndexedItem("image", "b"), IndexedItem("image", "c")
    assert scope_hash("m", [a, b]) == scope_hash("m", [b, a])
    assert scope_hash("m", [a, b]) != scope_hash("m", [a, c])
    assert scope_hash("m", [a, b]) != scope_hash("m2", [a, b])
    # A video joining the set changes it, so a cached projection computed before it existed
    # is detected as stale.
    assert scope_hash("m", [a, b]) != scope_hash("m", [a, b, IndexedItem("video", "clip.mp4")])
    # The kind itself is NOT hashed: names identify items on their own, and hashing the kind
    # would give every pre-video projection a new hash, costing every user a recomputed fit on
    # the first load after the upgrade.
    assert scope_hash("m", [IndexedItem("image", "x")]) == scope_hash("m", [IndexedItem("video", "x")])
    assert scope_hash("m", []) != scope_hash("m2", [])
    # Concatenation ambiguity: ["ab"] must differ from ["a", "b"].
    assert scope_hash("m", [IndexedItem("image", "ab")]) != scope_hash("m", [a, b])


def test_projection_params_is_stable_json() -> None:
    params = json.loads(projection_params(n_points=100))
    assert params["n_neighbors"] == 15
    assert json.loads(projection_params(n_points=5))["n_neighbors"] == 4
    assert json.loads(projection_params(n_points=1))["n_neighbors"] == 1
    assert projection_params(n_points=100) == projection_params(n_points=100)


def test_clustering_reports_the_eps_dbscan_actually_ran_at(monkeypatch) -> None:
    """One clustering resolves eps once, and reports the value it used.

    The endpoints report `cluster_eps` out of this call so a later request can
    reproduce the clustering. Resolving a second time — which routing the
    reported value back through the entry point would do — re-runs the
    k-distance fit and the whole KD-tree budget shrink (measured at ~65% of a
    50k-point request), and re-applies the 0.01 floor, which re-inflates a
    budget-shrunk eps onto a different shrink grid: the number reported would
    not be the number DBSCAN used.
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
    _, diagnostics = cluster_with_diagnostics(coords, eps=None, min_samples=2)

    assert calls["resolve"] == 1, "one clustering must resolve eps exactly once"
    assert captured["eps"] == diagnostics.resolved_eps, "DBSCAN must run at exactly the eps reported"


def test_diagnostics_count_the_clusters_that_were_found() -> None:
    coords = _blob_coords()
    labels, diagnostics = cluster_with_diagnostics(coords, min_samples=2)

    sizes = sorted(int((labels == cluster).sum()) for cluster in set(labels) if cluster >= 0)
    assert sizes, "the blob fixture must cluster, or this asserts nothing"
    assert diagnostics.cluster_count == len(sizes)
    assert diagnostics.largest_cluster == sizes[-1]
    assert diagnostics.smallest_cluster == sizes[0]
    assert diagnostics.unclustered == int((labels < 0).sum())
    assert diagnostics.skipped is None
    assert diagnostics.eps is not None and diagnostics.eps.adaptive_eps is not None


def test_diagnostics_name_the_point_cap_that_left_a_map_unclustered() -> None:
    """The gate a 170k-image gallery hits: every point noise, nothing in the response saying why."""
    coords = np.zeros((MAX_CLUSTERED_POINTS + 1, 2), dtype=np.float32)
    labels, diagnostics = cluster_with_diagnostics(coords, eps=0.2, min_samples=2)

    assert set(labels) == {-1}
    assert diagnostics.skipped is not None and "MAX_CLUSTERED_POINTS" in diagnostics.skipped
    assert diagnostics.unclustered == coords.shape[0]
    assert diagnostics.cluster_count == 0
    # Nothing was resolved, so nothing about eps may be reported as if it had been.
    assert diagnostics.eps is None and diagnostics.resolved_eps is None
    assert "MAX_CLUSTERED_POINTS" in diagnostics.summary()


def test_diagnostics_name_the_pair_budget_when_the_shrink_bottoms_out(monkeypatch) -> None:
    # Coincident points are every other point's neighbor at any radius, so no
    # eps meets the budget and the shrink stops at its floor.
    import invokeai.app.services.image_index.projection as projection

    monkeypatch.setattr(projection, "MAX_NEIGHBOR_PAIRS", 100)
    coords = np.zeros((100, 2), dtype=np.float32)
    labels, diagnostics = cluster_with_diagnostics(coords, min_samples=2)

    assert set(labels) == {-1}
    assert diagnostics.skipped is not None and "MAX_NEIGHBOR_PAIRS" in diagnostics.skipped
    # eps resolution DID run here, so the shrink it bottomed out on is reported.
    assert diagnostics.eps is not None
    assert diagnostics.eps.neighbor_pairs == coords.shape[0] ** 2
    assert diagnostics.resolved_eps is not None and diagnostics.resolved_eps <= projection.MIN_BUDGETED_EPS


def test_diagnostics_report_each_link_of_the_eps_chain() -> None:
    """A wrong chain is worse than none: it misdirects the next investigation."""
    # Span 30, so the clamp (5% of span = 1.5) binds an eps of 5.0.
    coords = np.array([[0.0, 0.0], [30.0, 30.0], [10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    resolution = resolve_cluster_eps(coords, eps=5.0, min_samples=2)

    assert resolution.requested_eps == 5.0
    assert resolution.adaptive_eps is None, "an explicit eps must not be reported as an adaptive one"
    assert resolution.coord_span == 30.0
    assert resolution.span_clamped_eps == 5.0, "a requested eps passes the clamp untouched"
    assert resolution.floored_eps == 5.0, "5.0 is above the floor; the floor must not move it"
    assert resolution.resolved_eps == 5.0, "4 pairs is under any budget"


def test_diagnostics_separate_the_span_clamp_from_the_eps_floor() -> None:
    """Both can produce a too-tight eps, and the log has to say which one did."""
    # A 0.02-wide map: the clamp takes the derived eps to 0.0049, and the 0.01
    # floor then lifts it back. One number could not distinguish the two.
    rng = np.random.default_rng(4)
    coords = rng.uniform(0.0, 0.02, size=(30, 2)).astype(np.float32)
    resolution = resolve_cluster_eps(coords, None, 10)

    assert resolution.adaptive_eps is not None
    assert resolution.span_clamped_eps < resolution.adaptive_eps, "the clamp bound the derived value"
    assert resolution.span_clamped_eps < 0.01, "and took it below the floor"
    assert resolution.floored_eps == 0.01, "and the floor put it back"
    assert "floored_eps=0.01" in cluster_with_diagnostics(coords, min_samples=10)[1].signature()


def test_the_pair_budget_shrink_can_succeed_short_of_its_floor(monkeypatch) -> None:
    """The shrink's successful exit — clusters at a reduced eps, rather than giving up."""
    import invokeai.app.services.image_index.projection as projection

    rng = np.random.default_rng(11)
    coords = np.concatenate([rng.normal(scale=0.05, size=(200, 2)) + c for c in ([0.0, 0.0], [10.0, 10.0])]).astype(
        np.float32
    )
    monkeypatch.setattr(projection, "MAX_NEIGHBOR_PAIRS", 20_000)
    labels, diagnostics = cluster_with_diagnostics(coords, eps=2.0, min_samples=2)

    assert diagnostics.skipped is None, "the shrink met the budget, so DBSCAN must have run"
    assert diagnostics.eps is not None
    assert diagnostics.resolved_eps < diagnostics.eps.floored_eps, "the budget must have shrunk eps"
    assert diagnostics.resolved_eps > projection.MIN_BUDGETED_EPS, "it must have stopped short of the floor"
    assert diagnostics.eps.neighbor_pairs <= 20_000
    assert diagnostics.cluster_count > 0, "a met budget must still cluster, not degrade to noise"
    # The reported histogram describes the labels that were actually served.
    sizes = sorted(int((labels == cluster).sum()) for cluster in set(labels) if cluster >= 0)
    assert [diagnostics.cluster_count, diagnostics.largest_cluster, diagnostics.smallest_cluster] == [
        len(sizes),
        sizes[-1],
        sizes[0],
    ]


def test_diagnostics_signature_is_stable_across_identical_runs() -> None:
    """The endpoint logs one line per distinct outcome, so timing must stay out of the key."""
    coords = _blob_coords()
    first = cluster_with_diagnostics(coords, min_samples=2)[1]
    second = cluster_with_diagnostics(coords, min_samples=2)[1]

    assert first.signature() == second.signature()
    assert first.signature() in first.summary()
    assert cluster_with_diagnostics(coords, min_samples=3)[1].signature() != first.signature()


def _blob_coords() -> np.ndarray:
    rng = np.random.default_rng(3)
    return np.concatenate([rng.standard_normal((20, 2)) * 0.1 + offset for offset in ([0, 0], [5, 5])]).astype(
        np.float32
    )
