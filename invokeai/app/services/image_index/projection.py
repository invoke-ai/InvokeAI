"""Pure projection/clustering math for the image map.

umap-learn (and its numba JIT) and scikit-learn are imported lazily inside
the functions: importing numba costs seconds and these modules must not slow
app startup or break the app if the dependency stack is unhealthy.

Parameters follow PhotoMapAI's tuning: UMAP(n_neighbors=min(15, N-1),
n_components=2, min_dist=0.05, metric="cosine") and DBSCAN(min_samples=10)
over the 2D coordinates. Unlike PhotoMapAI (fixed eps=0.2 plus a user-facing
slider), eps defaults to an adaptive value: UMAP's output scale grows as the
point count shrinks, so any fixed eps that works for a dense thousand-image
map labels a small gallery as all noise.
"""

import hashlib
import json
import warnings
from typing import Optional

import numpy as np

from invokeai.app.services.image_index.image_index_common import EMBEDDING_DTYPE

DEFAULT_CLUSTER_EPS = 0.2
DEFAULT_CLUSTER_MIN_SAMPLES = 10
DEFAULT_UMAP_SEED = 42


def projection_params(seed: int = DEFAULT_UMAP_SEED, n_points: int = 0) -> str:
    """JSON description of the projection parameters, stored with the cache."""
    return json.dumps(
        {
            "metric": "cosine",
            "min_dist": 0.05,
            "n_components": 2,
            "n_neighbors": min(15, n_points - 1) if n_points > 1 else 1,
            "seed": seed,
        },
        sort_keys=True,
    )


def compute_umap(embeddings: np.ndarray, seed: int = DEFAULT_UMAP_SEED) -> np.ndarray:
    """Project an (N, D) embedding matrix to (N, 2) with UMAP.

    Returns an empty (0, 2) array for empty input. A single point maps to the
    origin; 2-3 points fall back to a PCA projection because umap-learn
    cannot fit them (n_neighbors must exceed 1, and spectral initialization
    needs fewer components than points). Raises on UMAP failure — the caller
    decides what a failed fit means for its cache.
    """
    if embeddings.shape[0] == 0:
        return np.empty((0, 2), dtype=EMBEDDING_DTYPE)
    if embeddings.shape[0] == 1:
        return np.zeros((1, 2), dtype=EMBEDDING_DTYPE)
    if embeddings.shape[0] <= 3:
        centered = embeddings.astype(np.float64) - embeddings.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:2].T
        if coords.shape[1] < 2:
            coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])))
        return coords.astype(EMBEDDING_DTYPE)

    from umap import UMAP

    with warnings.catch_warnings():
        # UMAP warns about TBB versions and small-N spectral fallbacks; both
        # are noise at this call site.
        warnings.filterwarnings("ignore")
        n_neighbors = min(15, embeddings.shape[0] - 1)
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=2,
            min_dist=0.05,
            metric="cosine",
            random_state=seed,
        )
        coords = umap_model.fit_transform(embeddings)
    return np.asarray(coords, dtype=EMBEDDING_DTYPE)


# Above this many points, per-request DBSCAN is skipped (everything is served
# as unclustered) — sklearn DBSCAN materializes all radius neighborhoods, and
# the memory cost grows with the pair count.
MAX_CLUSTERED_POINTS = 50_000

# eps is clamped to this fraction of the projection's coordinate span. UMAP's
# output scale is data-dependent, so an absolute eps close to the span makes
# every point a neighbor of every other (O(N^2) memory in sklearn's DBSCAN).
MAX_EPS_SPAN_FRACTION = 0.05


def adaptive_cluster_eps(coords: np.ndarray, min_samples: int = DEFAULT_CLUSTER_MIN_SAMPLES) -> float:
    """Median k-distance of the coordinates: each point's distance to its
    min_samples-th nearest neighbor, aggregated with the median.

    This is the standard DBSCAN eps heuristic: a point whose min_samples-th
    neighbor is within eps is a core point, so the median k-distance makes
    roughly half the points core points — clusters form wherever density is
    above the map's typical density, at any gallery size or UMAP scale.
    """
    n_points = coords.shape[0]
    k = min(min_samples, n_points - 1)
    if k < 1:
        return DEFAULT_CLUSTER_EPS

    from sklearn.neighbors import NearestNeighbors

    # k + 1 neighbors because each point's nearest neighbor is itself.
    distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)
    eps = float(np.median(distances[:, -1]))
    # Coincident points give a zero k-distance; zero eps clusters nothing.
    return eps if eps > 0 else DEFAULT_CLUSTER_EPS


# sklearn's DBSCAN materializes every point's radius neighborhood as int64
# index arrays, so its memory is proportional to the neighbor-pair count.
# 50M pairs is ~400MB — comparable to the worst case the old fixed eps=0.2
# default could produce, and far below OOM territory.
MAX_NEIGHBOR_PAIRS = 50_000_000

# Hard lower bound for the budget shrink. A fully coincident map can never fit
# the pair budget at any positive eps, so the shrink needs a floor to
# terminate; compute_clusters skips clustering when the budget is still
# unmet here rather than letting DBSCAN allocate.
MIN_BUDGETED_EPS = 1e-6


def _shrink_eps_to_pair_budget(coords: np.ndarray, eps: float) -> float:
    """Shrink eps until DBSCAN's neighbor-pair count fits MAX_NEIGHBOR_PAIRS.

    The span clamp alone cannot bound memory: a dense blob concentrates most
    pairs in a small region, so a modest eps on a wide map can still
    materialize billions of pairs. Counting pairs with a KD-tree is cheap.
    """
    if coords.shape[0] < 2:
        return eps

    from sklearn.neighbors import KDTree

    tree = KDTree(coords)
    # Iterate until the budget is met rather than a fixed count: 12 rounds of
    # 0.7 only covers a 71x reduction, and a tight blob can need far more.
    # Bounded below by MIN_BUDGETED_EPS so a fully coincident map terminates.
    while eps > MIN_BUDGETED_EPS:
        pairs = int(tree.query_radius(coords, r=eps, count_only=True).sum())
        if pairs <= MAX_NEIGHBOR_PAIRS:
            return eps
        eps *= 0.7
    return eps


def resolve_cluster_eps(
    coords: np.ndarray,
    eps: Optional[float] = None,
    min_samples: int = DEFAULT_CLUSTER_MIN_SAMPLES,
) -> float:
    """The eps compute_clusters will actually use for these coordinates.

    Resolution order: adaptive default when eps is None, then the span clamp,
    then the neighbor-pair budget. Exposed so callers (the /points endpoint)
    can report the effective eps and other callers can reuse it verbatim.
    """
    if eps is None:
        # Cap at the API's own upper bound (le=2.0) so a reported adaptive
        # eps can always be passed back explicitly. Values that large only
        # arise on sparse structureless maps, where a smaller eps just means
        # more (correct) noise.
        eps = min(adaptive_cluster_eps(coords, min_samples), 2.0)
    span = float(np.ptp(coords, axis=0).max()) if coords.shape[0] > 1 else 0.0
    if span > 0:
        eps = min(eps, span * MAX_EPS_SPAN_FRACTION)
    # Floor at the API's lower bound (ge=0.01) for pass-back validity, but do
    # it BEFORE the budget shrink, never after: applied afterwards it silently
    # re-inflated eps past the neighbor-pair budget the shrink had just
    # computed, which is how a near-coincident map reached 400M pairs against
    # a 50M budget (~4GB). The budget bounds memory and therefore wins; an eps
    # below the API's floor is a cosmetic pass-back wart, an OOM is not.
    return _shrink_eps_to_pair_budget(coords, max(eps, 0.01))


def compute_clusters(
    coords: np.ndarray,
    eps: Optional[float] = None,
    min_samples: int = DEFAULT_CLUSTER_MIN_SAMPLES,
) -> np.ndarray:
    """DBSCAN-cluster 2D coordinates. Label -1 is noise ("unclustered").

    Cheap enough to run per request, which is what makes the eps parameter
    live-adjustable without recomputing the UMAP. When eps is None it is
    derived from the data with adaptive_cluster_eps; see resolve_cluster_eps
    for the clamps applied either way. Clustering is skipped entirely above
    MAX_CLUSTERED_POINTS.
    """
    if coords.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    if coords.shape[0] > MAX_CLUSTERED_POINTS:
        return np.full((coords.shape[0],), -1, dtype=np.int64)

    return cluster_at_eps(coords, resolve_cluster_eps(coords, eps, min_samples), min_samples)


def cluster_at_eps(
    coords: np.ndarray,
    eps: float,
    min_samples: int = DEFAULT_CLUSTER_MIN_SAMPLES,
) -> np.ndarray:
    """DBSCAN at an eps `resolve_cluster_eps` has ALREADY produced.

    Split out so a caller that needs to report the effective eps can resolve it
    once and cluster at exactly that value. Passing a resolved eps back into
    compute_clusters instead would re-resolve it — and since resolution floors
    at 0.01 before applying the neighbour-pair budget, a budget-shrunk eps comes
    back out different, so the reported value would not be the one used.
    """
    if coords.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    if coords.shape[0] > MAX_CLUSTERED_POINTS:
        return np.full((coords.shape[0],), -1, dtype=np.int64)

    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import KDTree

    if coords.shape[0] > 1:
        # The shrink bottoms out at MIN_BUDGETED_EPS, which a fully coincident
        # map cannot satisfy at any positive radius. Skip clustering rather
        # than hand DBSCAN a neighborhood it would materialize into GBs — same
        # response shape as the MAX_CLUSTERED_POINTS skip above.
        pairs = int(KDTree(coords).query_radius(coords, r=eps, count_only=True).sum())
        if pairs > MAX_NEIGHBOR_PAIRS:
            return np.full((coords.shape[0],), -1, dtype=np.int64)

    return DBSCAN(eps=eps, min_samples=min_samples).fit(coords).labels_


def scope_hash(model_id: str, image_names: list[str]) -> str:
    """Fingerprint of an accessible-image set under one model.

    Order-insensitive: the caller may pass names in any order. Comparing the
    stored hash against a freshly derived one detects staleness from any
    cause — new/deleted images, board visibility changes, shares.
    """
    digest = hashlib.sha256()
    digest.update(model_id.encode("utf-8"))
    for name in sorted(image_names):
        digest.update(b"\x00")
        digest.update(name.encode("utf-8"))
    return digest.hexdigest()
