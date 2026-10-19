"""Pure projection/clustering math for the image map.

umap-learn (and its numba JIT) and scikit-learn are imported lazily inside
the functions: importing numba costs seconds and these modules must not slow
app startup or break the app if the dependency stack is unhealthy.

Parameters follow PhotoMapAI's tuning: UMAP(n_neighbors=min(15, N-1),
n_components=2, min_dist=0.05, metric="cosine") and DBSCAN(min_samples=10)
over the 2D coordinates. eps defaults to an adaptive value derived from the
coordinates themselves — UMAP's output scale grows as the point count
shrinks, so any fixed eps that works for a dense thousand-image map labels a
small gallery as all noise — and the clustering-strength control lets a user
override it per map.
"""

import hashlib
import json
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from invokeai.app.services.image_index.image_index_common import EMBEDDING_DTYPE, IndexedItem

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
#
# Raised from 50k to cover a reported 170k-item gallery, which the old value
# left entirely unclustered with nothing in the response saying so. The
# neighbor-pair budget below, not this number, is what actually bounds
# DBSCAN's memory; this bounds the per-request CPU and the size of the label
# arrays the endpoint caches. Both grow with it — see the cluster cache in
# the image_map router — so it is set from measurement, not headroom.
MAX_CLUSTERED_POINTS = 300_000

# A DERIVED eps is clamped to this fraction of the projection's coordinate
# span (a requested one is not — see resolve_cluster_eps). UMAP's output scale
# is data-dependent, so an eps close to the span makes every point a neighbor
# of every other.
#
# 0.05 here was correct for a median k-distance, which sits low in the
# distribution. The quantile scan deliberately climbs past the median, and at
# 0.05 the clamp overrode the scan's answer outright on a map of a few
# well-separated blobs. PhotoMapAI records making exactly this mistake by
# copying the 0.05 from here. The pair budget below, not this, is what bounds
# memory; this is only a sanity bound on an automatically chosen number, so it
# is set where it catches absurdity and nothing else.
MAX_EPS_SPAN_FRACTION = 0.25


# Quantiles of the k-distance distribution, ascending, that the scan walks.
# Coarse on purpose: each step costs a DBSCAN fit, and near the knee the
# resulting eps values are within a few percent of each other anyway.
CANDIDATE_QUANTILES: tuple[int, ...] = (40, 50, 60, 70, 80, 90)

# A cluster holding more than this share of the gallery is a blob, not a
# cluster: it is what the map looks like just before every point merges into
# one component. Ported from PhotoMapAI, where this was measured against real
# albums and the last candidate below it consistently landed on the value a
# human had tuned by hand.
MAX_TOP_CLUSTER_SHARE = 0.25


def _k_distances(coords: np.ndarray, min_samples: int) -> Optional[np.ndarray]:
    """Each point's distance to its min_samples-th nearest neighbor.

    None when the map is too small to have one. Imported lazily because
    sklearn's neighbors module is not free to import.
    """
    n_points = coords.shape[0]
    k = min(min_samples, n_points - 1)
    if k < 1:
        return None

    from sklearn.neighbors import NearestNeighbors

    # k + 1 neighbors because each point's nearest neighbor is itself.
    distances, _ = NearestNeighbors(n_neighbors=k + 1).fit(coords).kneighbors(coords)
    return distances[:, -1]


def _top_cluster_share(coords: np.ndarray, eps: float, min_samples: int) -> float:
    """Share of the gallery held by the largest cluster at eps.

    Noise is excluded, so an eps that clusters nothing scores 0 and the scan
    keeps climbing. Returns 1.0 for a candidate whose neighborhoods would
    exceed the pair budget: that eps is unaffordable, which for the scan's
    purposes is the same answer as "it has collapsed the map".
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import KDTree

    if coords.shape[0] > 1:
        tree = KDTree(coords)
        if _count_neighbor_pairs(tree, coords, eps, MAX_NEIGHBOR_PAIRS) > MAX_NEIGHBOR_PAIRS:
            return 1.0

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(coords).labels_
    clustered = labels[labels >= 0]
    if clustered.size == 0:
        return 0.0
    return float(np.bincount(clustered).max()) / float(labels.size)


def adaptive_cluster_eps(coords: np.ndarray, min_samples: int = DEFAULT_CLUSTER_MIN_SAMPLES) -> float:
    """The loosest eps that has not yet collapsed the map into one cluster.

    Candidates are quantiles of the k-distance distribution, so they are
    expressed in the units of THESE coordinates — which is what makes the
    result independent of gallery size and of UMAP's arbitrary output scale.
    The scan walks them upward and keeps the last one whose largest cluster
    stays under MAX_TOP_CLUSTER_SHARE.

    This replaces taking the median outright. The median makes about half the
    points core points by construction, which measures at 29-35% noise on a
    large gallery — far more unclustered than a hand-tuned eps gives, and the
    reason a 170k-image map read as mostly noise even once it was clustering
    at all. Walking up the quantiles instead stops right before one cluster
    swallows the map. Ported from PhotoMapAI, which reports the rule
    reproducing hand-tuned values on real albums (0.119 against a hand-set
    0.12 on 38k images; 0.051 against 0.05 on 86k).

    If even the smallest candidate is over the share there is no structure to
    separate, and the smallest is returned: going lower only turns the blob
    into noise without revealing anything.
    """
    distances = _k_distances(coords, min_samples)
    if distances is None:
        return DEFAULT_CLUSTER_EPS

    # Coincident points give a zero k-distance, and eps=0 clusters nothing.
    candidates = [eps for eps in (float(np.percentile(distances, q)) for q in CANDIDATE_QUANTILES) if eps > 0]
    if not candidates:
        return DEFAULT_CLUSTER_EPS

    # Probed in order, stopping at the first candidate that blobs. Not
    # searched: the top share is NOT monotone in eps, so the passing
    # candidates are not a prefix. Raising eps does grow every core-point
    # cluster, but a border point can be claimed by a different cluster that
    # has just come into reach, and the largest cluster then loses it — which
    # on a random blob map perturbs the share downward often enough to change
    # the chosen quantile. A binary search over the candidates read a pass
    # after a failure and kept climbing, picking an eps 1.3x looser than this
    # rule on some maps and none on others.
    best = candidates[0]
    for eps in candidates:
        if _top_cluster_share(coords, eps, min_samples) > MAX_TOP_CLUSTER_SHARE:
            # Later candidates are only looser, and the rule wants the last
            # one before the map collapses — so nothing after this matters.
            break
        best = eps

    return best


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


# Query points per chunk when counting neighbor pairs. The count only has to
# answer "is this over the budget?", and an eps far above it blows past the
# budget within the first chunk — so chunking turns the expensive probes into
# a fixed small cost while an under-budget eps still gets an exact full count.
# Without it, resolving a requested eps of 2.0 on a dense 150k-point map took
# 50s of CPU: thirteen full passes counting billions of pairs, only to reject
# every one of them.
_PAIR_COUNT_CHUNK = 4096


def _count_neighbor_pairs(tree, coords: np.ndarray, eps: float, cap: int) -> int:
    """Neighbor pairs at eps, abandoned once the count passes `cap`.

    Exact whenever the result is within the cap, which is the only case a
    caller acts on the number itself; above it the return is a lower bound.
    """
    total = 0
    for start in range(0, coords.shape[0], _PAIR_COUNT_CHUNK):
        chunk = coords[start : start + _PAIR_COUNT_CHUNK]
        total += int(tree.query_radius(chunk, r=eps, count_only=True).sum())
        if total > cap:
            break

    return total


def _shrink_eps_to_pair_budget(coords: np.ndarray, eps: float) -> tuple[float, Optional[int]]:
    """Shrink eps until DBSCAN's neighbor-pair count fits MAX_NEIGHBOR_PAIRS.

    The span clamp alone cannot bound memory: a dense blob concentrates most
    pairs in a small region, so a modest eps on a wide map can still
    materialize billions of pairs. Counting pairs with a KD-tree is cheap.

    Returns the eps and the pair count AT that eps (None when there is nothing
    to count), so a caller reporting diagnostics does not repeat the walk.
    """
    if coords.shape[0] < 2:
        return eps, None

    from sklearn.neighbors import KDTree

    tree = KDTree(coords)
    pairs = _count_neighbor_pairs(tree, coords, eps, MAX_NEIGHBOR_PAIRS)
    # Iterate until the budget is met rather than a fixed count: 12 rounds of
    # 0.7 only covers a 71x reduction, and a tight blob can need far more.
    # Bounded below by MIN_BUDGETED_EPS so a fully coincident map terminates.
    while pairs > MAX_NEIGHBOR_PAIRS and eps > MIN_BUDGETED_EPS:
        eps *= 0.7
        pairs = _count_neighbor_pairs(tree, coords, eps, MAX_NEIGHBOR_PAIRS)
    return eps, pairs


@dataclass(frozen=True)
class EpsResolution:
    """Every step of `resolve_cluster_eps`, for callers that report why.

    Each field is a gate that can silently turn a real clustering into an
    all-noise map, so the resolution keeps them rather than only its result.
    """

    requested_eps: Optional[float]
    """What the caller asked for; None means the adaptive default was used."""
    adaptive_eps: Optional[float]
    """Median k-distance, before the 2.0 cap. Only computed when requested_eps is None."""
    coord_span: float
    """Widest coordinate extent of the projection; 0.0 for a single point."""
    span_clamped_eps: float
    """After the MAX_EPS_SPAN_FRACTION clamp, before the floor. Equal to the
    requested eps when the caller supplied one: the clamp guards the
    heuristic, not the user."""
    floored_eps: float
    """After the 0.01 floor. Reported apart from the clamp because the two are
    separate causes of an over-tight eps, and a single number cannot say which
    one produced it."""
    resolved_eps: float
    """What DBSCAN will actually run with, after the neighbor-pair budget."""
    neighbor_pairs: Optional[int]
    """Pair count at resolved_eps; None when there was nothing to count."""


def resolve_cluster_eps(
    coords: np.ndarray,
    eps: Optional[float] = None,
    min_samples: int = DEFAULT_CLUSTER_MIN_SAMPLES,
) -> EpsResolution:
    """The eps clustering will actually use for these coordinates, and every
    intermediate value it passed through.

    Resolution order: adaptive default when eps is None, then the span clamp,
    then the 0.01 floor, then the neighbor-pair budget. The intermediates are
    returned rather than discarded because each one is a gate that can leave a
    map entirely unclustered, and the resolved value alone does not say which
    one bound it.
    """
    requested_eps = eps
    adaptive_eps: Optional[float] = None
    span = float(np.ptp(coords, axis=0).max()) if coords.shape[0] > 1 else 0.0
    if eps is None:
        # Cap at the API's own upper bound (le=2.0) so a reported adaptive
        # eps can always be passed back explicitly. Values that large only
        # arise on sparse structureless maps, where a smaller eps just means
        # more (correct) noise.
        adaptive_eps = adaptive_cluster_eps(coords, min_samples)
        eps = min(adaptive_eps, 2.0)
        # The span clamp applies ONLY to a derived value. A number the user
        # typed into the clustering-strength control is theirs to keep:
        # silently retuning it makes the control lie about what it does, and
        # on a small map (span 2, clamp 0.1) it would override most of the
        # range the control offers. A derived eps a large fraction of the
        # whole map's width, by contrast, means there is no density structure
        # to find, so bounding that one is a sanity check on the heuristic.
        # The pair budget below is the memory bound and applies to both.
        if span > 0:
            eps = min(eps, span * MAX_EPS_SPAN_FRACTION)
    # Floor at the API's lower bound (ge=0.01) for pass-back validity, but do
    # it BEFORE the budget shrink, never after: applied afterwards it silently
    # re-inflated eps past the neighbor-pair budget the shrink had just
    # computed, which is how a near-coincident map reached 400M pairs against
    # a 50M budget (~4GB). The budget bounds memory and therefore wins; an eps
    # below the API's floor is a cosmetic pass-back wart, an OOM is not.
    span_clamped_eps = eps
    floored_eps = max(span_clamped_eps, 0.01)
    resolved_eps, neighbor_pairs = _shrink_eps_to_pair_budget(coords, floored_eps)
    return EpsResolution(
        requested_eps=requested_eps,
        adaptive_eps=adaptive_eps,
        coord_span=span,
        span_clamped_eps=span_clamped_eps,
        floored_eps=floored_eps,
        resolved_eps=resolved_eps,
        neighbor_pairs=neighbor_pairs,
    )


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
    MAX_CLUSTERED_POINTS. Callers that need to know WHY the labels came out as
    they did call cluster_with_diagnostics, which this delegates to.
    """
    return cluster_with_diagnostics(coords, eps, min_samples)[0]


def _g(value: Optional[float]) -> str:
    """One float format for the whole diagnostic line, so its columns line up."""
    return "None" if value is None else f"{value:.6g}"


@dataclass(frozen=True)
class ClusterDiagnostics:
    """Why a clustering came out the way it did.

    Four gates can turn a real clustering into an all-noise map — the point
    cap, the span clamp, the pair budget, and an eps floored at 0.01 — and the
    served response shows none of them: every point simply reads
    "unclustered". This records what each gate saw, so that outcome can be
    explained from a log line instead of a reproduction.
    """

    n_points: int
    min_samples: int
    skipped: Optional[str]
    """Why DBSCAN did not run at all; None when it did."""
    eps: Optional[EpsResolution]
    """None when the run was skipped before eps was resolved."""
    cluster_count: int
    largest_cluster: int
    smallest_cluster: int
    unclustered: int
    duration_ms: float

    @property
    def resolved_eps(self) -> Optional[float]:
        return self.eps.resolved_eps if self.eps is not None else None

    def signature(self) -> str:
        """Greppable key=value line, timing excluded.

        Timing is what changes between two otherwise identical runs, so a
        caller that logs one line per distinct outcome compares this.
        """
        fields = [f"points={self.n_points}", f"min_samples={self.min_samples}"]
        if self.eps is None:
            fields.append("eps=unresolved")
        else:
            fields += [
                f"requested_eps={_g(self.eps.requested_eps)}",
                f"adaptive_eps={_g(self.eps.adaptive_eps)}",
                f"coord_span={_g(self.eps.coord_span)}",
                f"span_clamped_eps={_g(self.eps.span_clamped_eps)}",
                f"floored_eps={_g(self.eps.floored_eps)}",
                f"resolved_eps={_g(self.eps.resolved_eps)}",
                f"neighbor_pairs={self.eps.neighbor_pairs}",
            ]
        fields += [
            f"clusters={self.cluster_count}",
            f"largest={self.largest_cluster}",
            f"smallest={self.smallest_cluster}",
            f"unclustered={self.unclustered}",
        ]
        if self.skipped is not None:
            fields.append(f"skipped={self.skipped!r}")
        return " ".join(fields)

    def summary(self) -> str:
        return f"{self.signature()} took={self.duration_ms:.1f}ms"


def cluster_with_diagnostics(
    coords: np.ndarray,
    eps: Optional[float] = None,
    min_samples: int = DEFAULT_CLUSTER_MIN_SAMPLES,
) -> tuple[np.ndarray, ClusterDiagnostics]:
    """`compute_clusters`, plus a record of every gate the run passed through.

    The single implementation of the resolve-then-cluster sequence, so the
    pair count the eps resolution already produced is also the one that
    decides the pair-budget skip rather than being measured a second time.
    """
    started = time.perf_counter()
    n_points = coords.shape[0]

    def finish(labels: np.ndarray, resolution: Optional[EpsResolution], skipped: Optional[str]):
        _, sizes = np.unique(labels[labels >= 0], return_counts=True)
        return labels, ClusterDiagnostics(
            n_points=n_points,
            min_samples=min_samples,
            skipped=skipped,
            eps=resolution,
            cluster_count=int(sizes.size),
            largest_cluster=int(sizes.max()) if sizes.size else 0,
            smallest_cluster=int(sizes.min()) if sizes.size else 0,
            unclustered=int((labels < 0).sum()),
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )

    if n_points == 0:
        return finish(np.empty((0,), dtype=np.int64), None, None)
    if n_points > MAX_CLUSTERED_POINTS:
        return finish(
            np.full((n_points,), -1, dtype=np.int64),
            None,
            f"point count above MAX_CLUSTERED_POINTS={MAX_CLUSTERED_POINTS}",
        )

    resolution = resolve_cluster_eps(coords, eps, min_samples)
    if resolution.neighbor_pairs is not None and resolution.neighbor_pairs > MAX_NEIGHBOR_PAIRS:
        return finish(
            np.full((n_points,), -1, dtype=np.int64),
            resolution,
            f"neighbor pairs above MAX_NEIGHBOR_PAIRS={MAX_NEIGHBOR_PAIRS} at the pair budget's floor",
        )

    from sklearn.cluster import DBSCAN

    return finish(DBSCAN(eps=resolution.resolved_eps, min_samples=min_samples).fit(coords).labels_, resolution, None)


def scope_hash(model_id: str, items: list[IndexedItem]) -> str:
    """Fingerprint of an accessible-item set under one model.

    Order-insensitive: the caller may pass items in any order. Comparing the
    stored hash against a freshly derived one detects staleness from any
    cause — new/deleted items, board visibility changes, shares.

    Names alone identify the set: image and video names are both a UUID plus a
    kind-specific extension, so one name cannot name two items. Leaving the
    kind out of the digest also means a gallery with no videos keeps the
    projection it had before videos were indexable, rather than every user
    paying for a recomputed fit on the first load after an upgrade.
    """
    digest = hashlib.sha256()
    digest.update(model_id.encode("utf-8"))
    for name in sorted(item.name for item in items):
        digest.update(b"\x00")
        digest.update(name.encode("utf-8"))
    return digest.hexdigest()
