import asyncio
import threading
import time
from collections import OrderedDict
from datetime import date
from typing import AbstractSet, Any, Callable, Literal, Optional, TypeVar

import numpy as np
from fastapi import File, HTTPException, Query, UploadFile, status
from fastapi.routing import APIRouter
from PIL import Image
from pydantic import BaseModel, Field

from invokeai.app.api.auth_dependencies import AdminUserOrDefault, CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.api.routers._access import (
    assert_board_read_access,
    assert_image_read_access,
    assert_video_read_access,
)
from invokeai.app.services.image_files.image_files_common import ImageFileNotFoundException
from invokeai.app.services.image_index.cluster_labels import (
    MAX_CUSTOM_VOCAB_TERM_LENGTH,
    MAX_CUSTOM_VOCAB_TERMS,
    normalize_custom_vocab_terms,
)
from invokeai.app.services.image_index.image_index_base import (
    ImageIndexServiceBase,
    TextSearchUnavailableError,
    VocabBuildState,
)
from invokeai.app.services.image_index.image_index_common import (
    ImageIndexStatus,
    IndexedItem,
    MediaKind,
)
from invokeai.app.services.image_index.projection import (
    DEFAULT_CLUSTER_MIN_SAMPLES,
    MIN_BUDGETED_EPS,
    ClusterDiagnostics,
    cluster_with_diagnostics,
    compute_clusters,
    scope_hash,
)
from invokeai.app.services.image_records.image_records_common import ImageRecordNotFoundException
from invokeai.app.services.video_records.video_records_common import VideoRecordNotFoundException

image_map_router = APIRouter(prefix="/v1/image_map", tags=["image_map"])

ImageMapState = Literal["disabled", "model_missing", "empty", "computing", "ready"]


class ImageMapPoint(BaseModel):
    """One gallery item's position on the 2D semantic map."""

    x: float = Field(description="UMAP x coordinate")
    y: float = Field(description="UMAP y coordinate")
    image_name: str = Field(
        description="The image or video this point represents; `kind` says which namespace the name belongs to"
    )
    kind: MediaKind = Field(description="Whether this point is an image or a video")
    cluster: int = Field(description="DBSCAN cluster label; -1 means unclustered")


class ImageMapPointsResponse(BaseModel):
    """The current user's semantic map."""

    points: list[ImageMapPoint] = Field(description="The projected points")
    state: ImageMapState = Field(
        description="disabled: indexing is off; model_missing: indexing is enabled but the configured embedding "
        "model is not installed; empty: nothing to show; computing: a projection is being built, or the index is "
        "switching to a replacement embedding model; ready: points are served"
    )
    model_id: Optional[str] = Field(
        default=None, description="Active encoder fingerprint; clients must discard cached labels when it changes"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="The configured embedding model's name; only set when state is model_missing, so the client "
        "can tell the user which model to install",
    )
    stale: bool = Field(
        description="True when the accessible image set has changed since this projection was computed; a refresh has been requested"
    )
    point_count: int = Field(description="Number of points returned")
    cluster_eps: Optional[float] = Field(
        default=None,
        description="The effective DBSCAN eps used for these points (adaptive default resolved, clamps applied). "
        "Pass it back explicitly to get an identical clustering from a later request.",
    )
    visible_hash: Optional[str] = Field(
        default=None,
        description="Fingerprint of the visible image set these points were computed over; compare across "
        "image-map responses to detect accessible-set drift between requests.",
    )
    updated_at: Optional[str] = Field(default=None, description="When the served projection was computed")


class ImageMapProjectionStatus(BaseModel):
    """Status of the current user's cached projection."""

    state: ImageMapState = Field(description="Projection cache state")
    stale: bool = Field(description="Whether the cached projection lags the accessible image set")
    point_count: int = Field(description="Points in the cached projection")
    updated_at: Optional[str] = Field(default=None, description="When the cached projection was computed")


class ImageMapStatusResponse(BaseModel):
    """Combined index + projection status for the current user."""

    enabled: bool = Field(description="Whether the embedding index is running")
    model_id: Optional[str] = Field(
        default=None, description="Active encoder fingerprint; clients must discard cached labels when it changes"
    )
    model_name: Optional[str] = Field(
        default=None,
        description="The configured embedding model's name; only set when the projection state is model_missing",
    )
    index: Optional[ImageIndexStatus] = Field(
        default=None,
        description="Embedding index progress counts. Admin-only: the counts aggregate over all users' images, so they are omitted for regular users.",
    )
    projection: ImageMapProjectionStatus = Field(description="The user's projection cache status")


class ImageMapRefreshResponse(BaseModel):
    """Result of a projection refresh request."""

    enqueued: bool = Field(description="True if the recompute was accepted (or already pending)")


ClusterEpsQuery = Query(
    default=None,
    # Floored at what the resolver itself can produce, not at a round number:
    # `_shrink_eps_to_pair_budget` walks down to MIN_BUDGETED_EPS on a map with
    # near-coincident coordinates (duplicate generations project to the same
    # spot), and /points reports that value as `cluster_eps` for the client to
    # pass back here. A higher bound rejects the server's own output, and the
    # client treats the 422 as "no labels" with nothing logged anywhere.
    ge=MIN_BUDGETED_EPS,
    le=2.0,
    description="DBSCAN eps for clustering. Defaults to an adaptive value derived from the projection's "
    "k-distance distribution; that default is clamped relative to the coordinate span, a supplied value is "
    "not. Either way it is reduced if needed to keep DBSCAN's neighbourhoods inside a memory budget, and "
    "the value actually used is returned as `cluster_eps`.",
)
ClusterMinSamplesQuery = Query(
    default=DEFAULT_CLUSTER_MIN_SAMPLES, ge=2, le=100, description="DBSCAN min_samples for clustering"
)


def _scope(current_user) -> tuple[str, bool]:
    """(projection cache key, admin-wide scope flag) for the requesting user."""
    return current_user.user_id, current_user.is_admin


# A recompute takes minutes on any gallery large enough for this to matter, so
# refusing a second request inside this window costs a caller nothing — while a
# loop of them would otherwise pin the single index worker for every user and
# fan an event into every connected admin's socket per iteration.
MIN_REFRESH_INTERVAL_SECONDS = 10.0

_refresh_claims: dict[str, float] = {}
# Guards both module-level caches below. Held only for dict work, never across
# clustering or a DB call.
_state_lock = threading.Lock()


def _claim_refresh_slot(user_id: str) -> bool:
    """Whether this user may enqueue a refresh now, claiming their interval if so."""
    now = time.monotonic()
    with _state_lock:
        # Prune on write: entries older than the window can never throttle
        # anyone, so this keeps the map to users who refreshed recently rather
        # than one entry per user who ever has.
        for uid, claimed_at in list(_refresh_claims.items()):
            if now - claimed_at >= MIN_REFRESH_INTERVAL_SECONDS:
                del _refresh_claims[uid]
        if user_id in _refresh_claims:
            return False
        _refresh_claims[user_id] = now
        return True


def _release_refresh_slot(user_id: str) -> None:
    with _state_lock:
        _refresh_claims.pop(user_id, None)


# /points is refreshed on every gallery change, and between refreshes its
# inputs almost never change, so the clustering it repeats is usually
# identical work. Caching the labels turns the steady state into a dict
# lookup; entries are int64 label arrays, bounded at 8 bytes per point by the
# clustering cap (only a clustering that actually ran is stored, so the
# all-noise array a skipped clustering returns — free to recompute and
# unbounded in size — never lands here). At the current 300k cap that is
# 2.4MB per entry and ~77MB across a full pool, up from ~400KB and ~13MB when
# the cap was 50k; if the cap stays this high, the pool size below wants
# revisiting.
#
# ONE entry per user, rather than a shared pool of N. A shared pool made the
# cache worse than none: with more concurrent map users than slots, strict LRU
# over a round-robin access pattern misses every time, and any single caller
# could evict everyone else by varying `eps` across a handful of requests.
_CLUSTER_CACHE_USERS = 32
_ClusterCacheKey = tuple[str, Optional[str], str, Optional[float], int, tuple[MediaKind, ...]]
_cluster_cache: "OrderedDict[str, tuple[_ClusterCacheKey, np.ndarray, Optional[float]]]" = OrderedDict()


def _cluster_cache_get(user_id: str, key: _ClusterCacheKey) -> Optional[tuple[np.ndarray, Optional[float]]]:
    with _state_lock:
        entry = _cluster_cache.get(user_id)
        if entry is None or entry[0] != key:
            return None
        _cluster_cache.move_to_end(user_id)
        return entry[1], entry[2]


def _cluster_cache_put(user_id: str, key: _ClusterCacheKey, labels: np.ndarray, resolved_eps: Optional[float]) -> None:
    with _state_lock:
        _cluster_cache[user_id] = (key, labels, resolved_eps)
        _cluster_cache.move_to_end(user_id)
        while len(_cluster_cache) > _CLUSTER_CACHE_USERS:
            _cluster_cache.popitem(last=False)


# An all-noise map — every point "unclustered" — is the one clustering outcome
# the response cannot explain, and it has four separate causes (see
# ClusterDiagnostics). Reporting the diagnostics turns a user report into a
# single greppable line.
#
# Only that outcome is logged at INFO; a clustering that found clusters
# explains itself and goes to debug. The map refreshes on every gallery change
# (socket-driven, not polled), and each refresh moves the cache key, so
# without a guard a gallery stuck above the point cap would emit a line per
# refresh forever.
#
# The guard is per-user and keyed on the diagnostics minus their timing, so a
# repeat of the same outcome stays quiet while any change to the point set,
# the parameters or the resulting clustering speaks up. It also expires: the
# useful thing to be able to say is "open the map again while I watch the
# log", and a once-per-process line cannot be reproduced without a restart.
# Bounded like the cluster cache, under the same lock.
_CLUSTER_DIAGNOSTICS_USERS = 32
_CLUSTER_DIAGNOSTICS_REPEAT_AFTER_SECONDS = 600.0
_cluster_diagnostics_logged: "OrderedDict[str, tuple[str, float]]" = OrderedDict()


def _log_cluster_diagnostics(services, user_id: str, diagnostics: ClusterDiagnostics) -> None:
    """Report why this clustering came out as it did, at most once per outcome."""
    if diagnostics.n_points == 0:
        # Nothing was clustered because there was nothing to cluster; the
        # response already says `empty`.
        return

    explains_an_all_noise_map = diagnostics.skipped is not None or diagnostics.cluster_count == 0
    signature = diagnostics.signature()
    now = time.monotonic()
    with _state_lock:
        previous = _cluster_diagnostics_logged.get(user_id)
        # Recorded and moved to the end even when this request is about to be
        # suppressed: recency has to mean "last seen", not "last logged", or
        # the quiet users — the ones whose entry is doing its job — are the
        # first evicted, and eviction is what re-arms the line.
        _cluster_diagnostics_logged[user_id] = (signature, now)
        _cluster_diagnostics_logged.move_to_end(user_id)
        while len(_cluster_diagnostics_logged) > _CLUSTER_DIAGNOSTICS_USERS:
            _cluster_diagnostics_logged.popitem(last=False)
        if previous is not None and previous[0] == signature:
            if now - previous[1] < _CLUSTER_DIAGNOSTICS_REPEAT_AFTER_SECONDS:
                # Keep the ORIGINAL timestamp: refreshing it on every
                # suppressed request would hold the line off indefinitely.
                _cluster_diagnostics_logged[user_id] = (signature, previous[1])
                return

    message = f"Image map: clustered user '{user_id}': {diagnostics.summary()}"
    if explains_an_all_noise_map:
        services.logger.info(message)
    else:
        services.logger.debug(message)


def _active_model_id(services) -> Optional[str]:
    """Reconcile encoder deletion/reinstallation through the service's throttled lifecycle."""
    if services.configuration.image_index_enabled:
        services.image_index.try_activate()
    return services.image_index.model_id


def _inactive_state(services) -> ImageMapState:
    """Why there is no active model: an installed replacement still draining is not a missing model."""
    if not services.configuration.image_index_enabled:
        return "disabled"
    return "computing" if services.image_index.replacing_model else "model_missing"


async def _search_model_id(services) -> str:
    """The active model for a search; reconciled here too, so gallery search recovers without the map open."""
    model_id = await asyncio.to_thread(_active_model_id, services)
    if model_id is None:
        detail = (
            "The image index is switching embedding models; try again shortly"
            if _inactive_state(services) == "computing"
            else "The image index is not enabled; semantic search is unavailable"
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)
    return model_id


_T = TypeVar("_T")


def _with_model(indexer: ImageIndexServiceBase, model_id: str, operation: Callable[..., _T], *args: Any) -> _T:
    """Fence each executor step against the model that supplied its inputs."""
    with indexer.use_model():
        if indexer.model_id != model_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="The embedding model changed during this request; try again",
            )
        return operation(*args)


# Videos are indexed whether or not a client can render them, so serving them is opt-in: a
# client that resolves every name through the images endpoints (as the shipped gallery does)
# would turn a video into a broken tile and a selectable item that does not exist.
IncludeVideosQuery = Query(
    default=False,
    description="Include indexed videos among the returned items. Leave off unless the client "
    "resolves each item through the endpoint its `kind` names.",
)


def _served_kinds(include_videos: bool) -> tuple[MediaKind, ...]:
    return ("image", "video") if include_videos else ("image",)


SearchBoardQuery = Query(
    default=None,
    description="Restrict results to this board's items; 'none' selects items on no board. Omit to rank every "
    "accessible item.",
)
SearchCreatedDateQuery = Query(
    default=None,
    description="Restrict results to items created on this ISO date, as the date-based virtual boards list them.",
)


def _search_scope(
    board_id: Optional[str], created_date: Optional[date], current_user: CurrentUserOrDefault
) -> Optional[AbstractSet[str]]:
    """The item names a scoped search may return, or None when the search covers everything accessible.

    Membership comes from the gallery's own listing, intersected later with the indexed items, so
    items on archived boards (never indexed for search) cannot match. Image and video names never
    collide, so a flat name set is enough.
    """
    if board_id is None and created_date is None:
        return None
    if board_id is not None and board_id != "none":
        assert_board_read_access(board_id, current_user)
    names = ApiDependencies.invoker.services.gallery.get_item_names(
        starred_first=False,
        is_intermediate=False,
        board_id=board_id,
        user_id=current_user.user_id,
        is_admin=current_user.is_admin,
        created_date=created_date.isoformat() if created_date is not None else None,
    )
    return frozenset(names.item_names)


def _search_reference(image_name: Optional[str], video_name: Optional[str]) -> Optional[IndexedItem]:
    """The reference item for a similarity search, or None when none was given.

    Naming both is refused rather than resolved by precedence: it is a caller mistake, and the
    endpoint's "exactly one of" contract is what tells the caller so.
    """
    if image_name is not None and video_name is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide exactly one of q, image_name or video_name",
        )
    if image_name is not None:
        return IndexedItem("image", image_name)
    if video_name is not None:
        return IndexedItem("video", video_name)
    return None


def _reference_image(services, item: IndexedItem) -> Image.Image:
    """The pixels a reference item is embedded from.

    A video is represented by its thumbnail — the frame the index itself embeds — so an
    unindexed video answers the same query an indexed one would, and no decoder runs on a
    request thread.
    """
    if item.kind == "video":
        # Copied out of the context manager so the thumbnail's file handle is closed here
        # rather than whenever the returned image is collected.
        with Image.open(services.videos.get_path(item.name, thumbnail=True)) as thumbnail:
            return thumbnail.copy()
    return services.images.get_pil_image(item.name)


@image_map_router.get("/points", operation_id="get_image_map_points", response_model=ImageMapPointsResponse)
async def get_image_map_points(
    current_user: CurrentUserOrDefault,
    eps: Optional[float] = ClusterEpsQuery,
    min_samples: int = ClusterMinSamplesQuery,
    include_videos: bool = IncludeVideosQuery,
) -> ImageMapPointsResponse:
    """Gets the current user's semantic image map.

    Serves the cached UMAP projection (never blocks on a UMAP fit) and runs
    DBSCAN per request, so `eps` is live-adjustable. If the cache is missing
    or stale, a recompute is enqueued and reflected in `state`/`stale`.
    """
    services = ApiDependencies.invoker.services
    # Off the loop: the retry queries the model store.
    model_id = await asyncio.to_thread(_active_model_id, services)
    if model_id is None:
        # The service is also inert when indexing is enabled but the
        # configured model is not installed; tell the client which case
        # this is so it can show an actionable message.
        state = _inactive_state(services)
        return ImageMapPointsResponse(
            points=[],
            state=state,
            model_name=services.configuration.image_index_model if state == "model_missing" else None,
            stale=False,
            point_count=0,
        )

    user_id, is_admin = _scope(current_user)
    kinds = _served_kinds(include_videos)
    # Both reads are database work (on SQLite under the process-wide lock), and on a large gallery
    # the accessible listing is the expensive one — so they run off the event loop, like the
    # clustering below.
    current_items, record = await asyncio.to_thread(
        lambda: (
            services.image_index_records.list_accessible_embedded_items(None if is_admin else user_id, model_id),
            services.image_index_records.get_projection(user_id, model_id),
        )
    )

    if record is None:
        if not current_items:
            return ImageMapPointsResponse(points=[], state="empty", model_id=model_id, stale=False, point_count=0)
        enqueued = services.image_index.request_projection(user_id, all_images=is_admin)
        # stale means "a recompute is pending"; when nothing could be enqueued
        # (the indexer is not running) nothing is pending, and a client that
        # polls on stale would wait forever.
        return ImageMapPointsResponse(
            points=[], state="computing" if enqueued else "empty", model_id=model_id, stale=enqueued, point_count=0
        )

    current_hash = scope_hash(model_id, current_items)
    stale = record.scope_hash != current_hash

    if stale:
        services.image_index.request_projection(user_id, all_images=is_admin)

    # Serve only points still in the user's current accessible set, so items
    # un-shared (or deleted) since the projection was computed never leak out
    # of a stale cache. Filtering happens BEFORE clustering: labels computed
    # over hidden points would let density-chaining through an inaccessible
    # item leak its existence (and be wrong besides). All of this is CPU-bound
    # and O(point count) — including the accessibility mask, which is a Python
    # membership test per cached point — so it runs off the event loop. Hoisting
    # the mask into the coroutine to decide the retry first cost 54ms of event
    # loop time per poll on a 500k-image gallery, stalling every other request
    # in the process; the retry decision is made below instead, from the one
    # fact it actually needs.
    def build_points() -> tuple[list[ImageMapPoint], Optional[float], bool, str]:
        accessible = set(current_items)
        # The kind filter narrows what is SERVED, never what the scope hash covers: the
        # projection is fitted over every accessible item, so hashing a filtered set would
        # make every cached projection look permanently stale to an images-only client.
        visible_mask = np.fromiter(
            (item in accessible and item.kind in kinds for item in record.items),
            dtype=bool,
            count=len(record.items),
        )
        # A NaN cannot be serialized as valid JSON, so one corrupt coordinate
        # would fail the whole response; rows written before the projection
        # writer grew its isfinite guard are still out there in existing
        # databases.
        visible_mask &= np.isfinite(record.coords).all(axis=1)
        visible_items = [item for item, keep in zip(record.items, visible_mask, strict=True) if keep]
        visible_coords = record.coords[visible_mask]

        # Every input to the clustering is pinned by this key: which projection
        # row (its own scope hash plus updated_at, which moves on every rewrite),
        # which subset of it this user can currently see (current_hash), and the
        # two DBSCAN parameters. The cache is keyed by user on top of this, so
        # one user's labels can never be served to another even if all of these
        # collide.
        # The served kinds are part of the key: labels are computed over the visible points,
        # and an entry built for one kind filter is a different-length array for another —
        # which the zip below would refuse, 500ing whichever request read it second.
        cache_key: _ClusterCacheKey = (record.scope_hash, record.updated_at, current_hash, eps, min_samples, kinds)
        cached_labels = _cluster_cache_get(user_id, cache_key)
        if cached_labels is not None:
            labels, resolved_eps = cached_labels
            return (
                [
                    ImageMapPoint(x=float(x), y=float(y), image_name=item.name, kind=item.kind, cluster=int(label))
                    for item, (x, y), label in zip(visible_items, visible_coords, labels, strict=True)
                ],
                resolved_eps,
                True,
                scope_hash(model_id, visible_items),
            )

        diagnostics = None
        try:
            labels, diagnostics = cluster_with_diagnostics(visible_coords, eps=eps, min_samples=min_samples)
            resolved_eps = diagnostics.resolved_eps
            # eps was resolved exactly when the point cap let the clustering
            # proceed, so this is "the label array is bounded by the cap" —
            # asked of the diagnostics rather than re-derived from a second
            # copy of the constant, which can drift from the one the
            # clustering actually applied. Above the cap the labels are a
            # constant -1 array sized to the full point count: free to
            # recompute, unbounded in size, and never worth storing.
            if diagnostics.eps is not None:
                _cluster_cache_put(user_id, cache_key, labels, resolved_eps)
        except Exception:
            # Clustering is a presentation detail; the coordinates are the
            # data. sklearn raises on a non-finite cached row (which a build
            # predating the writer's isfinite guard could have left behind),
            # and an unhandled raise here 500s this user on EVERY request until
            # their gallery changes. Serve the map unclustered instead. Nothing
            # is cached on this path: a one-off failure must not stick.
            services.logger.exception(f"Image map: clustering failed for user '{user_id}'; serving points unclustered")
            resolved_eps = None
            labels = np.full((visible_coords.shape[0],), -1, dtype=np.int64)
        if diagnostics is not None:
            # Outside the try: a logging failure must not be reported as a
            # clustering failure, nor cost the caller their cached labels.
            _log_cluster_diagnostics(services, user_id, diagnostics)
        return (
            [
                ImageMapPoint(x=float(x), y=float(y), image_name=item.name, kind=item.kind, cluster=int(label))
                for item, (x, y), label in zip(visible_items, visible_coords, labels, strict=True)
            ],
            resolved_eps,
            bool(visible_mask.any()),
            scope_hash(model_id, visible_items),
        )

    points, resolved_eps, any_visible, visible_hash = await asyncio.to_thread(build_points)

    retrying = False
    # Only items of a kind this caller is served can evidence a failed fit. An accessible set
    # that is all videos, read by a client that asked for images, is empty for a reason the
    # projection cannot fix — and asking anyway starts the exact cycle `failed_scope` exists to
    # stop: the worker short-circuits on the matching scope hash without ever marking the scope
    # failed, emits projection_ready, and the client comes straight back with another /points.
    servable_exists = any(item.kind in kinds for item in current_items)
    if not stale and servable_exists and not any_visible:
        # Nothing servable over a gallery that HAS embedded items of this kind: a failed
        # fit, not a result — and it is stamped with the current scope, so
        # staleness will never ask for it again. `failed_scope` makes the
        # service refuse this once the scope's single retry is spent, so a
        # permanent failure settles into an honest "empty" instead of a
        # request/event cycle driven by every poll.
        retrying = services.image_index.request_projection(user_id, all_images=is_admin, failed_scope=current_hash)

    state: ImageMapState = "ready" if points else ("computing" if retrying else "empty")
    return ImageMapPointsResponse(
        points=points,
        state=state,
        model_id=model_id,
        stale=stale,
        point_count=len(points),
        cluster_eps=resolved_eps,
        visible_hash=visible_hash,
        updated_at=record.updated_at,
    )


class ImageMapSearchResult(BaseModel):
    """One semantic search hit."""

    image_name: str = Field(description="The matching image or video; `kind` says which namespace the name belongs to")
    kind: MediaKind = Field(description="Whether this hit is an image or a video")
    score: float = Field(description="Cosine similarity to the query; higher is more similar")


class ImageMapSearchResponse(BaseModel):
    """Semantic search results over the user's embedded gallery items."""

    results: list[ImageMapSearchResult] = Field(description="Ranked results, most similar first")


@image_map_router.get("/search", operation_id="search_image_map", response_model=ImageMapSearchResponse)
async def search_image_map(
    current_user: CurrentUserOrDefault,
    q: Optional[str] = Query(default=None, max_length=500, description="Text query to embed and search with"),
    image_name: Optional[str] = Query(default=None, description="Reference image for similarity search"),
    video_name: Optional[str] = Query(default=None, description="Reference video for similarity search"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum number of results"),
    include_videos: bool = IncludeVideosQuery,
    board_id: Optional[str] = SearchBoardQuery,
    created_date: Optional[date] = SearchCreatedDateQuery,
) -> ImageMapSearchResponse:
    """Ranks the user's accessible images and videos by semantic similarity.

    Provide exactly one of `q` (text search — requires the embedding model's
    text encoder to be installed), `image_name`, or `video_name` (visual
    similarity — uses the reference item's stored embedding when it exists,
    and otherwise embeds its file on demand, so unindexed items such as assets
    can be reference items too). A video is represented by its thumbnail, the
    same frame the index embedded.
    """
    services = ApiDependencies.invoker.services
    reference = _search_reference(image_name, video_name)
    if (q is None) == (reference is None):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Provide exactly one of q, image_name or video_name",
        )

    model_id = await _search_model_id(services)

    user_id, is_admin = _scope(current_user)
    scope_user = None if is_admin else user_id
    # Resolved before embedding so an unreadable or empty scope costs no encoder pass.
    within = await asyncio.to_thread(_search_scope, board_id, created_date, current_user)
    if within is not None and not within:
        return ImageMapSearchResponse(results=[])

    if q is not None:
        try:
            query_embedding = await asyncio.to_thread(
                _with_model, services.image_index, model_id, services.image_index.embed_text, q
            )
        except TextSearchUnavailableError as e:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    else:
        assert reference is not None
        if reference.kind == "video":
            assert_video_read_access(reference.name, current_user)
        else:
            assert_image_read_access(reference.name, current_user)
        found, matrix = services.image_index_records.get_embeddings([reference], model_id)
        if found:
            query_embedding = matrix[0]
        else:
            # Not in the index (assets and intermediates are never indexed) —
            # embed the stored file on demand for this one query.
            def embed_from_file() -> np.ndarray:
                pil = _reference_image(services, reference)
                # Capped exactly as the uploaded/downloaded path is: the encoder
                # downscales to ~224px either way, and convert("RGB") on a large
                # stored image materializes hundreds of MB on a request thread.
                # This branch is the *normal* one for such images, since assets
                # and intermediates are never indexed.
                if pil.width * pil.height > MAX_SEARCH_IMAGE_PIXELS:
                    raise HTTPException(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        detail="The reference image has too many pixels",
                    )
                return services.image_index.embed_image(pil)

            try:
                query_embedding = await asyncio.to_thread(_with_model, services.image_index, model_id, embed_from_file)
            except HTTPException:
                raise
            except (ImageFileNotFoundException, ImageRecordNotFoundException, VideoRecordNotFoundException, OSError):
                # The stored file is missing or undecodable — the only failure
                # here that is really about this item. Both record and file
                # exceptions are plain Exceptions rather than OSError, so they
                # have to be named; PIL's decode failures are OSError subclasses.
                services.logger.warning(
                    f"Image search: cannot read {reference.kind} '{reference.name}' for on-demand embed", exc_info=True
                )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="This item could not be embedded for search (its file may be missing)",
                )
            except Exception:
                # An encoder fault, a stopped index, an OOM. Reporting these as
                # "file may be missing" sent anyone debugging to the wrong place.
                services.logger.error(
                    f"Image search: failed to embed {reference.kind} '{reference.name}' on demand", exc_info=True
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="This item could not be embedded for search",
                )

    results = await asyncio.to_thread(
        _with_model,
        services.image_index,
        model_id,
        services.image_index.search_similar,
        scope_user,
        query_embedding,
        limit,
        _served_kinds(include_videos),
        within,
    )
    return ImageMapSearchResponse(
        results=[ImageMapSearchResult(image_name=item.name, kind=item.kind, score=score) for item, score in results]
    )


# One-off reference images are small; cap what we will buffer from an upload
# or a URL download before decoding, and how many pixels we will decode.
MAX_SEARCH_IMAGE_BYTES = 32 * 1024 * 1024
MAX_SEARCH_IMAGE_PIXELS = 64_000_000
_URL_TIMEOUT_SECONDS = 10.0
_URL_TOTAL_DEADLINE_SECONDS = 30.0
_URL_MAX_REDIRECTS = 5

# Downloads run on the loop's shared default executor; a handful of slow
# remote servers must not be able to park every thread in it.
_download_slots = asyncio.Semaphore(4)


def _assert_url_host_allowed(url: str) -> None:
    """Reject URLs whose scheme is not http(s) or whose host resolves to any
    non-global address (loopback, RFC1918, link-local, reserved).

    Resolving via getaddrinfo covers hostnames like 'localhost' and the
    non-canonical IP notations (decimal, hex, octal, shortened) that a plain
    ip_address() literal parse misses. DNS rebinding between this check and
    the actual request is accepted residual risk: the endpoint is
    authenticated, and nothing about the response is revealed beyond whether
    it decoded as an image.
    """
    import ipaddress
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="image_url must be an http(s) URL")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(parsed.hostname, port, proto=socket.IPPROTO_TCP)
    except OSError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="image_url host could not be resolved"
        )
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="image_url must not target a private or loopback address",
            )


def _download_search_image(url: str) -> bytes:
    """Fetch a reference image for a one-off similarity search.

    Redirects are followed manually so every hop is re-validated by
    _assert_url_host_allowed — requests' automatic redirect handling would
    happily follow a public URL into a private address. The wall-clock
    deadline bounds slow-trickle servers that never trip the per-read
    timeout.
    """
    import time
    from urllib.parse import urljoin

    import requests

    deadline = time.monotonic() + _URL_TOTAL_DEADLINE_SECONDS

    def _remaining() -> float:
        """Budget left, as a positive number; raises once it is gone.

        Checked before each hop and used as the per-request timeout, so the deadline bounds the
        whole exchange rather than only the body. Without it the connect and read timeouts apply
        afresh to every redirect, and a server that stalls each of the allowed hops before sending
        a byte holds a thread for minutes — uninterruptibly, since this runs under
        `asyncio.to_thread` on the executor that also serves embedding and search.
        """
        left = deadline - time.monotonic()

        if left <= 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="The image download took too long",
            )

        return left

    try:
        for _ in range(_URL_MAX_REDIRECTS + 1):
            _assert_url_host_allowed(url)
            hop_timeout = min(_URL_TIMEOUT_SECONDS, _remaining())
            with requests.get(url, stream=True, timeout=hop_timeout, allow_redirects=False) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        break
                    url = urljoin(url, location)
                    continue
                response.raise_for_status()
                chunks: list[bytes] = []
                received = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    received += len(chunk)
                    if received > MAX_SEARCH_IMAGE_BYTES:
                        raise HTTPException(
                            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            detail="The reference image is too large",
                        )
                    _remaining()
                    chunks.append(chunk)
                return b"".join(chunks)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The image could not be downloaded from image_url",
        )
    raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Too many redirects for image_url")


@image_map_router.post(
    "/search_by_image", operation_id="search_image_map_by_image", response_model=ImageMapSearchResponse
)
async def search_image_map_by_image(
    current_user: CurrentUserOrDefault,
    image: Optional[UploadFile] = File(default=None, description="Reference image file"),
    image_url: Optional[str] = Query(default=None, max_length=2000, description="URL of a reference image"),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum number of results"),
    include_videos: bool = IncludeVideosQuery,
    board_id: Optional[str] = SearchBoardQuery,
    created_date: Optional[date] = SearchCreatedDateQuery,
) -> ImageMapSearchResponse:
    """Ranks the user's accessible images by similarity to an arbitrary reference image.

    Provide exactly one of `image` (multipart upload) or `image_url` (the
    server downloads it). The reference is embedded for this query only —
    nothing is stored.
    """
    services = ApiDependencies.invoker.services
    if (image is None) == (image_url is None):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Provide exactly one of image or image_url"
        )

    model_id = await _search_model_id(services)

    user_id, is_admin = _scope(current_user)
    scope_user = None if is_admin else user_id
    within = await asyncio.to_thread(_search_scope, board_id, created_date, current_user)
    if within is not None and not within:
        return ImageMapSearchResponse(results=[])

    if image is not None:
        data = await image.read(MAX_SEARCH_IMAGE_BYTES + 1)
        if len(data) > MAX_SEARCH_IMAGE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="The reference image is too large"
            )
    else:
        assert image_url is not None
        async with _download_slots:
            data = await asyncio.to_thread(_download_search_image, image_url)

    def embed_bytes() -> np.ndarray:
        from io import BytesIO

        from PIL import Image

        pil = Image.open(BytesIO(data))
        # The encoder downscales to ~224px anyway; refuse decompression bombs
        # before convert() materializes hundreds of MB of RGB.
        if pil.width * pil.height > MAX_SEARCH_IMAGE_PIXELS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="The reference image has too many pixels"
            )
        return services.image_index.embed_image(pil)

    try:
        query_embedding = await asyncio.to_thread(_with_model, services.image_index, model_id, embed_bytes)
    except HTTPException:
        raise
    except Exception:
        services.logger.warning("Image search: failed to decode or embed an uploaded reference image", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="The reference could not be decoded as an image",
        )

    results = await asyncio.to_thread(
        _with_model,
        services.image_index,
        model_id,
        services.image_index.search_similar,
        scope_user,
        query_embedding,
        limit,
        _served_kinds(include_videos),
        within,
    )
    return ImageMapSearchResponse(
        results=[ImageMapSearchResult(image_name=item.name, kind=item.kind, score=score) for item, score in results]
    )


class ImageMapClusterLabel(BaseModel):
    """The best vocabulary label for one cluster."""

    label: str = Field(description="Best-matching vocabulary phrase")
    alternates: list[str] = Field(description="Runner-up phrases")
    score: float = Field(description="Cosine similarity of the best phrase to the cluster centroid")


class ImageMapClusterLabelsResponse(BaseModel):
    """Automatic labels for the current user's visible clusters, keyed by cluster id."""

    labels: dict[str, ImageMapClusterLabel] = Field(description="Cluster id -> label; noise (-1) is omitted")
    visible_hash: Optional[str] = Field(
        default=None,
        description="Fingerprint of the visible image set these labels were clustered over; clients must discard "
        "labels whose visible_hash does not match their points response",
    )
    updated_at: Optional[str] = Field(
        default=None,
        description="The projection these labels were computed against; clients must discard labels whose projection does not match their points",
    )


@image_map_router.get(
    "/cluster_labels", operation_id="get_image_map_cluster_labels", response_model=ImageMapClusterLabelsResponse
)
async def get_image_map_cluster_labels(
    current_user: CurrentUserOrDefault,
    eps: Optional[float] = ClusterEpsQuery,
    min_samples: int = ClusterMinSamplesQuery,
    top_k: int = Query(default=3, ge=1, le=10, description="Candidate labels per cluster"),
    include_videos: bool = IncludeVideosQuery,
) -> ImageMapClusterLabelsResponse:
    """Labels the user's visible clusters with the most similar vocabulary phrases.

    Clustering mirrors `/points` (same eps semantics, computed over the
    caller's currently-accessible points), so cluster ids line up with the
    served map. Pass the `cluster_eps` reported by `/points` so both requests
    cluster with the same eps; cluster ids can still differ if the accessible
    set changes between the requests, so compare the two responses'
    `visible_hash` (and `updated_at`) and discard labels on mismatch.
    Requires the embedding model's text encoder; the first call per model
    embeds the whole vocabulary and is slow.
    """
    from invokeai.app.services.image_index.cluster_labels import label_clusters

    services = ApiDependencies.invoker.services
    model_id = services.image_index.model_id
    if model_id is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The image index is not enabled")

    user_id, is_admin = _scope(current_user)
    record = services.image_index_records.get_projection(user_id, model_id)
    if record is None or record.point_count == 0:
        return ImageMapClusterLabelsResponse(labels={})

    try:
        vocabulary, vocab_embeddings = await asyncio.to_thread(
            _with_model, services.image_index, model_id, services.image_index.get_vocab_embeddings
        )
    except TextSearchUnavailableError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))

    current_items = services.image_index_records.list_accessible_embedded_items(None if is_admin else user_id, model_id)

    def build() -> tuple[dict[int, dict], str]:
        accessible = set(current_items)
        # Exactly the mask /points applies, kind filter included: the two responses' visible
        # hashes are compared by the client, so they must cover the same set.
        kinds = _served_kinds(include_videos)
        mask = np.fromiter(
            (item in accessible and item.kind in kinds for item in record.items),
            dtype=bool,
            count=len(record.items),
        )
        # Exactly the mask /points applies, non-finite filter included. Without
        # it this endpoint's visible_hash is computed over a different set of
        # items than the one /points reports, so every response the client
        # receives fails the hash comparison it is told to make and all labels
        # are discarded — and the clustering below is handed a NaN, which
        # sklearn raises on, 500ing this user until their gallery changes.
        mask &= np.isfinite(record.coords).all(axis=1)
        visible_items = [item for item, keep in zip(record.items, mask, strict=True) if keep]
        visible_hash = scope_hash(model_id, visible_items)
        if not visible_items:
            return {}, visible_hash
        visible_coords = record.coords[mask]
        try:
            # Resolves the client-supplied eps exactly once, as /points does, so
            # both endpoints agree on the clustering these labels describe. The
            # diagnostics /points reports cover this call too: same points, same
            # parameters, same result.
            cluster_ids = compute_clusters(visible_coords, eps=eps, min_samples=min_samples)
        except Exception:
            # /points degrades to unclustered rather than 500ing; labels over no
            # clusters is the same degradation, and the client already discards
            # a label set whose visible_hash does not match its points.
            services.logger.exception(f"Image map: clustering failed for user '{user_id}'; serving no labels")
            return {}, visible_hash
        # Nothing clustered, nothing to label. Worth checking before the gather
        # below: above MAX_CLUSTERED_POINTS every id is -1 by design, and on a
        # gallery that large the fancy-index would copy gigabytes purely to
        # hand label_clusters a set of noise rows it discards.
        clustered = cluster_ids >= 0
        if not clustered.any():
            return {}, visible_hash

        cluster_by_item = dict(zip(visible_items, cluster_ids, strict=True))
        # The accessible matrix comes from the same LRU the search endpoint
        # uses — this endpoint fires after every points refresh, and a full
        # BLOB read per request would not scale to large galleries.
        accessible_items, accessible_matrix = services.image_index.get_accessible_embeddings(
            None if is_admin else user_id
        )
        row_by_item = {item: index for index, item in enumerate(accessible_items)}
        # Noise rows are excluded here rather than inside label_clusters: they
        # contribute to no centroid, so gathering them only widens the copy.
        found_items = [
            item
            for item, is_clustered in zip(visible_items, clustered, strict=True)
            if is_clustered and item in row_by_item
        ]
        if not found_items:
            return {}, visible_hash
        embeddings = accessible_matrix[[row_by_item[item] for item in found_items]]
        aligned = np.fromiter((cluster_by_item[item] for item in found_items), dtype=np.int64, count=len(found_items))
        return label_clusters(aligned, embeddings, vocabulary, vocab_embeddings, top_k=top_k), visible_hash

    labels, visible_hash = await asyncio.to_thread(_with_model, services.image_index, model_id, build)
    return ImageMapClusterLabelsResponse(
        labels={
            str(cluster_id): ImageMapClusterLabel(
                alternates=info["alternates"], label=info["label"], score=info["score"]
            )
            for cluster_id, info in labels.items()
        },
        visible_hash=visible_hash,
        updated_at=record.updated_at,
    )


class ImageMapImageLabelsResponse(BaseModel):
    """The best vocabulary labels for one gallery item."""

    label: str = Field(description="Best-matching vocabulary phrase")
    alternates: list[str] = Field(description="Runner-up phrases")
    score: float = Field(description="Cosine similarity of the best phrase to the item's embedding")


@image_map_router.get(
    "/image_labels", operation_id="get_image_map_image_labels", response_model=ImageMapImageLabelsResponse
)
async def get_image_map_image_labels(
    current_user: CurrentUserOrDefault,
    image_name: str = Query(description="The image or video to label"),
    kind: MediaKind = Query(default="image", description="Which namespace image_name belongs to"),
    top_k: int = Query(default=3, ge=1, le=10, description="Number of candidate labels"),
) -> ImageMapImageLabelsResponse:
    """Labels one gallery item with the vocabulary phrases most similar to its stored embedding.

    Serves map hover cards, so it only covers items the index has embedded;
    an unindexed item (assets, intermediates, not-yet-indexed) is a 404
    rather than an on-demand embed — a hover must never queue encoder work.
    Requires the embedding model's text encoder, like /cluster_labels.
    """
    services = ApiDependencies.invoker.services
    model_id = services.image_index.model_id
    if model_id is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The image index is not enabled")

    item = IndexedItem(kind, image_name)
    if item.kind == "video":
        assert_video_read_access(item.name, current_user)
    else:
        assert_image_read_access(item.name, current_user)

    try:
        vocabulary, vocab_embeddings = await asyncio.to_thread(
            _with_model, services.image_index, model_id, services.image_index.get_vocab_embeddings
        )
    except TextSearchUnavailableError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    if not vocabulary:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The labeling vocabulary is empty")

    # A BLOB read and a database transaction; off the event loop like the
    # clustering endpoints, since this one fires per hovered point.
    #
    # A stored row whose blob length disagrees with its `dim` column raises out
    # of `blob_to_embedding`, and a row whose dim disagrees with the vocabulary
    # matrix raises out of the matmul below. Both are this one item's data
    # being unusable, so both answer 404 like the degenerate-vector case — an
    # unhandled 500 here would refire on every hover of that point, which is
    # once per pointer sweep rather than once per user action.
    try:
        found, matrix = await asyncio.to_thread(services.image_index_records.get_embeddings, [item], model_id)
        if not found:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="This item has no stored embedding to label"
            )

        # float64 for the norm, and a degenerate row is refused rather than
        # passed through — exactly as `_normalize_query_vector` does for search
        # queries. Dividing by a zero or non-finite norm makes every score NaN,
        # and `argpartition` then returns arbitrary rows: the card would present
        # three unrelated vocabulary phrases as this item's tags, with a
        # `score` that serializes as JSON null against a schema that declares it
        # a float. The writer rejects such rows now, but ones predating that
        # guard are still out there — the same assumption this file already
        # makes about cached coords.
        vector = matrix[0]
        norm = float(np.linalg.norm(vector.astype(np.float64)))
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError("degenerate stored embedding")

        scores = vocab_embeddings @ (vector / norm).astype(vocab_embeddings.dtype)
    except HTTPException:
        raise
    except ValueError:
        services.logger.warning(
            f"Image map: cannot label {item.kind} '{item.name}' from its stored embedding", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="This item's stored embedding cannot be labeled"
        )

    # Bounded by the score vector too, not just the phrase list: a cached vocab
    # matrix shorter than its phrase list would otherwise put `kth` out of
    # bounds and 500.
    count = min(top_k, len(vocabulary), int(scores.shape[0]))
    top = np.argpartition(-scores, count - 1)[:count]
    top = top[np.argsort(-scores[top])]
    return ImageMapImageLabelsResponse(
        label=vocabulary[int(top[0])],
        alternates=[vocabulary[int(index)] for index in top[1:]],
        score=float(scores[int(top[0])]),
    )


@image_map_router.post(
    "/refresh",
    operation_id="refresh_image_map",
    response_model=ImageMapRefreshResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def refresh_image_map(current_user: CurrentUserOrDefault) -> ImageMapRefreshResponse:
    """Requests a recompute of the current user's image map projection."""
    # The Retry the map offers alongside its "model not installed" message
    # lands here, so give the install a chance to have landed too.
    _active_model_id(ApiDependencies.invoker.services)
    user_id, is_admin = _scope(current_user)
    if not _claim_refresh_slot(user_id):
        # Throttled, not failed: `enqueued` already means "was this accepted",
        # so a client needs no new status code or field to understand it.
        return ImageMapRefreshResponse(enqueued=False)

    try:
        enqueued = ApiDependencies.invoker.services.image_index.request_projection(
            user_id, all_images=is_admin, user_initiated=True
        )
    except Exception:
        # A raise between the claim and the release (the invoker is absent
        # during a startup or shutdown window, say) would otherwise leave the
        # claim held with nothing queued against it.
        _release_refresh_slot(user_id)
        raise
    if not enqueued:
        # Nothing was actually queued (the indexer is not running), so this
        # request must not consume the user's interval — otherwise a client
        # polling through a restart is locked out of the first real refresh.
        _release_refresh_slot(user_id)
    return ImageMapRefreshResponse(enqueued=enqueued)


class ImageMapVocabResponse(BaseModel):
    """The supplementary cluster-labeling vocabulary and its embedding build state."""

    terms: list[str] = Field(description="The stored supplementary terms, normalized, sorted alphabetically")
    state: VocabBuildState = Field(
        description="unavailable: the index is not running; idle: embeddings will build when labels are next "
        "requested; building: an embedding (re)build is queued or running; ready: embeddings are serving; "
        "error: the last build failed (see error)"
    )
    error: Optional[str] = Field(default=None, description="Why the last embedding build failed; only set on error")
    max_terms: int = Field(description="Maximum number of supplementary terms the server accepts")
    max_term_length: int = Field(description="Maximum length of one term, in characters")


class ImageMapVocabUpdate(BaseModel):
    """Replacement supplementary vocabulary."""

    terms: list[str] = Field(
        max_length=MAX_CUSTOM_VOCAB_TERMS,
        description="The full supplementary term list; replaces what is stored. Terms are normalized "
        "(lowercased, whitespace collapsed) and deduplicated server-side.",
    )


def _vocab_response(services) -> ImageMapVocabResponse:
    terms = services.image_index_records.get_custom_vocab_terms()
    state, error = services.image_index.get_vocab_build_state()
    return ImageMapVocabResponse(
        terms=terms,
        state=state,
        error=error,
        max_terms=MAX_CUSTOM_VOCAB_TERMS,
        max_term_length=MAX_CUSTOM_VOCAB_TERM_LENGTH,
    )


@image_map_router.get("/vocab", operation_id="get_image_map_vocab", response_model=ImageMapVocabResponse)
def get_image_map_vocab(current_user: CurrentUserOrDefault) -> ImageMapVocabResponse:
    """Gets the supplementary cluster-labeling vocabulary.

    The list is server-wide: cluster labels are computed against one shared
    vocabulary. Readable by any user; only admins may change it.
    """
    return _vocab_response(ApiDependencies.invoker.services)


@image_map_router.put("/vocab", operation_id="update_image_map_vocab", response_model=ImageMapVocabResponse)
def update_image_map_vocab(update: ImageMapVocabUpdate, current_user: AdminUserOrDefault) -> ImageMapVocabResponse:
    """Replaces the supplementary cluster-labeling vocabulary. Admin-only.

    The stored embeddings are invalidated and rebuilt in the background (the
    response's `state` reflects this); labels served in the meantime still use
    the previous vocabulary. Works while the index is disabled too — terms
    persist and take effect when indexing next runs.
    """
    try:
        terms = normalize_custom_vocab_terms(update.terms)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    services = ApiDependencies.invoker.services
    services.image_index_records.set_custom_vocab_terms(terms)
    # After the commit, so the rebuild can only ever read the new rows.
    services.image_index.invalidate_vocab()
    return _vocab_response(services)


@image_map_router.get("/status", operation_id="get_image_map_status", response_model=ImageMapStatusResponse)
def get_image_map_status(
    current_user: CurrentUserOrDefault, include_videos: bool = IncludeVideosQuery
) -> ImageMapStatusResponse:
    """Gets embedding index progress and the user's projection cache status."""
    services = ApiDependencies.invoker.services
    model_id = _active_model_id(services)
    if model_id is None:
        state = _inactive_state(services)
        return ImageMapStatusResponse(
            enabled=False,
            model_name=services.configuration.image_index_model if state == "model_missing" else None,
            projection=ImageMapProjectionStatus(state=state, stale=False, point_count=0),
        )

    user_id, is_admin = _scope(current_user)
    record = services.image_index_records.get_projection(user_id, model_id)
    if record is None:
        projection = ImageMapProjectionStatus(state="empty", stale=False, point_count=0)
    else:
        current_items = services.image_index_records.list_accessible_embedded_items(
            None if is_admin else user_id, model_id
        )
        # Count only currently-accessible points; a stale record's raw count
        # would reveal the size of a since-revoked scope. Kinds the caller did not ask for are
        # excluded too, so this count matches the points /points would serve it.
        kinds = _served_kinds(include_videos)
        visible_count = len({item for item in set(record.items) & set(current_items) if item.kind in kinds})
        projection = ImageMapProjectionStatus(
            state="ready" if visible_count else "empty",
            stale=record.scope_hash != scope_hash(model_id, current_items),
            point_count=visible_count,
            updated_at=record.updated_at,
        )
    # Index counts aggregate over all users' images; expose them to admins only.
    index = services.image_index.get_status() if is_admin else None
    return ImageMapStatusResponse(enabled=True, model_id=model_id, index=index, projection=projection)
