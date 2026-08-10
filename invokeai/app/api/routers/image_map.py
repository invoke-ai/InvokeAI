import asyncio
import threading
import time
from collections import OrderedDict
from typing import Literal, Optional

import numpy as np
from fastapi import Query, status
from fastapi.routing import APIRouter
from pydantic import BaseModel, Field

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.image_index.image_index_common import ImageIndexStatus
from invokeai.app.services.image_index.projection import (
    DEFAULT_CLUSTER_MIN_SAMPLES,
    MAX_CLUSTERED_POINTS,
    cluster_at_eps,
    compute_clusters,
    resolve_cluster_eps,
    scope_hash,
)

image_map_router = APIRouter(prefix="/v1/image_map", tags=["image_map"])

ImageMapState = Literal["disabled", "model_missing", "empty", "computing", "ready"]


class ImageMapPoint(BaseModel):
    """One image's position on the 2D semantic map."""

    x: float = Field(description="UMAP x coordinate")
    y: float = Field(description="UMAP y coordinate")
    image_name: str = Field(description="The image this point represents")
    cluster: int = Field(description="DBSCAN cluster label; -1 means unclustered")


class ImageMapPointsResponse(BaseModel):
    """The current user's semantic map."""

    points: list[ImageMapPoint] = Field(description="The projected points")
    state: ImageMapState = Field(
        description="disabled: indexing is off; model_missing: indexing is enabled but the configured embedding "
        "model is not installed; empty: nothing to show; computing: a projection is being built; ready: points are served"
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
    ge=0.01,
    le=2.0,
    description="DBSCAN eps for clustering. Defaults to an adaptive value derived from the projection's "
    "k-distance distribution. Clamped server-side relative to the projection's coordinate span.",
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


# /points is polled, and between polls its inputs almost never change, so the
# clustering it repeats is usually identical work. Caching the labels turns the
# steady state into a dict lookup; the entries are int64 label arrays (~400KB at
# the 50k-point clustering cap), so a handful of them costs a few MB.
_CLUSTER_CACHE_ENTRIES = 8
_cluster_cache: "OrderedDict[tuple[str, str, Optional[str], Optional[float], int], tuple[np.ndarray, Optional[float]]]" = OrderedDict()


def _cluster_cache_get(key) -> Optional[tuple[np.ndarray, Optional[float]]]:
    with _state_lock:
        hit = _cluster_cache.get(key)
        if hit is not None:
            _cluster_cache.move_to_end(key)
        return hit


def _cluster_cache_put(key, value: tuple[np.ndarray, Optional[float]]) -> None:
    with _state_lock:
        _cluster_cache[key] = value
        _cluster_cache.move_to_end(key)
        while len(_cluster_cache) > _CLUSTER_CACHE_ENTRIES:
            _cluster_cache.popitem(last=False)


@image_map_router.get("/points", operation_id="get_image_map_points", response_model=ImageMapPointsResponse)
async def get_image_map_points(
    current_user: CurrentUserOrDefault,
    eps: Optional[float] = ClusterEpsQuery,
    min_samples: int = ClusterMinSamplesQuery,
) -> ImageMapPointsResponse:
    """Gets the current user's semantic image map.

    Serves the cached UMAP projection (never blocks on a UMAP fit) and runs
    DBSCAN per request, so `eps` is live-adjustable. If the cache is missing
    or stale, a recompute is enqueued and reflected in `state`/`stale`.
    """
    services = ApiDependencies.invoker.services
    model_id = services.image_index.model_id
    if model_id is None:
        # The service is also inert when indexing is enabled but the
        # configured model is not installed; tell the client which case
        # this is so it can show an actionable message.
        if services.configuration.image_index_enabled:
            return ImageMapPointsResponse(
                points=[],
                state="model_missing",
                model_name=services.configuration.image_index_model,
                stale=False,
                point_count=0,
            )
        return ImageMapPointsResponse(points=[], state="disabled", stale=False, point_count=0)

    user_id, is_admin = _scope(current_user)
    current_names = services.image_index_records.list_accessible_embedded_images(
        None if is_admin else user_id, model_id
    )
    record = services.image_index_records.get_projection(user_id, model_id)

    if record is None:
        if not current_names:
            return ImageMapPointsResponse(points=[], state="empty", stale=False, point_count=0)
        enqueued = services.image_index.request_projection(user_id, all_images=is_admin)
        # stale means "a recompute is pending"; when nothing could be enqueued
        # (the indexer is not running) nothing is pending, and a client that
        # polls on stale would wait forever.
        return ImageMapPointsResponse(
            points=[], state="computing" if enqueued else "empty", stale=enqueued, point_count=0
        )

    current_hash = scope_hash(model_id, current_names)
    stale = record.scope_hash != current_hash

    # Computed once, here, because the retry decision below depends on it: a row
    # can carry point_count > 0 and still contribute nothing, if every one of
    # its coordinates is non-finite. Deciding on the cached count instead left
    # exactly that row serving "empty, not stale" with nothing ever asking for a
    # recompute — the same terminal blankness this retry exists to remove.
    accessible = set(current_names)
    visible_mask = np.fromiter(
        (name in accessible for name in record.image_names), dtype=bool, count=len(record.image_names)
    )
    # A NaN cannot be serialized as valid JSON, so one corrupt coordinate would
    # fail the whole response; rows written before the projection writer grew
    # its isfinite guard are still out there in existing databases.
    visible_mask &= np.isfinite(record.coords).all(axis=1)

    retrying = False
    if stale:
        services.image_index.request_projection(user_id, all_images=is_admin)
    elif current_names and not visible_mask.any():
        # Nothing servable over a gallery that HAS embedded images: a failed
        # fit, not a result — and it is stamped with the current scope, so
        # staleness will never ask for it again. `failed_scope` makes the
        # service refuse this once the scope's single retry is spent, so a
        # permanent failure settles into an honest "empty" instead of a
        # request/event cycle driven by every poll.
        retrying = services.image_index.request_projection(user_id, all_images=is_admin, failed_scope=current_hash)

    # Serve only points still in the user's current accessible set, so images
    # un-shared (or deleted) since the projection was computed never leak out
    # of a stale cache. Filtering happens BEFORE clustering: labels computed
    # over hidden points would let density-chaining through an inaccessible
    # image leak its existence (and be wrong besides). The clustering and
    # response assembly are CPU-bound, so they run off the event loop.
    def build_points() -> tuple[list[ImageMapPoint], Optional[float]]:
        visible_names = [name for name, keep in zip(record.image_names, visible_mask, strict=True) if keep]
        visible_coords = record.coords[visible_mask]

        # Every input to the clustering is pinned by this key: which projection
        # row (its own scope hash plus updated_at, which moves on every rewrite),
        # which subset of it this user can currently see (current_hash), and the
        # two DBSCAN parameters. user_id is in the key too, so one user's labels
        # can never be served to another even if the rest collides.
        cache_key = (user_id, record.scope_hash, record.updated_at, current_hash, eps, min_samples)
        cached_labels = _cluster_cache_get(cache_key)
        if cached_labels is not None:
            labels, resolved_eps = cached_labels
            return [
                ImageMapPoint(x=float(x), y=float(y), image_name=name, cluster=int(label))
                for name, (x, y), label in zip(visible_names, visible_coords, labels, strict=True)
            ], resolved_eps

        try:
            if 0 < visible_coords.shape[0] <= MAX_CLUSTERED_POINTS:
                resolved_eps = resolve_cluster_eps(visible_coords, eps, min_samples)
                # cluster_at_eps, not compute_clusters: the latter re-resolves
                # what was just resolved, and the second pass re-applies the
                # 0.01 floor to a budget-shrunk eps — so the value reported
                # here would not be the value DBSCAN used.
                labels = cluster_at_eps(visible_coords, resolved_eps, min_samples)
            else:
                resolved_eps = None
                labels = compute_clusters(visible_coords, eps=eps, min_samples=min_samples)
        except Exception:
            # Clustering is a presentation detail; the coordinates are the
            # data. sklearn raises on a non-finite cached row (which a build
            # predating the writer's isfinite guard could have left behind),
            # and an unhandled raise here 500s this user on EVERY request until
            # their gallery changes. Serve the map unclustered instead.
            services.logger.exception(f"Image map: clustering failed for user '{user_id}'; serving points unclustered")
            resolved_eps = None
            labels = np.full((visible_coords.shape[0],), -1, dtype=np.int64)
        else:
            # Only a real result is cached: caching the degraded all-noise
            # fallback would make a one-off failure stick for as long as the
            # entry survives.
            _cluster_cache_put(cache_key, (labels, resolved_eps))
        return [
            ImageMapPoint(x=float(x), y=float(y), image_name=name, cluster=int(label))
            for name, (x, y), label in zip(visible_names, visible_coords, labels, strict=True)
        ], resolved_eps

    points, resolved_eps = await asyncio.to_thread(build_points)
    state: ImageMapState = "ready" if points else ("computing" if retrying else "empty")
    return ImageMapPointsResponse(
        points=points,
        state=state,
        stale=stale,
        point_count=len(points),
        cluster_eps=resolved_eps,
        updated_at=record.updated_at,
    )


@image_map_router.post(
    "/refresh",
    operation_id="refresh_image_map",
    response_model=ImageMapRefreshResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def refresh_image_map(current_user: CurrentUserOrDefault) -> ImageMapRefreshResponse:
    """Requests a recompute of the current user's image map projection."""
    user_id, is_admin = _scope(current_user)
    if not _claim_refresh_slot(user_id):
        # Throttled, not failed: `enqueued` already means "was this accepted",
        # so a client needs no new status code or field to understand it.
        return ImageMapRefreshResponse(enqueued=False)

    enqueued = ApiDependencies.invoker.services.image_index.request_projection(user_id, all_images=is_admin)
    if not enqueued:
        # Nothing was actually queued (the indexer is not running), so this
        # request must not consume the user's interval — otherwise a client
        # polling through a restart is locked out of the first real refresh.
        _release_refresh_slot(user_id)
    return ImageMapRefreshResponse(enqueued=enqueued)


@image_map_router.get("/status", operation_id="get_image_map_status", response_model=ImageMapStatusResponse)
async def get_image_map_status(current_user: CurrentUserOrDefault) -> ImageMapStatusResponse:
    """Gets embedding index progress and the user's projection cache status."""
    services = ApiDependencies.invoker.services
    model_id = services.image_index.model_id
    if model_id is None:
        model_missing = services.configuration.image_index_enabled
        return ImageMapStatusResponse(
            enabled=False,
            model_name=services.configuration.image_index_model if model_missing else None,
            projection=ImageMapProjectionStatus(
                state="model_missing" if model_missing else "disabled", stale=False, point_count=0
            ),
        )

    user_id, is_admin = _scope(current_user)
    record = services.image_index_records.get_projection(user_id, model_id)
    if record is None:
        projection = ImageMapProjectionStatus(state="empty", stale=False, point_count=0)
    else:
        current_names = services.image_index_records.list_accessible_embedded_images(
            None if is_admin else user_id, model_id
        )
        # Count only currently-accessible points; a stale record's raw count
        # would reveal the size of a since-revoked scope.
        visible_count = len(set(record.image_names) & set(current_names))
        projection = ImageMapProjectionStatus(
            state="ready" if visible_count else "empty",
            stale=record.scope_hash != scope_hash(model_id, current_names),
            point_count=visible_count,
            updated_at=record.updated_at,
        )
    # Index counts aggregate over all users' images; expose them to admins only.
    index = services.image_index.get_status() if is_admin else None
    return ImageMapStatusResponse(enabled=True, index=index, projection=projection)
