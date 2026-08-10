import asyncio
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

    stale = record.scope_hash != scope_hash(model_id, current_names)
    retrying = False
    if stale:
        services.image_index.request_projection(user_id, all_images=is_admin)
    elif record.point_count == 0 and current_names:
        # An empty projection over a gallery that HAS embedded images is a
        # failed fit, not a result — and it is stamped with the current scope,
        # so staleness will never ask for it again. Request the recompute; the
        # service grants one retry per scope per process, so a transient
        # failure heals here while a permanent one cannot spin.
        retrying = services.image_index.request_projection(user_id, all_images=is_admin)

    # Serve only points still in the user's current accessible set, so images
    # un-shared (or deleted) since the projection was computed never leak out
    # of a stale cache. Filtering happens BEFORE clustering: labels computed
    # over hidden points would let density-chaining through an inaccessible
    # image leak its existence (and be wrong besides). The clustering and
    # response assembly are CPU-bound, so they run off the event loop.
    def build_points() -> tuple[list[ImageMapPoint], Optional[float]]:
        accessible = set(current_names)
        mask = np.fromiter(
            (name in accessible for name in record.image_names), dtype=bool, count=len(record.image_names)
        )
        # Drop non-finite rows as well. A NaN cannot be serialized as valid
        # JSON, so one corrupt coordinate would fail the whole response — and
        # rows written before the projection writer grew its isfinite guard are
        # still out there in existing databases.
        mask &= np.isfinite(record.coords).all(axis=1)
        visible_names = [name for name, keep in zip(record.image_names, mask, strict=True) if keep]
        visible_coords = record.coords[mask]
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
    enqueued = ApiDependencies.invoker.services.image_index.request_projection(user_id, all_images=is_admin)
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
