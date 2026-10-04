"""Contracts of the intermediates manager: summaries, previews and cleanup operations."""

from datetime import datetime
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

from invokeai.app.services.shared.media_references import MediaReferenceOwnerKind

IntermediatesCleanupMode = Literal["safe", "force"]
"""`safe` keeps everything a saved document still references; `force` deletes those too. Neither
touches active work, recent uploads or durable media."""

IntermediatesScopeKind = Literal["selection", "owner", "everyone", "matching"]
IntermediatesOperationStatus = Literal["pending", "running", "completed", "failed"]
IntermediatesSummarySort = Literal["reclaimable_bytes", "project_name"]

# Intermediates younger than this are never collected: it covers the window between a browser
# uploading an intermediate and either promoting it to a durable asset, saving a document that
# references it, or enqueueing the job that consumes it.
RECENT_GRACE_SECONDS = 30 * 60

PREVIEW_TTL_SECONDS = 10 * 60

# An open editor refreshes its hold every few minutes; a tab that closes without releasing it
# stops protecting its media after this long.
BROWSER_HOLD_TTL_SECONDS = 15 * 60
MAX_BROWSER_HOLD_NAMES = 50_000
# One open editor (tab) holds one lease.
BROWSER_HOLD_LEASE_ID_PATTERN = r"^[A-Za-z0-9_-]+$"
# Leases one account can hold at once; past it the least recently refreshed lease lapses.
MAX_BROWSER_HOLD_LEASES_PER_USER = 8
# A force preview lists at most this many broken documents; `affected_documents_total` counts them all.
MAX_AFFECTED_DOCUMENTS = 200
# A force preview acknowledges every document it would break by identity; past this many the
# caller narrows the scope instead of the server holding an unbounded set per preview.
MAX_ACKNOWLEDGED_DOCUMENTS = 10_000

MediaName = Annotated[str, Field(min_length=1, max_length=255)]


class IntermediatesBrowserHoldRequest(BaseModel):
    images: list[MediaName] = Field(default_factory=list, max_length=MAX_BROWSER_HOLD_NAMES)
    videos: list[MediaName] = Field(default_factory=list, max_length=MAX_BROWSER_HOLD_NAMES)


class IntermediatesScopeTarget(BaseModel):
    """One row of the manager: an owner's project, or their unassigned intermediates."""

    user_id: str = Field(min_length=1, max_length=255, description="The owning account")
    project_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="The originating project; null selects the owner's unassigned intermediates",
    )


class IntermediatesScope(BaseModel):
    """What a cleanup acts on.

    `selection` names rows; `owner` names an account; `everyone` is every account (administrators);
    `matching` is every row the summary filters match, minus `excluded`, resolved by the server so
    a client never has to enumerate rows it has not loaded.
    """

    kind: IntermediatesScopeKind = Field(description="How the targets were chosen")
    targets: list[IntermediatesScopeTarget] = Field(
        default_factory=list, max_length=1000, description="The selected rows; only read for a `selection` scope"
    )
    user_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description=(
            "The account an `owner` scope targets, or the owner filter of a `matching` scope"
            " (null: the caller's account, or every account for administrators)"
        ),
    )
    project_id: Optional[str] = Field(
        default=None, min_length=1, max_length=255, description="`matching` only: the summary's project filter"
    )
    search: Optional[str] = Field(default=None, max_length=200, description="`matching` only: the summary's search")
    excluded: list[IntermediatesScopeTarget] = Field(
        default_factory=list, max_length=1000, description="`matching` only: matching rows to leave out"
    )


class IntermediatesKindCounts(BaseModel):
    """How one media kind's intermediates split under the cleanup policy."""

    safe: int = Field(default=0, description="Unreferenced, inactive and old enough: deleted by either mode")
    referenced: int = Field(default=0, description="Named by a saved document: kept by safe mode")
    active: int = Field(default=0, description="Produced or consumed by pending, waiting or running work: always kept")
    recent: int = Field(default=0, description="Created inside the grace window: always kept")

    @property
    def total(self) -> int:
        return self.safe + self.referenced + self.active + self.recent


class IntermediatesRow(IntermediatesScopeTarget):
    user_display_name: Optional[str] = Field(default=None, description="The owner's display name, if known")
    user_email: Optional[str] = Field(default=None, description="The owner's email, if known")
    project_name: Optional[str] = Field(default=None, description="The project's name; null for unassigned rows")
    cover_image_name: Optional[str] = Field(
        default=None, description="The newest durable image on the project's board, for a thumbnail"
    )
    images: IntermediatesKindCounts = Field(default_factory=IntermediatesKindCounts)
    videos: IntermediatesKindCounts = Field(default_factory=IntermediatesKindCounts)
    reclaimable_bytes: int = Field(default=0, description="Measured size of the safe items")
    referenced_bytes: int = Field(default=0, description="Measured size of the referenced items a force clear adds")
    unknown_size_count: int = Field(default=0, description="Safe or referenced items whose size is not yet measured")


class IntermediatesSummaryTotals(BaseModel):
    """Totals over every row that matches the request, not just the returned page."""

    rows: int = Field(default=0)
    safe_images: int = Field(default=0)
    safe_videos: int = Field(default=0)
    in_use_images: int = Field(default=0)
    in_use_videos: int = Field(default=0)
    reclaimable_bytes: int = Field(default=0)
    unknown_size_count: int = Field(default=0)


class IntermediatesSummary(BaseModel):
    items: list[IntermediatesRow]
    total: int = Field(description="Rows matching the request")
    offset: int
    limit: int
    totals: IntermediatesSummaryTotals
    recent_grace_seconds: int = Field(description="How long a new intermediate is protected")
    measuring: bool = Field(description="Whether sizes are still being measured in the background")
    can_manage_everyone: bool = Field(description="Whether the caller may target other accounts")


class IntermediatesPreviewRequest(BaseModel):
    mode: IntermediatesCleanupMode
    scope: IntermediatesScope


class IntermediatesImpact(BaseModel):
    delete_images: int = Field(default=0)
    delete_videos: int = Field(default=0)
    keep_referenced_images: int = Field(default=0)
    keep_referenced_videos: int = Field(default=0)
    keep_active_images: int = Field(default=0)
    keep_active_videos: int = Field(default=0)
    keep_recent_images: int = Field(default=0)
    keep_recent_videos: int = Field(default=0)
    reclaimable_bytes: int = Field(default=0, description="Measured size of the items that would be deleted")
    unknown_size_count: int = Field(default=0, description="Items to delete whose size is not yet measured")


class IntermediatesAffectedDocument(BaseModel):
    """A saved document a force clear would leave pointing at deleted media."""

    kind: MediaReferenceOwnerKind = Field(
        description=(
            "`client_state` is the legacy editor's persisted state; `quarantined_project` is a project kept for repair"
        )
    )
    user_id: str
    user_display_name: Optional[str] = Field(default=None, description="The owner's display name, if known")
    user_email: Optional[str] = Field(default=None, description="The owner's email, if known")
    owner_id: str = Field(description="The project or workflow id, or the client state key")
    name: Optional[str] = Field(default=None, description="The document's name, if it still exists")
    references: int = Field(description="How many of the targeted items the document names")


class IntermediatesPreview(BaseModel):
    preview_id: str
    mode: IntermediatesCleanupMode
    scope: IntermediatesScope
    created_at: datetime
    expires_at: datetime
    target_rows: int = Field(description="Rows the scope resolved to")
    impact: IntermediatesImpact
    affected_documents: list[IntermediatesAffectedDocument] = Field(
        default_factory=list,
        description=(
            "Documents a force clear would break. Confirming acknowledges these documents: the operation deletes"
            " referenced targets only while every document naming them is acknowledged, so a document saved after"
            " the preview keeps its media. A non-administrator's force clear keeps media other accounts'"
            " documents name, so these are always the caller's own. Bounded; see `affected_documents_total`"
        ),
    )
    affected_documents_total: int = Field(
        default=0, description="How many documents a force clear would break, including any not listed"
    )


class IntermediatesOperationRequest(BaseModel):
    preview_id: str = Field(min_length=1, max_length=64, description="A preview is confirmed at most once")


class IntermediatesOperationProgress(BaseModel):
    processed_images: int = Field(default=0)
    processed_videos: int = Field(default=0)
    deleted_images: int = Field(default=0)
    deleted_videos: int = Field(default=0)
    retained_images: int = Field(default=0, description="Targets the final check kept: promoted, protected or gone")
    retained_videos: int = Field(default=0)
    failed_images: int = Field(default=0, description="Targets whose deletion raised; a new cleanup picks them up")
    failed_videos: int = Field(default=0)
    reclaimed_bytes: int = Field(default=0, description="Bytes whose files are confirmed removed")
    unknown_size_count: int = Field(
        default=0, description="Deleted items whose size was never measured; their bytes are not in reclaimed_bytes"
    )
    pending_disk_cleanup: int = Field(
        default=0, description="Deleted records whose files could not be purged yet; the journal retries at startup"
    )


class IntermediatesOperation(BaseModel):
    """A cleanup run. Operations live in server memory: a restart forgets them, and the live policy
    is the retry, so a new preview and confirmation picks up whatever an interrupted run left."""

    operation_id: str
    user_id: str = Field(description="The account that confirmed the operation")
    mode: IntermediatesCleanupMode
    scope: IntermediatesScope = Field(description="The scope as requested, so a client can request it again")
    status: IntermediatesOperationStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = Field(default=None, description="Why the operation stopped, when it failed")
    target_images: int = Field(
        description=(
            "Image deletions the preview expected. The scope is paged live, so rows that became deletable"
            " since the preview are collected too and the deleted count may exceed this"
        )
    )
    target_videos: int = Field(description="Video deletions the preview expected; see `target_images`")
    progress: IntermediatesOperationProgress


class IntermediatesOperationList(BaseModel):
    items: list[IntermediatesOperation] = Field(description="The caller's retained operations, newest first")


class IntermediatesPreviewNotFoundError(Exception):
    """The preview expired, was consumed, or belongs to another caller."""


class IntermediatesOperationNotFoundError(Exception):
    pass


class IntermediatesScopeForbiddenError(Exception):
    """The caller may not target the requested rows."""


class IntermediatesScopeInvalidError(Exception):
    """The scope is malformed for its kind."""


class IntermediatesUnavailableError(Exception):
    """Cleanup cannot run right now, e.g. image storage maintenance is active."""
