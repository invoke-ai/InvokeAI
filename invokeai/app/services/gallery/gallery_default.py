from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from invokeai.app.services.board_access.board_access_common import ItemAction, item_allows
from invokeai.app.services.gallery.gallery_base import GalleryMediaKind, GalleryServiceABC
from invokeai.app.services.gallery.gallery_common import (
    MAX_USAGE_DOCUMENTS,
    MAX_USAGE_NAMES,
    BoardMediaSummary,
    GalleryItem,
    GalleryItemKind,
    GalleryItemLocation,
    GalleryItemNames,
    GalleryItemNamesResult,
    GalleryItemRef,
    GalleryMediaUsage,
    GalleryMediaUsageDocument,
    GalleryMediaUsageDocumentKind,
    GalleryMediaUsageOthers,
    GalleryMediaUsageQueue,
)
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries import Queries
from invokeai.app.services.shared.database.queries.gallery import Filters
from invokeai.app.services.shared.media_references import MediaKind, MediaReferences
from invokeai.app.services.shared.pagination import OffsetPaginatedResults, SQLiteDirection
from invokeai.app.services.video_records.video_records_common import coerce_media_origin
from invokeai.app.services.virtual_boards.virtual_boards_common import VirtualSubBoardDTO

# A reference index owner: (owner kind, owner account, owner id).
_Owner = tuple[str, str, str]

_LEGACY_EDITOR_ID = "legacy_editor"


def _document_of(owner_kind: str, owner_id: str) -> tuple[GalleryMediaUsageDocumentKind, str]:
    """The document a reference owner belongs to, as the usage check lists it: a project's content and its
    workspaces are one project, and the legacy editor's client state is one document however many keys it uses."""
    if owner_kind in ("project", "project_workspace"):
        return "project", owner_id
    if owner_kind == "client_state":
        return "legacy_editor", _LEGACY_EDITOR_ID
    if owner_kind == "workflow":
        return "workflow", owner_id
    return "quarantined_project", owner_id


def _add(references: MediaReferences, kind: MediaKind, name: str) -> None:
    (references.images if kind == "image" else references.videos).add(name)


@dataclass
class _DocumentUse:
    names: MediaReferences = field(default_factory=MediaReferences)
    owners: set[_Owner] = field(default_factory=set)


def _capped(names: set[str]) -> list[str]:
    return sorted(names)[:MAX_USAGE_NAMES]


class GalleryService(GalleryServiceABC):
    """A gallery of images and videos as one list (see `queries/gallery.py`)."""

    __invoker: Invoker

    def __init__(self, database: Database) -> None:
        super().__init__()
        self._queries = database.queries

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

    def list_items(
        self,
        offset: int = 0,
        limit: int = 10,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        starred: Optional[bool] = None,
        media_kind: Optional[GalleryMediaKind] = None,
    ) -> OffsetPaginatedResults[GalleryItem]:
        filters = Filters(
            origin=origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=user_id,
            is_admin=is_admin,
            created_from=created_from,
            created_to=created_to,
            starred=starred,
            media_kind=media_kind,
        )
        rows, total = self._queries.gallery.page(
            filters,
            offset=offset,
            limit=limit,
            starred_first=starred_first,
            descending=order_dir == SQLiteDirection.Descending,
        )
        items = [self._to_item(row) for row in rows]
        return OffsetPaginatedResults[GalleryItem](items=items, offset=offset, limit=limit, total=total)

    def get_item_location(
        self,
        kind: GalleryItemKind,
        name: str,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        starred: Optional[bool] = None,
        media_kind: Optional[GalleryMediaKind] = None,
    ) -> Optional[GalleryItemLocation]:
        """Finds item's exact rank without building or returning the full name list."""
        filters = Filters(
            origin=origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=user_id,
            is_admin=is_admin,
            created_from=created_from,
            created_to=created_to,
            starred=starred,
            media_kind=media_kind,
        )
        row = self._queries.gallery.location(
            filters, kind=kind.value, name=name, descending=order_dir == SQLiteDirection.Descending
        )
        if row is None:
            return None
        found_kind, found_name, index, total = row
        return GalleryItemLocation(
            kind=GalleryItemKind(found_kind), name=found_name, index=int(index), total=int(total)
        )

    def _names(self, filters: Filters, starred_first: bool, order_dir: SQLiteDirection) -> tuple[Sequence[Any], int]:
        """The ordered (kind, name, starred) rows and the starred count. Shared by both name-list shapes so the
        deprecated `(kind, name)` variant and the flat one can never drift apart in ordering or filtering."""
        rows = self._queries.gallery.names(
            filters, starred_first=starred_first, descending=order_dir == SQLiteDirection.Descending
        )
        return rows, (sum(1 for row in rows if row[2]) if starred_first else 0)

    def list_item_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
        created_date: Optional[str] = None,
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        starred: Optional[bool] = None,
        media_kind: Optional[GalleryMediaKind] = None,
    ) -> GalleryItemNamesResult:
        filters = Filters(
            origin=origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=user_id,
            is_admin=is_admin,
            created_date=created_date,
            created_from=created_from,
            created_to=created_to,
            starred=starred,
            media_kind=media_kind,
        )
        rows, starred_count = self._names(filters, starred_first, order_dir)
        refs = [GalleryItemRef(kind=GalleryItemKind(kind), name=name) for kind, name, _ in rows]
        return GalleryItemNamesResult(items=refs, starred_count=starred_count, total_count=len(refs))

    def get_item_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
        created_date: Optional[str] = None,
        created_from: Optional[str] = None,
        created_to: Optional[str] = None,
        starred: Optional[bool] = None,
        media_kind: Optional[GalleryMediaKind] = None,
    ) -> GalleryItemNames:
        filters = Filters(
            origin=origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=user_id,
            is_admin=is_admin,
            created_date=created_date,
            created_from=created_from,
            created_to=created_to,
            starred=starred,
            media_kind=media_kind,
        )
        rows, starred_count = self._names(filters, starred_first, order_dir)
        # The raw names, deliberately: building one model per row is what made the deprecated variant expensive.
        names = [name for _, name, _ in rows]
        return GalleryItemNames(item_names=names, starred_count=starred_count, total_count=len(names))

    def get_dates(self, user_id: Optional[str] = None, is_admin: bool = False) -> list[VirtualSubBoardDTO]:
        counts, covers = self._queries.gallery.date_counts(user_id if user_id is not None and not is_admin else None)
        boards: list[VirtualSubBoardDTO] = []
        for day in counts:
            cover_kind, cover_name = covers.get(day.day, (None, None))
            boards.append(
                VirtualSubBoardDTO(
                    virtual_board_id=f"by_date:{day.day}",
                    board_name=day.day,
                    date=day.day,
                    image_count=day.images,
                    asset_count=day.assets,
                    video_count=day.videos,
                    asset_video_count=day.asset_videos,
                    cover_image_name=cover_name if cover_kind == "image" else None,
                    cover_video_name=cover_name if cover_kind == "video" else None,
                )
            )
        return boards

    def get_board_media_summaries(self, board_ids: list[str]) -> dict[str, BoardMediaSummary]:
        summaries = {board_id: BoardMediaSummary() for board_id in board_ids}
        if not board_ids:
            return summaries
        for summary in self._queries.gallery.board_summaries(board_ids):
            summaries[summary.board_id] = BoardMediaSummary(
                cover_image_name=summary.cover_image_name,
                cover_video_name=summary.cover_video_name,
                image_count=summary.images,
                video_count=summary.videos,
                asset_count=summary.assets,
                asset_video_count=summary.asset_videos,
            )
        return summaries

    def get_media_usage(
        self, image_names: Collection[str], video_names: Collection[str], user_id: str, is_admin: bool
    ) -> GalleryMediaUsage:
        checked = MediaReferences(
            images=self._deletable("image", image_names, user_id, is_admin),
            videos=self._deletable("video", video_names, user_id, is_admin),
        )
        return self._usage(checked, user_id)

    def get_board_media_usage(self, board_id: str, user_id: str, contributor_id: Optional[str]) -> GalleryMediaUsage:
        # The enumeration `DELETE /boards/{id}?include_images=true` deletes.
        services = self.__invoker.services
        images = services.board_image_records.get_all_board_image_names_for_board(
            board_id, None, None, user_id=contributor_id
        )
        videos = services.board_video_records.get_all_board_video_names_for_board(
            board_id, None, None, user_id=contributor_id
        )
        return self._usage(MediaReferences(images=set(images), videos=set(videos)), user_id)

    def _deletable(self, kind: MediaKind, names: Collection[str], user_id: str, is_admin: bool) -> set[str]:
        """The existing named items the caller may delete, in one query. The delete endpoints decide each item with
        `item_allows`; no board role grants a permanent delete, so the board an item is on does not enter it."""
        if not names:
            return set()
        return {
            name
            for name, owner in self._queries.gallery.item_owners(kind, set(names))
            if item_allows(is_item_owner=owner == user_id, is_admin=is_admin, board=None, action=ItemAction.Delete)
        }

    def _usage(self, checked: MediaReferences, user_id: str) -> GalleryMediaUsage:
        """Who uses the checked media: the caller's documents and queue work by name, everyone else's as counts."""
        by_kind: tuple[tuple[MediaKind, set[str]], ...] = (("image", checked.images), ("video", checked.videos))

        def read(
            q: Queries,
        ) -> tuple[list[tuple[MediaKind, str, _Owner]], dict[_Owner, str], dict[str, tuple[int, Optional[int]]]]:
            rows = [
                (kind, media, (owner_kind, owner_user, owner_id))
                for kind, names in by_kind
                if names
                for media, owner_kind, owner_user, owner_id in q.media_references.owners(kind, names)
            ]
            callers = {owner for _, _, owner in rows if owner[1] == user_id}
            # Only the caller's own documents are named. A project's revisions are read in the same transaction as
            # its references, which are written in the transaction that saves it: they describe the state the names
            # were read from, which a client compares with the copy it holds.
            projects = sorted(
                {owner_id for owner_kind, _, owner_id in callers if _document_of(owner_kind, owner_id)[0] == "project"}
            )
            return rows, q.media_references.document_names(callers), q.projects.revisions(user_id, projects)

        rows, document_names, revisions = (
            self._queries.run(read, read_only=True) if not checked.is_empty() else ([], {}, {})
        )
        active = self.__invoker.services.intermediates.get_active_queue_uses(checked)

        in_use = MediaReferences()
        mine: dict[tuple[GalleryMediaUsageDocumentKind, str], _DocumentUse] = {}
        other_documents: set[tuple[GalleryMediaUsageDocumentKind, str, str]] = set()
        other_media = MediaReferences()
        for kind, media, owner in rows:
            owner_kind, owner_user, owner_id = owner
            document = _document_of(owner_kind, owner_id)
            _add(in_use, kind, media)
            if owner_user == user_id:
                use = mine.setdefault(document, _DocumentUse())
                _add(use.names, kind, media)
                use.owners.add(owner)
            else:
                # Another account's workspace in the caller's project is that account's document.
                other_documents.add((document[0], owner_user, document[1]))
                _add(other_media, kind, media)

        queue_items = 0
        queue_media = MediaReferences()
        other_queue_items = 0
        for item in active:
            # As the queue's own owner checks: an item stored without an owner is no caller's.
            if item.user_id == user_id:
                queue_items += 1
                target = queue_media
            else:
                other_queue_items += 1
                target = other_media
            for names in (in_use, target):
                names.images |= item.images
                names.videos |= item.videos

        documents: list[GalleryMediaUsageDocument] = []
        for (kind, document_id), use in mine.items():
            revision, workspace_revision = (
                revisions.get(document_id, (None, None)) if kind == "project" else (None, None)
            )
            documents.append(
                GalleryMediaUsageDocument(
                    kind=kind,
                    id=document_id,
                    # A project's own name before a workspace's: sorted owners put `project` first.
                    name=next((document_names[o] for o in sorted(use.owners) if o in document_names), None),
                    image_names=_capped(use.names.images),
                    video_names=_capped(use.names.videos),
                    image_count=len(use.names.images),
                    video_count=len(use.names.videos),
                    revision=revision,
                    workspace_revision=workspace_revision,
                )
            )
        documents.sort(key=lambda d: (d.kind, (d.name or "").casefold(), d.id))
        return GalleryMediaUsage(
            checked_image_count=len(checked.images),
            checked_video_count=len(checked.videos),
            in_use_image_count=len(in_use.images),
            in_use_video_count=len(in_use.videos),
            documents=documents[:MAX_USAGE_DOCUMENTS],
            documents_total=len(documents),
            queue=GalleryMediaUsageQueue(
                item_count=queue_items,
                image_names=_capped(queue_media.images),
                video_names=_capped(queue_media.videos),
                image_count=len(queue_media.images),
                video_count=len(queue_media.videos),
            ),
            other_accounts=GalleryMediaUsageOthers(
                document_count=len(other_documents),
                queue_item_count=other_queue_items,
                image_count=len(other_media.images),
                video_count=len(other_media.videos),
            ),
        )

    def _to_item(self, row: Sequence[Any]) -> GalleryItem:
        kind, name, width, height, category, starred, is_intermediate, board_id, created_at, duration, fps, origin = row
        urls = self.__invoker.services.urls
        item_kind = GalleryItemKind(kind)
        if item_kind == GalleryItemKind.IMAGE:
            full_url, thumbnail_url = urls.get_image_url(name), urls.get_image_url(name, thumbnail=True)
            duration = fps = media_origin = None
        else:
            full_url, thumbnail_url = urls.get_video_url(name), urls.get_video_url(name, thumbnail=True)
            media_origin = coerce_media_origin(origin)
        return GalleryItem(
            kind=item_kind,
            name=name,
            full_url=full_url,
            thumbnail_url=thumbnail_url,
            width=width,
            height=height,
            category=ImageCategory(category),
            starred=bool(starred),
            is_intermediate=bool(is_intermediate),
            board_id=board_id,
            created_at=created_at,
            duration=duration,
            fps=fps,
            media_origin=media_origin,
        )
