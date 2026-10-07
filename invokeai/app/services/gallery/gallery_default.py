from typing import Any, Optional, Sequence

from invokeai.app.services.gallery.gallery_base import GalleryServiceABC
from invokeai.app.services.gallery.gallery_common import (
    BoardMediaSummary,
    GalleryItem,
    GalleryItemKind,
    GalleryItemNames,
    GalleryItemNamesResult,
    GalleryItemRef,
)
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.database.database import Database
from invokeai.app.services.shared.database.queries.gallery import Filters
from invokeai.app.services.shared.pagination import OffsetPaginatedResults, SQLiteDirection
from invokeai.app.services.video_records.video_records_common import coerce_media_origin
from invokeai.app.services.virtual_boards.virtual_boards_common import VirtualSubBoardDTO


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
