import sqlite3
from typing import Optional, Union, cast

from invokeai.app.services.gallery.gallery_base import GalleryServiceABC
from invokeai.app.services.gallery.gallery_common import (
    GalleryItem,
    GalleryItemKind,
    GalleryItemNamesResult,
    GalleryItemRef,
)
from invokeai.app.services.image_records.image_records_common import ImageCategory, ResourceOrigin
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase


class SqliteGalleryService(GalleryServiceABC):
    """Implements a polymorphic gallery via UNION ALL across the `images` and `videos` tables.

    Filters are applied identically on each half. The two halves expose a common column set so
    the result is shape-compatible (a literal `kind` discriminator + a `name` alias + duration/fps
    that are NULL for images).
    """

    __invoker: Invoker

    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

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
    ) -> OffsetPaginatedResults[GalleryItem]:
        image_half, image_params, image_count_query = self._build_half(
            kind="image",
            origin=origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=user_id,
            is_admin=is_admin,
        )
        video_half, video_params, video_count_query = self._build_half(
            kind="video",
            origin=origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=user_id,
            is_admin=is_admin,
        )
        project_half, project_params, project_count_query = self._build_half(
            kind="canvas_project",
            origin=origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=user_id,
            is_admin=is_admin,
        )

        if starred_first:
            order_clause = f"ORDER BY starred DESC, created_at {order_dir.value}"
        else:
            order_clause = f"ORDER BY created_at {order_dir.value}"

        union_query = f"""--sql
        SELECT * FROM (
            {image_half}
            UNION ALL
            {video_half}
            UNION ALL
            {project_half}
        )
        {order_clause}
        LIMIT ? OFFSET ?
        ;
        """

        with self._db.transaction() as cursor:
            cursor.execute(union_query, image_params + video_params + project_params + [limit, offset])
            rows = cast(list[sqlite3.Row], cursor.fetchall())

            cursor.execute(image_count_query, image_params)
            image_count = cast(int, cursor.fetchone()[0])
            cursor.execute(video_count_query, video_params)
            video_count = cast(int, cursor.fetchone()[0])
            cursor.execute(project_count_query, project_params)
            project_count = cast(int, cursor.fetchone()[0])

        urls = self.__invoker.services.urls
        items = [self._row_to_item(row, urls) for row in rows]
        return OffsetPaginatedResults[GalleryItem](
            items=items,
            offset=offset,
            limit=limit,
            total=image_count + video_count + project_count,
        )

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
    ) -> GalleryItemNamesResult:
        image_half, image_params, _ = self._build_half(
            kind="image",
            origin=origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=user_id,
            is_admin=is_admin,
            names_only=True,
        )
        video_half, video_params, _ = self._build_half(
            kind="video",
            origin=origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=user_id,
            is_admin=is_admin,
            names_only=True,
        )
        project_half, project_params, _ = self._build_half(
            kind="canvas_project",
            origin=origin,
            categories=categories,
            is_intermediate=is_intermediate,
            board_id=board_id,
            search_term=search_term,
            user_id=user_id,
            is_admin=is_admin,
            names_only=True,
        )

        if starred_first:
            order_clause = f"ORDER BY starred DESC, created_at {order_dir.value}"
        else:
            order_clause = f"ORDER BY created_at {order_dir.value}"

        union_query = f"""--sql
        SELECT * FROM (
            {image_half}
            UNION ALL
            {video_half}
            UNION ALL
            {project_half}
        )
        {order_clause}
        ;
        """

        with self._db.transaction() as cursor:
            cursor.execute(union_query, image_params + video_params + project_params)
            rows = cast(list[sqlite3.Row], cursor.fetchall())

            starred_count = 0
            if starred_first:
                starred_count = sum(1 for r in rows if r["starred"])

        refs = [GalleryItemRef(kind=GalleryItemKind(row["kind"]), name=row["name"]) for row in rows]
        return GalleryItemNamesResult(items=refs, starred_count=starred_count, total_count=len(refs))

    def _build_half(
        self,
        kind: str,
        origin: Optional[ResourceOrigin],
        categories: Optional[list[ImageCategory]],
        is_intermediate: Optional[bool],
        board_id: Optional[str],
        search_term: Optional[str],
        user_id: Optional[str],
        is_admin: bool,
        names_only: bool = False,
    ) -> tuple[str, list[Union[int, str, bool]], str]:
        """Builds one half of the union (either `images` or `videos`).

        Returns `(query_with_select, params, count_query)`. Both halves emit the same columns so
        UNION ALL is shape-compatible: `kind`, `name`, `width`, `height`, `category`, `starred`,
        `is_intermediate`, `board_id`, `created_at`, `duration`, `fps`.

        `names_only=True` selects only `kind`, `name`, `starred`, `created_at` (the minimum needed
        for ordering + the counts result).
        """
        # Canvas projects have no `category` column — they're conceptually GENERAL — and no
        # `metadata` column. We project literals to keep the UNION shape-compatible and skip the
        # category filter / metadata search on this half.
        is_canvas_project = kind == "canvas_project"

        if kind == "image":
            base_table = "images"
            join_table = "board_images"
            name_col = "image_name"
            category_col = "image_category"
            origin_col = "image_origin"
            duration_expr = "NULL"
            fps_expr = "NULL"
            image_count_expr = "NULL"
            category_select = f"{base_table}.{category_col}"
        elif kind == "video":
            base_table = "videos"
            join_table = "board_videos"
            name_col = "video_name"
            category_col = "video_category"
            origin_col = "video_origin"
            duration_expr = f"{base_table}.duration"
            fps_expr = f"{base_table}.fps"
            image_count_expr = "NULL"
            category_select = f"{base_table}.{category_col}"
        elif kind == "canvas_project":
            base_table = "canvas_projects"
            join_table = "board_canvas_projects"
            name_col = "project_name"
            origin_col = "project_origin"
            duration_expr = "NULL"
            fps_expr = "NULL"
            image_count_expr = f"{base_table}.image_count"
            # Canvas projects don't have a category column — emit a literal so the UNION matches.
            category_select = f"'{ImageCategory.GENERAL.value}'"
            category_col = None  # unused, projects bypass the category filter
        else:
            raise ValueError(f"Unknown kind: {kind}")

        if names_only:
            select_cols = (
                f"'{kind}' AS kind, "
                f"{base_table}.{name_col} AS name, "
                f"{base_table}.starred AS starred, "
                f"{base_table}.created_at AS created_at"
            )
        else:
            select_cols = (
                f"'{kind}' AS kind, "
                f"{base_table}.{name_col} AS name, "
                f"{base_table}.width AS width, "
                f"{base_table}.height AS height, "
                f"{category_select} AS category, "
                f"{base_table}.starred AS starred, "
                f"{base_table}.is_intermediate AS is_intermediate, "
                f"{join_table}.board_id AS board_id, "
                f"{base_table}.created_at AS created_at, "
                f"{duration_expr} AS duration, "
                f"{fps_expr} AS fps, "
                f"{image_count_expr} AS image_count"
            )

        from_clause = f"FROM {base_table} LEFT JOIN {join_table} ON {join_table}.{name_col} = {base_table}.{name_col}"

        conditions = ""
        params: list[Union[int, str, bool]] = []

        if origin is not None:
            conditions += f" AND {base_table}.{origin_col} = ? "
            params.append(origin.value)

        if categories is not None and category_col is not None:
            # Canvas projects don't carry a category — when callers filter by category, they
            # implicitly exclude projects unless GENERAL is in the list. Make that explicit.
            category_strings = [c.value for c in set(categories)]
            placeholders = ",".join("?" * len(category_strings))
            conditions += f" AND {base_table}.{category_col} IN ( {placeholders} ) "
            for c in category_strings:
                params.append(c)
        elif categories is not None and is_canvas_project:
            # GENERAL not in categories → exclude all canvas projects.
            if ImageCategory.GENERAL not in set(categories):
                conditions += " AND 1=0 "

        if is_intermediate is not None:
            conditions += f" AND {base_table}.is_intermediate = ? "
            params.append(is_intermediate)

        if board_id == "none":
            conditions += f" AND {join_table}.board_id IS NULL "
            if user_id is not None and not is_admin:
                conditions += f" AND {base_table}.user_id = ? "
                params.append(user_id)
        elif board_id is not None:
            conditions += f" AND {join_table}.board_id = ? "
            params.append(board_id)
        elif user_id is not None and not is_admin:
            # No board_id supplied — still enforce per-user isolation so
            # non-admin callers cannot enumerate other users' items.
            conditions += f" AND {base_table}.user_id = ? "
            params.append(user_id)

        if search_term:
            if is_canvas_project:
                # No `metadata` column on canvas_projects — search the `name` field instead.
                conditions += f" AND ({base_table}.name LIKE ? OR {base_table}.created_at LIKE ?) "
            else:
                conditions += f" AND ({base_table}.metadata LIKE ? OR {base_table}.created_at LIKE ?) "
            params.append(f"%{search_term.lower()}%")
            params.append(f"%{search_term.lower()}%")

        half_query = f"SELECT {select_cols} {from_clause} WHERE 1=1 {conditions}"
        count_query = f"SELECT COUNT(*) {from_clause} WHERE 1=1 {conditions}"
        return half_query, params, count_query

    def _row_to_item(self, row: sqlite3.Row, urls) -> GalleryItem:
        kind = GalleryItemKind(row["kind"])
        name = row["name"]
        duration: Optional[float] = None
        fps: Optional[float] = None
        image_count: Optional[int] = None
        if kind == GalleryItemKind.IMAGE:
            full_url = urls.get_image_url(name)
            thumbnail_url = urls.get_image_url(name, thumbnail=True)
        elif kind == GalleryItemKind.VIDEO:
            full_url = urls.get_video_url(name)
            thumbnail_url = urls.get_video_url(name, thumbnail=True)
            duration = row["duration"]
            fps = row["fps"]
        else:
            full_url = urls.get_canvas_project_url(name)
            thumbnail_url = urls.get_canvas_project_url(name, thumbnail=True)
            image_count = row["image_count"]
        return GalleryItem(
            kind=kind,
            name=name,
            full_url=full_url,
            thumbnail_url=thumbnail_url,
            width=row["width"],
            height=row["height"],
            category=ImageCategory(row["category"]),
            starred=bool(row["starred"]),
            is_intermediate=bool(row["is_intermediate"]),
            board_id=row["board_id"],
            created_at=row["created_at"],
            duration=duration,
            fps=fps,
            image_count=image_count,
        )
