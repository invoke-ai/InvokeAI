from typing import Optional
from urllib.parse import quote_plus

from invokeai.app.services.canvas_project_files.canvas_project_files_common import (
    CanvasProjectFileDeleteException,
    CanvasProjectFileSaveException,
)
from invokeai.app.services.canvas_project_records.canvas_project_records_common import (
    CanvasProjectNamesResult,
    CanvasProjectRecord,
    CanvasProjectRecordChanges,
    CanvasProjectRecordDeleteException,
    CanvasProjectRecordNotFoundException,
    CanvasProjectRecordSaveException,
)
from invokeai.app.services.canvas_projects.canvas_projects_base import CanvasProjectServiceABC
from invokeai.app.services.canvas_projects.canvas_projects_common import CanvasProjectDTO, canvas_project_record_to_dto
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    InvalidOriginException,
    ResourceOrigin,
)
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection


class CanvasProjectService(CanvasProjectServiceABC):
    __invoker: Invoker

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

    def create(
        self,
        zip_bytes: bytes,
        name: str,
        app_version: str,
        width: int,
        height: int,
        image_count: int,
        thumbnail_bytes: Optional[bytes] = None,
        project_origin: ResourceOrigin = ResourceOrigin.INTERNAL,
        board_id: Optional[str] = None,
        is_intermediate: Optional[bool] = False,
        starred: Optional[bool] = False,
        user_id: Optional[str] = None,
    ) -> CanvasProjectDTO:
        if project_origin not in ResourceOrigin:
            raise InvalidOriginException

        project_name = self.__invoker.services.names.create_canvas_project_name()

        # Reuse the image subfolder strategy for canvas project organization.
        from invokeai.app.services.image_files.image_subfolder_strategy import create_subfolder_strategy

        strategy_name = self.__invoker.services.configuration.image_subfolder_strategy
        strategy = create_subfolder_strategy(strategy_name)
        # Canvas projects don't have a true category — use GENERAL as a stand-in so TypeStrategy
        # bucket them alongside general assets rather than under a dedicated folder.
        project_subfolder = strategy.get_subfolder(project_name, ImageCategory.GENERAL, is_intermediate or False)

        try:
            self.__invoker.services.canvas_project_records.save(
                project_name=project_name,
                project_origin=project_origin,
                name=name,
                app_version=app_version,
                width=width,
                height=height,
                image_count=image_count,
                has_thumbnail=thumbnail_bytes is not None,
                is_intermediate=is_intermediate,
                starred=starred,
                user_id=user_id,
                project_subfolder=project_subfolder,
            )
            if board_id is not None:
                try:
                    self.__invoker.services.board_canvas_project_records.add_project_to_board(
                        board_id=board_id, project_name=project_name
                    )
                except Exception as e:
                    self.__invoker.services.logger.warning(
                        f"Failed to add canvas project to board {board_id}: {str(e)}"
                    )

            self.__invoker.services.canvas_project_files.save(
                zip_bytes=zip_bytes,
                project_name=project_name,
                thumbnail_bytes=thumbnail_bytes,
                project_subfolder=project_subfolder,
            )

            project_dto = self.get_dto(project_name)
            self._on_changed(project_dto)
            return project_dto
        except CanvasProjectRecordSaveException:
            self.__invoker.services.logger.error("Failed to save canvas project record")
            raise
        except CanvasProjectFileSaveException:
            self.__invoker.services.logger.error("Failed to save canvas project file")
            raise
        except Exception as e:
            self.__invoker.services.logger.error(f"Problem saving canvas project record and file: {str(e)}")
            raise e

    def update(self, project_name: str, changes: CanvasProjectRecordChanges) -> CanvasProjectDTO:
        try:
            self.__invoker.services.canvas_project_records.update(project_name, changes)
            project_dto = self.get_dto(project_name)
            self._on_changed(project_dto)
            return project_dto
        except CanvasProjectRecordSaveException:
            self.__invoker.services.logger.error("Failed to update canvas project record")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem updating canvas project record")
            raise e

    def replace_file(
        self,
        project_name: str,
        zip_bytes: bytes,
        width: int,
        height: int,
        image_count: int,
        app_version: str,
        thumbnail_bytes: Optional[bytes] = None,
    ) -> CanvasProjectDTO:
        try:
            record = self.__invoker.services.canvas_project_records.get(project_name)

            # Overwrite the ZIP (and thumbnail if provided) on disk. The files service `save()`
            # writes through, so no manual delete is needed.
            self.__invoker.services.canvas_project_files.save(
                zip_bytes=zip_bytes,
                project_name=project_name,
                thumbnail_bytes=thumbnail_bytes,
                project_subfolder=record.project_subfolder,
            )

            # When a caller updates a project without supplying a fresh thumbnail, we keep the
            # existing one — `has_thumbnail` then mirrors the record's previous value.
            has_thumbnail = thumbnail_bytes is not None or record.has_thumbnail

            self.__invoker.services.canvas_project_records.update_file_metadata(
                project_name=project_name,
                width=width,
                height=height,
                image_count=image_count,
                has_thumbnail=has_thumbnail,
                app_version=app_version,
            )

            project_dto = self.get_dto(project_name)
            self._on_changed(project_dto)
            return project_dto
        except CanvasProjectRecordNotFoundException:
            self.__invoker.services.logger.error("Canvas project record not found")
            raise
        except CanvasProjectFileSaveException:
            self.__invoker.services.logger.error("Failed to write canvas project file")
            raise
        except CanvasProjectRecordSaveException:
            self.__invoker.services.logger.error("Failed to update canvas project record")
            raise
        except Exception as e:
            self.__invoker.services.logger.error(f"Problem replacing canvas project file: {str(e)}")
            raise e

    def get_record(self, project_name: str) -> CanvasProjectRecord:
        try:
            return self.__invoker.services.canvas_project_records.get(project_name)
        except CanvasProjectRecordNotFoundException:
            self.__invoker.services.logger.error("Canvas project record not found")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting canvas project record")
            raise e

    def get_dto(self, project_name: str) -> CanvasProjectDTO:
        try:
            record = self.__invoker.services.canvas_project_records.get(project_name)
            # Cache-buster: append `updated_at` as a query param so the browser refetches the ZIP
            # and thumbnail after an in-place replace. The path itself is stable (UUID-based), so
            # without this the browser would keep serving the stale cached bytes.
            version = quote_plus(str(record.updated_at))
            project_url = f"{self.__invoker.services.urls.get_canvas_project_url(project_name)}?v={version}"
            thumbnail_url: Optional[str] = None
            if record.has_thumbnail:
                base_thumb = self.__invoker.services.urls.get_canvas_project_url(project_name, thumbnail=True)
                thumbnail_url = f"{base_thumb}?v={version}"
            return canvas_project_record_to_dto(
                project_record=record,
                project_url=project_url,
                thumbnail_url=thumbnail_url,
                board_id=self.__invoker.services.board_canvas_project_records.get_board_for_project(project_name),
            )
        except CanvasProjectRecordNotFoundException:
            self.__invoker.services.logger.error("Canvas project record not found")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting canvas project DTO")
            raise e

    def get_path(self, project_name: str, thumbnail: bool = False) -> str:
        try:
            record = self.__invoker.services.canvas_project_records.get(project_name)
            return str(
                self.__invoker.services.canvas_project_files.get_path(
                    project_name, thumbnail=thumbnail, project_subfolder=record.project_subfolder
                )
            )
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting canvas project path")
            raise e

    def get_url(self, project_name: str, thumbnail: bool = False) -> str:
        try:
            return self.__invoker.services.urls.get_canvas_project_url(project_name, thumbnail=thumbnail)
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting canvas project URL")
            raise e

    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        project_origin: Optional[ResourceOrigin] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> OffsetPaginatedResults[CanvasProjectDTO]:
        try:
            results = self.__invoker.services.canvas_project_records.get_many(
                offset=offset,
                limit=limit,
                starred_first=starred_first,
                order_dir=order_dir,
                project_origin=project_origin,
                is_intermediate=is_intermediate,
                board_id=board_id,
                search_term=search_term,
                user_id=user_id,
                is_admin=is_admin,
            )
            project_dtos: list[CanvasProjectDTO] = []
            for r in results.items:
                # Cache-buster: same scheme as get_dto so the listing also surfaces fresh URLs
                # after an in-place project file replace.
                version = quote_plus(str(r.updated_at))
                project_url = f"{self.__invoker.services.urls.get_canvas_project_url(r.project_name)}?v={version}"
                thumbnail_url: Optional[str] = None
                if r.has_thumbnail:
                    base_thumb = self.__invoker.services.urls.get_canvas_project_url(
                        r.project_name, thumbnail=True
                    )
                    thumbnail_url = f"{base_thumb}?v={version}"
                project_dtos.append(
                    canvas_project_record_to_dto(
                        project_record=r,
                        project_url=project_url,
                        thumbnail_url=thumbnail_url,
                        board_id=self.__invoker.services.board_canvas_project_records.get_board_for_project(
                            r.project_name
                        ),
                    )
                )
            return OffsetPaginatedResults[CanvasProjectDTO](
                items=project_dtos, offset=results.offset, limit=results.limit, total=results.total
            )
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting paginated canvas project DTOs")
            raise e

    def delete(self, project_name: str) -> None:
        try:
            record = self.__invoker.services.canvas_project_records.get(project_name)
            self.__invoker.services.canvas_project_files.delete(
                project_name, project_subfolder=record.project_subfolder
            )
            self.__invoker.services.canvas_project_records.delete(project_name)
            self._on_deleted(project_name)
        except CanvasProjectRecordDeleteException:
            self.__invoker.services.logger.error("Failed to delete canvas project record")
            raise
        except CanvasProjectFileDeleteException:
            self.__invoker.services.logger.error("Failed to delete canvas project file")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem deleting canvas project record and file")
            raise e

    def delete_projects_on_board(self, board_id: str) -> None:
        try:
            project_names = (
                self.__invoker.services.board_canvas_project_records.get_all_board_project_names_for_board(
                    board_id, is_intermediate=None
                )
            )
            for project_name in project_names:
                try:
                    record = self.__invoker.services.canvas_project_records.get(project_name)
                    self.__invoker.services.canvas_project_files.delete(
                        project_name, project_subfolder=record.project_subfolder
                    )
                except Exception:
                    pass
            self.__invoker.services.canvas_project_records.delete_many(project_names)
            for project_name in project_names:
                self._on_deleted(project_name)
        except CanvasProjectRecordDeleteException:
            self.__invoker.services.logger.error("Failed to delete canvas project records")
            raise
        except CanvasProjectFileDeleteException:
            self.__invoker.services.logger.error("Failed to delete canvas project files")
            raise
        except Exception as e:
            self.__invoker.services.logger.error(f"Problem deleting canvas project records and files: {str(e)}")
            raise e

    def get_project_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        project_origin: Optional[ResourceOrigin] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> CanvasProjectNamesResult:
        try:
            return self.__invoker.services.canvas_project_records.get_project_names(
                starred_first=starred_first,
                order_dir=order_dir,
                project_origin=project_origin,
                is_intermediate=is_intermediate,
                board_id=board_id,
                search_term=search_term,
                user_id=user_id,
                is_admin=is_admin,
            )
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting canvas project names")
            raise e
