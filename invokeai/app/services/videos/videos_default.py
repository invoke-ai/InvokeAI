from pathlib import Path
from typing import Optional

from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    InvalidImageCategoryException,
    InvalidOriginException,
    ResourceOrigin,
)
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.shared.sqlite.sqlite_common import SQLiteDirection
from invokeai.app.services.video_files.video_files_common import (
    VideoFileDeleteException,
    VideoFileNotFoundException,
    VideoFileSaveException,
)
from invokeai.app.services.video_records.video_records_common import (
    VideoNamesResult,
    VideoRecord,
    VideoRecordChanges,
    VideoRecordDeleteException,
    VideoRecordNotFoundException,
    VideoRecordSaveException,
)
from invokeai.app.services.videos.videos_base import VideoServiceABC
from invokeai.app.services.videos.videos_common import VideoDTO, video_record_to_dto


class VideoService(VideoServiceABC):
    __invoker: Invoker

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

    def create(
        self,
        source_path: Path,
        width: int,
        height: int,
        duration: float,
        fps: Optional[float],
        video_origin: ResourceOrigin,
        video_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        board_id: Optional[str] = None,
        is_intermediate: Optional[bool] = False,
        metadata: Optional[str] = None,
        workflow: Optional[str] = None,
        graph: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> VideoDTO:
        if video_origin not in ResourceOrigin:
            raise InvalidOriginException
        if video_category not in ImageCategory:
            raise InvalidImageCategoryException

        video_name = self.__invoker.services.names.create_video_name()

        # Reuse the image subfolder strategy for video organization.
        from invokeai.app.services.image_files.image_subfolder_strategy import create_subfolder_strategy

        strategy_name = self.__invoker.services.configuration.image_subfolder_strategy
        strategy = create_subfolder_strategy(strategy_name)
        video_subfolder = strategy.get_subfolder(video_name, video_category, is_intermediate or False)

        try:
            self.__invoker.services.video_records.save(
                video_name=video_name,
                video_origin=video_origin,
                video_category=video_category,
                width=width,
                height=height,
                duration=duration,
                fps=fps,
                has_workflow=workflow is not None or graph is not None,
                is_intermediate=is_intermediate,
                node_id=node_id,
                metadata=metadata,
                session_id=session_id,
                user_id=user_id,
                video_subfolder=video_subfolder,
            )
            if board_id is not None:
                try:
                    self.__invoker.services.board_video_records.add_video_to_board(
                        board_id=board_id, video_name=video_name
                    )
                except Exception as e:
                    self.__invoker.services.logger.warning(f"Failed to add video to board {board_id}: {str(e)}")

            self.__invoker.services.video_files.save(
                source_path=source_path,
                video_name=video_name,
                video_subfolder=video_subfolder,
                metadata=metadata,
                workflow=workflow,
                graph=graph,
            )

            video_dto = self.get_dto(video_name)
            self._on_changed(video_dto)
            return video_dto
        except VideoRecordSaveException:
            self.__invoker.services.logger.error("Failed to save video record")
            raise
        except VideoFileSaveException:
            self.__invoker.services.logger.error("Failed to save video file")
            raise
        except Exception as e:
            self.__invoker.services.logger.error(f"Problem saving video record and file: {str(e)}")
            raise e

    def update(self, video_name: str, changes: VideoRecordChanges) -> VideoDTO:
        try:
            self.__invoker.services.video_records.update(video_name, changes)
            video_dto = self.get_dto(video_name)
            self._on_changed(video_dto)
            return video_dto
        except VideoRecordSaveException:
            self.__invoker.services.logger.error("Failed to update video record")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem updating video record")
            raise e

    def get_record(self, video_name: str) -> VideoRecord:
        try:
            return self.__invoker.services.video_records.get(video_name)
        except VideoRecordNotFoundException:
            self.__invoker.services.logger.error("Video record not found")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting video record")
            raise e

    def get_dto(self, video_name: str) -> VideoDTO:
        try:
            video_record = self.__invoker.services.video_records.get(video_name)
            return video_record_to_dto(
                video_record=video_record,
                video_url=self.__invoker.services.urls.get_video_url(video_name),
                thumbnail_url=self.__invoker.services.urls.get_video_url(video_name, thumbnail=True),
                board_id=self.__invoker.services.board_video_records.get_board_for_video(video_name),
            )
        except VideoRecordNotFoundException:
            self.__invoker.services.logger.error("Video record not found")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting video DTO")
            raise e

    def get_metadata(self, video_name: str) -> Optional[MetadataField]:
        try:
            return self.__invoker.services.video_records.get_metadata(video_name)
        except VideoRecordNotFoundException:
            self.__invoker.services.logger.error("Video record not found")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting video metadata")
            raise e

    def get_workflow(self, video_name: str) -> Optional[str]:
        try:
            record = self.__invoker.services.video_records.get(video_name)
            return self.__invoker.services.video_files.get_workflow(video_name, video_subfolder=record.video_subfolder)
        except VideoFileNotFoundException:
            self.__invoker.services.logger.error("Video file not found")
            raise
        except Exception:
            self.__invoker.services.logger.error("Problem getting video workflow")
            raise

    def get_graph(self, video_name: str) -> Optional[str]:
        try:
            record = self.__invoker.services.video_records.get(video_name)
            return self.__invoker.services.video_files.get_graph(video_name, video_subfolder=record.video_subfolder)
        except VideoFileNotFoundException:
            self.__invoker.services.logger.error("Video file not found")
            raise
        except Exception:
            self.__invoker.services.logger.error("Problem getting video graph")
            raise

    def get_path(self, video_name: str, thumbnail: bool = False) -> str:
        try:
            record = self.__invoker.services.video_records.get(video_name)
            return str(
                self.__invoker.services.video_files.get_path(
                    video_name, thumbnail=thumbnail, video_subfolder=record.video_subfolder
                )
            )
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting video path")
            raise e

    def get_url(self, video_name: str, thumbnail: bool = False) -> str:
        try:
            return self.__invoker.services.urls.get_video_url(video_name, thumbnail=thumbnail)
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting video URL")
            raise e

    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        video_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> OffsetPaginatedResults[VideoDTO]:
        try:
            results = self.__invoker.services.video_records.get_many(
                offset,
                limit,
                starred_first,
                order_dir,
                video_origin,
                categories,
                is_intermediate,
                board_id,
                search_term,
                user_id,
                is_admin,
            )
            video_dtos = [
                video_record_to_dto(
                    video_record=r,
                    video_url=self.__invoker.services.urls.get_video_url(r.video_name),
                    thumbnail_url=self.__invoker.services.urls.get_video_url(r.video_name, thumbnail=True),
                    board_id=self.__invoker.services.board_video_records.get_board_for_video(r.video_name),
                )
                for r in results.items
            ]
            return OffsetPaginatedResults[VideoDTO](
                items=video_dtos, offset=results.offset, limit=results.limit, total=results.total
            )
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting paginated video DTOs")
            raise e

    def delete(self, video_name: str) -> None:
        try:
            record = self.__invoker.services.video_records.get(video_name)
            self.__invoker.services.video_files.delete(video_name, video_subfolder=record.video_subfolder)
            self.__invoker.services.video_records.delete(video_name)
            self._on_deleted(video_name)
        except VideoRecordDeleteException:
            self.__invoker.services.logger.error("Failed to delete video record")
            raise
        except VideoFileDeleteException:
            self.__invoker.services.logger.error("Failed to delete video file")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem deleting video record and file")
            raise e

    def delete_videos_on_board(self, board_id: str, user_id: Optional[str] = None) -> None:
        try:
            # When ``user_id`` is set the lookup filters to videos owned by that user so the
            # cascade doesn't destroy other users' contributions to a public/shared board.
            video_names = self.__invoker.services.board_video_records.get_all_board_video_names_for_board(
                board_id, categories=None, is_intermediate=None, user_id=user_id
            )
            # Only delete records for files we actually managed to remove. Otherwise a
            # transient FS error would leave the file orphaned on disk with no record
            # pointing at it — the API would report success and the user would have no
            # way to clean up the leak. The board itself will still be deleted by the
            # caller, so any preserved records cascade to "uncategorized" via the
            # board_videos FK.
            deleted_video_names: list[str] = []
            for video_name in video_names:
                try:
                    record = self.__invoker.services.video_records.get(video_name)
                    self.__invoker.services.video_files.delete(video_name, video_subfolder=record.video_subfolder)
                    deleted_video_names.append(video_name)
                except Exception as e:
                    self.__invoker.services.logger.error(
                        f"Failed to delete video file {video_name}; keeping record: {str(e)}"
                    )
            self.__invoker.services.video_records.delete_many(deleted_video_names)
            for video_name in deleted_video_names:
                self._on_deleted(video_name)
        except VideoRecordDeleteException:
            self.__invoker.services.logger.error("Failed to delete video records")
            raise
        except VideoFileDeleteException:
            self.__invoker.services.logger.error("Failed to delete video files")
            raise
        except Exception as e:
            self.__invoker.services.logger.error(f"Problem deleting video records and files: {str(e)}")
            raise e

    def get_video_names(
        self,
        starred_first: bool = True,
        order_dir: SQLiteDirection = SQLiteDirection.Descending,
        video_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
        search_term: Optional[str] = None,
        user_id: Optional[str] = None,
        is_admin: bool = False,
    ) -> VideoNamesResult:
        try:
            return self.__invoker.services.video_records.get_video_names(
                starred_first=starred_first,
                order_dir=order_dir,
                video_origin=video_origin,
                categories=categories,
                is_intermediate=is_intermediate,
                board_id=board_id,
                search_term=search_term,
                user_id=user_id,
                is_admin=is_admin,
            )
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting video names")
            raise e
