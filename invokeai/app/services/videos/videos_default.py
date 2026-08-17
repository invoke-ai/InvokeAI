from pathlib import Path
from typing import Optional

from PIL import Image

from invokeai.app.invocations.fields import MetadataField
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    InvalidImageCategoryException,
    InvalidOriginException,
    ResourceOrigin,
)
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.bulk_media_delete import StagedMediaDeleteAdapter, delete_media_by_names
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
        first_frame: Optional[Image.Image] = None,
        move_source: bool = True,
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

        record_saved = False
        board_attached = False
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
            record_saved = True
            if board_id is not None:
                # Board attachment is deliberately best-effort, mirroring ImageService.create:
                # this is reachable when the board is deleted between the caller's access
                # check and this insert, and failing the whole create here would destroy a
                # just-generated video over a cosmetic categorization problem. The returned
                # DTO reports the video's *actual* board (None on fallback), so callers are
                # not told the attachment succeeded.
                try:
                    self.__invoker.services.board_video_records.add_video_to_board(
                        board_id=board_id, video_name=video_name
                    )
                    board_attached = True
                except Exception as e:
                    self.__invoker.services.logger.warning(f"Failed to add video to board {board_id}: {str(e)}")

            self.__invoker.services.video_files.save(
                source_path=source_path,
                video_name=video_name,
                video_subfolder=video_subfolder,
                metadata=metadata,
                workflow=workflow,
                graph=graph,
                first_frame=first_frame,
                move_source=move_source,
            )

            video_dto = self.get_dto(video_name)
            self._on_changed(video_dto)
            return video_dto
        except VideoRecordSaveException:
            self.__invoker.services.logger.error("Failed to save video record")
            raise
        except Exception as e:
            # Roll back any DB-side state we created so the gallery doesn't end up with a
            # ghost record whose file endpoints 404. Most commonly triggered by
            # VideoFileSaveException (disk save or sidecar write failure), but we also
            # need to unwind on any unexpected post-record failure.
            if board_attached:
                try:
                    self.__invoker.services.board_video_records.remove_video_from_board(video_name=video_name)
                except Exception as rollback_err:
                    self.__invoker.services.logger.error(
                        f"Failed to roll back board attachment for {video_name}: {str(rollback_err)}"
                    )
            if record_saved:
                try:
                    self.__invoker.services.video_records.delete(video_name)
                except Exception as rollback_err:
                    self.__invoker.services.logger.error(
                        f"Failed to roll back video record for {video_name}: {str(rollback_err)}"
                    )
            # The disk layer cleans up after itself when the save fails, but a failure
            # after a successful file save (e.g. building the DTO) would still leave the
            # files on disk with no record pointing at them. delete() skips files that
            # don't exist, so this is a no-op when nothing was written.
            try:
                self.__invoker.services.video_files.delete(video_name, video_subfolder=video_subfolder)
            except Exception as rollback_err:
                self.__invoker.services.logger.error(
                    f"Failed to roll back video files for {video_name}: {str(rollback_err)}"
                )
            if isinstance(e, VideoFileSaveException):
                self.__invoker.services.logger.error("Failed to save video file")
            else:
                self.__invoker.services.logger.error(f"Problem saving video record and file: {str(e)}")
            raise

    def copy(self, source_video_name: str, board_id: Optional[str] = None, user_id: Optional[str] = None) -> VideoDTO:
        """Duplicate a video without moving the source or exposing partial attachment semantics.

        A lost board attachment is fatal here as it is for images, but it is caught *after* the copy
        rather than before: this path goes through ``create``, which deliberately swallows a failed
        attachment so a freshly generated video is never lost to it. That leaves no way to refuse up
        front, so the copy is made and then withdrawn — through ``delete``, which takes the files
        with the record.
        """
        record = self.get_record(source_video_name)
        metadata = self.get_metadata(source_video_name)
        first_frame: Optional[Image.Image] = None

        try:
            thumbnail_path = Path(self.get_path(source_video_name, thumbnail=True))
            if thumbnail_path.exists():
                with Image.open(thumbnail_path) as thumbnail:
                    first_frame = thumbnail.copy()
        except Exception:
            # A thumbnail is an optimization only. ``create`` extracts frame zero when absent.
            first_frame = None

        created = self.create(
            source_path=Path(self.get_path(source_video_name)),
            width=record.width,
            height=record.height,
            duration=record.duration,
            fps=record.fps,
            video_origin=record.video_origin,
            video_category=record.video_category,
            board_id=board_id,
            is_intermediate=False,
            metadata=metadata.model_dump_json() if metadata is not None else None,
            workflow=self.get_workflow(source_video_name),
            graph=self.get_graph(source_video_name),
            user_id=user_id,
            first_frame=first_frame,
            # The source path belongs to this service; a copy may never consume it.
            move_source=False,
        )

        if board_id is not None and created.board_id != board_id:
            try:
                self.delete(created.video_name)
            except Exception as cleanup_error:
                self.__invoker.services.logger.error(
                    f"Failed to remove video copy {created.video_name} after board attachment failed: {cleanup_error}"
                )
            raise RuntimeError(f"Copy of {source_video_name} did not reach board {board_id}")

        return created

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
        token: object | None = None
        record_deleted = False
        try:
            record = self.__invoker.services.video_records.get(video_name)
            token = self.__invoker.services.video_files.stage_delete(video_name, video_subfolder=record.video_subfolder)
            self.__invoker.services.video_records.delete(video_name)
            record_deleted = True
            try:
                self.__invoker.services.video_files.commit_delete(token)
            except Exception as cleanup_error:
                self.__invoker.services.logger.error(f"Failed to purge staged video files: {cleanup_error}")
            self._on_deleted(video_name)
        except VideoRecordDeleteException:
            if token is not None:
                self.__invoker.services.video_files.rollback_delete(token)
            self.__invoker.services.logger.error("Failed to delete video record")
            raise
        except VideoFileDeleteException:
            self.__invoker.services.logger.error("Failed to delete video file")
            raise
        except Exception as e:
            if token is not None and not record_deleted:
                try:
                    self.__invoker.services.video_files.rollback_delete(token)
                except Exception as rollback_error:
                    self.__invoker.services.logger.error(f"Failed to restore video files: {rollback_error}")
            self.__invoker.services.logger.error("Problem deleting video record and file")
            raise e

    def delete_videos_on_board(self, board_id: str, user_id: Optional[str] = None) -> tuple[list[str], list[str]]:
        # When ``user_id`` is set the lookup filters to videos owned by that user so the
        # cascade doesn't destroy other users' contributions to a public/shared board.
        video_names = self.__invoker.services.board_video_records.get_all_board_video_names_for_board(
            board_id, categories=None, is_intermediate=None, user_id=user_id
        )
        return self.delete_videos_by_names(video_names)

    def delete_videos_by_names(self, video_names: list[str]) -> tuple[list[str], list[str]]:
        """Delete exactly these videos, returning ``(deleted, failed)``.

        Split from ``delete_videos_on_board`` so a caller that must decide whether the board may go
        *before* destroying anything can enumerate first and delete second.
        """
        try:
            records = self.__invoker.services.video_records
            files = self.__invoker.services.video_files
            return delete_media_by_names(
                video_names,
                StagedMediaDeleteAdapter(
                    kind="video",
                    stage=lambda name: files.stage_delete(name, video_subfolder=records.get(name).video_subfolder),
                    delete_records=records.delete_many,
                    rollback=files.rollback_delete,
                    commit=files.commit_delete,
                    notify_deleted=self._on_deleted,
                    log_error=self.__invoker.services.logger.error,
                ),
            )
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
