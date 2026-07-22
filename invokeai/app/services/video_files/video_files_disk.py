import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.video_files.video_files_base import VideoFileStorageBase
from invokeai.app.services.video_files.video_files_common import (
    VideoFileDeleteException,
    VideoFileNotFoundException,
    VideoFileSaveException,
)
from invokeai.app.util.thumbnails import make_thumbnail
from invokeai.app.util.video_thumbnails import extract_video_frame, get_video_thumbnail_name
from invokeai.backend.util.logging import InvokeAILogger


@dataclass
class _StagedDelete:
    directory: Path
    files: list[tuple[Path, Path]]


class DiskVideoFileStorage(VideoFileStorageBase):
    """Stores video files on disk under {outputs}/videos/, with first-frame WebP thumbnails under
    {outputs}/videos/thumbnails/ and optional JSON sidecars for metadata/workflow/graph under
    {outputs}/videos/sidecars/."""

    def __init__(self, output_folder: Union[str, Path]):
        self.__output_folder = output_folder if isinstance(output_folder, Path) else Path(output_folder)
        self.__thumbnails_folder = self.__output_folder / "thumbnails"
        self.__sidecars_folder = self.__output_folder / "sidecars"
        self.__validate_storage_folders()

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker
        self.__recover_staged_deletes()

    def save(
        self,
        source_path: Path,
        video_name: str,
        thumbnail_size: int = 256,
        video_subfolder: str = "",
        metadata: Optional[str] = None,
        workflow: Optional[str] = None,
        graph: Optional[str] = None,
    ) -> None:
        logger = InvokeAILogger.get_logger()
        try:
            self.__validate_storage_folders()
            video_path = self.get_path(video_name, video_subfolder=video_subfolder)
            video_path.parent.mkdir(parents=True, exist_ok=True)

            # Move if the source is on the same filesystem; otherwise copy then unlink.
            try:
                shutil.move(str(source_path), str(video_path))
            except Exception:
                shutil.copy2(str(source_path), str(video_path))
                try:
                    Path(source_path).unlink(missing_ok=True)
                except Exception:
                    pass
            logger.info(f"Video file written: {video_path}")

            thumbnail_name = get_video_thumbnail_name(video_name)
            thumbnail_path = self.get_path(thumbnail_name, thumbnail=True, video_subfolder=video_subfolder)
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)

            # Thumbnail extraction is best-effort — if both imageio and cv2 fail, we still want
            # the video record + file in place and the invocation to complete. A missing
            # thumbnail leaves the gallery with a broken-image placeholder for that item, which
            # is annoying but not fatal.
            try:
                frame = extract_video_frame(video_path, frame_index=0)
            except Exception as e:
                logger.warning(f"Thumbnail extraction raised for {video_name}: {e}")
                frame = None
            if frame is not None:
                thumbnail = make_thumbnail(frame, thumbnail_size)
                thumbnail.save(thumbnail_path, "WEBP")
                logger.info(f"Thumbnail written: {thumbnail_path}")
            else:
                logger.warning(
                    f"Could not extract a thumbnail frame for {video_name}; gallery thumbnail will be missing."
                )

            if metadata is not None or workflow is not None or graph is not None:
                sidecar_path = self.__get_sidecar_path(video_name, video_subfolder=video_subfolder)
                sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                sidecar = {
                    "invokeai_metadata": metadata,
                    "invokeai_workflow": workflow,
                    "invokeai_graph": graph,
                }
                with open(sidecar_path, "w", encoding="utf-8") as f:
                    json.dump(sidecar, f)
                logger.info(f"Sidecar written: {sidecar_path}")
        except Exception as e:
            # By this point the source MP4 has usually already been moved into permanent
            # storage, so bailing out without cleanup would orphan the video (and any
            # partially written thumbnail/sidecar) on disk with no DB record through which
            # it can be managed — the caller rolls the record back on this exception.
            try:
                self.delete(video_name, video_subfolder=video_subfolder)
            except Exception as cleanup_err:
                logger.error(f"Failed to clean up partially saved files for {video_name}: {cleanup_err}")
            raise VideoFileSaveException from e

    def delete(self, video_name: str, video_subfolder: str = "") -> None:
        token = self.stage_delete(video_name, video_subfolder)
        self.commit_delete(token)

    def stage_delete(self, video_name: str, video_subfolder: str = "") -> _StagedDelete:
        candidates = [
            self.get_path(video_name, video_subfolder=video_subfolder),
            self.get_path(video_name, thumbnail=True, video_subfolder=video_subfolder),
            self.__get_sidecar_path(video_name, video_subfolder=video_subfolder),
        ]
        staging_dir = Path(tempfile.mkdtemp(prefix=".delete_", dir=self.__output_folder))
        staged: list[tuple[Path, Path]] = []
        try:
            with open(staging_dir / "manifest.json", "w", encoding="utf-8") as manifest:
                manifest.write(json.dumps({"video_name": video_name, "video_subfolder": video_subfolder}))
                manifest.flush()
                os.fsync(manifest.fileno())
            for index, source in enumerate(candidates):
                if source.exists():
                    destination = staging_dir / str(index)
                    source.replace(destination)
                    staged.append((source, destination))
            return _StagedDelete(directory=staging_dir, files=staged)
        except Exception as e:
            for source, destination in reversed(staged):
                if destination.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.replace(source)
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise VideoFileDeleteException from e

    def commit_delete(self, token: object) -> None:
        if not isinstance(token, _StagedDelete):
            raise VideoFileDeleteException("Invalid staged-delete token")
        shutil.rmtree(token.directory)

    def rollback_delete(self, token: object) -> None:
        if not isinstance(token, _StagedDelete):
            raise VideoFileDeleteException("Invalid staged-delete token")
        try:
            for source, destination in reversed(token.files):
                if destination.exists():
                    source.parent.mkdir(parents=True, exist_ok=True)
                    destination.replace(source)
            shutil.rmtree(token.directory, ignore_errors=True)
        except Exception as e:
            raise VideoFileDeleteException from e

    def get_path(self, video_name: str, thumbnail: bool = False, video_subfolder: str = "") -> Path:
        base_folder = self.__thumbnails_folder if thumbnail else self.__output_folder
        filename = get_video_thumbnail_name(video_name) if thumbnail else video_name

        basename = Path(filename).name
        if basename != filename:
            raise ValueError("Invalid video name, potential directory traversal detected")

        if video_subfolder:
            self._validate_subfolder(video_subfolder)
            video_path = base_folder / video_subfolder / basename
        else:
            video_path = base_folder / basename

        resolved_base = base_folder.resolve()
        resolved_video_path = video_path.resolve()
        if not resolved_video_path.is_relative_to(resolved_base):
            raise ValueError("Video path outside outputs folder, potential directory traversal detected")
        return resolved_video_path

    def get_workflow(self, video_name: str, video_subfolder: str = "") -> Optional[str]:
        sidecar = self.__read_sidecar(video_name, video_subfolder)
        if sidecar is None:
            return None
        workflow = sidecar.get("invokeai_workflow")
        return workflow if isinstance(workflow, str) else None

    def get_graph(self, video_name: str, video_subfolder: str = "") -> Optional[str]:
        sidecar = self.__read_sidecar(video_name, video_subfolder)
        if sidecar is None:
            return None
        graph = sidecar.get("invokeai_graph")
        return graph if isinstance(graph, str) else None

    def validate_path(self, path: Union[str, Path]) -> bool:
        path = path if isinstance(path, Path) else Path(path)
        return path.exists()

    @staticmethod
    def _validate_subfolder(subfolder: str) -> None:
        """Validates a subfolder path to prevent directory traversal."""
        if not subfolder:
            return
        if "\\" in subfolder:
            raise ValueError("Backslashes not allowed in subfolder path")
        if subfolder.startswith("/"):
            raise ValueError("Absolute paths not allowed in subfolder path")
        for part in subfolder.split("/"):
            if part == "..":
                raise ValueError("Parent directory references not allowed in subfolder path")
            if part == "":
                raise ValueError("Empty path segments not allowed in subfolder path")

    def __get_sidecar_path(self, video_name: str, video_subfolder: str = "") -> Path:
        sidecar_name = Path(video_name).stem + ".json"
        if video_subfolder:
            self._validate_subfolder(video_subfolder)
            sidecar_path = self.__sidecars_folder / video_subfolder / sidecar_name
        else:
            sidecar_path = self.__sidecars_folder / sidecar_name
        resolved_base = self.__sidecars_folder.resolve()
        resolved_sidecar_path = sidecar_path.resolve()
        if not resolved_sidecar_path.is_relative_to(resolved_base):
            raise ValueError("Sidecar path outside outputs folder, potential directory traversal detected")
        return resolved_sidecar_path

    def __read_sidecar(self, video_name: str, video_subfolder: str = "") -> Optional[dict]:
        path = self.__get_sidecar_path(video_name, video_subfolder=video_subfolder)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            raise VideoFileNotFoundException from e

    def __validate_storage_folders(self) -> None:
        for folder in (self.__output_folder, self.__thumbnails_folder, self.__sidecars_folder):
            folder.mkdir(parents=True, exist_ok=True)

    def __recover_staged_deletes(self) -> None:
        logger = InvokeAILogger.get_logger()
        for staging_dir in self.__output_folder.glob(".delete_*"):
            manifest_path = staging_dir / "manifest.json"
            if not manifest_path.is_file():
                if not any(staging_dir.iterdir()):
                    staging_dir.rmdir()
                continue
            try:
                with open(manifest_path, encoding="utf-8") as manifest:
                    data = json.load(manifest)
                video_name = data["video_name"]
                video_subfolder = data.get("video_subfolder", "")
                candidates = [
                    self.get_path(video_name, video_subfolder=video_subfolder),
                    self.get_path(video_name, thumbnail=True, video_subfolder=video_subfolder),
                    self.__get_sidecar_path(video_name, video_subfolder=video_subfolder),
                ]
                token = _StagedDelete(
                    directory=staging_dir,
                    files=[(source, staging_dir / str(index)) for index, source in enumerate(candidates)],
                )
                if self.__invoker.services.video_records.get(video_name) is None:
                    self.commit_delete(token)
                else:
                    self.rollback_delete(token)
            except Exception as error:
                from invokeai.app.services.video_records.video_records_common import VideoRecordNotFoundException

                if isinstance(error, VideoRecordNotFoundException):
                    shutil.rmtree(staging_dir, ignore_errors=True)
                else:
                    logger.error(f"Failed to recover staged video deletion {staging_dir}: {error}")
