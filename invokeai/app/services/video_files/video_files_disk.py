import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Collection, Optional, Sequence, Union

from PIL import Image

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.video_files.video_files_base import VideoFileStorageBase
from invokeai.app.services.video_files.video_files_common import (
    EMBEDDED_METADATA_KEYS,
    GRAPH_KEY,
    METADATA_KEY,
    WORKFLOW_KEY,
    VideoFileDeleteException,
    VideoFileNotFoundException,
    VideoFileSaveException,
)
from invokeai.app.services.video_records.video_records_common import VideoRecordNotFoundException
from invokeai.app.util.mp4_metadata import read_mp4_tags, write_mp4_tags
from invokeai.app.util.thumbnails import make_thumbnail
from invokeai.app.util.video_thumbnails import extract_representative_video_frame, get_video_thumbnail_name
from invokeai.backend.util.logging import InvokeAILogger


@dataclass
class _StagedDelete:
    directory: Path
    files: list[tuple[Path, Path]]


@dataclass
class _PendingDelete:
    directory: Path
    videos: list[tuple[str, str]]


# Prefix of the same-directory temp file a metadata remux writes before replacing the video.
_EMBED_TEMP_PREFIX = ".embed_"


class DiskVideoFileStorage(VideoFileStorageBase):
    """Stores video files on disk under {outputs}/videos/, with representative-frame WebP thumbnails under
    {outputs}/videos/thumbnails/.

    Metadata, workflow and graph are embedded in the MP4 itself as keyed metadata (the same three keys a PNG
    carries as text chunks), so a downloaded video stays recallable anywhere. A JSON sidecar under
    {outputs}/videos/sidecars/ is written only when embedding fails, and is still read for videos stored before
    embedding existed."""

    def __init__(self, output_folder: Union[str, Path]):
        self.__output_folder = output_folder if isinstance(output_folder, Path) else Path(output_folder)
        self.__thumbnails_folder = self.__output_folder / "thumbnails"
        self.__sidecars_folder = self.__output_folder / "sidecars"
        self.__validate_storage_folders()

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker
        self.__recover_staged_deletes()
        self.__sweep_embed_temps()

    def save(
        self,
        source_path: Path,
        video_name: str,
        thumbnail_size: int = 256,
        video_subfolder: str = "",
        metadata: Optional[str] = None,
        workflow: Optional[str] = None,
        graph: Optional[str] = None,
        first_frame: Optional[Image.Image] = None,
        move_source: bool = True,
        duration: Optional[float] = None,
        fps: Optional[float] = None,
    ) -> None:
        logger = InvokeAILogger.get_logger()
        try:
            self.__validate_storage_folders()
            video_path = self.get_path(video_name, video_subfolder=video_subfolder)
            video_path.parent.mkdir(parents=True, exist_ok=True)

            if move_source:
                # Move if the source is on the same filesystem; otherwise copy then unlink.
                try:
                    shutil.move(str(source_path), str(video_path))
                except Exception:
                    shutil.copy2(str(source_path), str(video_path))
                    try:
                        Path(source_path).unlink(missing_ok=True)
                    except Exception:
                        pass
            else:
                shutil.copy2(str(source_path), str(video_path))
            logger.info(f"Video file written: {video_path}")

            tags = {
                key: value
                for key, value in ((METADATA_KEY, metadata), (WORKFLOW_KEY, workflow), (GRAPH_KEY, graph))
                if value is not None
            }
            embedded = self.__embed_tags(video_path, tags)

            thumbnail_name = get_video_thumbnail_name(video_name)
            thumbnail_path = self.get_path(thumbnail_name, thumbnail=True, video_subfolder=video_subfolder)
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)

            # Thumbnail extraction is best-effort — if both imageio and cv2 fail, we still want
            # the video record + file in place and the invocation to complete. A missing
            # thumbnail leaves the gallery with a broken-image placeholder for that item, which
            # is annoying but not fatal. The upload path already decoded a representative frame
            # to prove decodability and passes it in, saving the decode-worker spawns per
            # upload; this fallback runs the same seek ladder for generated/derived videos.
            frame = first_frame
            if frame is None:
                try:
                    frame = extract_representative_video_frame(video_path, duration, fps)
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

            if tags and not embedded:
                # Nothing recallable may be lost to a remux failure: fall back to the sidecar the reader
                # still understands.
                sidecar_path = self.__get_sidecar_path(video_name, video_subfolder=video_subfolder)
                sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                sidecar = {
                    METADATA_KEY: metadata,
                    WORKFLOW_KEY: workflow,
                    GRAPH_KEY: graph,
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
        candidates = self.__delete_candidates(video_name, video_subfolder)
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

    def begin_delete(self, videos: Sequence[tuple[str, str]]) -> _PendingDelete:
        # Resolving each path validates the names before anything is journalled; the paths are re-derived at commit.
        for name, subfolder in videos:
            self.__delete_candidates(name, subfolder)
        directory = Path(tempfile.mkdtemp(prefix=".delete_", dir=self.__output_folder))
        try:
            with open(directory / "manifest.json", "w", encoding="utf-8") as manifest:
                json.dump({"version": 2, "videos": videos}, manifest)
                manifest.flush()
                os.fsync(manifest.fileno())
            self.__fsync_directory(directory)
            self.__fsync_directory(self.__output_folder)
            return _PendingDelete(directory=directory, videos=list(videos))
        except Exception as error:
            shutil.rmtree(directory, ignore_errors=True)
            raise VideoFileDeleteException from error

    def abandon_delete(self, token: object) -> None:
        if not isinstance(token, _PendingDelete):
            raise VideoFileDeleteException("Invalid pending-delete token")
        shutil.rmtree(token.directory, ignore_errors=True)

    def commit_delete(self, token: object, video_names: Optional[Collection[str]] = None) -> None:
        if isinstance(token, _PendingDelete):
            self.__commit_pending_delete(token, video_names)
            return
        if not isinstance(token, _StagedDelete):
            raise VideoFileDeleteException("Invalid staged-delete token")
        shutil.rmtree(token.directory)

    def __commit_pending_delete(self, token: _PendingDelete, video_names: Optional[Collection[str]]) -> None:
        selected = None if video_names is None else set(video_names)
        purged: list[tuple[str, str]] = []
        try:
            for name, subfolder in token.videos:
                if selected is not None and name not in selected:
                    continue
                self.__purge_files(name, subfolder)
                purged.append((name, subfolder))
            self.__persist_purges(purged)
        except OSError as error:
            # A retry must re-check the record before touching the live files.
            raise VideoFileDeleteException from error
        shutil.rmtree(token.directory, ignore_errors=True)

    def __delete_candidates(self, name: str, subfolder: str) -> list[Path]:
        return [
            self.get_path(name, video_subfolder=subfolder),
            self.get_path(name, thumbnail=True, video_subfolder=subfolder),
            self.__get_sidecar_path(name, video_subfolder=subfolder),
        ]

    def __purge_files(self, name: str, subfolder: str) -> None:
        for path in self.__delete_candidates(name, subfolder):
            path.unlink(missing_ok=True)

    @staticmethod
    def __fsync_directory(directory: Path) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def __persist_purges(self, videos: Sequence[tuple[str, str]]) -> None:
        parents = {path.parent for name, subfolder in videos for path in self.__delete_candidates(name, subfolder)}
        for parent in parents:
            if parent.exists():
                self.__fsync_directory(parent)

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

    def get_file_size_bytes(self, video_name: str, video_subfolder: str = "") -> Optional[int]:
        try:
            size = self.get_path(video_name, video_subfolder=video_subfolder).stat().st_size
        except FileNotFoundError:
            return None
        for companion in (
            self.get_path(video_name, thumbnail=True, video_subfolder=video_subfolder),
            self.__get_sidecar_path(video_name, video_subfolder=video_subfolder),
        ):
            try:
                size += companion.stat().st_size
            except FileNotFoundError:
                pass
        return size

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
        return self.__read_embedded_or_sidecar(video_name, WORKFLOW_KEY, video_subfolder)

    def get_graph(self, video_name: str, video_subfolder: str = "") -> Optional[str]:
        return self.__read_embedded_or_sidecar(video_name, GRAPH_KEY, video_subfolder)

    def __embed_tags(self, video_path: Path, tags: dict[str, str]) -> bool:
        """Make ``video_path``'s keyed metadata carry exactly ``tags`` — the record being saved, which
        may be empty. False (and a warning) on any failure, leaving the original file untouched so the
        caller can fall back to a sidecar."""
        logger = InvokeAILogger.get_logger()
        tmp_path: Optional[Path] = None
        try:
            # A file that already carries exactly this record (a re-uploaded InvokeAI download, or an
            # untagged file saved without one) needs no rewrite. A file carrying a key the record lacks
            # — an upload whose embedded workflow failed validation — is rewritten so the rejected value
            # cannot be read back as the record.
            if read_mp4_tags(video_path, keys=EMBEDDED_METADATA_KEYS) == tags:
                return True
            # Same directory, so the final os.replace is an atomic rename and never a cross-device copy.
            fd, tmp_name = tempfile.mkstemp(prefix=_EMBED_TEMP_PREFIX, suffix=".mp4", dir=video_path.parent)
            os.close(fd)
            tmp_path = Path(tmp_name)
            write_mp4_tags(video_path, tmp_path, tags)
            os.replace(tmp_path, video_path)
            return True
        except Exception as e:
            logger.warning(f"Could not embed metadata in {video_path.name}; writing a sidecar instead: {e}")
            return False
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    def __sweep_embed_temps(self) -> None:
        """Remove remux temp files a crash left behind: they are never referenced by a record."""
        for stale in self.__output_folder.rglob(f"{_EMBED_TEMP_PREFIX}*.mp4"):
            try:
                stale.unlink()
            except OSError as e:
                InvokeAILogger.get_logger().warning(f"Could not remove stale metadata temp file {stale}: {e}")

    def __read_embedded_or_sidecar(self, video_name: str, key: str, video_subfolder: str) -> Optional[str]:
        video_path = self.get_path(video_name, video_subfolder=video_subfolder)
        if video_path.exists():
            try:
                value = read_mp4_tags(video_path, keys=(key,)).get(key)
            except OSError as e:
                raise VideoFileNotFoundException from e
            if value is not None:
                return value
        sidecar = self.__read_sidecar(video_name, video_subfolder)
        if sidecar is None:
            return None
        value = sidecar.get(key)
        return value if isinstance(value, str) else None

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
                if data.get("version") == 2:
                    videos = [(entry[0], entry[1]) for entry in data["videos"]]
                    for video_name, video_subfolder in videos:
                        try:
                            self.__invoker.services.video_records.get(video_name)
                        except VideoRecordNotFoundException:
                            self.__purge_files(video_name, video_subfolder)
                    self.__persist_purges(videos)
                    shutil.rmtree(staging_dir)
                    continue
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
                # get() raises VideoRecordNotFoundException when the record is gone —
                # handled below by purging the staging dir (the delete had committed in
                # the DB, so finish it). A surviving record means the delete never
                # committed: restore the files.
                self.__invoker.services.video_records.get(video_name)
                self.rollback_delete(token)
            except Exception as error:
                if isinstance(error, VideoRecordNotFoundException):
                    shutil.rmtree(staging_dir, ignore_errors=True)
                else:
                    logger.error(f"Failed to recover staged video deletion {staging_dir}: {error}")
