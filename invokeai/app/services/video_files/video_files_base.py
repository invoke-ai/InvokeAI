from abc import ABC, abstractmethod
from pathlib import Path
from typing import Collection, Optional, Sequence

from PIL import Image


class VideoFileStorageBase(ABC):
    """Low-level service responsible for storing and retrieving video files."""

    @abstractmethod
    def get_path(self, video_name: str, thumbnail: bool = False, video_subfolder: str = "") -> Path:
        """Gets the internal path to a video or its thumbnail."""
        pass

    @abstractmethod
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
        """Saves a video by moving the file at `source_path` into storage, then writes a sibling
        WEBP thumbnail, plus an optional sidecar JSON of metadata/workflow/graph.
        The thumbnail is extracted from a representative frame (the first informative one on a
        seek ladder starting ~1s in; see ``extract_representative_video_frame``), located using
        ``duration``/``fps`` when the caller knows them. A caller that already decoded a
        representative frame can pass it as `first_frame` to skip the extraction.

        `source_path` is **consumed** by default: almost every caller hands over a temp file it just
        wrote, and moving it is both cheaper and the correct lifetime. A caller whose source is a
        file the server still owns — copying an existing video — must pass ``move_source=False``,
        or the copy will take the original's bytes with it.
        """
        pass

    @abstractmethod
    def get_file_size_bytes(self, video_name: str, video_subfolder: str = "") -> Optional[int]:
        """Bytes the video, its thumbnail and sidecar occupy; None when the video file is missing."""
        pass

    @abstractmethod
    def delete(self, video_name: str, video_subfolder: str = "") -> None:
        """Deletes a video file and its thumbnail (if one exists)."""
        pass

    @abstractmethod
    def stage_delete(self, video_name: str, video_subfolder: str = "") -> object:
        """Moves a video's files out of service and returns a rollback token."""
        pass

    @abstractmethod
    def begin_delete(self, videos: Sequence[tuple[str, str]]) -> object:
        """Durably journals a conditional delete before video records are removed."""
        pass

    @abstractmethod
    def abandon_delete(self, token: object) -> None:
        """Discards a pending journal after record deletion fails."""
        pass

    @abstractmethod
    def commit_delete(self, token: object, video_names: Optional[Collection[str]] = None) -> None:
        """Permanently removes files represented by a staged-delete token."""
        pass

    @abstractmethod
    def rollback_delete(self, token: object) -> None:
        """Restores files represented by a staged-delete token."""
        pass

    @abstractmethod
    def get_workflow(self, video_name: str, video_subfolder: str = "") -> Optional[str]:
        """Gets the workflow JSON embedded in (or stored beside) a video, if any."""
        pass

    @abstractmethod
    def get_graph(self, video_name: str, video_subfolder: str = "") -> Optional[str]:
        """Gets the graph JSON embedded in (or stored beside) a video, if any."""
        pass

    @abstractmethod
    def validate_path(self, path: str) -> bool:
        """Validates the path given for a video or thumbnail."""
        pass
