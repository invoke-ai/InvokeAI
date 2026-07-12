from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


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
    ) -> None:
        """Saves a video by moving/copying the file at `source_path` into storage, then writes a sibling
        WEBP thumbnail extracted from the first frame, plus an optional sidecar JSON of metadata/workflow/graph.
        """
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
    def commit_delete(self, token: object) -> None:
        """Permanently removes files represented by a staged-delete token."""
        pass

    @abstractmethod
    def rollback_delete(self, token: object) -> None:
        """Restores files represented by a staged-delete token."""
        pass

    @abstractmethod
    def get_workflow(self, video_name: str, video_subfolder: str = "") -> Optional[str]:
        """Gets the workflow JSON sidecar of a video, if any."""
        pass

    @abstractmethod
    def get_graph(self, video_name: str, video_subfolder: str = "") -> Optional[str]:
        """Gets the graph JSON sidecar of a video, if any."""
        pass

    @abstractmethod
    def validate_path(self, path: str) -> bool:
        """Validates the path given for a video or thumbnail."""
        pass
