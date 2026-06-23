import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Optional

from PIL.Image import Image as PILImageType


class ImageFileStorageBase(ABC):
    """Low-level service responsible for storing and retrieving image files."""

    @abstractmethod
    def get(self, image_name: str, image_subfolder: str = "") -> PILImageType:
        """Retrieves an image as PIL Image."""
        pass

    @abstractmethod
    def get_bytes(self, image_name: str, thumbnail: bool = False, image_subfolder: str = "") -> bytes:
        """Retrieves the raw bytes of an image or its thumbnail."""
        pass

    @abstractmethod
    def get_path(self, image_name: str, thumbnail: bool = False, image_subfolder: str = "") -> Path:
        """Gets the internal path to an image or thumbnail."""
        pass

    def get_local_path(self, image_name: str, thumbnail: bool = False, image_subfolder: str = "") -> Optional[Path]:
        """Return a real local filesystem path for the file, or ``None``.

        Backends that store files on the local filesystem return a usable
        ``Path`` so callers can stream it directly (e.g. add it to a zip without
        loading the whole file into memory). Remote backends (e.g. S3) return
        ``None``; callers should fall back to :meth:`open_stream` / :meth:`get_bytes`.
        """
        return None

    def open_stream(self, image_name: str, thumbnail: bool = False, image_subfolder: str = "") -> BinaryIO:
        """Return a readable binary stream of the file's bytes. The caller closes it.

        The default buffers the whole payload via :meth:`get_bytes`. Backends that
        can stream (e.g. S3) should override this to read in chunks and avoid
        loading the entire file into memory (e.g. when zipping large downloads).
        """
        return io.BytesIO(self.get_bytes(image_name, thumbnail=thumbnail, image_subfolder=image_subfolder))

    # TODO: We need to validate paths before starlette makes the FileResponse, else we get a
    # 500 internal server error. I don't like having this method on the service.
    @abstractmethod
    def validate_path(self, path: str) -> bool:
        """Validates the path given for an image or thumbnail."""
        pass

    @abstractmethod
    def save(
        self,
        image: PILImageType,
        image_name: str,
        metadata: Optional[str] = None,
        workflow: Optional[str] = None,
        graph: Optional[str] = None,
        thumbnail_size: int = 256,
        image_subfolder: str = "",
    ) -> None:
        """Saves an image and a 256x256 WEBP thumbnail. Returns a tuple of the image name, thumbnail name, and created timestamp."""
        pass

    @abstractmethod
    def delete(self, image_name: str, image_subfolder: str = "") -> None:
        """Deletes an image and its thumbnail (if one exists)."""
        pass

    @abstractmethod
    def get_workflow(self, image_name: str, image_subfolder: str = "") -> Optional[str]:
        """Gets the workflow of an image."""
        pass

    @abstractmethod
    def get_graph(self, image_name: str, image_subfolder: str = "") -> Optional[str]:
        """Gets the graph of an image."""
        pass
