from pathlib import Path

from invokeai.app.services.asset_files.asset_files_base import AssetFilesServiceBase
from invokeai.app.services.asset_files.asset_files_common import (
    AssetFileDeleteException,
    AssetFileNotFoundException,
    AssetFileSaveException,
)
from invokeai.app.services.invoker import Invoker


class DiskAssetFileStorage(AssetFilesServiceBase):
    """Stores 3D asset files (Gaussian-splat .ply / .splat) on disk."""

    def __init__(self, asset_files_folder: Path):
        self._asset_files_folder = asset_files_folder
        self._validate_storage_folder()

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def save(self, asset_name: str, data: bytes) -> None:
        try:
            self._validate_storage_folder()
            path = self._resolve_path(asset_name)
            path.write_bytes(data)
        except Exception as e:
            raise AssetFileSaveException from e

    def get_path(self, asset_name: str) -> Path:
        path = self._resolve_path(asset_name)
        if not path.exists():
            raise AssetFileNotFoundException
        return path

    def get_url(self, asset_name: str) -> str:
        return self._invoker.services.urls.get_asset_url(asset_name)

    def delete(self, asset_name: str) -> None:
        try:
            path = self._resolve_path(asset_name)
            if not path.exists():
                raise AssetFileNotFoundException
            path.unlink()
        except AssetFileNotFoundException as e:
            raise AssetFileNotFoundException from e
        except Exception as e:
            raise AssetFileDeleteException from e

    def _resolve_path(self, asset_name: str) -> Path:
        """Resolves an asset name to a path, guarding against path traversal."""
        # Only bare filenames are allowed - reject any path separators / traversal.
        if not asset_name or asset_name != Path(asset_name).name:
            raise ValueError(f"Invalid asset name: {asset_name}")
        path = (self._asset_files_folder / asset_name).resolve()
        if not path.is_relative_to(self._asset_files_folder.resolve()):
            raise ValueError(f"Invalid asset name: {asset_name}")
        return path

    def _validate_storage_folder(self) -> None:
        """Creates the storage folder if it does not exist."""
        self._asset_files_folder.mkdir(parents=True, exist_ok=True)
