import pathlib
import shutil
import sqlite3
from logging import Logger

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.model_install.model_install_default import ModelInstallService
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration

LEGACY_CORE_MODELS = {
    # OpenPose
    "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true": "any/annotators/dwpose/yolox_l.onnx",
    "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true": "any/annotators/dwpose/dw-ll_ucoco_384.onnx",
    # DepthAnything
    "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth?download=true": "any/annotators/depth_anything/depth_anything_vitl14.pth",
    "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitb14.pth?download=true": "any/annotators/depth_anything/depth_anything_vitb14.pth",
    "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vits14.pth?download=true": "any/annotators/depth_anything/depth_anything_vits14.pth",
    # Lama inpaint
    "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt": "core/misc/lama/lama.pt",
    # RealESRGAN upscale
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth": "core/upscaling/realesrgan/RealESRGAN_x4plus.pth",
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth": "core/upscaling/realesrgan/RealESRGAN_x4plus_anime_6B.pth",
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth": "core/upscaling/realesrgan/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth": "core/upscaling/realesrgan/RealESRGAN_x2plus.pth",
}


class Migration10Callback:
    def __init__(self, app_config: InvokeAIAppConfig, logger: Logger) -> None:
        self._app_config = app_config
        self._logger = logger

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._rename_convert_cache()
        self._migrate_downloaded_models_cache()
        self._remove_unused_core_models()

    def _rename_convert_cache(self) -> None:
        """Rename models/.cache to models/.convert_cache."""
        legacy_convert_path = self._app_config.root_path / "models" / ".cache"
        configured_convert_dir = self._app_config.convert_cache_dir
        configured_convert_path = self._app_config.convert_cache_path
        # old convert dir was in use, and current convert dir has not been changed
        if legacy_convert_path.exists() and configured_convert_dir == pathlib.Path("models/.convert_cache"):
            self._logger.info(
                f"Migrating legacy convert cache directory from {str(legacy_convert_path)} to {str(configured_convert_path)}"
            )
            shutil.rmtree(configured_convert_path, ignore_errors=True)  # shouldn't be needed, but just in case...
            shutil.move(legacy_convert_path, configured_convert_path)

    def _migrate_downloaded_models_cache(self) -> None:
        """Move used core models to modsl/.download_cache."""
        self._logger.info(f"Migrating legacy core models to {str(self._app_config.download_cache_path)}")
        for url, legacy_dest in LEGACY_CORE_MODELS.items():
            legacy_dest_path = self._app_config.models_path / legacy_dest
            if not legacy_dest_path.exists():
                continue
            # this returns a unique directory path
            new_path = ModelInstallService._download_cache_path(url, self._app_config)
            new_path.mkdir(parents=True, exist_ok=True)
            shutil.move(legacy_dest_path, new_path / legacy_dest_path.name)

    def _remove_unused_core_models(self) -> None:
        """Remove unused core models and their directories."""
        self._logger.info("Removing defunct core models.")
        for dir in ["face_restoration", "misc", "upscaling"]:
            path_to_remove = self._app_config.models_path / "core" / dir
            shutil.rmtree(path_to_remove, ignore_errors=True)
        shutil.rmtree(self._app_config.models_path / "any" / "annotators", ignore_errors=True)


def build_migration_10(app_config: InvokeAIAppConfig, logger: Logger) -> Migration:
    """
    Build the migration from database version 9 to 10.

    This migration does the following:
    - Moves "core" models previously downloaded with download_with_progress_bar() into new
      "models/.download_cache" directory.
    - Renames "models/.cache" to "models/.convert_cache".
    """
    migration_10 = Migration(
        from_version=9,
        to_version=10,
        callback=Migration10Callback(app_config=app_config, logger=logger),
    )

    return migration_10
