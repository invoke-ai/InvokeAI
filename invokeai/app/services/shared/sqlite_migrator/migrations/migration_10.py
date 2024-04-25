import shutil
import sqlite3
from logging import Logger

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration

LEGACY_CORE_MODELS = [
    # OpenPose
    "any/annotators/dwpose/yolox_l.onnx",
    "any/annotators/dwpose/dw-ll_ucoco_384.onnx",
    # DepthAnything
    "any/annotators/depth_anything/depth_anything_vitl14.pth",
    "any/annotators/depth_anything/depth_anything_vitb14.pth",
    "any/annotators/depth_anything/depth_anything_vits14.pth",
    # Lama inpaint
    "core/misc/lama/lama.pt",
    # RealESRGAN upscale
    "core/upscaling/realesrgan/RealESRGAN_x4plus.pth",
    "core/upscaling/realesrgan/RealESRGAN_x4plus_anime_6B.pth",
    "core/upscaling/realesrgan/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth",
    "core/upscaling/realesrgan/RealESRGAN_x2plus.pth",
]


class Migration10Callback:
    def __init__(self, app_config: InvokeAIAppConfig, logger: Logger) -> None:
        self._app_config = app_config
        self._logger = logger

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._remove_convert_cache()
        self._remove_downloaded_models()
        self._remove_unused_core_models()

    def _remove_convert_cache(self) -> None:
        """Rename models/.cache to models/.convert_cache."""
        self._logger.info("Removing .cache directory. Converted models will now be cached in .convert_cache.")
        legacy_convert_path = self._app_config.root_path / "models" / ".cache"
        shutil.rmtree(legacy_convert_path, ignore_errors=True)

    def _remove_downloaded_models(self) -> None:
        """Remove models from their old locations; they will re-download when needed."""
        self._logger.info(
            "Removing legacy just-in-time models. Downloaded models will now be cached in .download_cache."
        )
        for model_path in LEGACY_CORE_MODELS:
            legacy_dest_path = self._app_config.models_path / model_path
            legacy_dest_path.unlink(missing_ok=True)

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
