# Copyright (c) 2024 Lincoln D. Stein and the InvokeAI Development Team
"""
Init file for the model loader.
"""
from importlib import import_module
from pathlib import Path
from typing import Optional

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.backend.util.logging import InvokeAILogger

from .convert_cache.convert_cache_default import ModelConvertCache
from .load_base import AnyModelLoader, LoadedModel
from .model_cache.model_cache_default import ModelCache

# This registers the subclasses that implement loaders of specific model types
loaders = [x.stem for x in Path(Path(__file__).parent, "model_loaders").glob("*.py") if x.stem != "__init__"]
for module in loaders:
    import_module(f"{__package__}.model_loaders.{module}")

__all__ = ["AnyModelLoader", "LoadedModel", "ModelCache", "ModelConvertCache"]


def get_standalone_loader(app_config: Optional[InvokeAIAppConfig]) -> AnyModelLoader:
    app_config = app_config or InvokeAIAppConfig.get_config()
    logger = InvokeAILogger.get_logger(config=app_config)
    return AnyModelLoader(
        app_config=app_config,
        logger=logger,
        ram_cache=ModelCache(
            logger=logger, max_cache_size=app_config.ram_cache_size, max_vram_cache_size=app_config.vram_cache_size
        ),
        convert_cache=ModelConvertCache(app_config.models_convert_cache_path),
    )
