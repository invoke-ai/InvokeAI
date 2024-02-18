# Copyright (c) 2024 Lincoln D. Stein and the InvokeAI Development Team
"""
Init file for the model loader.
"""
from importlib import import_module
from pathlib import Path

from .convert_cache.convert_cache_default import ModelConvertCache
from .load_base import LoadedModel, ModelLoaderBase
from .load_default import ModelLoader
from .model_cache.model_cache_default import ModelCache
from .model_loader_registry import ModelLoaderRegistry, ModelLoaderRegistryBase

# This registers the subclasses that implement loaders of specific model types
loaders = [x.stem for x in Path(Path(__file__).parent, "model_loaders").glob("*.py") if x.stem != "__init__"]
for module in loaders:
    import_module(f"{__package__}.model_loaders.{module}")

__all__ = [
    "LoadedModel",
    "ModelCache",
    "ModelConvertCache",
    "ModelLoaderBase",
    "ModelLoader",
    "ModelLoaderRegistryBase",
    "ModelLoaderRegistry",
]
