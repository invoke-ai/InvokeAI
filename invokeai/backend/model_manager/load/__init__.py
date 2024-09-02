# Copyright (c) 2024 Lincoln D. Stein and the InvokeAI Development Team
"""
Init file for the model loader.
"""

from importlib import import_module
from pathlib import Path

from invokeai.backend.model_manager.load.load_base import LoadedModel, LoadedModelWithoutConfig, ModelLoaderBase
from invokeai.backend.model_manager.load.load_default import ModelLoader
from invokeai.backend.model_manager.load.model_cache.model_cache_default import ModelCache
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry, ModelLoaderRegistryBase

# This registers the subclasses that implement loaders of specific model types
loaders = [x.stem for x in Path(Path(__file__).parent, "model_loaders").glob("*.py") if x.stem != "__init__"]
for module in loaders:
    import_module(f"{__package__}.model_loaders.{module}")

__all__ = [
    "LoadedModel",
    "LoadedModelWithoutConfig",
    "ModelCache",
    "ModelLoaderBase",
    "ModelLoader",
    "ModelLoaderRegistryBase",
    "ModelLoaderRegistry",
]
