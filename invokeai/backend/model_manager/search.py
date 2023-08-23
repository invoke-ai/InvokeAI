# Copyright 2023, Lincoln D. Stein and the InvokeAI Team
"""
Abstract base class for recursive directory search for models.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Set, Optional, Callable, Union, types
from pathlib import Path

import invokeai.backend.util.logging as logger


class ModelSearchBase(ABC):
    """Hierarchical directory model search class"""

    def __init__(self, logger: types.ModuleType = logger):
        """
        Initialize a recursive model directory search.
        :param directories: List of directory Paths to recurse through
        :param logger: Logger to use
        """
        self.logger = logger
        self._items_scanned = 0
        self._models_found = 0
        self._scanned_dirs = set()
        self._scanned_paths = set()
        self._pruned_paths = set()

    @abstractmethod
    def on_search_started(self):
        """
        Called before the scan starts.
        """
        pass

    @abstractmethod
    def on_model_found(self, model: Path):
        """
        Process a found model. Raise an exception if something goes wrong.
        :param model: Model to process - could be a directory or checkpoint.
        """
        pass

    @abstractmethod
    def on_search_completed(self):
        """
        Perform some activity when the scan is completed. May use instance
        variables, items_scanned and models_found
        """
        pass

    def search(self, directories: List[Union[Path, str]]):
        self.on_search_started()
        for dir in directories:
            self.walk_directory(dir)
        self.on_search_completed()

    def walk_directory(self, path: Union[Path, str]):
        for root, dirs, files in os.walk(path, followlinks=True):
            if str(Path(root).name).startswith("."):
                self._pruned_paths.add(root)
            if any([Path(root).is_relative_to(x) for x in self._pruned_paths]):
                continue

            self._items_scanned += len(dirs) + len(files)
            for d in dirs:
                path = Path(root) / d
                if path in self._scanned_paths or path.parent in self._scanned_dirs:
                    self._scanned_dirs.add(path)
                    continue
                if any(
                    [
                        (path / x).exists()
                        for x in ["config.json", "model_index.json", "learned_embeds.bin", "pytorch_lora_weights.bin"]
                    ]
                ):
                    try:
                        self.on_model_found(path)
                        self._models_found += 1
                        self._scanned_dirs.add(path)
                    except Exception as e:
                        self.logger.warning(str(e))

            for f in files:
                path = Path(root) / f
                if path.parent in self._scanned_dirs:
                    continue
                if path.suffix in {".ckpt", ".bin", ".pth", ".safetensors", ".pt"}:
                    try:
                        self.on_model_found(path)
                        self._models_found += 1
                    except Exception as e:
                        self.logger.warning(str(e))


class ModelSearch(ModelSearchBase):
    """
    Implementation of ModelSearch with callbacks.
    Usage:
       search = ModelSearch()
       search.model_found = lambda path : 'anime' in path.as_posix()
       found = search.list_models(['/tmp/models1','/tmp/models2'])
       # returns all models that have 'anime' in the path
    """

    _model_set: Set[Path]
    search_started: Callable[[Path], None]
    search_completed: Callable[[Set[Path]], None]
    model_found: Callable[[Path], bool]

    def __init__(self, logger: types.ModuleType = logger):
        super().__init__(logger)
        self._model_set = set()
        self.search_started = None
        self.search_completed = None
        self.model_found = None

    def on_search_started(self):
        self._model_set = set()
        if self.search_started:
            self.search_started()

    def on_model_found(self, model: Path):
        if not self.model_found:
            self._model_set.add(model)
            return
        if self.model_found(model):
            self._model_set.add(model)

    def on_search_completed(self):
        if self.search_completed:
            self.search_completed(self._model_set)

    def list_models(self, directories: List[Union[Path, str]]) -> List[Path]:
        """Return list of models found"""
        self.search(directories)
        return list(self._model_set)
