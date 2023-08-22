# Copyright 2023, Lincoln D. Stein and the InvokeAI Team
"""
Abstract base class for recursive directory search for models.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Set, types
from pathlib import Path

import invokeai.backend.util.logging as logger


class ModelSearch(ABC):
    def __init__(self, directories: List[Path], logger: types.ModuleType = logger):
        """
        Initialize a recursive model directory search.
        :param directories: List of directory Paths to recurse through
        :param logger: Logger to use
        """
        self.directories = directories
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

    def search(self):
        self.on_search_started()
        for dir in self.directories:
            self.walk_directory(dir)
        self.on_search_completed()

    def walk_directory(self, path: Path):
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
                        for x in {"config.json", "model_index.json", "learned_embeds.bin", "pytorch_lora_weights.bin"}
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


class FindModels(ModelSearch):
    def on_search_started(self):
        self.models_found: Set[Path] = set()

    def on_model_found(self, model: Path):
        self.models_found.add(model)

    def on_search_completed(self):
        pass

    def list_models(self) -> List[Path]:
        self.search()
        return list(self.models_found)
