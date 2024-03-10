# Copyright 2023, Lincoln D. Stein and the InvokeAI Team
"""
Abstract base class and implementation for recursive directory search for models.

Example usage:
```
  from invokeai.backend.model_manager import ModelSearch, ModelProbe

  def find_main_models(model: Path) -> bool:
    info = ModelProbe.probe(model)
    if info.model_type == 'main' and info.base_type == 'sd-1':
        return True
    else:
        return False

  search = ModelSearch(on_model_found=report_it)
  found = search.search('/tmp/models')
  print(found)   #  list of matching model paths
  print(search.stats)  #  search stats
```
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from invokeai.backend.util.logging import InvokeAILogger


@dataclass
class SearchStats:
    """Statistics about the search.

    Attributes:
        items_scanned: number of items scanned
        models_found: number of models found
        models_filtered: number of models that passed the filter
    """

    items_scanned = 0
    models_found = 0
    models_filtered = 0


class ModelSearch:
    """Searches a directory tree for models, using a callback to filter the results.

    Usage:
        search = ModelSearch()
        search.model_found = lambda path : 'anime' in path.as_posix()
        found = search.list_models(['/tmp/models1','/tmp/models2'])
        # returns all models that have 'anime' in the path
    """

    def __init__(
        self,
        on_search_started: Optional[Callable[[Path], None]] = None,
        on_model_found: Optional[Callable[[Path], bool]] = None,
        on_search_completed: Optional[Callable[[set[Path]], None]] = None,
    ) -> None:
        """Create a new ModelSearch object.

        Args:
            on_search_started: callback to be invoked when the search starts
            on_model_found: callback to be invoked when a model is found. The callback should return True if the model
                should be included in the results.
            on_search_completed: callback to be invoked when the search is completed
        """
        self.stats = SearchStats()
        self.logger = InvokeAILogger.get_logger()
        self.on_search_started = on_search_started
        self.on_model_found = on_model_found
        self.on_search_completed = on_search_completed
        self.models_found: set[Path] = set()

    def search_started(self) -> None:
        self.models_found = set()
        if self.on_search_started:
            self.on_search_started(self._directory)

    def model_found(self, model: Path) -> None:
        self.stats.models_found += 1
        if self.on_model_found is None or self.on_model_found(model):
            self.stats.models_filtered += 1
            self.models_found.add(model)

    def search_completed(self) -> None:
        if self.on_search_completed is not None:
            self.on_search_completed(self.models_found)

    def search(self, directory: Path) -> set[Path]:
        self._directory = Path(directory)
        self._directory = self._directory.resolve()
        self.stats = SearchStats()  # zero out
        self.search_started()  # This will initialize _models_found to empty
        self._walk_directory(self._directory)
        self.search_completed()
        return self.models_found

    def _walk_directory(self, path: Path, max_depth: int = 20) -> None:
        """Recursively walk the directory tree, looking for models."""
        absolute_path = Path(path)
        if (
            len(absolute_path.parts) - len(self._directory.parts) > max_depth
            or not absolute_path.exists()
            or absolute_path.parent in self.models_found
        ):
            return
        entries = os.scandir(absolute_path.as_posix())
        entries = [entry for entry in entries if not entry.name.startswith(".")]
        dirs = [entry for entry in entries if entry.is_dir()]
        file_names = [entry.name for entry in entries if entry.is_file()]
        if any(
            x in file_names
            for x in [
                "config.json",
                "model_index.json",
                "learned_embeds.bin",
                "pytorch_lora_weights.bin",
                "image_encoder.txt",
            ]
        ):
            try:
                self.model_found(absolute_path)
                return
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self.logger.warning(str(e))
                return

        for n in file_names:
            if n.endswith((".ckpt", ".bin", ".pth", ".safetensors", ".pt")):
                try:
                    self.model_found(absolute_path / n)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    self.logger.warning(str(e))

        for d in dirs:
            self._walk_directory(absolute_path / d)
