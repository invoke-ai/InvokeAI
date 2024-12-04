from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class CacheRecord:
    """
    Elements of the cache:

    key: Unique key for each model, same as used in the models database.
    model: Model in memory.
    state_dict: A read-only copy of the model's state dict in RAM. It will be
                used as a template for creating a copy in the VRAM.
    size: Size of the model
    loaded: True if the model's state dict is currently in VRAM

    Before a model is executed, the state_dict template is copied into VRAM,
    and then injected into the model. When the model is finished, the VRAM
    copy of the state dict is deleted, and the RAM version is reinjected
    into the model.

    The state_dict should be treated as a read-only attribute. Do not attempt
    to patch or otherwise modify it. Instead, patch the copy of the state_dict
    after it is loaded into the execution device (e.g. CUDA) using the `LoadedModel`
    context manager call `model_on_device()`.
    """

    key: str
    model: Any
    device: torch.device
    state_dict: Optional[Dict[str, torch.Tensor]]
    size: int
    loaded: bool = False
    _locks: int = 0

    def lock(self) -> None:
        """Lock this record."""
        self._locks += 1

    def unlock(self) -> None:
        """Unlock this record."""
        self._locks -= 1
        assert self._locks >= 0

    @property
    def locked(self) -> bool:
        """Return true if record is locked."""
        return self._locks > 0
