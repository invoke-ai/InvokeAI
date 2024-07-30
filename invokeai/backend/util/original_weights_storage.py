from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple

import torch

from invokeai.backend.util.devices import TorchDevice


class OriginalWeightsStorage:
    def __init__(self, cached_weights: Optional[Dict[str, torch.Tensor]] = None):
        self._weights = {}
        self._changed_weights = set()
        if cached_weights:
            self._weights.update(cached_weights)

    def save(self, key: str, weight: torch.Tensor, copy: bool = True):
        self._changed_weights.add(key)
        if key in self._weights:
            return

        self._weights[key] = weight.detach().to(device=TorchDevice.CPU_DEVICE, copy=copy)

    def get(self, key: str, copy: bool = False) -> Optional[torch.Tensor]:
        weight = self._weights.get(key, None)
        if weight is not None and copy:
            weight = weight.clone()
        return weight

    def contains(self, key: str) -> bool:
        return key in self._weights

    def get_changed_weights(self) -> Iterator[Tuple[str, torch.Tensor]]:
        for key in self._changed_weights:
            yield key, self._weights[key]
