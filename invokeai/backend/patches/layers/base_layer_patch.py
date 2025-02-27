from abc import ABC, abstractmethod

import torch


class BaseLayerPatch(ABC):
    @abstractmethod
    def get_parameters(self, orig_parameters: dict[str, torch.Tensor], weight: float) -> dict[str, torch.Tensor]:
        """Get the parameter residual updates that should be applied to the original parameters. Parameters omitted
        from the returned dict are not updated.
        """
        ...

    @abstractmethod
    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        """Move all internal tensors to the specified device and dtype."""
        ...

    @abstractmethod
    def calc_size(self) -> int:
        """Calculate the total size of all internal tensors in bytes."""
        ...
