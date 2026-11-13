import copy
from abc import ABC, abstractmethod

import torch


class BaseLayerPatch(ABC):
    def __copy__(self):
        """Shallow copy that also copies sub-patches, so the copy's `to()` cannot move the original's.

        The sidecar forwards move `copy.copy(patch)` to the input device. Wrapper patches delegate `to()` to
        their sub-patches, which a plain shallow copy would share with the cached patch.
        """
        clone = object.__new__(type(self))
        clone.__dict__ = {name: _copy_sub_patches(value) for name, value in self.__dict__.items()}
        return clone

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


def _held_patches(value: object) -> list[BaseLayerPatch]:
    """The patches an attribute holds: the value itself if it is a patch, or the patch items of a list or tuple."""
    if isinstance(value, BaseLayerPatch):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, BaseLayerPatch)]
    return []


def _copy_sub_patches(value: object) -> object:
    """`value` with every held patch replaced by a copy; containers without patches and other values are shared."""
    if isinstance(value, BaseLayerPatch):
        return copy.copy(value)
    if not _held_patches(value):
        return value
    items = [copy.copy(item) if isinstance(item, BaseLayerPatch) else item for item in value]  # type: ignore[attr-defined]
    return items if isinstance(value, list) else tuple(items)


def sub_patches(patch: BaseLayerPatch) -> list[BaseLayerPatch]:
    """The direct sub-patches of a wrapper patch: attributes that are patches, alone or in a list or tuple."""
    return [item for value in vars(patch).values() for item in _held_patches(value)]
