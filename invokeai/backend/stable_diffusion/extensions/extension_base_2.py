from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, TypeVar

import torch
from diffusers import UNet2DConditionModel

from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType


@dataclass
class CallbackMetadata:
    callback_type: ExtensionCallbackType
    order: int


@dataclass
class CallbackFunctionWithMetadata:
    metadata: CallbackMetadata
    func: Callable[[DenoiseContext], None]


# A TypeVar that represents any subclass of ExtensionBase.
TExtensionBaseSubclass = TypeVar("TExtensionBaseSubclass", bound="ExtensionBase")


def callback(callback_type: ExtensionCallbackType, order: int = 0):
    """A decorator that marks an extension method as a callback."""

    def _decorator(func: Callable[[TExtensionBaseSubclass, DenoiseContext], None]):
        func._metadata = CallbackMetadata(callback_type, order)  # type: ignore
        return func

    return _decorator


class ExtensionBase:
    def __init__(self):
        self._callbacks: dict[ExtensionCallbackType, List[CallbackFunctionWithMetadata]] = {}

        # Register all of the callback methods for this instance.
        for func_name in dir(self):
            func = getattr(self, func_name)
            metadata = getattr(func, "_metadata", None)
            if metadata is not None and isinstance(metadata, CallbackMetadata):
                if metadata.callback_type not in self._callbacks:
                    self._callbacks[metadata.callback_type] = []
                self._callbacks[metadata.callback_type].append(CallbackFunctionWithMetadata(metadata, func))

    def get_callbacks(self):
        return self._callbacks

    @contextmanager
    def patch_attention_processor(self, attention_processor_cls: object):
        yield None

    @contextmanager
    def patch_unet(self, state_dict: Dict[str, torch.Tensor], unet: UNet2DConditionModel):
        yield None
