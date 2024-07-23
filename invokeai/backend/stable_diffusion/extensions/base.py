from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch
from diffusers import UNet2DConditionModel

if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
    from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType


@dataclass
class CallbackMetadata:
    callback_type: ExtensionCallbackType
    order: int


@dataclass
class CallbackFunctionWithMetadata:
    metadata: CallbackMetadata
    function: Callable[[DenoiseContext], None]


def callback(callback_type: ExtensionCallbackType, order: int = 0):
    def _decorator(function):
        function._ext_metadata = CallbackMetadata(
            callback_type=callback_type,
            order=order,
        )
        return function

    return _decorator


class ExtensionBase:
    def __init__(self):
        self._callbacks: Dict[ExtensionCallbackType, List[CallbackFunctionWithMetadata]] = {}

        # Register all of the callback methods for this instance.
        for func_name in dir(self):
            func = getattr(self, func_name)
            metadata = getattr(func, "_ext_metadata", None)
            if metadata is not None and isinstance(metadata, CallbackMetadata):
                if metadata.callback_type not in self._callbacks:
                    self._callbacks[metadata.callback_type] = []
                self._callbacks[metadata.callback_type].append(CallbackFunctionWithMetadata(metadata, func))

    def get_callbacks(self):
        return self._callbacks

    @contextmanager
    def patch_extension(self, ctx: DenoiseContext):
        yield None

    @contextmanager
    def patch_unet(self, unet: UNet2DConditionModel, cached_weights: Optional[Dict[str, torch.Tensor]] = None):
        yield None
