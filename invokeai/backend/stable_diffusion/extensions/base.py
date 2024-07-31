from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
from diffusers import UNet2DConditionModel

if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
    from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
    from invokeai.backend.stable_diffusion.extension_override_type import ExtensionOverrideType


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


@dataclass
class OverrideMetadata:
    override_type: ExtensionOverrideType


@dataclass
class OverrideFunctionWithMetadata:
    metadata: OverrideMetadata
    function: Callable[..., Any]


def override(override_type: ExtensionOverrideType):
    def _decorator(function):
        function._ext_metadata = OverrideMetadata(
            override_type=override_type,
        )
        return function

    return _decorator


class ExtensionBase:
    def __init__(self):
        self._callbacks: Dict[ExtensionCallbackType, List[CallbackFunctionWithMetadata]] = {}
        self._overrides: Dict[ExtensionOverrideType, OverrideFunctionWithMetadata] = {}

        # Register all of the callback methods for this instance.
        for func_name in dir(self):
            func = getattr(self, func_name)
            metadata = getattr(func, "_ext_metadata", None)
            if metadata is not None:
                if isinstance(metadata, CallbackMetadata):
                    if metadata.callback_type not in self._callbacks:
                        self._callbacks[metadata.callback_type] = []
                    self._callbacks[metadata.callback_type].append(CallbackFunctionWithMetadata(metadata, func))
                elif isinstance(metadata, OverrideMetadata):
                    if metadata.override_type in self._overrides:
                        raise RuntimeError(
                            f"Override {metadata.override_type} defined multiple times in {type(self).__qualname__}"
                        )
                    self._overrides[metadata.override_type] = OverrideFunctionWithMetadata(metadata, func)

    def get_callbacks(self):
        return self._callbacks

    def get_overrides(self):
        return self._overrides

    @contextmanager
    def patch_extension(self, ctx: DenoiseContext):
        yield None

    @contextmanager
    def patch_unet(self, unet: UNet2DConditionModel, cached_weights: Optional[Dict[str, torch.Tensor]] = None):
        yield None
