from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List

from diffusers import UNet2DConditionModel

if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
    from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
    from invokeai.backend.util.original_weights_storage import OriginalWeightsStorage


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
    def patch_unet(self, unet: UNet2DConditionModel, original_weights: OriginalWeightsStorage):
        """A context manager for applying patches to the UNet model. The context manager's lifetime spans the entire
        diffusion process. Weight unpatching is handled upstream, and is achieved by saving unchanged weights by
        `original_weights.save` function. Note that this enables some performance optimization by avoiding redundant
        operations. All other patches (e.g. changes to tensor shapes, function monkey-patches, etc.) should be unpatched
        by this context manager.

        Args:
            unet (UNet2DConditionModel): The UNet model on execution device to patch.
            original_weights (OriginalWeightsStorage): A storage with copy of the model's original weights in CPU, for
                unpatching purposes. Extension should save tensor which being modified in this storage, also extensions
                can access original weights values.
        """
        yield
