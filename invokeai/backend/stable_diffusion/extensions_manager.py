from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

import torch
from diffusers import UNet2DConditionModel

from invokeai.app.services.session_processor.session_processor_common import CanceledException
from invokeai.backend.util.original_weights_storage import OriginalWeightsStorage

if TYPE_CHECKING:
    from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
    from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
    from invokeai.backend.stable_diffusion.extensions.base import CallbackFunctionWithMetadata, ExtensionBase


class ExtensionsManager:
    def __init__(self, is_canceled: Optional[Callable[[], bool]] = None):
        self._is_canceled = is_canceled

        # A list of extensions in the order that they were added to the ExtensionsManager.
        self._extensions: List[ExtensionBase] = []
        self._ordered_callbacks: Dict[ExtensionCallbackType, List[CallbackFunctionWithMetadata]] = {}

    def add_extension(self, extension: ExtensionBase):
        self._extensions.append(extension)
        self._regenerate_ordered_callbacks()

    def _regenerate_ordered_callbacks(self):
        """Regenerates self._ordered_callbacks. Intended to be called each time a new extension is added."""
        self._ordered_callbacks = {}

        # Fill the ordered callbacks dictionary.
        for extension in self._extensions:
            for callback_type, callbacks in extension.get_callbacks().items():
                if callback_type not in self._ordered_callbacks:
                    self._ordered_callbacks[callback_type] = []
                self._ordered_callbacks[callback_type].extend(callbacks)

        # Sort each callback list.
        for callback_type, callbacks in self._ordered_callbacks.items():
            # Note that sorted() is stable, so if two callbacks have the same order, the order that they extensions were
            # added will be preserved.
            self._ordered_callbacks[callback_type] = sorted(callbacks, key=lambda x: x.metadata.order)

    def run_callback(self, callback_type: ExtensionCallbackType, ctx: DenoiseContext):
        if self._is_canceled and self._is_canceled():
            raise CanceledException

        callbacks = self._ordered_callbacks.get(callback_type, [])
        for cb in callbacks:
            cb.function(ctx)

    @contextmanager
    def patch_extensions(self, ctx: DenoiseContext):
        if self._is_canceled and self._is_canceled():
            raise CanceledException

        with ExitStack() as exit_stack:
            for ext in self._extensions:
                exit_stack.enter_context(ext.patch_extension(ctx))

            yield None

    @contextmanager
    def patch_unet(self, unet: UNet2DConditionModel, cached_weights: Optional[Dict[str, torch.Tensor]] = None):
        if self._is_canceled and self._is_canceled():
            raise CanceledException

        original_weights = OriginalWeightsStorage(cached_weights)
        try:
            with ExitStack() as exit_stack:
                for ext in self._extensions:
                    exit_stack.enter_context(ext.patch_unet(unet, original_weights))

                yield None

        finally:
            with torch.no_grad():
                for param_key, weight in original_weights.get_changed_weights():
                    unet.get_parameter(param_key).copy_(weight)
