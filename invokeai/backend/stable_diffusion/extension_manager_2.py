from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.extension_base_2 import CallbackFunctionWithMetadata, ExtensionBase


class ExtensionManager:
    def __init__(self):
        self._extensions: list[ExtensionBase] = []
        self._ordered_callbacks: dict[ExtensionCallbackType, list[CallbackFunctionWithMetadata]] = {}

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
            self._ordered_callbacks[callback_type] = sorted(callbacks, key=lambda x: x.metadata.order)

    def run_callback(self, callback_type: ExtensionCallbackType, ctx: DenoiseContext):
        cbs = self._ordered_callbacks.get(callback_type, [])

        for cb in cbs:
            cb.func(ctx)
