from unittest import mock

from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extensions.base import ExtensionBase, callback


class MockExtension(ExtensionBase):
    """A mock ExtensionBase subclass for testing purposes."""

    def __init__(self, x: int):
        super().__init__()
        self._x = x

    @callback(ExtensionCallbackType.PRE_DENOISE_LOOP)
    def set_step_index(self, ctx: DenoiseContext):
        ctx.step_index = self._x


def test_extension_base_callback_registration():
    """Test that a callback can be successfully registered with an extension."""
    val = 5
    mock_extension = MockExtension(val)

    mock_ctx = mock.MagicMock()

    callbacks = mock_extension.get_callbacks()
    pre_denoise_loop_cbs = callbacks.get(ExtensionCallbackType.PRE_DENOISE_LOOP, [])
    assert len(pre_denoise_loop_cbs) == 1

    # Call the mock callback.
    pre_denoise_loop_cbs[0].function(mock_ctx)

    # Confirm that the callback ran.
    assert mock_ctx.step_index == val


def test_extension_base_empty_callback_type():
    """Test that an empty list is returned when no callbacks are registered for a given callback type."""
    mock_extension = MockExtension(5)

    # There should be no callbacks registered for POST_DENOISE_LOOP.
    callbacks = mock_extension.get_callbacks()

    post_denoise_loop_cbs = callbacks.get(ExtensionCallbackType.POST_DENOISE_LOOP, [])
    assert len(post_denoise_loop_cbs) == 0
