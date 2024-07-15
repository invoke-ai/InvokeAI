from unittest import mock

import pytest

from invokeai.backend.stable_diffusion.denoise_context import DenoiseContext
from invokeai.backend.stable_diffusion.extension_callback_type import ExtensionCallbackType
from invokeai.backend.stable_diffusion.extension_manager_2 import ExtensionManager
from invokeai.backend.stable_diffusion.extensions.extension_base_2 import (
    ExtensionBase,
    callback,
)


class MockExtension(ExtensionBase):
    """A mock ExtensionBase subclass for testing purposes."""

    def __init__(self, x: int):
        super().__init__()
        self._x = x

    # Note that order is not specified. It should default to 0.
    @callback(ExtensionCallbackType.PRE_DENOISE_LOOP)
    def set_step_index(self, ctx: DenoiseContext):
        ctx.step_index = self._x


class MockExtensionLate(ExtensionBase):
    """A mock ExtensionBase subclass with a high order value on its PRE_DENOISE_LOOP callback."""

    def __init__(self, x: int):
        super().__init__()
        self._x = x

    @callback(ExtensionCallbackType.PRE_DENOISE_LOOP, order=1000)
    def set_step_index(self, ctx: DenoiseContext):
        ctx.step_index = self._x


def test_extension_manager_run_callback():
    """Test that run_callback runs all callbacks for the given callback type."""

    em = ExtensionManager()
    mock_extension_1 = MockExtension(1)
    em.add_extension(mock_extension_1)

    mock_ctx = mock.MagicMock()
    em.run_callback(ExtensionCallbackType.PRE_DENOISE_LOOP, mock_ctx)

    assert mock_ctx.step_index == 1


def test_extension_manager_run_callback_no_callbacks():
    """Test that run_callback does not raise an error when there are no callbacks for the given callback type."""
    em = ExtensionManager()
    mock_ctx = mock.MagicMock()
    em.run_callback(ExtensionCallbackType.PRE_DENOISE_LOOP, mock_ctx)


@pytest.mark.parametrize(
    ["extension_1", "extension_2"],
    # Regardless of initialization order, we expect MockExtensionLate to run last.
    [(MockExtension(1), MockExtensionLate(2)), (MockExtensionLate(2), MockExtension(1))],
)
def test_extension_manager_order_callbacks(extension_1: ExtensionBase, extension_2: ExtensionBase):
    """Test that run_callback runs callbacks in the correct order."""
    em = ExtensionManager()
    em.add_extension(extension_1)
    em.add_extension(extension_2)

    mock_ctx = mock.MagicMock()
    em.run_callback(ExtensionCallbackType.PRE_DENOISE_LOOP, mock_ctx)

    assert mock_ctx.step_index == 2
