class CustomModuleMixin:
    """A mixin class for custom modules that enables device autocasting of module parameters."""

    _device_autocasting_enabled = False

    def set_device_autocasting_enabled(self, enabled: bool):
        """Pass True to enable autocasting of module parameters to the same device as the input tensor. Pass False to
        disable autocasting, which results in slightly faster execution speed when we know that device autocasting is
        not needed.
        """
        self._device_autocasting_enabled = enabled
