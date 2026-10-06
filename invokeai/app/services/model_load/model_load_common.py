"""Shared rules for when a model record change invalidates loaded instances of that model."""

from invokeai.backend.model_manager.configs.factory import AnyModelConfig

_LOAD_AFFECTING_SETTINGS: tuple[str, ...] = ("fp8_storage", "cpu_only")


def load_settings_changed(previous: AnyModelConfig, updated: AnyModelConfig) -> bool:
    """Return True if any setting that influences how the model is loaded changed.

    Such settings are read by the loader during `_load_model` and baked into the resulting
    nn.Module, so a cached entry built under the old value must be evicted for the change
    to take effect, and a load that started from the old value must not be admitted after
    that eviction.
    """
    if getattr(previous, "cpu_only", None) != getattr(updated, "cpu_only", None):
        return True
    previous_settings = getattr(previous, "default_settings", None)
    updated_settings = getattr(updated, "default_settings", None)
    for field in _LOAD_AFFECTING_SETTINGS:
        if getattr(previous_settings, field, None) != getattr(updated_settings, field, None):
            return True
    return False
