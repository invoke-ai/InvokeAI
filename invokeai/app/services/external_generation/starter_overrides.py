from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig, ExternalModelCapabilities
from invokeai.backend.model_manager.starter_models import STARTER_MODELS

_UNSET_CAPABILITIES = ExternalModelCapabilities()


def apply_external_starter_overrides(config: ExternalApiModelConfig) -> ExternalApiModelConfig:
    """Fill a starter-sourced record's unset fields from its starter definition.

    A record that never set a field (the default capability matrix, no default settings, no panel schema)
    follows the shipped starter definition, so definition updates reach it. A field the record sets, by an
    install that copied the definition or by an edit in the model editor, is the record's own and wins.
    """
    source = config.source or f"external://{config.provider_id}/{config.provider_model_id}"
    starter_match = next((starter for starter in STARTER_MODELS if starter.source == source), None)
    if starter_match is None:
        return config

    updates: dict[str, object] = {}
    if starter_match.capabilities is not None and config.capabilities == _UNSET_CAPABILITIES:
        updates["capabilities"] = starter_match.capabilities
    if starter_match.default_settings is not None and config.default_settings is None:
        updates["default_settings"] = starter_match.default_settings
    if starter_match.panel_schema is not None and config.panel_schema is None:
        updates["panel_schema"] = starter_match.panel_schema

    if not updates:
        return config

    return config.model_copy(update=updates)
