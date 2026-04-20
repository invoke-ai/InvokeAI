from unittest.mock import MagicMock

from invokeai.app.services.external_generation.startup import sync_configured_external_starter_models
from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig, ExternalModelCapabilities


def _build_installed_model(source: str) -> ExternalApiModelConfig:
    provider_id, provider_model_id = source.removeprefix("external://").split("/", 1)
    return ExternalApiModelConfig(
        key=f"{provider_id}-{provider_model_id}",
        name=provider_model_id,
        source=source,
        provider_id=provider_id,
        provider_model_id=provider_model_id,
        capabilities=ExternalModelCapabilities(modes=["txt2img"]),
    )


def test_sync_configured_external_starter_models_queues_missing_models_for_configured_providers() -> None:
    model_manager = MagicMock()
    model_manager.store.search_by_attr.return_value = [
        _build_installed_model("external://openai/gpt-image-1"),
    ]
    logger = MagicMock()

    queued_sources = sync_configured_external_starter_models(
        configured_provider_ids={"gemini", "openai"},
        model_manager=model_manager,
        logger=logger,
    )

    assert "external://openai/gpt-image-1" not in queued_sources
    assert "external://gemini/gemini-2.5-flash-image" in queued_sources
    assert "external://gemini/gemini-3.1-flash-image-preview" in queued_sources
    assert "external://gemini/gemini-3-pro-image-preview" in queued_sources

    install_calls = [call.args[0] for call in model_manager.install.heuristic_import.call_args_list]
    assert "external://openai/gpt-image-1" not in install_calls
    assert "external://gemini/gemini-2.5-flash-image" in install_calls
    assert "external://gemini/gemini-3.1-flash-image-preview" in install_calls
    assert "external://gemini/gemini-3-pro-image-preview" in install_calls


def test_sync_configured_external_starter_models_skips_when_no_provider_is_configured() -> None:
    model_manager = MagicMock()
    logger = MagicMock()

    queued_sources = sync_configured_external_starter_models(
        configured_provider_ids=set(),
        model_manager=model_manager,
        logger=logger,
    )

    assert queued_sources == []
    model_manager.store.search_by_attr.assert_not_called()
    model_manager.install.heuristic_import.assert_not_called()
