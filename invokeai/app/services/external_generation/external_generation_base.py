from __future__ import annotations

from abc import ABC, abstractmethod
from logging import Logger

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.external_generation.external_generation_common import (
    ExternalGenerationRequest,
    ExternalGenerationResult,
    ExternalProviderStatus,
)


class ExternalProvider(ABC):
    provider_id: str

    def __init__(self, app_config: InvokeAIAppConfig, logger: Logger) -> None:
        self._app_config = app_config
        self._logger = logger

    @abstractmethod
    def is_configured(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        raise NotImplementedError

    def generate_generic(self, model_id: str, payload: dict[str, object]) -> dict[str, object]:
        """Submit provider-specific JSON for endpoints outside normalized image generation."""
        raise NotImplementedError(f"Provider '{self.provider_id}' does not support generic media generation")

    def get_status(self) -> ExternalProviderStatus:
        return ExternalProviderStatus(provider_id=self.provider_id, configured=self.is_configured())


class ExternalGenerationServiceBase(ABC):
    @abstractmethod
    def generate(self, request: ExternalGenerationRequest) -> ExternalGenerationResult:
        raise NotImplementedError

    @abstractmethod
    def get_provider_statuses(self) -> dict[str, ExternalProviderStatus]:
        raise NotImplementedError

    @abstractmethod
    def generate_generic(self, provider_id: str, model_id: str, payload: dict[str, object]) -> dict[str, object]:
        raise NotImplementedError
