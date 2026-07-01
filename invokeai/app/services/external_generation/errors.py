class ExternalGenerationError(Exception):
    """Base error for external generation."""


class ExternalProviderNotFoundError(ExternalGenerationError):
    """Raised when no provider is registered for a model."""


class ExternalProviderNotConfiguredError(ExternalGenerationError):
    """Raised when a provider is missing required credentials."""


class ExternalProviderCapabilityError(ExternalGenerationError):
    """Raised when a request is not supported by provider capabilities."""


class ExternalProviderRequestError(ExternalGenerationError):
    """Raised when a provider rejects the request or returns an error."""


class ExternalProviderRateLimitError(ExternalProviderRequestError):
    """Raised when a provider returns HTTP 429 (rate limit exceeded)."""

    retry_after: float | None

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
