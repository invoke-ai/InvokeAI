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
