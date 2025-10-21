"""Cloud model loader for cloud-based image generation services."""

import os
from pathlib import Path
from typing import Optional

from invokeai.app.services.cloud_providers.google_gemini_provider import GoogleGeminiProvider
from invokeai.app.services.cloud_providers.provider_base import CloudModelProviderBase
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.load.load_base import LoadedModel, ModelLoaderBase
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.taxonomy import (
    BaseModelType,
    CloudProviderType,
    ModelFormat,
    ModelType,
    SubModelType,
)


class CloudModelWrapper:
    """Lightweight wrapper for cloud model providers.

    This wrapper is what gets "cached" when a cloud model is loaded. It's
    essentially just a holder for the provider instance and configuration.
    Unlike local models, cloud models don't need to be loaded into memory.
    """

    def __init__(self, config: AnyModelConfig, provider: CloudModelProviderBase):
        """Initialize cloud model wrapper.

        Args:
            config: Cloud model configuration
            provider: Cloud provider instance (Gemini, OpenAI, etc.)
        """
        self.config = config
        self.provider = provider

    async def generate(self, request):
        """Generate images via cloud provider.

        Args:
            request: CloudGenerationRequest

        Returns:
            CloudGenerationResponse
        """
        return await self.provider.generate_image(request)


@ModelLoaderRegistry.register(base=BaseModelType.CloudAPI, type=ModelType.Main, format=ModelFormat.CloudREST)
class CloudModelLoader(ModelLoaderBase):
    """Loader for cloud-based image generation models.

    Unlike traditional model loaders that load weights from disk into memory,
    this loader creates a lightweight wrapper around a cloud provider instance.
    The actual "model" lives in the cloud and is accessed via API calls.
    """

    def load_model(self, model_config: AnyModelConfig, submodel_type: Optional[SubModelType] = None) -> LoadedModel:
        """'Load' a cloud model by creating a provider wrapper.

        Args:
            model_config: Cloud model configuration
            submodel_type: Not used for cloud models

        Returns:
            LoadedModel containing the cloud model wrapper

        Raises:
            ValueError: If provider is not supported or API key is missing
        """
        # Get API key from environment
        api_key = self._get_api_key(model_config.provider)

        # Create provider instance based on provider type
        provider = self._create_provider(model_config.provider, api_key, model_config.provider_settings)

        # Wrap in CloudModelWrapper
        cloud_model = CloudModelWrapper(config=model_config, provider=provider)

        # Add to cache with nominal size (it's just a wrapper, not actual model weights)
        cache_record = self._ram_cache.put(
            key=model_config.key,
            model=cloud_model,
            size=1024,  # Nominal size in bytes (1KB)
        )

        return LoadedModel(config=model_config, cache_record=cache_record, cache=self._ram_cache)

    def get_size_fs(
        self, config: AnyModelConfig, model_path: Path, submodel_type: Optional[SubModelType] = None
    ) -> int:
        """Get filesystem size of model.

        Cloud models have no filesystem footprint.

        Returns:
            0 (cloud models don't use disk space)
        """
        return 0

    def _get_api_key(self, provider: CloudProviderType) -> str:
        """Get API key for the specified provider from environment.

        Attempts to load from .env file first, then falls back to environment variables.

        Args:
            provider: Cloud provider type

        Returns:
            API key string

        Raises:
            ValueError: If API key is not found
        """
        # Try to load from .env file
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            # python-dotenv not installed, skip .env loading
            pass

        # Map provider to environment variable name
        key_mapping = {
            CloudProviderType.GoogleGemini: "GOOGLE_API_KEY",
            CloudProviderType.GoogleImagen: "GOOGLE_API_KEY",  # Imagen uses same key
            CloudProviderType.OpenAI: "OPENAI_API_KEY",
        }

        env_var = key_mapping.get(provider)
        if not env_var:
            raise ValueError(f"Unknown cloud provider: {provider}")

        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(
                f"API key not found for {provider}. "
                f"Please set {env_var} in .env file or environment variables."
            )

        return api_key

    def _create_provider(
        self, provider_type: CloudProviderType, api_key: str, config: dict
    ) -> CloudModelProviderBase:
        """Create a provider instance for the specified type.

        Args:
            provider_type: Type of cloud provider
            api_key: API key for authentication
            config: Provider-specific configuration

        Returns:
            CloudModelProviderBase instance

        Raises:
            ValueError: If provider type is not supported
        """
        if provider_type == CloudProviderType.GoogleGemini:
            return GoogleGeminiProvider(api_key=api_key, config=config)
        elif provider_type == CloudProviderType.GoogleImagen:
            # Import here to avoid circular imports
            from invokeai.app.services.cloud_providers.google_imagen_provider import GoogleImagenProvider

            return GoogleImagenProvider(api_key=api_key, config=config)
        elif provider_type == CloudProviderType.OpenAI:
            # Import here to avoid circular imports
            from invokeai.app.services.cloud_providers.openai_provider import OpenAIProvider

            return OpenAIProvider(api_key=api_key, config=config)
        else:
            raise ValueError(f"Unsupported cloud provider: {provider_type}")
