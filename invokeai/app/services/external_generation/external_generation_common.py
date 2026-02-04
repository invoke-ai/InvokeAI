from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL.Image import Image as PILImageType

from invokeai.backend.model_manager.configs.external_api import ExternalApiModelConfig, ExternalGenerationMode


@dataclass(frozen=True)
class ExternalReferenceImage:
    image: PILImageType
    weight: float | None = None
    mode: str | None = None


@dataclass(frozen=True)
class ExternalGenerationRequest:
    model: ExternalApiModelConfig
    mode: ExternalGenerationMode
    prompt: str
    negative_prompt: str | None
    seed: int | None
    num_images: int
    width: int
    height: int
    steps: int | None
    guidance: float | None
    init_image: PILImageType | None
    mask_image: PILImageType | None
    reference_images: list[ExternalReferenceImage]
    metadata: dict[str, Any] | None


@dataclass(frozen=True)
class ExternalGeneratedImage:
    image: PILImageType
    seed: int | None = None


@dataclass(frozen=True)
class ExternalGenerationResult:
    images: list[ExternalGeneratedImage]
    seed_used: int | None = None
    provider_request_id: str | None = None
    provider_metadata: dict[str, Any] | None = None
    content_filters: dict[str, str] | None = None


@dataclass(frozen=True)
class ExternalProviderStatus:
    provider_id: str
    configured: bool
    message: str | None = None
