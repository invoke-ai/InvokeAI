"""The shape of a starter model entry.

Split out so the per-architecture modules can import it without importing each other.
"""

from typing import Optional, Self

from pydantic import BaseModel, Field, model_validator

from invokeai.backend.architectures import LatentSpaceFacet, generative_bases, get, latent_space_with_channels
from invokeai.backend.model_manager.configs.external_api import (
    ExternalApiModelDefaultSettings,
    ExternalModelCapabilities,
    ExternalModelPanelSchema,
)
from invokeai.backend.model_manager.taxonomy import (
    AnyVariant,
    BaseModelType,
    ModelFormat,
    ModelType,
)


class StarterModelWithoutDependencies(BaseModel):
    description: str
    source: str
    name: str
    base: BaseModelType
    type: ModelType
    format: Optional[ModelFormat] = None
    variant: Optional[AnyVariant] = None
    is_installed: bool = False
    capabilities: ExternalModelCapabilities | None = None
    default_settings: ExternalApiModelDefaultSettings | None = None
    panel_schema: ExternalModelPanelSchema | None = None
    # allows us to track what models a user has installed across name changes within starter models
    # if you update a starter model name, please add the old one to this list for that starter model
    previous_names: list[str] = []
    compatible_bases: list[BaseModelType] = Field(
        default=[], description="Other bases this model also serves, besides `base`."
    )

    @model_validator(mode="after")
    def _derive_compatible_bases(self) -> Self:
        # A TAE serves every base whose latents resolve to the same space object as its own base's, the
        # same pairing `latent_space_with_channels` makes for an installed TAE record. Starters carry no
        # channel count, so the base's primary space stands in for it.
        if self.type is ModelType.TinyAutoencoder:
            facet = get(self.base, LatentSpaceFacet)
            if facet is not None:
                space = facet.primary
                self.compatible_bases = sorted(
                    (
                        b
                        for b in generative_bases()
                        if b is not self.base and latent_space_with_channels(b, space.channels) is space
                    ),
                    key=lambda b: b.value,
                )
        return self


class StarterModel(StarterModelWithoutDependencies):
    # Optional list of model source dependencies that need to be installed before this model can be used
    dependencies: Optional[list[StarterModelWithoutDependencies]] = None


class StarterModelBundle(BaseModel):
    name: str
    models: list[StarterModel]
