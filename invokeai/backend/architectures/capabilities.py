"""The architecture table, in the shape a client fetches it.

One row per architecture, plus a row per variant that answers differently. A client fetches this
once and joins it against model records locally: look up `(base, variant)`, fall back to
`(base, None)`.

This is not `ExternalModelCapabilities`, and deliberately not a subclass of it. That describes one
external provider's model — aspect ratios, resolution presets, mask format, per-request image
limits — and is stored on each such record. This describes an architecture, is the same for every
model of that architecture, and is not stored anywhere. Merging them would put fifteen irrelevant
fields on a schema webv2 already consumes.

Also deliberately not a computed field on `AnyModelConfig`: that would add these fields to all 115
config schemas and risk them being persisted into model records.

Only what a client reads is served. The registry knows more -- the modes an architecture generates
in, its metadata prefix, its latent compression -- and those stay backend facts until a client has
a use for them: a field nobody reads is a contract nobody checks, and it goes stale unnoticed.

The models here are response-only and stay open. `extra="forbid"` on one would reject nothing at
runtime -- nothing parses these -- and only stamp `additionalProperties: false` into `openapi.json`,
which turns every field this table grows into a breaking change for a strictly-validating client.
"""

from pydantic import BaseModel, Field

from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import (
    ControlKind,
    FeaturesFacet,
    NegativePromptUsage,
    SchedulerSet,
)
from invokeai.backend.architectures.facets.vae import VaeFacet
from invokeai.backend.architectures.registry import generative_bases, get, require
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import AnyVariant, BaseModelType


class NegativePromptPolicy(BaseModel):
    visible: bool = Field(description="Whether to show a negative prompt field at all.")
    usage: NegativePromptUsage = Field(
        description="'always', 'cfg-gated' (only above CFG 1), or 'never'.",
    )


class ArchitectureFeatures(BaseModel):
    """What a UI may offer for this architecture."""

    negative_prompt: NegativePromptPolicy
    dimension_grid: int = Field(
        description="Width and height must be a multiple of this. A variant row may carry its own."
    )
    guidance_label: str = Field(description="What to call the guidance slider: 'CFG' or 'Guidance'.")
    guidance_min: float = Field(
        description=(
            "Lowest guidance value the slider offers: 1.0, which is no guidance, or the denoise node's own "
            "floor where that is higher."
        )
    )
    guidance_max: float | None = Field(
        default=None,
        description="Highest guidance value the denoise node accepts; null means it enforces no ceiling.",
    )
    scheduler_set: SchedulerSet | None = Field(
        default=None, description="Which scheduler family to offer; null means no choice."
    )
    scheduler_applies_to_graph: bool = False
    control_kinds: list[ControlKind] = Field(default_factory=list, description="Sorted.")
    max_reference_images: int = 0
    reference_images_require_variant: str | None = Field(
        default=None,
        description="If set, reference images are only accepted for models of this variant.",
    )
    supports_regional_guidance: bool = False
    regional_negative: bool = Field(
        default=False, description="Whether a region's negative prompt is masked, rather than applied globally."
    )
    clip_skip_max: int | None = None
    supports_seamless: bool = False
    supports_cfg_rescale: bool = False
    sd_vae_override: bool = False
    color_compensation: bool = False
    vae_precision: bool = False


class VaeAcceptance(BaseModel):
    """One VAE this architecture's decode accepts."""

    base: BaseModelType
    latent_channels: int | None = Field(
        default=None,
        description="Null unless the base ships VAEs of more than one latent width; only wan does.",
    )


class ArchitectureVae(BaseModel):
    """Which VAEs an architecture's decode accepts, beyond its own base.

    Served because the clients keep their own copy of this and it drifts: widening a backend list
    without the picker leaves a VAE that loads but cannot be chosen. A variant row carries its own
    list where its decoder differs -- Wan TI2V-5B takes the 48-channel VAE, A14B the 16-channel one.
    """

    accepted: list[VaeAcceptance]


class ArchitectureCapabilities(BaseModel):
    """One row of the table."""

    base: BaseModelType
    variant: str | None = Field(
        default=None,
        description="Null for the architecture's own row. A variant row overrides it.",
    )
    features: ArchitectureFeatures
    defaults: MainModelDefaultSettings | None = Field(
        default=None, description="Recommended generation parameters, if the architecture has any."
    )
    vae: ArchitectureVae | None = Field(
        default=None, description="Null where the architecture declares no VAE compatibility beyond its own base."
    )


def _features_of(facet: FeaturesFacet, variant: AnyVariant | None = None) -> ArchitectureFeatures:
    return ArchitectureFeatures(
        negative_prompt=NegativePromptPolicy(
            visible=facet.negative_prompt.visible,
            usage=facet.negative_prompt.usage,
        ),
        dimension_grid=facet.resolve_dimension_grid(variant),
        guidance_label=facet.guidance_label,
        guidance_min=facet.guidance_min,
        guidance_max=facet.guidance_max,
        scheduler_set=facet.resolve_scheduler_set(variant),
        scheduler_applies_to_graph=facet.scheduler_applies_to_graph,
        control_kinds=sorted(facet.control_kinds),
        max_reference_images=facet.max_reference_images,
        reference_images_require_variant=facet.reference_images_require_variant,
        supports_regional_guidance=facet.supports_regional_guidance,
        regional_negative=facet.regional_negative,
        clip_skip_max=facet.clip_skip_max,
        supports_seamless=facet.supports_seamless,
        supports_cfg_rescale=facet.supports_cfg_rescale,
        sd_vae_override=facet.sd_vae_override,
        color_compensation=facet.color_compensation,
        vae_precision=facet.vae_precision,
    )


def _vae_of(facet: VaeFacet | None, variant: AnyVariant | None = None) -> ArchitectureVae | None:
    # Optional: most architectures accept only their own base, which needs no row.
    if facet is None:
        return None
    return ArchitectureVae(
        accepted=[
            VaeAcceptance(base=c.base, latent_channels=c.latent_channels)
            for c in sorted(facet.resolve(variant), key=lambda c: (c.base.value, c.latent_channels or 0))
        ]
    )


def architecture_capabilities() -> list[ArchitectureCapabilities]:
    """Every row, base rows first, then the variant rows that override them.

    A variant gets its own row only where something served actually differs: its recommended
    parameters (`DefaultSettingsFacet.by_variant`), its dimension grid
    (`FeaturesFacet.dimension_grid_by_variant`), its schedulers (`FeaturesFacet.scheduler_set_by_variant`)
    or the VAEs it decodes with (`VaeFacet.by_variant`).
    Those mappings are the whole rule — a row is emitted for the union of their keys, and every row
    is rendered in full, so a client never has to know which fields a variant row is allowed to omit.

    Differences too small to have earned a mapping are expressed on the base row instead, by
    `features.reference_images_require_variant`; Qwen-Image is the only one.

    Sorted by base value, then variant, so the response is stable and diffable.
    """
    rows: list[ArchitectureCapabilities] = []
    for base in sorted(generative_bases(), key=lambda b: b.value):
        # Both are REQUIRED, so `validate()` has already refused to start without them.
        # `require()` rather than `assert`: this is an API path, and `python -O` drops asserts.
        features = require(base, FeaturesFacet)
        defaults = require(base, DefaultSettingsFacet)
        vae_facet = get(base, VaeFacet)

        base_defaults = defaults.resolve()

        rows.append(
            ArchitectureCapabilities(
                base=base,
                features=_features_of(features),
                defaults=base_defaults,
                vae=_vae_of(vae_facet),
            )
        )
        differing = (
            set(defaults.by_variant)
            | set(features.dimension_grid_by_variant)
            | set(features.scheduler_set_by_variant)
            | set(vae_facet.by_variant if vae_facet is not None else ())
        ) - {None}
        for variant in sorted(differing):
            rows.append(
                ArchitectureCapabilities(
                    base=base,
                    # `.value`, not `str()`: these are `str`-mixin enums, and `str()` on one yields
                    # "FluxVariantType.DevFill" rather than the "dev_fill" a client stores and sends.
                    variant=variant.value,
                    features=_features_of(features, variant),
                    defaults=defaults.by_variant.get(variant, base_defaults),
                    vae=_vae_of(vae_facet, variant),
                )
            )
    return rows
