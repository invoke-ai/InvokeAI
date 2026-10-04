"""Which VAEs an architecture can decode with.

Not derivable from the latent geometry. Eight architectures share a 16-channel space at 8x
compression, but their VAEs are not interchangeable: SD 3.5 and CogView 4 are 16-channel too and
belong to neither the Wan/Qwen family nor FLUX's. Compatibility is a fact about the decoder class,
so it is declared.

Nor is it derivable from the VAE's `base` alone. The same physical file is registered under
`anima` or `qwen-image` depending on which family it was installed for -- byte-identical, 194
tensors -- and a `wan` VAE may be either the 16-channel Wan 2.1 file (the same family again) or
TI2V-5B's 48-channel Wan2.2-VAE, which fits nothing else. `VAE_Checkpoint_Wan_Config` already
records `latent_channels`; this facet is what finally reads it.

Nor, for Wan, is it a property of the architecture: A14B decodes with the 16-channel VAE and
TI2V-5B with the 48-channel one, and `wan_model_loader` rejects the other pairing. `by_variant`
says so, the same way `FeaturesFacet.dimension_grid_by_variant` does for the grid.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

from invokeai.backend.architectures.facet import Facet
from invokeai.backend.architectures.registry import get
from invokeai.backend.model_manager.taxonomy import AnyVariant, BaseModelType


@dataclass(frozen=True)
class VaeCompatibility:
    """One VAE an architecture accepts, as a model record can be matched against it."""

    base: BaseModelType

    latent_channels: int | None = None
    """Required channel count, where the base alone is ambiguous.

    Only `wan` is: its VAEs carry `latent_channels` of 16 or 48 and only the 16-channel ones belong
    to the shared family. `None` means the base is unambiguous and the field is not consulted --
    which is also what happens for VAE configs that do not carry it.
    """

    def matches(self, vae_base: BaseModelType, vae_latent_channels: int | None = None) -> bool:
        if vae_base != self.base:
            return False
        if self.latent_channels is None:
            return True
        return vae_latent_channels == self.latent_channels


@dataclass(frozen=True)
class VaeFacet(Facet):
    """The VAEs a model of this architecture can be paired with.

    Includes the architecture's own base wherever that is accepted, so the set is complete on its
    own and a consumer never has to add an implicit "or its own".
    """

    REQUIRED: ClassVar[bool] = False

    accepted: frozenset[VaeCompatibility]
    """What the architecture accepts, and what any variant not named in `by_variant` accepts."""

    by_variant: Mapping[AnyVariant, frozenset[VaeCompatibility]] = field(default_factory=dict)
    """A variant whose decoder is a different one. Replaces `accepted` for that variant, never extends it."""

    def __post_init__(self) -> None:
        if None in self.by_variant:
            raise ValueError(
                "VaeFacet.by_variant has a None key. `accepted` is the answer for every variant not "
                "named here, so a None key would never be read."
            )

        empty = sorted(
            str(getattr(variant, "value", variant)) for variant, entries in self.by_variant.items() if not entries
        )
        if not self.accepted or empty:
            raise ValueError(
                f"VaeFacet accepts no VAE at all for {', '.join(empty) or 'the architecture itself'}. The picker "
                "would offer nothing and a model that needs a VAE could never generate."
            )

    def resolve(self, variant: AnyVariant | None = None) -> frozenset[VaeCompatibility]:
        if variant is None:
            return self.accepted
        return self.by_variant.get(variant, self.accepted)

    def accepts(
        self, vae_base: BaseModelType, vae_latent_channels: int | None = None, variant: AnyVariant | None = None
    ) -> bool:
        return any(entry.matches(vae_base, vae_latent_channels) for entry in self.resolve(variant))

    @property
    def accepted_bases(self) -> frozenset[BaseModelType]:
        """The bases alone, across every variant, for comparing against a loader's `ui_model_base` --
        one loader serves all variants and cannot express a channel constraint."""
        entries = self.accepted.union(*self.by_variant.values())
        return frozenset(entry.base for entry in entries)


def accepts_vae(
    base: BaseModelType,
    vae_base: BaseModelType,
    vae_latent_channels: int | None = None,
    variant: AnyVariant | None = None,
) -> bool:
    """Whether a model of `base` (and `variant`, where it matters) can decode with this VAE.

    Architectures that declare no `VaeFacet` accept only their own base -- the SD family and
    anything whose loader has no VAE input at all.
    """
    facet = get(base, VaeFacet)

    if facet is None:
        return vae_base == base

    return facet.accepts(vae_base, vae_latent_channels, variant)


def accepted_vae_bases(base: BaseModelType) -> list[BaseModelType]:
    """The VAE bases a loader for `base` offers, as its `vae_model` field's `ui_model_base`.

    Bases only, across every variant: one loader serves all variants and the field cannot express a
    channel constraint, so the loader's own validation narrows further with `accepts_vae`. Sorted by
    value because frozenset iteration order follows the hash seed, and this lands in the schema.
    """
    facet = get(base, VaeFacet)
    bases = facet.accepted_bases if facet is not None else frozenset({base})
    return sorted(bases, key=lambda b: b.value)
