"""Which VAEs each architecture accepts.

The facts here were duplicated three times before -- once per loader as `ui_model_base`, once in
webv2's VAE picker, once more in its related-models linker -- and all three disagreed. Anima's
loader declared nothing at all while its description named three families; FLUX.2 Klein's offered
FLUX VAEs that crash in `decode()`. The loaders now derive `ui_model_base` from the facet through
`accepted_vae_bases`.
"""

import pytest

from invokeai.app.invocations.baseinvocation import InvocationRegistry
from invokeai.app.services.shared.graph import Graph  # noqa: F401 -- imports every invocation
from invokeai.backend.architectures import architecture_capabilities, generative_bases, get
from invokeai.backend.architectures.facets.latent_space import LatentSpace, LatentSpaceFacet
from invokeai.backend.architectures.facets.vae import VaeCompatibility, VaeFacet, accepted_vae_bases, accepts_vae
from invokeai.backend.architectures.registry import require
from invokeai.backend.model_manager.taxonomy import BaseModelType, WanVariantType


def test_flux2_no_longer_offers_a_vae_that_crashes() -> None:
    """Klein advertised FLUX VAEs. Decoding a FLUX.2 latent with one raises
    `AutoEncoder.decode() got an unexpected keyword argument 'return_dict'` -- the legacy FLUX
    `AutoEncoder` against a 32-channel latent. Verified by generating with it."""
    assert accepts_vae(BaseModelType.Flux2, BaseModelType.Flux2) is True
    assert accepts_vae(BaseModelType.Flux2, BaseModelType.Flux) is False


def test_anima_accepts_the_wan_family_under_each_base_it_registers_as() -> None:
    """One 194-tensor checkpoint, three registrations. All three decode the same latent."""
    for vae_base in (BaseModelType.Anima, BaseModelType.QwenImage):
        assert accepts_vae(BaseModelType.Anima, vae_base) is True, vae_base.value

    assert accepts_vae(BaseModelType.Anima, BaseModelType.Wan, 16) is True
    # TI2V-5B's Wan2.2-VAE is the same class but a 48-channel latent space; it fits nothing else.
    assert accepts_vae(BaseModelType.Anima, BaseModelType.Wan, 48) is False


def test_anima_no_longer_offers_a_flux_vae_that_decodes_to_noise() -> None:
    """Anima denoises in WAN21_16; a FLUX VAE reads the same 16 channels in a different basis.

    `anima_l2i` used to take a `FluxAutoEncoder` without raising and without the Wan per-channel
    denormalisation, so nothing failed and nothing was logged. Measured on real weights: one Anima
    denoise decoded by both VAEs in the same graph, 8.67 dB PSNR between the two images -- a magenta,
    moire-patterned image in which the subject is barely discernible. It is not a fallback, so it is
    not offered; `tests/app/invocations/test_anima_vae.py` pins the node's refusal.
    """
    assert accepts_vae(BaseModelType.Anima, BaseModelType.Flux) is False


def test_an_architecture_without_the_facet_accepts_only_its_own_base() -> None:
    assert accepts_vae(BaseModelType.StableDiffusionXL, BaseModelType.StableDiffusionXL) is True
    assert accepts_vae(BaseModelType.StableDiffusionXL, BaseModelType.Flux) is False


# The generic loader serves the families that declare no facet at all; it names them itself.
GENERIC_VAE_NODES = {"vae_loader"}


def test_every_node_with_a_vae_input_takes_its_bases_from_a_facet() -> None:
    """Anima's loader declared no bases, so its picker offered every VAE installed; a loader with its
    own literal list drifts from what the graph accepts. Each list must be one the facet answers."""
    facet_lists = {tuple(b.value for b in accepted_vae_bases(base)) for base in generative_bases()}
    mismatched = sorted(
        f"{cls.get_type()}: {vae.get('ui_model_base')}"
        for cls in InvocationRegistry.get_invocation_classes()
        if cls.get_type() not in GENERIC_VAE_NODES
        and (vae := cls.model_json_schema().get("properties", {}).get("vae_model")) is not None
        and tuple(vae.get("ui_model_base") or ()) not in facet_lists
    )

    assert mismatched == [], (
        "These nodes take a VAE but their `ui_model_base` is not a facet's list. "
        "Use `ui_model_base=accepted_vae_bases(<base>)`."
    )


def _declared_latent_space(base: BaseModelType, latent_channels: int | None) -> LatentSpace | None:
    """The space a VAE registered under `base` decodes, or None if `latent_channels` matches none.

    `latent_channels` narrows only where the base has alternates -- today only `wan`, whose 48 is
    TI2V-5B's Wan2.2-VAE. Everywhere else the base has exactly one space and the field is unused.
    """
    facet = require(base, LatentSpaceFacet)
    if latent_channels is None:
        return facet.primary
    spaces = (facet.primary, *facet.alternates)
    return next((space for space in spaces if space.channels == latent_channels), None)


def test_every_accepted_vae_shares_the_architectures_latent_space() -> None:
    """Compatibility is about the basis, not the channel count, and nothing else enforces that.

    Eight architectures denoise in 16 channels at 8x and their VAEs are still not interchangeable:
    WAN21_16 and FLUX_16 have different `rgb_factors` and only WAN21_16 carries an `rgb_bias`.
    Anima declared `VaeCompatibility(Flux)` and `anima_l2i` decoded it without raising -- 8.67 dB
    PSNR against the correct decode of the same latent, measured on real weights; see
    `test_anima_no_longer_offers_a_flux_vae_that_decodes_to_noise`.

    There is no exception list. A new entry that fails this is a silently wrong decode, so the fix
    is to drop the entry, not to name it here.
    """

    def _sort_key(entry: VaeCompatibility) -> tuple[str, int]:
        return (entry.base.value, entry.latent_channels or 0)

    violations = []
    for base in generative_bases():
        facet = get(base, VaeFacet)
        if facet is None:
            continue

        own = require(base, LatentSpaceFacet)

        # Per variant, against the one space that variant denoises in -- not against any space the
        # architecture declares. Against the union, Wan A14B accepting TI2V-5B's 48-channel VAE
        # passed, and `wan_model_loader` rejects exactly that pairing.
        for variant in (None, *facet.by_variant):
            expected = own.resolve_variant(variant)
            label = base.value if variant is None else f"{base.value}/{variant.value}"

            for entry in sorted(facet.resolve(variant), key=_sort_key):
                space = _declared_latent_space(entry.base, entry.latent_channels)
                if space is None:
                    violations.append(
                        f"{label} accepts {entry.base.value} at {entry.latent_channels} channels, "
                        f"but {entry.base.value} declares no such latent space"
                    )
                elif space != expected:
                    violations.append(
                        f"{label} denoises in {expected.channels}ch/{expected.spatial_compression}x but "
                        f"accepts a {entry.base.value} VAE, whose latent space is a different basis of "
                        f"{space.channels}ch/{space.spatial_compression}x"
                    )

    assert violations == []


def test_wan_offers_each_variant_only_the_vae_its_decode_accepts() -> None:
    """`wan_model_loader` and `wan_l2i` refuse a 48-channel VAE for A14B and a 16-channel one for
    TI2V-5B. The served rows said both widths were fine for both, so the variant row stated a
    pairing that fails at enqueue -- the one fact `latent_channels` is on the contract for."""
    assert accepts_vae(BaseModelType.Wan, BaseModelType.Wan, 16) is True
    assert accepts_vae(BaseModelType.Wan, BaseModelType.Wan, 48) is False
    assert accepts_vae(BaseModelType.Wan, BaseModelType.Wan, 48, WanVariantType.TI2V_5B) is True
    assert accepts_vae(BaseModelType.Wan, BaseModelType.Wan, 16, WanVariantType.TI2V_5B) is False

    served = {
        row.variant: [(a.base, a.latent_channels) for a in row.vae.accepted]
        for row in architecture_capabilities()
        if row.base is BaseModelType.Wan and row.vae is not None
    }
    assert served == {None: [(BaseModelType.Wan, 16)], "ti2v_5b": [(BaseModelType.Wan, 48)]}


def test_a_variant_mapping_that_is_never_read_or_accepts_nothing_is_refused_at_declaration() -> None:
    """An empty set is served as `accepted: []` and leaves the picker with nothing to offer for that
    variant, with nothing pointing at the facet; a None key is dead weight `resolve` never reads."""
    wan16 = frozenset({VaeCompatibility(BaseModelType.Wan, latent_channels=16)})

    with pytest.raises(ValueError, match="None key"):
        VaeFacet(wan16, by_variant={None: wan16})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="ti2v_5b"):
        VaeFacet(wan16, by_variant={WanVariantType.TI2V_5B: frozenset()})
    with pytest.raises(ValueError, match="no VAE at all"):
        VaeFacet(frozenset())


def test_the_facet_is_only_declared_where_it_says_something_new() -> None:
    """A facet repeating "its own base, no constraints" would be noise; `accepts_vae` says that
    already for every architecture without one.

    Wan declares only its own base and is *not* redundant: it carries the channel split that tells
    A14B's 16-channel VAE from TI2V-5B's 48-channel one.
    """
    redundant = []
    for base in generative_bases():
        facet = get(base, VaeFacet)
        if facet is None:
            continue
        # Across every variant: a channel constraint that lives only in `by_variant` still says something.
        entries = facet.accepted.union(*facet.by_variant.values())
        says_nothing_new = facet.accepted_bases == frozenset({base}) and all(
            entry.latent_channels is None for entry in entries
        )
        if says_nothing_new:
            redundant.append(base.value)

    assert redundant == []
