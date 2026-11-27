"""The dimension grid, where it depends on the variant rather than only on the base.

`FeaturesFacet.dimension_grid` is pinned to a denoise node's `multiple_of` by
`test_features.py`, which is a field constraint and therefore one number per base. Wan has two: the
A14B family denoises in the 16-channel space at 8x and takes multiples of 16, TI2V-5B in the
48-channel Wan2.2 space at 16x and takes multiples of 32, and the transformer's `patch_size=(1, 2,
2)` doubles both. The second constraint is enforced inside `wan_denoise.invoke()` -- after enqueue
and after the model has loaded -- so nothing a field-schema gate can see was ever wrong. 1280x720 is
the failure that motivated this: both multiples of 16, offered by the canvas, and `720 % 32 == 16`.
"""

import pytest

# `_validate_spatial_dimensions` is the enforcement site: it is what actually refuses an off-grid
# request. Reading the declaration against it, rather than against a number transcribed here, is the
# only way this file can fail for a real reason.
from invokeai.app.invocations.wan.wan_denoise import _validate_spatial_dimensions
from invokeai.backend.architectures import architecture_capabilities, generative_bases
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import WAN21_16, WAN22_48, LatentSpace, LatentSpaceFacet
from invokeai.backend.architectures.registry import ArchitectureError, require
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelVariantType, WanVariantType


def test_wan_declares_the_grid_ti2v_5b_actually_enforces() -> None:
    """Both directions, so the declaration cannot be a coarser number that merely happens to pass:
    the declared grid is accepted and half of it -- still a valid A14B size -- is not."""
    facet = require(BaseModelType.Wan, FeaturesFacet)
    grid = facet.resolve_dimension_grid(WanVariantType.TI2V_5B)
    assert grid == 32

    _validate_spatial_dimensions(WanVariantType.TI2V_5B, grid * 40, grid * 22)
    with pytest.raises(ValueError, match="multiples of 32"):
        _validate_spatial_dimensions(WanVariantType.TI2V_5B, grid * 40, grid * 22 + grid // 2)


def test_the_a14b_variants_keep_the_base_grid() -> None:
    """A variant the mapping does not name gets `dimension_grid`, which `test_features.py` has
    already pinned to `wan_denoise`'s own `multiple_of`."""
    facet = require(BaseModelType.Wan, FeaturesFacet)
    for variant in (WanVariantType.T2V_A14B, WanVariantType.I2V_A14B):
        assert facet.resolve_dimension_grid(variant) == facet.dimension_grid == 16
        _validate_spatial_dimensions(variant, 1280, 720)


def test_an_unknown_variant_falls_back_to_the_base_grid() -> None:
    """A model whose variant could not be identified, or whose variant enum the grid mapping says
    nothing about, still has to get an answer: the base grid, which is the one the architecture's own
    denoise node enforces on every request."""
    for base in generative_bases():
        facet = require(base, FeaturesFacet)
        assert facet.resolve_dimension_grid(None) == facet.dimension_grid
        assert facet.resolve_dimension_grid(ModelVariantType.Normal) == facet.dimension_grid


def test_a_none_key_is_refused() -> None:
    """`DefaultSettingsFacet.by_variant` spells its fallback `None`; this mapping's fallback is
    `dimension_grid`, so the same habit here would declare a grid that is silently never read."""
    with pytest.raises(ValueError, match="None key"):
        FeaturesFacet(
            negative_prompt=NegativePrompt(visible=True, usage="always"),
            dimension_grid=16,
            dimension_grid_by_variant={None: 32},
        )


def test_the_served_table_carries_the_variant_grid() -> None:
    """A client joins on `(base, variant)` and falls back to `(base, None)`. Without the TI2V-5B row
    carrying 32, that join hands it 16 and it offers sizes the graph rejects."""
    rows = {(r.base, r.variant): r for r in architecture_capabilities()}

    assert rows[(BaseModelType.Wan, None)].features.dimension_grid == 16
    assert rows[(BaseModelType.Wan, "ti2v_5b")].features.dimension_grid == 32


def test_a_variant_row_appears_wherever_a_grid_override_is_declared() -> None:
    """The declaration has to reach the table. A grid an architecture declares for a variant but
    that no row carries is invisible to the client doing the `(base, variant)` join, which is the
    only consumer that can act on it."""
    rows = {(r.base, r.variant): r for r in architecture_capabilities()}
    missing = []
    for base in generative_bases():
        for variant, grid in require(base, FeaturesFacet).dimension_grid_by_variant.items():
            row = rows.get((base, variant.value))
            if row is None or row.features.dimension_grid != grid:
                missing.append(f"{base.value}/{variant.value}: declared {grid}, served {row and row.features}")
    assert missing == []


def test_no_variant_row_repeats_its_base_row() -> None:
    """A variant row exists to override something. One identical to its base row is noise in a
    response clients diff, and a sign the emit rule has drifted from what actually differs."""
    by_base = {r.base: r for r in architecture_capabilities() if r.variant is None}
    redundant = [
        f"{r.base.value}/{r.variant}"
        for r in architecture_capabilities()
        if r.variant is not None
        and (r.features, r.defaults, r.vae) == (by_base[r.base].features, by_base[r.base].defaults, by_base[r.base].vae)
    ]
    assert redundant == []


def test_every_variant_row_is_rendered_in_full() -> None:
    """A variant row overrides its base row rather than patching it, so a client never has to know
    which fields a variant row may omit. A row emitted for a grid difference alone must still carry
    the architecture's recommended parameters."""
    by_base = {r.base: r for r in architecture_capabilities() if r.variant is None}
    for row in architecture_capabilities():
        if row.variant is None:
            continue
        base_row = by_base[row.base]
        assert (row.defaults is None) == (base_row.defaults is None), row.base.value
        assert (row.vae is None) == (base_row.vae is None), row.base.value


def test_the_variant_rows_of_a_base_stay_sorted() -> None:
    """The emit set is now a union of two mappings' keys, and an unsorted set iteration would
    reorder the response between runs -- `str` hashing is randomized per process."""
    for base in generative_bases():
        variants = [r.variant for r in architecture_capabilities() if r.base is base and r.variant is not None]
        assert variants == sorted(variants), base.value


def test_resolving_a_space_by_variant_agrees_with_resolving_it_by_sample() -> None:
    """Two ways to the same answer: generation has a tensor, the served table has a model record.
    If they disagreed, a preview and the size the client was told would describe different spaces."""
    import torch

    from invokeai.backend.model_manager.taxonomy import BaseModelType, WanVariantType

    facet = require(BaseModelType.Wan, LatentSpaceFacet)

    for variant, channels in ((WanVariantType.TI2V_5B, 48), (None, 16)):
        by_variant = facet.resolve_variant(variant)
        by_sample = facet.resolve(torch.zeros(1, channels, 8, 8))
        assert by_variant is by_sample, variant
        assert by_variant.channels == channels


def test_a_variant_cannot_name_a_space_the_facet_does_not_declare() -> None:
    """`by_variant` names one of this facet's spaces; it does not introduce one. A space missing
    from `alternates` is unreachable for `resolve`, so the two resolvers would disagree for every
    real sample."""
    stray = LatentSpace(channels=64, spatial_compression=8, rgb_factors=[[0.1, 0.1, 0.1]] * 64)

    with pytest.raises(ArchitectureError, match="does not declare"):
        LatentSpaceFacet(WAN21_16, alternates=(WAN22_48,), by_variant={WanVariantType.TI2V_5B: stray})
