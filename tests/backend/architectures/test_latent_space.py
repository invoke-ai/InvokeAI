"""The latent-space facet: the projection maths, and what each architecture declares.

Two tables carry this file. `DECLARED_LATENT_SPACES` says which space each of the sixteen
architectures denoises in; `PINNED_PREVIEW_PIXELS` says what each of the nine spaces actually
computes. Together they are what stops a wrong space reaching generation: swapping `FLUX_16` for
`SD3_16` in `defs/flux.py` moves one row of the first table, and the second table is what proves
the two rows are not interchangeable.
"""

import pytest
import torch

from invokeai.backend.architectures import generative_bases, resolve_latent_space
from invokeai.backend.architectures.facets import latent_space
from invokeai.backend.architectures.facets.latent_space import (
    COGVIEW4_16,
    FLUX2_32,
    FLUX_16,
    LTX2_128,
    MINIMAX_H3_24,
    SD3_16,
    SD15_4,
    SDXL_4,
    WAN21_16,
    WAN22_48,
    LatentSpace,
    LatentSpaceFacet,
)
from invokeai.backend.architectures.registry import get, require
from invokeai.backend.model_manager.taxonomy import BaseModelType

_SPACE_NAMES = {id(space): name for name, space in vars(latent_space).items() if isinstance(space, LatentSpace)}
"""The module-level name of each declared space, by identity.

Read off the facets module rather than transcribed, so it cannot fall behind. Assertions below
compare these names instead of the `LatentSpace` objects for two reasons: a failure then reads
`['SD3_16'] != ['FLUX_16']` rather than two screens of matrix, and a space that is an equal *copy*
of a declared one has no name here, so `==` on names is as strict as `is`."""


def _name(space: LatentSpace) -> str:
    return _SPACE_NAMES.get(id(space), f"an unnamed {space.channels}-channel space")


# The space each architecture denoises in. Derived twice: from `defs/<base>.py`, and from the
# pre-refactor dispatch this PR replaced, which is still readable at the base revision as
# `git show bfccc6e4643a760ca4e2d34376940d286272486a:invokeai/app/util/step_callback.py`. That
# `if/elif` chain is the independent source; where the two disagree the disagreement is the finding.
# It named its matrices rather than its spaces, so a row matches when the old branch selected the
# same factors, bias and smoothing kernel the space declares.
DECLARED_LATENT_SPACES: dict[BaseModelType, tuple[LatentSpace, ...]] = {
    # Old branch: ANIMA_LATENT_RGB_FACTORS + ANIMA_LATENT_RGB_BIAS, a byte-identical copy of the
    # Wan 2.1 matrix under a second name. Same for Krea-2 and Qwen-Image, which shared a third copy
    # (QWEN_IMAGE_*). The merge into one `WAN21_16` is the change; the numbers are unchanged.
    BaseModelType.Anima: (WAN21_16,),
    BaseModelType.CogView4: (COGVIEW4_16,),
    BaseModelType.ErnieImage: (FLUX2_32,),
    BaseModelType.Flux: (FLUX_16,),
    BaseModelType.Flux2: (FLUX2_32,),
    # The one row the `if/elif` chain cannot corroborate: it had no Ideogram 4 branch and would
    # have raised "Unsupported base model" here. The second, divergent copy of the projection is
    # the source instead -- `ideogram4_denoise.py` at the same revision built its own preview from
    # FLUX2_LATENT_RGB_FACTORS and FLUX2_LATENT_RGB_BIAS at 8x. Folding that copy in is what gives
    # Ideogram 4 a row at all.
    BaseModelType.Ideogram4: (FLUX2_32,),
    BaseModelType.Krea2: (WAN21_16,),
    BaseModelType.LTX2: (LTX2_128,),
    BaseModelType.MiniMaxH3: (MINIMAX_H3_24,),
    BaseModelType.QwenImage: (WAN21_16,),
    BaseModelType.StableDiffusion1: (SD15_4,),
    BaseModelType.StableDiffusion2: (SD15_4,),
    BaseModelType.StableDiffusion3: (SD3_16,),
    BaseModelType.StableDiffusionXL: (SDXL_4,),
    BaseModelType.StableDiffusionXLRefiner: (SDXL_4,),
    # The only architecture with more than one: the old chain read `sample.shape[-3] == 48` inside
    # the Wan branch. Every other row is a one-tuple, which is the declaration that an architecture
    # does not depend on the shape of its latents.
    BaseModelType.Wan: (WAN21_16, WAN22_48),
    # Old branch: FLUX_LATENT_RGB_FACTORS, commented "Z-Image uses FLUX-compatible VAE".
    BaseModelType.ZImage: (FLUX_16,),
}

# What each space projects a latent whose every channel holds the same value to.
#
# A `1 x C x 1 x 1` sample of `fill` projects to `fill * (column sums of rgb_factors) + rgb_bias`,
# which `preview()` then maps from -1..1 to 0..255 as `int(clamp((v + 1) / 2, 0, 1) * 255)` -- a
# truncating cast, not a round. Each row's comment carries the column sums, added by hand from the
# declared matrix; the bytes follow from them by that formula alone.
#
# Deriving them the other way round -- calling `preview()` and writing down the answer -- is how the
# WAN21 expectation went wrong before: a previous version of this test summed the very matrix under
# test, so any change to the matrix moved both sides at once. Its docstring claimed column sums of
# 0.3677/0.4577/0.9101 where the matrix really sums to 0.3887/0.8771/1.3152.
#
# `fill` is 1.0 everywhere except MiniMax H3, whose matrix sums past the clamp in two channels --
# a ones sample there reports 255 for anything summing above 0.881, which pins nothing. A quarter
# of that lands all three channels inside the range.
PINNED_PREVIEW_PIXELS: list[tuple[LatentSpace, float, tuple[int, int, int], tuple[int, int, int]]] = [
    # colsums (0.0192, 0.6051, -0.3178), no bias
    (SD15_4, 1.0, (129, 204, 86), (127, 127, 127)),
    # colsums (-0.2517, 0.7505, 0.0722), no bias, and then the 3x3 kernel. A 1x1 latent pads to
    # 3x3 with zeros, so only the kernel's centre weight survives: the projection is scaled by
    # 0.4711 before the mapping. Without the kernel these bytes would be (95, 223, 136).
    (SDXL_4, 1.0, (112, 172, 131), (127, 127, 127)),
    # colsums (0.06665435, 0.50958726, 0.32699234), no bias. R lands at 135.998, so the truncating
    # cast -- not a round -- is what makes it 135.
    (SD3_16, 1.0, (135, 192, 169), (127, 127, 127)),
    # colsums (0.66503498, 0.33521276, 0.26056899), no bias
    (COGVIEW4_16, 1.0, (212, 170, 160), (127, 127, 127)),
    # colsums (-0.0023, 0.2058, 0.2752), no bias
    (FLUX_16, 1.0, (127, 153, 162), (127, 127, 127)),
    # colsums (0.3887, 0.8771, 1.3152), bias (-0.1835, -0.0868, -0.3360)
    (WAN21_16, 1.0, (153, 228, 252), (104, 116, 84)),
    # colsums (-0.0122, 0.1637, 0.1435), bias (-0.0329, -0.0718, -0.0851)
    (FLUX2_32, 1.0, (121, 139, 134), (123, 118, 116)),
    # colsums (0.9628, 1.0896, -0.9290), bias (0.1189, 0.1415, -0.0034). See the note on `fill`.
    (MINIMAX_H3_24, 0.25, (173, 180, 97), (142, 145, 127)),
    # colsums (-0.5209, -0.1927, 0.9241), bias (0.0317, -0.0878, -0.1388)
    (WAN22_48, 1.0, (65, 91, 227), (131, 116, 109)),
    # colsums (-0.0698, -0.0317, 0.0266), bias (-0.2223, -0.2864, -0.2917)
    (LTX2_128, 0.25, (96, 89, 91), (99, 90, 90)),
]


class TestProjection:
    @pytest.mark.parametrize(
        ("space", "fill", "uniform_pixel", "zero_pixel"),
        [pytest.param(*row, id=_name(row[0])) for row in PINNED_PREVIEW_PIXELS],
    )
    def test_what_a_space_projects_a_uniform_latent_to(
        self,
        space: LatentSpace,
        fill: float,
        uniform_pixel: tuple[int, int, int],
        zero_pixel: tuple[int, int, int],
    ) -> None:
        """One reference colour per space, written down rather than recomputed.

        The zero sample is the bias on its own, which is the only thing separating two spaces that
        share a matrix -- and (127, 127, 127) is the assertion that a space declaring no bias has
        not quietly acquired one.
        """
        filled = torch.full((1, space.channels, 1, 1), fill)
        assert (
            space.preview(filled).getpixel((0, 0)),
            space.preview(torch.zeros(1, space.channels, 1, 1)).getpixel((0, 0)),
        ) == (uniform_pixel, zero_pixel)

    def test_every_declared_space_has_a_pinned_pixel(self) -> None:
        """A tenth space, or a matrix edited in place, has to land in the table above."""
        reachable = {
            _name(space)
            for base in generative_bases()
            for facet in [require(base, LatentSpaceFacet)]
            for space in (facet.primary, *facet.alternates)
        }
        assert reachable == {_name(row[0]) for row in PINNED_PREVIEW_PIXELS}

    def test_the_smoothing_kernel_reaches_beyond_the_pixel_it_centres_on(self) -> None:
        """SDXL is the only space with a kernel, and a uniform 1x1 latent only sees its centre.

        A 1x3 latent of ones sees the rest of the middle kernel row: the centre pixel is scaled by
        0.0964 + 0.4711 + 0.0964 = 0.6639, and each end pixel by 0.4711 + 0.0964 = 0.5675, because
        `conv2d(..., padding=1)` supplies a zero on its outer side. Dropping the kernel entirely
        would make all three (95, 223, 136); getting an off-centre weight wrong would move the ends
        without moving the centre.
        """
        image = SDXL_4.preview(torch.ones(1, 4, 1, 3))
        assert [image.getpixel((x, 0)) for x in range(3)] == [(109, 181, 132), (106, 191, 133), (109, 181, 132)]

    def test_the_preview_is_one_pixel_per_latent(self) -> None:
        assert WAN21_16.preview(torch.zeros(1, 16, 5, 7)).size == (7, 5)

    def test_a_sample_without_a_batch_dimension_is_accepted(self) -> None:
        assert WAN21_16.preview(torch.randn(16, 4, 4)).size == (4, 4)

    def test_a_uniform_sample_gives_a_uniform_preview(self) -> None:
        image = WAN21_16.preview(torch.zeros(1, 16, 3, 3))
        pixels = [image.getpixel((x, y)) for y in range(3) for x in range(3)]
        assert all(p == pixels[0] for p in pixels)


class TestResolution:
    def test_wan_picks_its_space_by_channel_count(self) -> None:
        """A14B and TI2V-5B are one `BaseModelType`; only the loaded checkpoint tells them apart."""
        assert resolve_latent_space(BaseModelType.Wan, torch.zeros(1, 16, 4, 4)) is WAN21_16
        assert resolve_latent_space(BaseModelType.Wan, torch.zeros(1, 48, 4, 4)) is WAN22_48

    def test_an_unknown_channel_count_falls_back_to_the_primary(self) -> None:
        assert resolve_latent_space(BaseModelType.Wan, torch.zeros(1, 7, 4, 4)) is WAN21_16

    def test_a_single_space_never_looks_at_the_sample(self) -> None:
        """The short circuit, pinned: an architecture with one space must not depend on tensor shape.

        A zero-dimensional tensor has no `shape[-3]`, so reading it would raise here.
        """
        facet = LatentSpaceFacet(WAN21_16)
        assert facet.resolve(torch.empty(0)) is WAN21_16


class TestWhatArchitecturesDeclare:
    @pytest.mark.parametrize(
        ("base", "expected"),
        [
            pytest.param(base, expected, id=base.value)
            for base, expected in sorted(DECLARED_LATENT_SPACES.items(), key=lambda item: item[0].value)
        ],
    )
    def test_the_space_each_architecture_denoises_in(
        self, base: BaseModelType, expected: tuple[LatentSpace, ...]
    ) -> None:
        """The single most load-bearing thing an architecture declares.

        A wrong space here is not a crash: the shapes still line up wherever the channel counts
        match, so FLUX declaring `SD3_16` would produce a plausible-looking preview in the wrong
        colours for every FLUX generation and nothing else would notice. Both directions are
        asserted -- what the facet declares, and what `resolve_latent_space` hands the step callback
        for a sample of that space's channel count.
        """
        facet = require(base, LatentSpaceFacet)
        declared = (facet.primary, *facet.alternates)
        assert [_name(space) for space in declared] == [_name(space) for space in expected]
        for space in expected:
            assert resolve_latent_space(base, torch.zeros(1, space.channels, 1, 1)) is space

    def test_every_architecture_is_pinned(self) -> None:
        """A new architecture has to land in the table above, rather than going unchecked."""
        assert set(DECLARED_LATENT_SPACES) == set(generative_bases())

    def test_every_architecture_declares_a_latent_space(self) -> None:
        """`REQUIRED = True` makes `validate()` enforce this at boot; this says what it means."""
        undeclared = sorted(b.value for b in generative_bases() if get(b, LatentSpaceFacet) is None)
        assert undeclared == []

    def test_each_matrix_has_one_row_per_channel(self) -> None:
        """A projection that disagrees with its own channel count fails only at generation time."""
        wrong = [
            (b.value, space.channels, len(space.rgb_factors))
            for b in generative_bases()
            for facet in [get(b, LatentSpaceFacet)]
            if facet is not None
            for space in (facet.primary, *facet.alternates)
            if len(space.rgb_factors) != space.channels or any(len(row) != 3 for row in space.rgb_factors)
        ]
        assert wrong == []

    def test_no_two_declared_spaces_hold_the_same_matrix(self) -> None:
        """Three byte-identical Wan 2.1 matrices lived under three names before the merge.

        Duplicates are how a projection gets fixed in one place and stays wrong in two others, so
        this fails the next time one is pasted rather than shared.
        """
        spaces: list[LatentSpace] = []
        for base in generative_bases():
            facet = get(base, LatentSpaceFacet)
            assert facet is not None
            for space in (facet.primary, *facet.alternates):
                if space not in spaces:
                    spaces.append(space)
        matrices = [tuple(tuple(row) for row in s.rgb_factors) for s in spaces]
        assert len(set(matrices)) == len(matrices)
