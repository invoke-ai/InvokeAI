"""The default-settings facet: what each architecture recommends, and what it deliberately does not.

ERNIE-Image's name-based detection is covered by tests/backend/model_manager/test_ernie_image_default_settings.py,
which now runs through this resolver.
"""

from typing import Any

import pytest

from invokeai.backend.architectures import generative_bases, resolve_default_settings
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.registry import get
from invokeai.backend.model_manager.taxonomy import (
    AnyVariant,
    BaseModelType,
    Flux2VariantType,
    FluxVariantType,
    Krea2VariantType,
    LTX2VariantType,
    WanVariantType,
    ZImageVariantType,
)


def test_the_facet_is_required() -> None:
    """Every architecture declares defaults, so `validate()` enforces it at boot.

    It was optional while four architectures legitimately had none. They now declare what their
    model cards recommend, and the SDXL refiner — the last holdout — declares SDXL's canvas. A
    missing prefill is quiet rather than loud, which is exactly why it needs a boot check.
    """
    assert DefaultSettingsFacet.REQUIRED is True


def test_every_architecture_declares_defaults() -> None:
    undeclared = sorted(b.value for b in generative_bases() if get(b, DefaultSettingsFacet) is None)
    assert undeclared == []


_SQUARE_1024 = {"width": 1024, "height": 1024}

# The whole matrix, hand-transcribed from `architectures/defs/` rather than read back out of it.
# Each row is *every* field its declaration sets — the comparison is against
# `model_dump(exclude_none=True)` — so a field nobody thought to pin cannot arrive unnoticed
# either.
#
# These numbers are persisted onto a model's config when it is identified and drive the generation
# sliders from then on, so editing one reaches users without passing through any UI review. This
# table is what makes such an edit a reviewable diff.
#
# Where the values come from:
#
# scheduler:  mirrors `BASE_GENERATION[base].defaults.scheduler` in webv2's
#             `baseGenerationPolicies.ts` for all fourteen bases that table knows. Which scheduler
#             to prefer is a product decision, not a model-card fact, and mirroring means nothing
#             changes for users when webv2 switches over to these values. MiniMax H3 and the SDXL
#             refiner are absent from that table and declare none — see
#             `test_every_architecture_with_a_scheduler_declares_which_one`.
# sd-1/2/XL:  the classic Stable Diffusion defaults, which every SD generation is built around,
#             each on its native canvas. 2.x is the judgment call: 768 is right for the
#             v-prediction checkpoints and wrong for the 512 `-base` ones, and nothing in the
#             config distinguishes them, so one of the two had to be picked.
# refiner:    a canvas and nothing else. It is a second pass over an SDXL latent, driven by the
#             UI's own refiner parameters, and is never run on its own.
# sd-3:       stable-diffusion-3.5-medium, 40 steps at guidance 4.5. Medium, not Large (28/3.5):
#             there is one `sd-3` row and no variant to tell them apart.
# cogview4:   THUDM/CogView4-6B, 50 steps at guidance 3.5 — true CFG, it takes a negative prompt.
# flux:       `guidance` is the distilled guidance embedding, not CFG, so cfg_scale stays at its
#             floor (1.0, meaning "off") for every FLUX and FLUX.2 variant. schnell is
#             timestep-distilled and ignores guidance entirely. Fill's 30.0 is corroborated
#             in-tree: `flux_denoise.py` warns when guidance drops below 25.0 for a Fill model.
#             dev's card uses 50 steps; 28 is the de-facto standard and what FLUX.2 [dev] declares,
#             so the two stay consistent.
# ernie:      the base model's 50/4.0, and Turbo's 8/1.0 — reached by name hint rather than
#             variant, because the two share an architecture and a config with nothing to probe.
# z-image:    Tongyi-MAI/Z-Image-Turbo, `num_inference_steps=9`, guidance 0 -> cfg_scale 1.0. The
#             undistilled base needs more steps and supports CFG.
# ideogram-4: cfg_scale 1.0, because the model is CFG-distilled and cannot do CFG at all —
#             `ideogram4_denoise` has no `cfg_scale` input, only `guidance_scale`, and the
#             FeaturesFacet says `negative_prompt: never`. This field previously held 7.0, taken
#             from the main weight in our own PRESETS. That number is the sampler's internal
#             guidance schedule, not a default anyone sets: the node reads `guidance_scale=None` as
#             "use the preset", and webv2 sends nothing unless the user overrides it through
#             Ideogram's own dedicated fields. Declaring 7.0 advertised a CFG default for a model
#             with no CFG. 48 steps is the preset default, V4_QUALITY_48.
# krea-2:     Diffusers' Krea-2 guidance 4.5 uses cond + 4.5 * (cond - uncond), equivalent to
#             InvokeAI's CFG convention at 5.5. Turbo is distilled; cfg_scale's floor is 1.
DEFAULT_SETTINGS_MATRIX: list[tuple[str, BaseModelType, AnyVariant | None, str | None, dict[str, Any]]] = [
    (
        "sd-1",
        BaseModelType.StableDiffusion1,
        None,
        None,
        {"scheduler": "euler_a", "steps": 30, "cfg_scale": 7.0, "width": 512, "height": 512},
    ),
    (
        "sd-2",
        BaseModelType.StableDiffusion2,
        None,
        None,
        {"scheduler": "euler_a", "steps": 30, "cfg_scale": 7.0, "width": 768, "height": 768},
    ),
    (
        "sdxl",
        BaseModelType.StableDiffusionXL,
        None,
        None,
        {"scheduler": "euler_a", "steps": 30, "cfg_scale": 7.0, **_SQUARE_1024},
    ),
    ("sdxl-refiner", BaseModelType.StableDiffusionXLRefiner, None, None, {**_SQUARE_1024}),
    (
        "sd-3",
        BaseModelType.StableDiffusion3,
        None,
        None,
        {"scheduler": "euler_a", "steps": 40, "cfg_scale": 4.5, **_SQUARE_1024},
    ),
    (
        "cogview4",
        BaseModelType.CogView4,
        None,
        None,
        {"scheduler": "euler_a", "steps": 50, "cfg_scale": 3.5, **_SQUARE_1024},
    ),
    (
        "flux-schnell",
        BaseModelType.Flux,
        FluxVariantType.Schnell,
        None,
        {"scheduler": "euler", "steps": 4, "cfg_scale": 1.0, **_SQUARE_1024},
    ),
    (
        "flux-dev-fill",
        BaseModelType.Flux,
        FluxVariantType.DevFill,
        None,
        {"scheduler": "euler", "steps": 50, "cfg_scale": 1.0, "guidance": 30.0, **_SQUARE_1024},
    ),
    (
        "flux-dev",
        BaseModelType.Flux,
        FluxVariantType.Dev,
        None,
        {"scheduler": "euler", "steps": 28, "cfg_scale": 1.0, "guidance": 3.5, **_SQUARE_1024},
    ),
    (
        "flux2-dev",
        BaseModelType.Flux2,
        Flux2VariantType.Dev,
        None,
        {"scheduler": "euler", "steps": 28, "cfg_scale": 1.0, "guidance": 3.5, **_SQUARE_1024},
    ),
    (
        "flux2-klein-4b-base",
        BaseModelType.Flux2,
        Flux2VariantType.Klein4BBase,
        None,
        {"scheduler": "euler", "steps": 28, "cfg_scale": 1.0, **_SQUARE_1024},
    ),
    (
        "flux2-klein-9b-base",
        BaseModelType.Flux2,
        Flux2VariantType.Klein9BBase,
        None,
        {"scheduler": "euler", "steps": 28, "cfg_scale": 1.0, **_SQUARE_1024},
    ),
    (
        "flux2-klein-distilled",
        BaseModelType.Flux2,
        None,
        None,
        {"scheduler": "euler", "steps": 4, "cfg_scale": 1.0, **_SQUARE_1024},
    ),
    (
        "ernie-image",
        BaseModelType.ErnieImage,
        None,
        None,
        {"scheduler": "euler", "steps": 50, "cfg_scale": 4.0, **_SQUARE_1024},
    ),
    (
        "ernie-image-turbo",
        BaseModelType.ErnieImage,
        None,
        "ERNIE-Image-Turbo",
        {"scheduler": "euler", "steps": 8, "cfg_scale": 1.0, **_SQUARE_1024},
    ),
    (
        "qwen-image",
        BaseModelType.QwenImage,
        None,
        None,
        {"scheduler": "euler_a", "steps": 40, "cfg_scale": 4.0, **_SQUARE_1024},
    ),
    (
        "z-image-base",
        BaseModelType.ZImage,
        ZImageVariantType.ZBase,
        None,
        {"scheduler": "euler", "steps": 50, "cfg_scale": 4.0, **_SQUARE_1024},
    ),
    (
        "z-image-turbo",
        BaseModelType.ZImage,
        None,
        None,
        {"scheduler": "euler", "steps": 9, "cfg_scale": 1.0, **_SQUARE_1024},
    ),
    (
        "ideogram-4",
        BaseModelType.Ideogram4,
        None,
        None,
        {"scheduler": "euler", "steps": 48, "cfg_scale": 1.0, **_SQUARE_1024},
    ),
    (
        "krea-2-base",
        BaseModelType.Krea2,
        Krea2VariantType.Base,
        None,
        {"scheduler": "euler", "steps": 28, "cfg_scale": 5.5, **_SQUARE_1024},
    ),
    (
        "krea-2-turbo",
        BaseModelType.Krea2,
        None,
        None,
        {"scheduler": "euler", "steps": 8, "cfg_scale": 1.0, **_SQUARE_1024},
    ),
    (
        "wan-ti2v-5b",
        BaseModelType.Wan,
        WanVariantType.TI2V_5B,
        None,
        {"scheduler": "euler", "steps": 30, "cfg_scale": 5.0, **_SQUARE_1024},
    ),
    (
        "wan-a14b",
        BaseModelType.Wan,
        None,
        None,
        {"scheduler": "euler", "steps": 40, "cfg_scale": 4.0, **_SQUARE_1024},
    ),
    ("minimax-h3", BaseModelType.MiniMaxH3, None, None, {"steps": 50, "cfg_scale": 1.0, "width": 1344, "height": 768}),
    ("ltx-2", BaseModelType.LTX2, None, None, {"steps": 30, "cfg_scale": 3.0, "width": 1248, "height": 704}),
    (
        "ltx-2-distilled",
        BaseModelType.LTX2,
        LTX2VariantType.Distilled,
        None,
        {"steps": 8, "cfg_scale": 1.0, "width": 1248, "height": 704},
    ),
    ("anima", BaseModelType.Anima, None, None, {"scheduler": "euler", "steps": 35, "cfg_scale": 4.5, **_SQUARE_1024}),
]


@pytest.mark.parametrize(
    ("base", "variant", "name", "expected"),
    [pytest.param(*row[1:], id=row[0]) for row in DEFAULT_SETTINGS_MATRIX],
)
def test_the_resolved_defaults_are_exactly_what_was_declared(
    base: BaseModelType, variant: AnyVariant | None, name: str | None, expected: dict[str, Any]
) -> None:
    settings = resolve_default_settings(base, variant, name)
    assert settings is not None, base.value
    assert settings.model_dump(exclude_none=True) == expected


def test_the_matrix_covers_every_architecture() -> None:
    """A new architecture whose defaults are pinned nowhere is the state this table exists to end,
    and parametrizing over the table cannot notice its own omission."""
    covered = {row[1].value for row in DEFAULT_SETTINGS_MATRIX}
    assert sorted({b.value for b in generative_bases()} - covered) == []


def test_an_unknown_variant_falls_back() -> None:
    """`None` is the fallback key, and it is what an unrecognised variant lands on."""
    assert resolve_default_settings(BaseModelType.Krea2, Krea2VariantType.Turbo) == resolve_default_settings(
        BaseModelType.Krea2, None
    )


def test_a_variant_from_another_architecture_falls_back_rather_than_matching() -> None:
    """`FluxVariantType.Dev` and `Flux2VariantType.Dev` are equal and hash alike — both are "dev".

    A mapping is only ever consulted for the architecture that declared it, so this cannot happen in
    practice. Pinned because the equality is surprising: a lookup keyed by the wrong enum would find
    an entry rather than miss it, and nothing but the fallback would reveal the mistake.

    Both architectures happen to declare 28 steps at guidance 3.5 for their `dev`, so the shared key
    is not observable there — which is exactly why the check uses a value the two do not share.
    """
    assert FluxVariantType.Dev == Flux2VariantType.Dev
    assert resolve_default_settings(BaseModelType.Flux2, FluxVariantType.Schnell) == resolve_default_settings(
        BaseModelType.Flux2, None
    )


def test_every_architecture_with_a_scheduler_declares_which_one() -> None:
    """The last piece webv2 still hardcodes.

    `BASE_GENERATION` in `baseGenerationPolicies.ts` carries a `defaults.scheduler` per base, and it
    was the one field the capabilities endpoint could not supply -- so adding an architecture still
    meant editing the frontend even when nothing about it was special. The values mirror what that
    table ships, deliberately: which scheduler to prefer is a product decision, not a model-card
    fact, and mirroring means nothing changes for users when webv2 switches over.

    The converse matters too. An architecture with no `scheduler_set` has no scheduler to choose --
    MiniMax H3 steps video and audio down two hardcoded flow schedules -- and declaring a default for
    it would put a control in the UI that reaches nothing.
    """
    from invokeai.backend.architectures import FeaturesFacet, get

    missing, spurious = [], []
    for base in generative_bases():
        features = get(base, FeaturesFacet)
        settings = resolve_default_settings(base)
        if features is None or settings is None:
            continue
        # The refiner declares a canvas but no generation settings; it is not run on its own.
        if settings.steps is None:
            continue
        if features.scheduler_set is not None and settings.scheduler is None:
            missing.append(base.value)
        if features.scheduler_set is None and settings.scheduler is not None:
            spurious.append(base.value)

    assert missing == [], f"scheduler_set declared but no default scheduler: {missing}"
    assert spurious == [], f"default scheduler but no scheduler_set: {spurious}"
