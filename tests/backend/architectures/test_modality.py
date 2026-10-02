"""What each architecture can generate, and the metadata strings it writes."""

from typing import get_args

from invokeai.app.invocations.metadata import GENERATION_MODES
from invokeai.backend.architectures import generative_bases
from invokeai.backend.architectures.facets.modality import ModalityFacet, generation_modes
from invokeai.backend.architectures.registry import get
from invokeai.backend.model_manager.taxonomy import BaseModelType


def test_the_declarations_reconstruct_generation_modes() -> None:
    """The gate. `GENERATION_MODES` is a type pydantic validates metadata against; this proves the
    architecture declarations and that literal describe the same 50 strings.

    Compared rather than generated from the declarations, deliberately: the literal has to stay a
    literal to be a type, and a mismatch in either direction is a bug worth naming. A string missing
    from the literal means metadata that will not validate; one missing from the declarations means
    a mode nothing can produce.
    """
    assert generation_modes() == frozenset(get_args(GENERATION_MODES))


def test_every_architecture_declares_its_modality() -> None:
    undeclared = sorted(b.value for b in generative_bases() if get(b, ModalityFacet) is None)
    assert undeclared == []


def test_the_slug_is_not_the_base_value() -> None:
    """Seven of fourteen slugs differ from the enum value, which is why they are declared.

    These strings sit in the metadata of every image a user has generated. Deriving them — by
    replacing `-` with `_`, say — would produce `ideogram_4` where the truth is `ideogram4`, and
    silently orphan every image already tagged with the old one.
    """
    divergent = {
        BaseModelType.StableDiffusion3: "sd3",
        BaseModelType.ZImage: "z_image",
        BaseModelType.ErnieImage: "ernie_image",
        BaseModelType.Ideogram4: "ideogram4",
        BaseModelType.QwenImage: "qwen_image",
        BaseModelType.Krea2: "krea2",
        BaseModelType.MiniMaxH3: "minimax_h3",
    }
    for base, slug in divergent.items():
        facet = get(base, ModalityFacet)
        assert facet is not None and facet.metadata_slug == slug, base.value
        assert slug != base.value, f"{base.value} would not need declaring"


def test_stable_diffusion_writes_unprefixed_modes() -> None:
    """SD 1.x and 2.x share the bare strings — the only two architectures with no prefix."""
    unprefixed = sorted(
        b.value for b in generative_bases() if (f := get(b, ModalityFacet)) is not None and f.metadata_slug is None
    )
    assert unprefixed == ["sd-1", "sd-2", "sdxl-refiner"]

    sd1 = get(BaseModelType.StableDiffusion1, ModalityFacet)
    assert sd1 is not None and "txt2img" in sd1.metadata_modes()


def test_the_refiner_generates_nothing() -> None:
    """Empty modes, so it contributes no metadata string despite having no prefix either."""
    facet = get(BaseModelType.StableDiffusionXLRefiner, ModalityFacet)
    assert facet is not None
    assert facet.modes == frozenset()
    assert facet.metadata_modes() == frozenset()


def test_the_text_to_image_only_architectures() -> None:
    for base in (BaseModelType.ErnieImage, BaseModelType.Ideogram4):
        facet = get(base, ModalityFacet)
        assert facet is not None and facet.modes == frozenset({"txt2img"}), base.value


def test_the_video_architectures() -> None:
    """Wan generates images at one frame and video above that; H3 is video-first."""
    wan = get(BaseModelType.Wan, ModalityFacet)
    h3 = get(BaseModelType.MiniMaxH3, ModalityFacet)
    assert wan is not None and h3 is not None

    assert "i2v" in wan.modes and "inpaint" in wan.modes
    assert h3.modes == frozenset({"txt2img", "t2v", "i2v", "lf2v", "flf2v", "extend_video", "ref2v"}), (
        "H3 has no img2img, inpaint or outpaint, but does have three keyframe-conditioned video modes "
        "and a reference-conditioned one"
    )
