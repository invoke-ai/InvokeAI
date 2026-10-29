"""What the minimax-h3 architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import MINIMAX_H3_24, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.variant import VariantFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType, MiniMaxH3VariantType, ModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import MiniMaxH3ConditioningInfo

register(
    BaseModelType.MiniMaxH3,
    LatentSpaceFacet(MINIMAX_H3_24),
    ConditioningFacet(MiniMaxH3ConditioningInfo),
    # H3 is guidance-distilled (cfg_scale 1.0 means no guidance) and was released for a fixed
    # 768px short edge; 1344x768 is its native 16:9 canvas. Dimensions must be multiples of 32.
    DefaultSettingsFacet({None: MainModelDefaultSettings(steps=50, cfg_scale=1.0, width=1344, height=768)}),
    # Video first, with a single-frame still-image path. No img2img, inpaint or outpaint.
    # lf2v / flf2v / extend_video are H3's keyframe-conditioned video modes: continue from a last
    # frame, interpolate between a first and last frame, and extend an existing clip.
    ModalityFacet(
        frozenset({"txt2img", "t2v", "i2v", "lf2v", "flf2v", "extend_video", "ref2v"}),
        metadata_slug="minimax_h3",
    ),
    # minimax_h3_denoise is explicit: 'guidance-distilled: no negative prompt, no CFG, one
    # forward per step'. It steps video and audio down two hardcoded flow schedules, so
    # there is no scheduler to choose either.
    FeaturesFacet(
        negative_prompt=NegativePrompt(visible=False, usage="never"),
        dimension_grid=32,
        guidance_label="Guidance",
    ),
    # FL2VA and Ref2VA are the same architecture with different transformer weights; only the
    # mains are labelled, and the turbo LoRAs are not variant-scoped.
    VariantFacet({ModelType.Main: MiniMaxH3VariantType}),
)
