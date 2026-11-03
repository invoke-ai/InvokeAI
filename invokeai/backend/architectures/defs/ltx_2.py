"""What the ltx-2 architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import LTX2_128, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.variant import VariantFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType, LTX2VariantType, ModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import LTX2ConditioningInfo

register(
    BaseModelType.LTX2,
    LatentSpaceFacet(LTX2_128),
    ConditioningFacet(LTX2ConditioningInfo),
    # Dev samples a guided 30-step shifted schedule at video CFG 3.0; Distilled runs a fixed
    # 8-sigma schedule with guidance off, so cfg_scale 1.0 means "none". Both are the released
    # pipeline's values for this generation. 1248x704 is the 32-multiple 16:9 canvas a 121-frame
    # clip fits on one 48 GB card at.
    DefaultSettingsFacet(
        {
            LTX2VariantType.Distilled: MainModelDefaultSettings(steps=8, cfg_scale=1.0, width=1248, height=704),
            None: MainModelDefaultSettings(steps=30, cfg_scale=3.0, width=1248, height=704),
        }
    ),
    # Every conditioning shape the family supports, all with synchronized audio generated alongside:
    # text-to-video, a held first and/or last frame, continuing an existing clip, and the two
    # whole-modality modes -- a picture for a given soundtrack, or a soundtrack for a given picture.
    ModalityFacet(frozenset({"t2v", "i2v", "lf2v", "flf2v", "extend_video", "a2v", "v2a"}), metadata_slug="ltx2"),
    FeaturesFacet(
        # The negative prompt only reaches the model through classifier-free guidance, which the
        # distilled variant runs without.
        negative_prompt=NegativePrompt(visible=True, usage="cfg-gated"),
        # The VAE's 32x spatial compression at patch size 1.
        dimension_grid=32,
        guidance_label="CFG",
        # ltx2_denoise.cfg_scale is ge=1.0 with no ceiling; the audio, STG and modality scales are
        # separate fields with their own controls, not this slider.
        guidance_min=1.0,
    ),
    VariantFacet({ModelType.Main: LTX2VariantType}),
)
