"""What the sd-3 architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import SD3_16, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.variant import VariantFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, PiDDecoderVariantType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import SD3ConditioningInfo

register(
    BaseModelType.StableDiffusion3,
    LatentSpaceFacet(SD3_16),
    ConditioningFacet(SD3ConditioningInfo),
    # stable-diffusion-3.5-medium's example: 40 steps at guidance 4.5. Medium rather than Large
    # (28/3.5) because there is one `sd-3` row and no variant to tell them apart, and Medium is the
    # smaller, more commonly run model.
    DefaultSettingsFacet(
        {None: MainModelDefaultSettings(scheduler="euler_a", steps=40, cfg_scale=4.5, width=1024, height=1024)}
    ),
    ModalityFacet(frozenset({"txt2img", "img2img", "inpaint", "outpaint"}), metadata_slug="sd3"),
    FeaturesFacet(
        negative_prompt=NegativePrompt(visible=True, usage="always"),
        dimension_grid=16,
        guidance_label="CFG",
        scheduler_set="standard",
    ),
    # SD3 mains carry no variant; only its PiD decoder checkpoints do.
    VariantFacet({ModelType.PiDDecoder: PiDDecoderVariantType}),
)
