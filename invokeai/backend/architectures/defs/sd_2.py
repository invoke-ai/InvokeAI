"""What the sd-2 architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import SD15_4, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.variant import VariantFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, ModelVariantType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import BasicConditioningInfo

# SD 2.x previews with the SD 1.x factors; same four-channel VAE.
register(
    BaseModelType.StableDiffusion2,
    LatentSpaceFacet(SD15_4),
    ConditioningFacet(BasicConditioningInfo),
    # 768 is right for the v-prediction checkpoints and wrong for the 512 `-base` ones, and
    # nothing here distinguishes them — SD 2.x has no variant modeled and we ship no starter
    # model for it. 768 is the deliberate choice of the two.
    DefaultSettingsFacet(
        {None: MainModelDefaultSettings(scheduler="euler_a", steps=30, cfg_scale=7.0, width=768, height=768)}
    ),
    ModalityFacet(frozenset({"txt2img", "img2img", "inpaint", "outpaint"})),
    FeaturesFacet(
        negative_prompt=NegativePrompt(visible=True, usage="always"),
        dimension_grid=8,
        guidance_label="CFG",
        scheduler_set="standard",
        scheduler_applies_to_graph=True,
        supports_regional_guidance=True,
        regional_negative=True,
        clip_skip_max=24,
        supports_seamless=True,
        supports_cfg_rescale=True,
        sd_vae_override=True,
        vae_precision=True,
    ),
    VariantFacet({ModelType.Main: ModelVariantType}),
)
