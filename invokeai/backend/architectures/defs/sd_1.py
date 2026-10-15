"""What the sd-1 architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import SD15_4, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.unet import UNetDownscaleFacet
from invokeai.backend.architectures.facets.variant import VariantFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, ModelVariantType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import BasicConditioningInfo

register(
    BaseModelType.StableDiffusion1,
    LatentSpaceFacet(SD15_4),
    UNetDownscaleFacet(max_unet_downscale=8),
    ConditioningFacet(BasicConditioningInfo),
    DefaultSettingsFacet(
        {None: MainModelDefaultSettings(scheduler="euler_a", steps=30, cfg_scale=7.0, width=512, height=512)}
    ),
    # SD 1.x and 2.x share the unprefixed mode strings: a bare `txt2img`.
    ModalityFacet(frozenset({"txt2img", "img2img", "inpaint", "outpaint"})),
    FeaturesFacet(
        negative_prompt=NegativePrompt(visible=True, usage="always"),
        dimension_grid=8,
        guidance_label="CFG",
        scheduler_set="standard",
        scheduler_applies_to_graph=True,
        control_kinds=frozenset({"controlnet", "t2i_adapter"}),
        max_reference_images=5,
        supports_regional_guidance=True,
        regional_negative=True,
        clip_skip_max=12,
        supports_seamless=True,
        supports_cfg_rescale=True,
        sd_vae_override=True,
        vae_precision=True,
    ),
    VariantFacet({ModelType.Main: ModelVariantType}),
)
